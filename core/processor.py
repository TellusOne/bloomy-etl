"""
Processamento de granules individuais
"""
import asyncio
import numpy as np
import xarray as xr
import rasterio
from pathlib import Path
from datetime import datetime
from typing import Optional
import earthaccess
from rasterio.mask import mask
from shapely.geometry import Polygon, mapping
from shapely.ops import transform as shapely_transform
from pyproj import Transformer
import logging

from config.settings import DEFAULT_QUALITY, DEFAULT_CONFIG

class GranuleProcessor:
    
    def __init__(
        self, 
        aoi_polygon, 
        temp_dir: Path, 
        cache_dir: Path,
        authenticator,  # 
        quality_thresholds=None,
        max_retries: int = 3
    ):
        self.logger = logging.getLogger(__name__)
        self.aoi_polygon = aoi_polygon
        self.temp_dir = temp_dir
        self.cache_dir = cache_dir
        self.authenticator = authenticator  
        self.quality = quality_thresholds or DEFAULT_QUALITY
        self.max_retries = max_retries
        self.config = DEFAULT_CONFIG
    
    async def process(
        self, 
        granule, 
        granule_idx: int, 
        total_granules: int,
        pbar: Optional = None
    ) -> Optional[xr.Dataset]:
        
        for attempt in range(self.max_retries):
            granule_date = None
            try:
                granule_date, tile_id = self._extract_metadata(granule)
                
                cache_file = self._get_cache_file(tile_id, granule_date)
                if cache_file.exists():
                    return self._load_from_cache(cache_file, pbar)
                
                if attempt == 0:
                    self.logger.debug(
                        f"[{granule_idx+1:02d}/{total_granules:02d}] "
                        f"{tile_id} {granule_date.date()} às {granule_date.strftime('%H:%M:%S')}"
                    )
                
                band_files = await self._download_bands(
                    granule, 
                    granule_idx
                )
                
                if not band_files:
                    if pbar:
                        pbar.update(1)
                    return None
                
                ds = self._process_rasters(
                    band_files, 
                    granule_date, 
                    tile_id, 
                    granule_idx
                )
                
                if ds is not None:
                    self._save_to_cache(ds, cache_file)
                    self._cleanup_temp_files(band_files)
                    if pbar:
                        pbar.update(1)
                
                return ds
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    self.logger.warning(f"Tried {attempt + 2}/{self.max_retries}: {e}")
                    await asyncio.sleep(2 ** attempt)
                else:
                    self.logger.error(f"Granule {granule_idx+1} fault: {e}")
                    if pbar:
                        pbar.update(1)
                    return None
    
    async def _download_bands(
        self, 
        granule, 
        granule_idx: int
    ) -> Optional[dict]:
        """Baixa bandas usando earthaccess.download()"""
        
        granule_links = earthaccess.results.DataGranule(granule, None).data_links()
        
        # Filtrar bandas necessárias
        required_bands = ["B02", "B04", "B08", "Fmask"]
        band_links = [
            link for link in granule_links 
            if any(f".{band}.tif" in link for band in required_bands)
        ]
        
        if len(band_links) < len(required_bands):
            self.logger.warning(
                f"⚠️  Granule {granule_idx+1} sem todas as bandas "
                f"({len(band_links)}/{len(required_bands)})"
            )
            return None
        
        self.logger.debug(f"   📥 Baixando {len(band_links)} bandas")
        
        # 🔥 Usar earthaccess.download() em thread separada (é síncrono)
        loop = asyncio.get_event_loop()
        try:
            downloaded_files = await loop.run_in_executor(
                None,
                self._download_with_earthaccess,
                band_links
            )
            
            if not downloaded_files or len(downloaded_files) < len(required_bands):
                self.logger.warning(f"⚠️  Download incompleto")
                return None
            
            # Organizar por banda
            band_files = {}
            for band in required_bands:
                found = next(
                    (str(f) for f in downloaded_files if f".{band}.tif" in str(f)), 
                    None
                )
                if found:
                    band_files[band] = found
            
            if len(band_files) != len(required_bands):
                self.logger.warning(f"⚠️  Bandas incompletas após download")
                return None
            
            return band_files
        
        except Exception as e:
            self.logger.error(f"❌ Erro ao baixar: {e}")
            return None
    
    def _download_with_earthaccess(self, links: list) -> list:
        try:
            self.authenticator.refresh_if_needed()
            downloaded = earthaccess.download(
                links,
                str(self.temp_dir)
            )
            
            return downloaded or []
        
        except Exception as e:
            self.logger.error(f"❌ earthaccess.download() fault: {e}")
            return []
    
    def _process_rasters(
        self, 
        band_files: dict, 
        granule_date: datetime, 
        tile_id: str, 
        granule_idx: int
    ) -> Optional[xr.Dataset]:
        
        try:
            from rasterio.warp import reproject, Resampling
            from rasterio.transform import from_bounds
            
            with rasterio.open(band_files["B04"]) as b4, \
                    rasterio.open(band_files["B08"]) as b8, \
                    rasterio.open(band_files["B02"]) as b2, \
                    rasterio.open(band_files["Fmask"]) as fm:
                
                source_crs = b4.crs.to_string()
                
                if not hasattr(self, '_grid_initialized'):
                    west, south, east, north = self.aoi_polygon.bounds
                    self.pixel_size = 0.00027  # ~30m em graus
                    self.width = int(np.ceil((east - west) / self.pixel_size))
                    self.height = int(np.ceil((north - south) / self.pixel_size))
                    self.grid_east = west + (self.width * self.pixel_size)
                    self.grid_north = south + (self.height * self.pixel_size)
                    self.dst_transform = from_bounds(west, south, self.grid_east, self.grid_north, self.width, self.height)
                    self.target_crs = "EPSG:4326"
                    self._grid_initialized = True
                    
                    self.logger.info(f"Grid WGS84: {self.height}x{self.width} pixels")
                    self.logger.info(f"Bounds: W={west:.6f} S={south:.6f} E={self.grid_east:.6f} N={self.grid_north:.6f}")
                
                self.logger.debug(f"   CRS: {source_crs} -> WGS84")
                
                red_wgs = np.full((self.height, self.width), np.nan, dtype=np.float32)
                nir_wgs = np.full((self.height, self.width), np.nan, dtype=np.float32)
                blue_wgs = np.full((self.height, self.width), np.nan, dtype=np.float32)
                fmask_wgs = np.full((self.height, self.width), 255, dtype=np.uint8)
                
                reproject(
                    source=rasterio.band(b4, 1),
                    destination=red_wgs,
                    src_transform=b4.transform,
                    src_crs=b4.crs,
                    dst_transform=self.dst_transform,
                    dst_crs=self.target_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=-9999,
                    dst_nodata=np.nan
                )
                
                reproject(
                    source=rasterio.band(b8, 1),
                    destination=nir_wgs,
                    src_transform=b8.transform,
                    src_crs=b8.crs,
                    dst_transform=self.dst_transform,
                    dst_crs=self.target_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=-9999,
                    dst_nodata=np.nan
                )
                
                reproject(
                    source=rasterio.band(b2, 1),
                    destination=blue_wgs,
                    src_transform=b2.transform,
                    src_crs=b2.crs,
                    dst_transform=self.dst_transform,
                    dst_crs=self.target_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=-9999,
                    dst_nodata=np.nan
                )
                
                reproject(
                    source=rasterio.band(fm, 1),
                    destination=fmask_wgs,
                    src_transform=fm.transform,
                    src_crs=fm.crs,
                    dst_transform=self.dst_transform,
                    dst_crs=self.target_crs,
                    resampling=Resampling.nearest,
                    src_nodata=255,
                    dst_nodata=255
                )
                
                red = red_wgs
                nir = nir_wgs
                blue = blue_wgs
                fmask = fmask_wgs
                
                outside_polygon = np.isnan(red) | np.isnan(nir) | np.isnan(blue)
                
                cloud_shadow_mask = np.isin(fmask, [2, 4]) & ~outside_polygon
                anomaly_low = ((red < self.quality.red_nir_low) | (nir < self.quality.red_nir_low)) & ~outside_polygon & ~np.isnan(red)
                anomaly_high = ((red > self.quality.red_nir_high) | (nir > self.quality.red_nir_high)) & ~outside_polygon & ~np.isnan(red)
                haze_mask = (blue > self.quality.blue_haze) & ~outside_polygon & ~np.isnan(blue)
                
                ndvi_prelim = (nir - red) / (nir + red + 1e-6)
                invalid_ndvi = ((ndvi_prelim > self.quality.ndvi_max) | (ndvi_prelim < self.quality.ndvi_min)) & ~outside_polygon
                
                mask_arr = outside_polygon | cloud_shadow_mask | anomaly_low | anomaly_high | haze_mask | invalid_ndvi
                
                total_pixels = mask_arr.size
                valid_pixels = total_pixels - np.sum(mask_arr)
                contamination_pct = (np.sum(haze_mask) + np.sum(anomaly_low | anomaly_high) + np.sum(invalid_ndvi)) / total_pixels * 100
                
                self.logger.debug(
                    f"   Mask: {valid_pixels}/{total_pixels} validos "
                    f"({valid_pixels/total_pixels*100:.1f}%) | "
                    f"Contamination: {contamination_pct:.1f}%"
                )
                
                if contamination_pct > self.quality.contamination_reject:
                    self.logger.warning(
                        f"Granule {granule_idx+1} with {contamination_pct:.1f}% "
                        f"Contamination"
                    )
                    return None
                
                if valid_pixels < total_pixels * (self.quality.valid_pixels_min / 100):
                    self.logger.warning(
                        f"Granule {granule_idx+1} with <{self.quality.valid_pixels_min}% "
                        f"Invalid pixels"
                    )
                    return None
                
                ndvi = (nir - red) / (nir + red + 1e-6)
                evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
                
                ndvi[mask_arr] = np.nan
                evi[mask_arr] = np.nan
                
                ndvi = np.clip(ndvi, -1.0, 1.0)
                evi = np.clip(evi, -3.0, 3.0)
                
                west, south, _, _ = self.aoi_polygon.bounds
                time_np = np.datetime64(granule_date.replace(tzinfo=None))
                ds = xr.Dataset(
                    {
                        "ndvi": (("time", "y", "x"), ndvi[np.newaxis, :, :]),
                        "evi": (("time", "y", "x"), evi[np.newaxis, :, :])
                    },
                    coords={"time": [time_np]},
                    attrs={
                        "crs": self.target_crs,
                        "transform": tuple(self.dst_transform)[:6],
                        "bounds": (west, south, self.grid_east, self.grid_north),
                        "date": granule_date.date().isoformat(),
                        "tile_id": tile_id,
                        "source_crs": source_crs,
                        "valid_pixels_pct": float(valid_pixels / total_pixels * 100),
                        "contamination_pct": float(contamination_pct)
                    }
                )
                
                return ds
        
        except Exception as e:
            self.logger.error(f"Error processing rasters: {e}")
            return None
    
    def _extract_metadata(self, granule) -> tuple:
        try:
            temporal_extent = granule['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
            granule_date = datetime.fromisoformat(temporal_extent.replace("Z", "+00:00"))
        except (KeyError, TypeError):
            temporal_extent = granule['umm']['TemporalExtent']['SingleDateTime']
            granule_date = datetime.fromisoformat(temporal_extent.replace("Z", "+00:00"))
        
        granule_name = granule['umm']['GranuleUR']
        tile_id = granule_name.split('.')[2]
        
        return granule_date, tile_id
    
    def _get_cache_file(self, tile_id: str, granule_date: datetime) -> Path:
        cache_key = f"{tile_id}_{granule_date.strftime('%Y%j')}"
        return self.cache_dir / f"{cache_key}.nc"
    
    def _load_from_cache(self, cache_file: Path, pbar) -> Optional[xr.Dataset]:
        self.logger.debug(f"Cache hit: {cache_file.name}")
        try:
            ds = xr.open_dataset(cache_file)
            if pbar:
                pbar.update(1)
            return ds
        except Exception as e:
            self.logger.warning(f"Cache wrong: {e}")
            cache_file.unlink()
            return None
    
    def _save_to_cache(self, ds: xr.Dataset, cache_file: Path):
        try:
            encoding = {
                var: {"zlib": True, "complevel": 5, "dtype": "float32"}
                for var in ds.data_vars
            }
            ds.to_netcdf(cache_file, encoding=encoding)
            self.logger.debug(f" Save at: {cache_file.name}")
        except Exception as e:
            self.logger.warning(f"Error when saved: {e}")
    
    def _cleanup_temp_files(self, band_files: dict):
        for file in band_files.values():
            try:
                Path(file).unlink()
            except:
                pass