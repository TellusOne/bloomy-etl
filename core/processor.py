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
    """Processa granules individuais"""
    
    def __init__(
        self, 
        aoi_polygon, 
        temp_dir: Path, 
        cache_dir: Path,
        authenticator,  # 🔥 Adicionar authenticator
        quality_thresholds=None,
        max_retries: int = 3
    ):
        self.logger = logging.getLogger(__name__)
        self.aoi_polygon = aoi_polygon
        self.temp_dir = temp_dir
        self.cache_dir = cache_dir
        self.authenticator = authenticator  # 🔥 Guardar referência
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
        """Processa um granule (sem session)"""
        
        for attempt in range(self.max_retries):
            granule_date = None
            try:
                # Extrair metadados
                granule_date, tile_id = self._extract_metadata(granule)
                
                # Verificar cache
                cache_file = self._get_cache_file(tile_id, granule_date)
                if cache_file.exists():
                    return self._load_from_cache(cache_file, pbar)
                
                if attempt == 0:
                    self.logger.debug(
                        f"[{granule_idx+1:02d}/{total_granules:02d}] "
                        f"{tile_id} {granule_date.date()} às {granule_date.strftime('%H:%M:%S')}"
                    )
                
                # Baixar bandas usando earthaccess
                band_files = await self._download_bands(
                    granule, 
                    granule_idx
                )
                
                if not band_files:
                    if pbar:
                        pbar.update(1)
                    return None
                
                # Processar rasters
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
                    self.logger.warning(f"⚠️  Tentativa {attempt + 2}/{self.max_retries}: {e}")
                    await asyncio.sleep(2 ** attempt)
                else:
                    self.logger.error(f"❌ Granule {granule_idx+1} falhou: {e}")
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
        """Download síncrono com earthaccess (thread-safe)"""
        try:
            # Renovar token se necessário
            self.authenticator.refresh_if_needed()
            
            # Download com sessão autenticada
            downloaded = earthaccess.download(
                links,
                str(self.temp_dir)
            )
            
            return downloaded or []
        
        except Exception as e:
            self.logger.error(f"❌ earthaccess.download() falhou: {e}")
            return []
    
    def _process_rasters(
        self, 
        band_files: dict, 
        granule_date: datetime, 
        tile_id: str, 
        granule_idx: int
    ) -> Optional[xr.Dataset]:
        """Processa rasters e calcula índices"""
        
        try:
            with rasterio.open(band_files["B04"]) as b4, \
                rasterio.open(band_files["B08"]) as b8, \
                rasterio.open(band_files["B02"]) as b2, \
                rasterio.open(band_files["Fmask"]) as fm:
                
                # Transformar AOI
                transformer = Transformer.from_crs(
                    "EPSG:4326", 
                    b4.crs.to_string(), 
                    always_xy=True
                )
                aoi_transformed = shapely_transform(transformer.transform, self.aoi_polygon)
                
                # Verificar interseção
                raster_bounds = Polygon.from_bounds(*b4.bounds)
                if not raster_bounds.intersects(aoi_transformed):
                    self.logger.debug(f"⚠️  Granule {granule_idx+1} não intersecta AOI")
                    return None
                
                aoi_geojson_transformed = mapping(aoi_transformed)
                
                # Ler região da AOI
                red, out_transform = mask(b4, [aoi_geojson_transformed], crop=True, all_touched=False, filled=False)
                nir, _ = mask(b8, [aoi_geojson_transformed], crop=True, all_touched=False, filled=False)
                blue, _ = mask(b2, [aoi_geojson_transformed], crop=True, all_touched=False, filled=False)
                fmask, _ = mask(fm, [aoi_geojson_transformed], crop=True, all_touched=False, filled=False)
                
                # Processar bandas
                red = red[0].astype("float32")
                nir = nir[0].astype("float32")
                blue = blue[0].astype("float32")
                fmask = fmask[0].astype("uint8")
                
                # Pixels fora do polígono
                outside_polygon = (red == -9999) | (nir == -9999) | (blue == -9999)
                red[outside_polygon] = np.nan
                nir[outside_polygon] = np.nan
                blue[outside_polygon] = np.nan
                
                # Filtros de qualidade
                cloud_shadow_mask = np.isin(fmask, [2, 4]) & ~outside_polygon
                anomaly_low = ((red < self.quality.red_nir_low) | (nir < self.quality.red_nir_low)) & ~outside_polygon & ~np.isnan(red)
                anomaly_high = ((red > self.quality.red_nir_high) | (nir > self.quality.red_nir_high)) & ~outside_polygon & ~np.isnan(red)
                haze_mask = (blue > self.quality.blue_haze) & ~outside_polygon & ~np.isnan(blue)
                
                ndvi_prelim = (nir - red) / (nir + red + 1e-6)
                invalid_ndvi = ((ndvi_prelim > self.quality.ndvi_max) | (ndvi_prelim < self.quality.ndvi_min)) & ~outside_polygon
                
                # Máscara combinada
                mask_arr = outside_polygon | cloud_shadow_mask | anomaly_low | anomaly_high | haze_mask | invalid_ndvi
                
                # Estatísticas
                total_pixels = mask_arr.size
                valid_pixels = total_pixels - np.sum(mask_arr)
                contamination_pct = (np.sum(haze_mask) + np.sum(anomaly_low | anomaly_high) + np.sum(invalid_ndvi)) / total_pixels * 100
                
                self.logger.debug(
                    f"   Máscara: {valid_pixels}/{total_pixels} válidos "
                    f"({valid_pixels/total_pixels*100:.1f}%) | "
                    f"Contaminação: {contamination_pct:.1f}%"
                )
                
                # Rejeitar se muita contaminação
                if contamination_pct > self.quality.contamination_reject:
                    self.logger.warning(
                        f"⚠️  Granule {granule_idx+1} com {contamination_pct:.1f}% "
                        f"contaminação - pulando"
                    )
                    return None
                
                # Rejeitar se poucos pixels válidos
                if valid_pixels < total_pixels * (self.quality.valid_pixels_min / 100):
                    self.logger.warning(
                        f"⚠️  Granule {granule_idx+1} com <{self.quality.valid_pixels_min}% "
                        f"pixels válidos - pulando"
                    )
                    return None
                
                # Calcular índices
                ndvi = (nir - red) / (nir + red + 1e-6)
                evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
                
                ndvi[mask_arr] = np.nan
                evi[mask_arr] = np.nan
                
                ndvi = np.clip(ndvi, -1.0, 1.0)
                evi = np.clip(evi, -3.0, 3.0)
                
                # Criar dataset
                time_np = np.datetime64(granule_date.replace(tzinfo=None))
                ds = xr.Dataset(
                    {
                        "ndvi": (("time", "y", "x"), ndvi[np.newaxis, :, :]),
                        "evi": (("time", "y", "x"), evi[np.newaxis, :, :])
                    },
                    coords={"time": [time_np]},
                    attrs={
                        "crs": b4.crs.to_string(),
                        "transform": tuple(out_transform),
                        "date": granule_date.date().isoformat(),
                        "tile_id": tile_id,
                        "valid_pixels_pct": float(valid_pixels / total_pixels * 100),
                        "contamination_pct": float(contamination_pct)
                    }
                )
                
                return ds
        
        except Exception as e:
            self.logger.error(f"❌ Erro ao processar rasters do granule {granule_idx+1}: {e}")
            return None
    
    # ... métodos auxiliares permanecem iguais ...
    def _extract_metadata(self, granule) -> tuple:
        """Extrai data e tile ID do granule"""
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
        """Retorna path do arquivo de cache"""
        cache_key = f"{tile_id}_{granule_date.strftime('%Y%j')}"
        return self.cache_dir / f"{cache_key}.nc"
    
    def _load_from_cache(self, cache_file: Path, pbar) -> Optional[xr.Dataset]:
        """Carrega dataset do cache"""
        self.logger.debug(f"   📦 Cache hit: {cache_file.name}")
        try:
            ds = xr.open_dataset(cache_file)
            if pbar:
                pbar.update(1)
            return ds
        except Exception as e:
            self.logger.warning(f"⚠️  Cache corrompido, reprocessando: {e}")
            cache_file.unlink()
            return None
    
    def _save_to_cache(self, ds: xr.Dataset, cache_file: Path):
        """Salva dataset no cache"""
        try:
            encoding = {
                var: {"zlib": True, "complevel": 5, "dtype": "float32"}
                for var in ds.data_vars
            }
            ds.to_netcdf(cache_file, encoding=encoding)
            self.logger.debug(f"   💾 Salvo no cache: {cache_file.name}")
        except Exception as e:
            self.logger.warning(f"⚠️  Erro ao salvar cache: {e}")
    
    def _cleanup_temp_files(self, band_files: dict):
        """Remove arquivos temporários"""
        for file in band_files.values():
            try:
                Path(file).unlink()
            except:
                pass