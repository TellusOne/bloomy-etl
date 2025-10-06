import json
import numpy as np
import xarray as xr
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ArcGISExporter:
    
    def __init__(self, nc_path: str, aoi_path: str = None):
        self.nc_path = Path(nc_path)
        self.aoi_path = Path(aoi_path) if aoi_path else None
        self.ds = None
        self.aoi_geom = None
        
    def load(self):
        logger.info(f"Loading {self.nc_path.name}")
        self.ds = xr.open_dataset(self.nc_path)
        
        if self.aoi_path and self.aoi_path.exists():
            with open(self.aoi_path, 'r') as f:
                data = json.load(f)
                self.aoi_geom = data['features'][0]['geometry'] if 'features' in data else data.get('geometry')
        
        logger.info(f"{len(self.ds.time)} timestamps, {self.ds.sizes['y']}x{self.ds.sizes['x']} pixels")
        logger.info(f"CRS: {self.ds.attrs.get('crs', 'NAO DEFINIDO')}")

    def export_geotiff(self, output_dir: Path, downsample=1):
        output_dir = Path(output_dir)
        geotiff_dir = output_dir / "geotiffs"
        geotiff_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exported {len(self.ds.time)} timestamps with GeoTIFF...")
        logger.info(f"Downsample: {downsample}x")
        
        import rasterio
        from rasterio.transform import Affine
        
        crs = self.ds.attrs.get('crs', 'EPSG:4326')
        transform = Affine(*self.ds.attrs['transform'])
        
        logger.info(f"CRS: {crs}")
        logger.info(f"Transform: {transform}")
        
        if downsample > 1:
            transform = transform * Affine.scale(downsample, downsample)
        
        dates = []
        
        for t_idx in range(len(self.ds.time)):
            date = str(self.ds.time.values[t_idx])[:10]
            dates.append(date)
            
            ndvi = self.ds.ndvi.isel(time=t_idx).values
            evi = self.ds.evi.isel(time=t_idx).values
            
            if downsample > 1:
                ndvi = ndvi[::downsample, ::downsample]
                evi = evi[::downsample, ::downsample]
            
            ny, nx = ndvi.shape
            
            output_path = geotiff_dir / f"{date}.tif"
            
            with rasterio.open(
                output_path, 'w',
                driver='GTiff',
                height=ny,
                width=nx,
                count=2,
                dtype=rasterio.float32,
                crs=crs,
                transform=transform,
                compress='lzw',
                nodata=-9999
            ) as dst:
                ndvi_out = np.where(np.isnan(ndvi), -9999, ndvi)
                evi_out = np.where(np.isnan(evi), -9999, evi)
                
                dst.write(ndvi_out.astype(np.float32), 1)
                dst.write(evi_out.astype(np.float32), 2)
                
                dst.set_band_description(1, 'NDVI')
                dst.set_band_description(2, 'EVI')
                
                dst.update_tags(1, date=date, variable='NDVI')
                dst.update_tags(2, date=date, variable='EVI')
            
            if (t_idx + 1) % 50 == 0:
                logger.info(f"Proccessed: {t_idx + 1}/{len(self.ds.time)}")
        
        logger.info(f"GeoTIFFs exported: {geotiff_dir}")
        
        bounds = self.ds.attrs.get('bounds', None)
        self._create_index(geotiff_dir, dates, crs, transform, ny, nx, bounds)
        
        self._create_style_file(output_dir)
        
        logger.info(f"Output: {output_dir.resolve()}")
        
    def _create_index(self, geotiff_dir, dates, crs, transform, height, width, bounds):
        

        
        if bounds is not None:
            west, south, east, north = bounds
            center_lon = (west + east) / 2
            center_lat = (south + north) / 2
        else:
            center_lon = transform.c + (width * transform.a) / 2
            center_lat = transform.f + (height * transform.e) / 2
            west = transform.c
            north = transform.f
            east = west + width * transform.a
            south = north + height * transform.e
        
        index = {
            "format": "GeoTIFF",
            "crs": str(crs),
            "transform": [transform.a, transform.b, transform.c, 
                         transform.d, transform.e, transform.f, 
                         0, 0, 1],
            "dimensions": {
                "height": int(height),
                "width": int(width)
            },
            "center": {
                "lon": float(center_lon),
                "lat": float(center_lat)
            },
            "bounds": {
                "west": float(west),
                "south": float(south),
                "east": float(east),
                "north": float(north)
            },
            "bands": [
                {"name": "NDVI", "description": "Normalized Difference Vegetation Index"},
                {"name": "EVI", "description": "Enhanced Vegetation Index"}
            ],
            "nodata": -9999,
            "dates": dates,
            "total_files": len(dates),
            "files": [f"{date}.tif" for date in dates]
        }
        
        with open(geotiff_dir.parent / "index.json", 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)
        
        logger.info(f"Index criado: {geotiff_dir.parent / 'index.json'}")
        logger.info(f"Centro: ({center_lon:.6f}, {center_lat:.6f})")

    def _create_style_file(self, output_dir):
        
        style = {
            "name": "NDVI_Style",
            "description": "Default NDVI color ramp",
            "type": "ClassBreaks",
            "field": "Band_1",
            "classes": [
                {"min": -0.2, "max": 0.0, "color": [139, 0, 0], "label": "Very Low"},
                {"min": 0.0, "max": 0.2, "color": [205, 133, 63], "label": "Low"},
                {"min": 0.2, "max": 0.4, "color": [255, 255, 0], "label": "Moderate"},
                {"min": 0.4, "max": 0.6, "color": [173, 255, 47], "label": "Moderate-High"},
                {"min": 0.6, "max": 0.8, "color": [34, 139, 34], "label": "High"},
                {"min": 0.8, "max": 1.0, "color": [0, 100, 0], "label": "Very High"}
            ]
        }
        
        with open(output_dir / "ndvi_style.json", 'w') as f:
            json.dump(style, f, indent=2)
        
        logger.info(f"Style created: {output_dir / 'ndvi_style.json'}")


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GeoTIFF Exporter for ArcGIS")
    parser.add_argument("--input", "-i", required=True, help="Input NetCDF file")
    parser.add_argument("--output", "-o", default="arcgis_export", help="Output directory")
    parser.add_argument("--downsample", type=int, default=1, help="Downsample factor")
    
    args = parser.parse_args()
    
    try:
        exporter = ArcGISExporter(args.input)
        exporter.load()
        exporter.export_geotiff(args.output, downsample=args.downsample)
        print(f"\Done! Find in: {Path(args.output).resolve()}")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()