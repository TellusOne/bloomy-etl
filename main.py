"""
HLS Data ETL Pipeline
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from core.pipeline import HLSPipeline
from core.arcgis import ArcGISExporter

if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="HLS Data ETL Pipeline")
    
    parser.add_argument("--aoi", help="Area of Interest GeoJSON file")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", required=True, help="Output/Input NetCDF file")
    parser.add_argument("--cloud-cover", type=int, default=20, help="Maximum cloud cover (%)")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    parser.add_argument("--cache-dir", help="Cache directory")
    parser.add_argument("--keep-cache", action="store_true", help="Keep cache after processing")
    
    parser.add_argument("--only-export", action="store_true", help="ONLY export existing NetCDF (do not process)")
    parser.add_argument("--export-geotiff", action="store_true", help="Export GeoTIFFs after processing")
    parser.add_argument("--geotiff-dir", help="GeoTIFF output directory (default: NetCDF_path + '_geotiffs')")
    parser.add_argument("--downsample", type=int, default=1, help="Downsample factor for GeoTIFFs (1=original, 2=half)")
    
    args = parser.parse_args()
    
    # EXPORT-ONLY MODE
    if args.only_export:
        logger.info("="*70)
        logger.info("EXPORT GEOTIFFS (without processing)")
        logger.info("="*70)
        logger.info(f"Input: {args.output}")
        logger.info(f"Downsample: {args.downsample}x")
        logger.info("="*70 + "\n")
        
        if not Path(args.output).exists():
            logger.error(f"ERROR: NetCDF not found: {args.output}")
            sys.exit(1)
        
        try:
            if args.geotiff_dir:
                geotiff_dir = Path(args.geotiff_dir)
            else:
                output_path = Path(args.output)
                geotiff_dir = output_path.parent / f"{output_path.stem}_geotiffs"
            
            exporter = ArcGISExporter(args.output, args.aoi)
            exporter.load()
            exporter.export_geotiff(geotiff_dir, downsample=args.downsample)
            
            logger.info("\n" + "="*70)
            logger.info("GEOTIFFS SUCCESSFULLY EXPORTED!")
            logger.info(f"Location: {geotiff_dir.resolve()}")
            logger.info("="*70)
            
        except Exception as e:
            logger.error(f"\nERROR exporting GeoTIFFs: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        return
    
    # PROCESS MODE (normal)
    if not args.aoi or not args.start or not args.end:
        logger.error("ERROR: --aoi, --start and --end are required for processing!")
        logger.error("Use --only-export to export an existing NetCDF without processing")
        sys.exit(1)
    
    logger.info("="*70)
    logger.info("HLS DATA PIPELINE")
    logger.info("="*70)
    logger.info(f"AOI: {args.aoi}")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Max cloud cover: {args.cloud_cover}%")
    if args.export_geotiff:
        logger.info(f"Export GeoTIFF: YES (downsample={args.downsample}x)")
    logger.info("="*70 + "\n")
    
    # Create pipeline
    pipeline = HLSPipeline(
        aoi_path=args.aoi,
        output_path=args.output,
        start_date=args.start,
        end_date=args.end,
        cloud_cover=args.cloud_cover,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        keep_cache=args.keep_cache
    )
    
    success = asyncio.run(pipeline.execute())
    
    if success:
        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        
        # EXPORT GEOTIFF (if requested)
        if args.export_geotiff:
            logger.info("\n" + "="*70)
            logger.info("EXPORTING GEOTIFFS FOR ARCGIS")
            logger.info("="*70 + "\n")
            
            try:
                # Define output directory
                if args.geotiff_dir:
                    geotiff_dir = Path(args.geotiff_dir)
                else:
                    output_path = Path(args.output)
                    geotiff_dir = output_path.parent / f"{output_path.stem}_geotiffs"
                
                # Export
                exporter = ArcGISExporter(args.output, args.aoi)
                exporter.load()
                exporter.export_geotiff(geotiff_dir, downsample=args.downsample)
                
                logger.info("\n" + "="*70)
                logger.info("GEOTIFFS SUCCESSFULLY EXPORTED!")
                logger.info(f"Location: {geotiff_dir.resolve()}")
                logger.info("="*70)
                
            except Exception as e:
                logger.error(f"\nERROR exporting GeoTIFFs: {e}")
                import traceback
                traceback.print_exc()
        
    else:
        logger.error("\n" + "="*70)
        logger.error("PIPELINE FAILED!")
        logger.error("="*70)
        sys.exit(1)

if __name__ == "__main__":
    main()