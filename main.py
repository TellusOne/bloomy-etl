"""
CLI para pipeline HLS
"""
import argparse
import asyncio
from dotenv import load_dotenv

from core.pipeline import HLSPipeline
from config.settings import QualityThresholds

load_dotenv()

def main():
    parser = argparse.ArgumentParser(
        description="🛰️  Pipeline HLS (Sentinel-2) - NDVI/EVI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Obrigatórios
    required = parser.add_argument_group('obrigatórios')
    required.add_argument("--aoi", type=str, required=True, help="Arquivo GeoJSON da AOI")
    required.add_argument("--output", type=str, required=True, help="Arquivo NetCDF de saída")
    required.add_argument("--start", type=str, required=True, help="Data início (YYYY-MM-DD)")
    required.add_argument("--end", type=str, required=True, help="Data fim (YYYY-MM-DD)")
    
    # Básicos
    parser.add_argument("--cloud-cover", type=int, default=20, help="Cobertura máxima de nuvem %% (padrão: 20)")
    parser.add_argument("--batch-size", type=int, default=10, help="Granules simultâneos (padrão: 10)")
    parser.add_argument("--max-retries", type=int, default=3, help="Tentativas em erro (padrão: 3)")
    parser.add_argument("--no-merge", action="store_true", help="NÃO mesclar mesmo dia")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    
    # Avançados
    parser.add_argument("--keep-cache", action="store_true", help="Manter cache após processamento")
    parser.add_argument("--disable-quality-filter", action="store_true", help="Desabilitar filtro temporal")
    parser.add_argument("--no-detect-events", action="store_true", help="NÃO detectar eventos")
    parser.add_argument("--cache-dir", type=str, default=None, help="Diretório de cache customizado")
    parser.add_argument("--temp-dir", type=str, default=None, help="Diretório temporário customizado")
    
    args = parser.parse_args()
    
    # Criar pipeline
    pipeline = HLSPipeline(
        aoi_path=args.aoi,
        output_path=args.output,
        start_date=args.start,
        end_date=args.end,
        cloud_cover=args.cloud_cover,
        batch_size=args.batch_size,
        merge_same_day=not args.no_merge,
        max_retries=args.max_retries,
        log_level=args.log_level,
        cache_dir=args.cache_dir,
        temp_dir=args.temp_dir,
        keep_cache=args.keep_cache,
        disable_quality_filter=args.disable_quality_filter,
        detect_events=not args.no_detect_events
    )
    
    # Executar
    success = asyncio.run(pipeline.execute())
    exit(0 if success else 1)

if __name__ == "__main__":
    main()