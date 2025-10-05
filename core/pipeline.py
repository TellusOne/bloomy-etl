"""
Orquestrador principal do pipeline HLS
"""
import asyncio
import aiohttp
import logging
import json
from pathlib import Path
from tqdm import tqdm
import xarray as xr
from shapely.geometry import shape

from core.authenticator import Authenticator
from core.searcher import GranuleSearcher
from core.processor import GranuleProcessor
from core.quality import QualityFilter, EventDetector
from core.merger import DatasetMerger
from utils.logger import setup_logger
from config.settings import QualityThresholds

class HLSPipeline:
    """Pipeline completo de processamento HLS"""
    
    def __init__(
        self,
        aoi_path: str,
        output_path: str,
        start_date: str,
        end_date: str,
        cloud_cover: int = 20,
        batch_size: int = 10,
        merge_same_day: bool = True,
        max_retries: int = 3,
        log_level: str = "INFO",
        quality_thresholds: QualityThresholds = None,
        cache_dir: str = None,
        temp_dir: str = None,
        keep_cache: bool = False,
        disable_quality_filter: bool = False,
        detect_events: bool = True
    ):
        self.aoi_path = Path(aoi_path)
        self.output_path = Path(output_path)
        self.start_date = start_date
        self.end_date = end_date
        self.cloud_cover = cloud_cover
        self.batch_size = batch_size
        self.merge_same_day = merge_same_day
        self.max_retries = max_retries
        self.keep_cache = keep_cache
        self.disable_quality_filter = disable_quality_filter
        self.detect_events_flag = detect_events
        
        # Setup logging
        log_file = self.output_path.parent / "pipeline.log"
        self.logger = setup_logger("HLSPipeline", log_level, log_file)
        
        # Diretórios
        self.cache_dir = Path(cache_dir) if cache_dir else self.output_path.parent / "cache"
        self.temp_dir = Path(temp_dir) if temp_dir else self.output_path.parent / "temp"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Componentes
        self.authenticator = Authenticator()
        self.searcher = None
        self.processor = None
        self.quality_filter = QualityFilter(quality_thresholds)
        self.event_detector = EventDetector(quality_thresholds)
        self.merger = DatasetMerger(merge_same_day)
        
        # Estado
        self.aoi_polygon = None
        self.granules = []
        self.stats = {"success": 0, "failed": 0, "skipped": 0}
    
    def load_aoi(self) -> bool:
        """Carrega área de interesse"""
        self.logger.info(f"📍 Carregando AOI de: {self.aoi_path}")
        
        try:
            with open(self.aoi_path, 'r') as f:
                aoi_geojson = json.load(f)
            
            if aoi_geojson['type'] == 'FeatureCollection':
                self.aoi_polygon = shape(aoi_geojson['features'][0]['geometry'])
            else:
                self.aoi_polygon = shape(aoi_geojson['geometry'] if 'geometry' in aoi_geojson else aoi_geojson)
            
            self.logger.info(f"✅ AOI carregada: {self.aoi_polygon.bounds}")
            return True
        
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar AOI: {e}")
            return False
    
    async def run_pipeline(self) -> list:
        """Processa todos os granules"""
        self.logger.info("🚀 Iniciando processamento em lotes...")
        
        all_datasets = []
        
        with tqdm(total=len(self.granules), desc="Processando granules", unit="granule") as pbar:
            # 🔥 SEM aiohttp.ClientSession
            for batch_idx in range(0, len(self.granules), self.batch_size):
                # Renovar token se necessário
                self.authenticator.refresh_if_needed()
                
                batch = self.granules[batch_idx:batch_idx + self.batch_size]
                batch_num = batch_idx // self.batch_size + 1
                total_batches = (len(self.granules) + self.batch_size - 1) // self.batch_size
                
                self.logger.info(f"📦 Lote {batch_num}/{total_batches} ({len(batch)} granules)")
                
                # Processar batch (SEM session)
                tasks = [
                    self.processor.process(
                        granule,
                        batch_idx + i,
                        len(self.granules),
                        pbar
                    )
                    for i, granule in enumerate(batch)
                ]
                
                batch_results = await asyncio.gather(*tasks)
                batch_datasets = [ds for ds in batch_results if ds is not None]
                all_datasets.extend(batch_datasets)
                
                self.logger.debug(f"✅ Lote {batch_num}: {len(batch_datasets)}/{len(batch)} sucesso")
                
                await asyncio.sleep(1)
        
        return all_datasets
    
    async def execute(self):
        """Executa pipeline completo"""
        try:
            # 1. Autenticar
            if not self.authenticator.login():
                return False
            
            # 2. Carregar AOI
            if not self.load_aoi():
                return False
            
            # 3. Buscar granules
            self.searcher = GranuleSearcher(self.authenticator.auth)
            if not self.searcher.search(self.aoi_polygon, self.start_date, self.end_date, self.cloud_cover):
                return False
            
            self.granules = self.searcher.get_granules()
            
            # 4. Inicializar processador
            self.processor = GranuleProcessor(
                self.aoi_polygon,
                self.temp_dir,
                self.cache_dir,
                self.authenticator,  # 🔥 Passar authenticator
                max_retries=self.max_retries
            )
            
            # 5. Processar granules
            all_datasets = await self.run_pipeline()
            
            if not all_datasets:
                self.logger.error("❌ Nenhum dataset processado")
                return False
            
            # 6. Filtrar qualidade
            if not self.disable_quality_filter:
                self.logger.info("🔍 Filtrando timestamps de baixa qualidade...")
                all_datasets = self.quality_filter.filter_timestamps(all_datasets)
                
                if not all_datasets:
                    self.logger.error("❌ Todos datasets filtrados")
                    return False
            
            # 7. Mesclar
            ds_combined = self.merger.merge_all(all_datasets)
            
            # 8. Detectar eventos
            if self.detect_events_flag:
                events = self.event_detector.detect_events(ds_combined)
                events_file = self.output_path.parent / f"{self.output_path.stem}_events.json"
                with open(events_file, 'w') as f:
                    json.dump(events, f, indent=2, default=str)
                self.logger.info(f"📋 Eventos salvos: {events_file}")
            
            # 9. Salvar
            self.logger.info(f"💾 Salvando dataset: {self.output_path}")
            encoding = {var: {"zlib": True, "complevel": 5} for var in ds_combined.data_vars}
            ds_combined.to_netcdf(self.output_path, encoding=encoding)
            self.logger.info(f"✅ Dataset salvo: {self.output_path}")
            
            # 10. Cleanup
            if not self.keep_cache:
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                self.logger.info("🧹 Arquivos temporários removidos")
            
            # 11. Stats
            self.logger.info(f"\n📊 Processamento completo:")
            self.logger.info(f"📅 Timestamps: {len(ds_combined.time)}")
            self.logger.info(f"📏 Dimensões: {ds_combined.sizes['y']} x {ds_combined.sizes['x']} pixels")
            
            return True
        
        except Exception as e:
            self.logger.error(f"❌ Erro fatal: {e}", exc_info=True)
            return False