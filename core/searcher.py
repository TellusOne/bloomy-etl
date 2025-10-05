"""
Busca de granules no NASA EarthData
"""
import logging
from earthaccess import DataGranules
from shapely.geometry import shape
import json

class GranuleSearcher:
    """Gerencia busca de granules"""
    
    def __init__(self, auth, concept_id: str = "C2021957295-LPCLOUD"):
        self.logger = logging.getLogger(__name__)
        self.auth = auth
        self.concept_id = concept_id
        self.granules = []
    
    def search(
        self, 
        aoi_polygon, 
        start_date: str, 
        end_date: str, 
        cloud_cover: int = 20
    ) -> bool:
        """Busca granules na AOI e período temporal"""
        
        self.logger.info(
            f"🔍 Buscando granules HLS S30 v2.0 "
            f"({start_date} até {end_date}, cloud_cover <= {cloud_cover}%)"
        )
        
        try:
            query = (
                DataGranules(self.auth)
                .concept_id(self.concept_id)
                .temporal(start_date, end_date)
                .polygon(aoi_polygon.exterior.coords)
                .cloud_cover(max_cover=cloud_cover)
                .day_night_flag("day")
            )
            
            self.granules = query.get_all()
            
            if not self.granules:
                self.logger.error("❌ Nenhum granule encontrado com os critérios especificados")
                return False
            
            self.logger.info(f"✅ {len(self.granules)} granules encontrados")
            return True
        
        except Exception as e:
            self.logger.error(f"❌ Erro na busca: {e}")
            return False
    
    def get_granules(self):
        """Retorna lista de granules"""
        return self.granules
    
    def get_granule_count(self) -> int:
        """Retorna número de granules encontrados"""
        return len(self.granules)