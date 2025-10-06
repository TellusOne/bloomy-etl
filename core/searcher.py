import logging
from earthaccess import DataGranules
from shapely.geometry import shape
import json

class GranuleSearcher:
    
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
        
        self.logger.info(
            f"({start_date} - {end_date}, cloud_cover <= {cloud_cover}%)"
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
                self.logger.error("Doesn't found any granules")
                return False
            
            self.logger.info(f"✅ {len(self.granules)} granules found")
            return True
        
        except Exception as e:
            self.logger.error(f"Error when search {e}")
            return False
    
    def get_granules(self):
        return self.granules
    
    def get_granule_count(self) -> int:
        return len(self.granules)