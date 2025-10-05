"""
Configurações e constantes do pipeline
"""
from dataclasses import dataclass
from datetime import timedelta

@dataclass
class QualityThresholds:
    """Thresholds de qualidade para filtros"""
    # Granule-level
    contamination_reject: float = 30.0
    valid_pixels_min: float = 10.0
    
    # Pixel-level
    red_nir_low: int = 100
    red_nir_high: int = 10000
    blue_haze: int = 1500
    ndvi_max: float = 0.95
    ndvi_min: float = -0.5
    
    # Temporal filtering
    valid_pct_min: float = 20.0
    contamination_high: float = 30.0
    contamination_moderate: float = 20.0
    ndvi_low: float = 0.15
    ndvi_std_high: float = 0.3
    ndvi_drop_threshold: float = 0.3
    recovery_threshold: float = 0.2
    
    # Event detection
    abrupt_drop: float = -0.3
    sustained_low: float = 0.3
    sustained_high_before: float = 0.5

@dataclass
class HLSConfig:
    """Configuração geral do pipeline"""
    concept_id: str = "C2021957295-LPCLOUD"  # HLS S30 v2.0
    required_bands: list = None
    token_expiry_hours: float = 1.5
    token_refresh_margin: timedelta = timedelta(minutes=10)
    
    def __post_init__(self):
        if self.required_bands is None:
            self.required_bands = ["B02", "B04", "B08", "Fmask"]

# Instâncias padrão
DEFAULT_QUALITY = QualityThresholds()
DEFAULT_CONFIG = HLSConfig()