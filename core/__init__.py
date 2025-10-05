"""
Componentes principais do pipeline
"""
from core.authenticator import Authenticator
from core.processor import GranuleProcessor
from core.searcher import GranuleSearcher
from core.quality import QualityFilter, EventDetector
from core.merger import DatasetMerger
from core.pipeline import HLSPipeline

__all__ = [
    'Authenticator',
    'GranuleProcessor',
    'GranuleSearcher',
    'QualityFilter',
    'EventDetector',
    'DatasetMerger',
    'HLSPipeline'
]