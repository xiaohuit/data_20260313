"""
Financial Data Pipeline — public API surface.
"""
from financial_pipeline.config import CONFIG, DataCategory, FRED_SERIES
from financial_pipeline.loader.pit_loader import PiTDataLoader, MarketSnapshot

__all__ = ["CONFIG", "DataCategory", "FRED_SERIES", "PiTDataLoader", "MarketSnapshot"]
