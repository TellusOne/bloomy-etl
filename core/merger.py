
import logging
import numpy as np
import xarray as xr
from typing import List
from collections import defaultdict
from datetime import datetime

class DatasetMerger:
    
    def __init__(self, merge_same_day: bool = True):
        self.logger = logging.getLogger(__name__)
        self.merge_same_day = merge_same_day
        self.stats = {"merged": 0}
    
    def merge_spatial_tiles(self, datasets: List[xr.Dataset]) -> List[xr.Dataset]:
        
        by_time = defaultdict(list)
        for ds in datasets:
            timestamp = str(ds.time.values[0])
            by_time[timestamp].append(ds)
        
        merged_datasets = []
        
        for timestamp, time_datasets in sorted(by_time.items()):
            if len(time_datasets) == 1:
                merged_datasets.append(time_datasets[0])
            else:
                tiles = [ds.attrs.get('tile_id', 'unknown') for ds in time_datasets]
                self.logger.debug(
                    f"📐 {timestamp[:10]}: Merging {len(time_datasets)} tiles: {tiles}"
                )
                
                try:
                    merged = xr.combine_by_coords(
                        time_datasets,
                        combine_attrs='drop_conflicts'
                    )
                    
                    merged = merged.mean(dim='time', skipna=True)
                    time_val = time_datasets[0].time.values[0]
                    merged = merged.expand_dims(time=[time_val])
                    
                    merged.attrs = time_datasets[0].attrs.copy()
                    merged.attrs['tile_id'] = '+'.join(tiles)
                    merged.attrs['num_tiles_merged'] = len(time_datasets)
                    
                    merged_datasets.append(merged)
                
                except Exception as e:
                    self.logger.warning(f"⚠️  Error when merging tiles: {e}")
                    merged_datasets.append(time_datasets[0])
        
        return merged_datasets
    
    def merge_temporal(self, datasets: List[xr.Dataset]) -> List[xr.Dataset]:
        
        if not self.merge_same_day:
            return datasets
        
        self.logger.info("Merging granules of the same day...")
        
        by_date = defaultdict(list)
        for ds in datasets:
            date_str = ds.attrs['date']
            by_date[date_str].append(ds)
        
        merged_datasets = []
        for date, date_datasets in sorted(by_date.items()):
            if len(date_datasets) == 1:
                merged_datasets.append(date_datasets[0])
            else:
                self.logger.debug(f"📅 {date}: {len(date_datasets)} granules → average")
                self.stats["merged"] += len(date_datasets) - 1
                
                merged = xr.concat(date_datasets, dim="time").mean(dim="time", skipna=True)
                
                timestamps = [ds.time.values[0] for ds in date_datasets]
                timestamps_ns = [ts.astype('datetime64[ns]').astype('int64') for ts in timestamps]
                mean_timestamp = np.datetime64(int(np.mean(timestamps_ns)), 'ns')
                
                merged = merged.expand_dims(time=[mean_timestamp])
                
                merged.attrs = date_datasets[0].attrs.copy()
                merged.attrs.update({
                    'valid_pixels_pct': float(np.mean([ds.attrs.get('valid_pixels_pct', 0) for ds in date_datasets])),
                    'num_granules_merged': len(date_datasets)
                })
                
                merged_datasets.append(merged)
        
        self.logger.info(f"✅ {len(by_date)} unique dates ({self.stats['merged']} granules merged)")
        return merged_datasets
    
    def merge_all(self, datasets: List[xr.Dataset]) -> xr.Dataset:
        
        self.logger.info("Verifyng spatial merge...")
        datasets = self.merge_spatial_tiles(datasets)
        
        datasets = self.merge_temporal(datasets)
        
        self.logger.info(f"💾 Concat {len(datasets)} datasets...")
        ds_combined = xr.concat(datasets, dim="time")
        ds_combined = ds_combined.sortby("time")
        
        # Metadados globais
        ds_combined.attrs.update({
            'processing_date': datetime.now().isoformat(),
            'total_timestamps': len(ds_combined.time),
            'date_range': f"{ds_combined.time.values[0]} to {ds_combined.time.values[-1]}"
        })
        
        return ds_combined