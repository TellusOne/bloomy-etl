import logging
import numpy as np
import xarray as xr
from typing import List, Dict
from collections import defaultdict

from config.settings import DEFAULT_QUALITY

from config.settings import DEFAULT_QUALITY

class QualityFilter:
    
    def __init__(self, thresholds=None):
        self.logger = logging.getLogger(__name__)
        self.quality = thresholds or DEFAULT_QUALITY
    
    def filter_timestamps(self, datasets: List[xr.Dataset]) -> List[xr.Dataset]:
        
        if len(datasets) < 3:
            self.logger.info("Skipping quality filter - not enough timestamps")
            return datasets
        
        # Ordenar por data
        datasets = sorted(datasets, key=lambda ds: ds.time.values[0])
        
        filtered = []
        removed_dates = []
        
        for i, ds in enumerate(datasets):
            valid_pct = ds.attrs.get('valid_pixels_pct', 0)
            contamination = ds.attrs.get('contamination_pct', 0)
            
            ndvi_mean = float(np.nanmean(ds.ndvi.values))
            ndvi_std = float(np.nanstd(ds.ndvi.values))
            
            reject = False
            reason = ""
            
            if valid_pct < self.quality.valid_pct_min:
                reject = True
                reason = f"Less valid pixels ({valid_pct:.1f}%)"
            elif contamination > self.quality.contamination_high:
                reject = True
                reason = f"Contammination ({contamination:.1f}%)"
            
            elif contamination > self.quality.contamination_moderate or \
                 (ndvi_mean < self.quality.ndvi_low and ndvi_std > self.quality.ndvi_std_high):
                
                prev_ds = datasets[i-1] if i > 0 else None
                next_ds = datasets[i+1] if i < len(datasets)-1 else None
                
                neighbor_ndvis = []
                if prev_ds is not None:
                    neighbor_ndvis.append(float(np.nanmean(prev_ds.ndvi.values)))
                if next_ds is not None:
                    neighbor_ndvis.append(float(np.nanmean(next_ds.ndvi.values)))
                
                if neighbor_ndvis:
                    avg_neighbor_ndvi = np.mean(neighbor_ndvis)
                    ndvi_drop = avg_neighbor_ndvi - ndvi_mean
                    
                    if ndvi_drop > self.quality.ndvi_drop_threshold:
                        if next_ds is not None:
                            next_ndvi = float(np.nanmean(next_ds.ndvi.values))
                            recovery = next_ndvi - ndvi_mean
                            
                            if recovery > self.quality.recovery_threshold:
                                reject = True
                                reason = (
                                    f"temporal anomaly(NDVI: {ndvi_mean:.3f}, "
                                    f"down: {ndvi_drop:.3f}, recovery: {recovery:.3f})"
                                )
                            else:
                                self.logger.info(
                                    f"⚠️  {ds.attrs.get('date')}: NDVI slow ({ndvi_mean:.3f}) "
                                )
                        else:
                            self.logger.info(
                                f"⚠️  {ds.attrs.get('date')}: NDVI slow, but its unique neighbor ({ndvi_mean:.3f})"
                            )
                    else:
                        if contamination > 25:
                            reject = True
                            reason = f"Moderate contammination ({contamination:.1f}%) + NDVI strange ({ndvi_mean:.3f})"
            
            if reject:
                date_str = ds.attrs.get('date', 'unknown')
                removed_dates.append((date_str, reason))
                self.logger.warning(f"Clean {date_str}: {reason}")
            else:
                filtered.append(ds)
        
        if removed_dates:
            self.logger.info(f"Cleaned {len(removed_dates)} timestamps with low quality")
        
        return filtered


class EventDetector:
    
    def __init__(self, thresholds=None):
        self.logger = logging.getLogger(__name__)
        self.quality = thresholds or DEFAULT_QUALITY
    
    def detect_events(self, ds_combined: xr.Dataset) -> Dict:
        
        self.logger.info("Detect events on temporial sereies...")
        
        events = {
            "abrupt_drops": [],
            "sustained_changes": [],
            "anomalies": []
        }
        
        ndvi_series = []
        dates = []
        
        for i in range(len(ds_combined.time)):
            ndvi_mean = float(np.nanmean(ds_combined.ndvi.isel(time=i).values))
            date = np.datetime_as_string(ds_combined.time.values[i], unit='D')
            ndvi_series.append(ndvi_mean)
            dates.append(date)
        
        for i in range(1, len(ndvi_series)):
            diff = ndvi_series[i] - ndvi_series[i-1]
            
            if diff < self.quality.abrupt_drop:
                if i < len(ndvi_series) - 1:
                    recovery = ndvi_series[i+1] - ndvi_series[i]
                    
                    if recovery > self.quality.recovery_threshold:
                        events["anomalies"].append({
                            "date": dates[i],
                            "ndvi_before": ndvi_series[i-1],
                            "ndvi_during": ndvi_series[i],
                            "ndvi_after": ndvi_series[i+1],
                            "type": "transient_anomaly"
                        })
                    else:
                        events["abrupt_drops"].append({
                            "date": dates[i],
                            "ndvi_before": ndvi_series[i-1],
                            "ndvi_after": ndvi_series[i],
                            "drop": abs(diff),
                            "type": "possible_fire_or_harvest"
                        })
                else:
                    events["abrupt_drops"].append({
                        "date": dates[i],
                        "ndvi_before": ndvi_series[i-1],
                        "ndvi_after": ndvi_series[i],
                        "drop": abs(diff),
                        "type": "recent_event"
                    })
            
            if i >= 2:
                if all(v < self.quality.sustained_low for v in ndvi_series[i-2:i+1]):
                    if i >= 3 and ndvi_series[i-3] > self.quality.sustained_high_before:
                        events["sustained_changes"].append({
                            "start_date": dates[i-2],
                            "end_date": dates[i],
                            "ndvi_before": ndvi_series[i-3],
                            "ndvi_sustained": np.mean(ndvi_series[i-2:i+1]),
                            "type": "sustained_change"
                        })
        
        if events["abrupt_drops"]:
            self.logger.info(f"{len(events['abrupt_drops'])} abrupt drops detected:")
            for event in events["abrupt_drops"]:
                self.logger.info(
                    f"   {event['date']}: NDVI {event['ndvi_before']:.3f} → "
                    f"{event['ndvi_after']:.3f} (fall: {event['drop']:.3f})"
                )
        
        if events["sustained_changes"]:
            self.logger.info(f" {len(events['sustained_changes'])} sustain changes")
        
        if events["anomalies"]:
            self.logger.info(f"  {len(events['anomalies'])} sustain changes")
        
        return events