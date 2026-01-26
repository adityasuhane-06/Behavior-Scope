"""
Missing Data Handler for Behavioral Analysis Pipeline.

Addresses critical reliability issue: missing frames currently treated as zeros,
biasing statistics downward and creating false negatives.

Key improvements:
1. Linear interpolation for short gaps (< 1 second)
2. Exclusion of long gaps (> 3 seconds) from statistics
3. Confidence weighting based on data completeness
4. Quality metrics reporting

Clinical rationale:
- Missing data is common due to occlusion, lighting, movement
- Treating missing as zero creates false "calm" periods
- Interpolation preserves behavioral continuity
- Quality weighting ensures reliable statistics

Engineering approach:
- Gap detection and classification
- Interpolation strategies by gap length
- Quality-aware statistical aggregation
- Metadata preservation for transparency
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
import warnings

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import zscore

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """
    Data quality assessment for a time series.
    
    Attributes:
        total_frames: Total expected frames
        valid_frames: Frames with valid detections
        missing_frames: Frames with missing data
        interpolated_frames: Frames filled by interpolation
        excluded_frames: Frames excluded from analysis
        completeness_ratio: Valid frames / total frames
        interpolation_ratio: Interpolated frames / total frames
        quality_score: Overall quality (0-1, higher = better)
        gap_analysis: Statistics about missing data gaps
    """
    total_frames: int
    valid_frames: int
    missing_frames: int
    interpolated_frames: int
    excluded_frames: int
    completeness_ratio: float
    interpolation_ratio: float
    quality_score: float
    gap_analysis: Dict[str, Union[int, float]]


class MissingDataHandler:
    """
    Handle missing data in behavioral analysis pipelines.
    
    Strategies:
    1. Short gaps (< 1s): Linear interpolation
    2. Medium gaps (1-3s): Weighted interpolation with uncertainty
    3. Long gaps (> 3s): Exclude from statistics
    4. Quality weighting: Weight statistics by data completeness
    
    Usage:
        handler = MissingDataHandler()
        cleaned_data, quality = handler.process_time_series(data, timestamps)
    """
    
    def __init__(
        self,
        short_gap_threshold: float = 1.0,
        long_gap_threshold: float = 3.0,
        min_quality_threshold: float = 0.5,
        interpolation_method: str = 'linear'
    ):
        """
        Initialize missing data handler.
        
        Args:
            short_gap_threshold: Threshold for short gaps (seconds)
            long_gap_threshold: Threshold for long gaps (seconds)
            min_quality_threshold: Minimum quality to proceed with analysis
            interpolation_method: Interpolation method ('linear', 'cubic', 'nearest')
        """
        self.short_gap_threshold = short_gap_threshold
        self.long_gap_threshold = long_gap_threshold
        self.min_quality_threshold = min_quality_threshold
        self.interpolation_method = interpolation_method
        
        logger.info(
            f"Missing data handler initialized: "
            f"short_gap={short_gap_threshold}s, long_gap={long_gap_threshold}s, "
            f"min_quality={min_quality_threshold}"
        )
    
    def process_feature_sequence(
        self,
        features: List,
        feature_name: str,
        timestamps: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, np.ndarray, DataQualityMetrics]:
        """
        Process a sequence of features with missing data handling.
        
        Args:
            features: List of feature objects (FaceFeatures, PoseFeatures, etc.)
            feature_name: Name of feature to extract
            timestamps: Optional timestamps (extracted from features if None)
            
        Returns:
            Tuple of (values, weights, quality_metrics)
        """
        if not features:
            logger.warning("Empty feature sequence provided")
            return np.array([]), np.array([]), self._empty_quality_metrics()
        
        # Extract timestamps if not provided
        if timestamps is None:
            timestamps = [f.timestamp for f in features]
        
        timestamps = np.array(timestamps)
        
        # Extract feature values and detection flags
        values, detection_flags = self._extract_feature_values(features, feature_name)
        
        # Detect missing data gaps
        gaps = self._detect_gaps(timestamps, detection_flags)
        
        # Process gaps according to strategy
        processed_values, weights = self._process_gaps(
            values, timestamps, detection_flags, gaps
        )
        
        # Compute quality metrics
        quality = self._compute_quality_metrics(
            len(timestamps), detection_flags, gaps, weights
        )
        
        logger.debug(
            f"Processed {feature_name}: {quality.completeness_ratio:.2%} complete, "
            f"{quality.interpolation_ratio:.2%} interpolated"
        )
        
        return processed_values, weights, quality
    
    def process_multiple_features(
        self,
        features: List,
        feature_names: List[str],
        timestamps: Optional[List[float]] = None
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, DataQualityMetrics]]:
        """
        Process multiple features simultaneously.
        
        Args:
            features: List of feature objects
            feature_names: List of feature names to extract
            timestamps: Optional timestamps
            
        Returns:
            Dictionary mapping feature names to (values, weights, quality)
        """
        results = {}
        
        for feature_name in feature_names:
            values, weights, quality = self.process_feature_sequence(
                features, feature_name, timestamps
            )
            results[feature_name] = (values, weights, quality)
        
        return results
    
    def _extract_feature_values(
        self,
        features: List,
        feature_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract feature values and detection flags.
        
        Returns:
            Tuple of (values, detection_flags)
        """
        values = []
        detection_flags = []
        
        for feature in features:
            # Check if feature was detected
            if hasattr(feature, 'face_detected'):
                detected = feature.face_detected
            elif hasattr(feature, 'pose_detected'):
                detected = feature.pose_detected
            else:
                detected = True  # Assume detected if no flag
            
            # Extract feature value
            if detected and hasattr(feature, feature_name):
                value = getattr(feature, feature_name)
                # Handle nested attributes (e.g., head_pose[0])
                if isinstance(value, (list, tuple)) and len(value) > 0:
                    value = value[0]  # Use first element
                values.append(float(value))
            else:
                values.append(np.nan)  # Missing value
            
            detection_flags.append(detected)
        
        return np.array(values), np.array(detection_flags)
    
    def _detect_gaps(
        self,
        timestamps: np.ndarray,
        detection_flags: np.ndarray
    ) -> List[Dict]:
        """
        Detect gaps in the data.
        
        Returns:
            List of gap dictionaries with start_idx, end_idx, duration
        """
        gaps = []
        in_gap = False
        gap_start = 0
        
        for i, detected in enumerate(detection_flags):
            if not detected and not in_gap:
                # Gap start
                gap_start = i
                in_gap = True
            elif detected and in_gap:
                # Gap end
                gap_duration = timestamps[i-1] - timestamps[gap_start] if i > gap_start else 0
                gaps.append({
                    'start_idx': gap_start,
                    'end_idx': i - 1,
                    'duration': gap_duration,
                    'type': self._classify_gap(gap_duration)
                })
                in_gap = False
        
        # Handle gap extending to end
        if in_gap:
            gap_duration = timestamps[-1] - timestamps[gap_start]
            gaps.append({
                'start_idx': gap_start,
                'end_idx': len(detection_flags) - 1,
                'duration': gap_duration,
                'type': self._classify_gap(gap_duration)
            })
        
        return gaps
    
    def _classify_gap(self, duration: float) -> str:
        """Classify gap by duration."""
        if duration < self.short_gap_threshold:
            return 'short'
        elif duration < self.long_gap_threshold:
            return 'medium'
        else:
            return 'long'
    
    def _process_gaps(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        detection_flags: np.ndarray,
        gaps: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process gaps according to strategy.
        
        Returns:
            Tuple of (processed_values, weights)
        """
        processed_values = values.copy()
        weights = np.ones_like(values, dtype=float)
        
        for gap in gaps:
            start_idx = gap['start_idx']
            end_idx = gap['end_idx']
            gap_type = gap['type']
            
            if gap_type == 'short':
                # Linear interpolation
                interpolated = self._interpolate_gap(
                    values, timestamps, start_idx, end_idx
                )
                processed_values[start_idx:end_idx+1] = interpolated
                # Reduce weight for interpolated values
                weights[start_idx:end_idx+1] = 0.7
                
            elif gap_type == 'medium':
                # Weighted interpolation with higher uncertainty
                interpolated = self._interpolate_gap(
                    values, timestamps, start_idx, end_idx
                )
                processed_values[start_idx:end_idx+1] = interpolated
                # Lower weight for medium gaps
                weights[start_idx:end_idx+1] = 0.4
                
            else:  # long gap
                # Exclude from analysis
                processed_values[start_idx:end_idx+1] = np.nan
                weights[start_idx:end_idx+1] = 0.0
        
        return processed_values, weights
    
    def _interpolate_gap(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        start_idx: int,
        end_idx: int
    ) -> np.ndarray:
        """
        Interpolate values for a gap.
        
        Uses surrounding valid values for interpolation.
        """
        # Find valid values before and after gap
        valid_before = None
        valid_after = None
        
        # Look backwards for valid value
        for i in range(start_idx - 1, -1, -1):
            if not np.isnan(values[i]):
                valid_before = i
                break
        
        # Look forwards for valid value
        for i in range(end_idx + 1, len(values)):
            if not np.isnan(values[i]):
                valid_after = i
                break
        
        gap_indices = np.arange(start_idx, end_idx + 1)
        
        if valid_before is not None and valid_after is not None:
            # Interpolate between valid points
            x_points = [timestamps[valid_before], timestamps[valid_after]]
            y_points = [values[valid_before], values[valid_after]]
            
            if self.interpolation_method == 'linear':
                interpolated = np.interp(
                    timestamps[gap_indices], x_points, y_points
                )
            else:
                # Use scipy for other methods
                f = interp1d(x_points, y_points, kind=self.interpolation_method)
                interpolated = f(timestamps[gap_indices])
                
        elif valid_before is not None:
            # Forward fill
            interpolated = np.full(len(gap_indices), values[valid_before])
            
        elif valid_after is not None:
            # Backward fill
            interpolated = np.full(len(gap_indices), values[valid_after])
            
        else:
            # No valid values - use zeros (will be weighted as 0)
            interpolated = np.zeros(len(gap_indices))
        
        return interpolated
    
    def _compute_quality_metrics(
        self,
        total_frames: int,
        detection_flags: np.ndarray,
        gaps: List[Dict],
        weights: np.ndarray
    ) -> DataQualityMetrics:
        """Compute data quality metrics."""
        
        valid_frames = np.sum(detection_flags)
        missing_frames = total_frames - valid_frames
        
        # Count interpolated and excluded frames
        interpolated_frames = np.sum((weights > 0) & (weights < 1.0))
        excluded_frames = np.sum(weights == 0.0)
        
        completeness_ratio = valid_frames / total_frames
        interpolation_ratio = interpolated_frames / total_frames
        
        # Overall quality score
        quality_score = (
            0.7 * completeness_ratio +
            0.2 * (1.0 - interpolation_ratio) +
            0.1 * (1.0 - excluded_frames / total_frames)
        )
        
        # Gap analysis
        gap_analysis = {
            'num_gaps': len(gaps),
            'short_gaps': len([g for g in gaps if g['type'] == 'short']),
            'medium_gaps': len([g for g in gaps if g['type'] == 'medium']),
            'long_gaps': len([g for g in gaps if g['type'] == 'long']),
            'total_gap_duration': sum(g['duration'] for g in gaps),
            'max_gap_duration': max([g['duration'] for g in gaps]) if gaps else 0.0
        }
        
        return DataQualityMetrics(
            total_frames=total_frames,
            valid_frames=int(valid_frames),
            missing_frames=int(missing_frames),
            interpolated_frames=int(interpolated_frames),
            excluded_frames=int(excluded_frames),
            completeness_ratio=float(completeness_ratio),
            interpolation_ratio=float(interpolation_ratio),
            quality_score=float(quality_score),
            gap_analysis=gap_analysis
        )
    
    def _empty_quality_metrics(self) -> DataQualityMetrics:
        """Return empty quality metrics for error cases."""
        return DataQualityMetrics(
            total_frames=0,
            valid_frames=0,
            missing_frames=0,
            interpolated_frames=0,
            excluded_frames=0,
            completeness_ratio=0.0,
            interpolation_ratio=0.0,
            quality_score=0.0,
            gap_analysis={}
        )


def compute_weighted_statistics(
    values: np.ndarray,
    weights: np.ndarray,
    exclude_nan: bool = True
) -> Dict[str, float]:
    """
    Compute weighted statistics for quality-aware aggregation.
    
    Args:
        values: Array of values
        weights: Array of weights (0-1)
        exclude_nan: Whether to exclude NaN values
        
    Returns:
        Dictionary with weighted statistics
    """
    if exclude_nan:
        valid_mask = ~np.isnan(values) & (weights > 0)
        values = values[valid_mask]
        weights = weights[valid_mask]
    
    if len(values) == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
            'effective_n': 0.0
        }
    
    # Normalize weights
    weights = weights / (np.sum(weights) + 1e-8)
    
    # Weighted statistics
    weighted_mean = np.sum(values * weights)
    weighted_var = np.sum(weights * (values - weighted_mean) ** 2)
    weighted_std = np.sqrt(weighted_var)
    
    # Effective sample size
    effective_n = 1.0 / np.sum(weights ** 2)
    
    return {
        'mean': float(weighted_mean),
        'std': float(weighted_std),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),  # Unweighted median
        'effective_n': float(effective_n)
    }


def assess_data_quality(
    features: List,
    feature_names: List[str]
) -> Dict[str, DataQualityMetrics]:
    """
    Assess data quality for multiple features.
    
    Args:
        features: List of feature objects
        feature_names: List of feature names to assess
        
    Returns:
        Dictionary mapping feature names to quality metrics
    """
    handler = MissingDataHandler()
    quality_results = {}
    
    for feature_name in feature_names:
        _, _, quality = handler.process_feature_sequence(features, feature_name)
        quality_results[feature_name] = quality
    
    return quality_results


def generate_quality_report(
    quality_metrics: Dict[str, DataQualityMetrics]
) -> str:
    """
    Generate human-readable quality report.
    
    Args:
        quality_metrics: Dictionary of quality metrics by feature
        
    Returns:
        Formatted quality report string
    """
    report = ["DATA QUALITY REPORT", "=" * 50]
    
    overall_quality = np.mean([q.quality_score for q in quality_metrics.values()])
    report.append(f"Overall Quality Score: {overall_quality:.2%}")
    report.append("")
    
    for feature_name, quality in quality_metrics.items():
        report.append(f"Feature: {feature_name}")
        report.append(f"  Completeness: {quality.completeness_ratio:.2%}")
        report.append(f"  Interpolated: {quality.interpolation_ratio:.2%}")
        report.append(f"  Quality Score: {quality.quality_score:.2%}")
        
        if quality.gap_analysis:
            gaps = quality.gap_analysis
            report.append(f"  Gaps: {gaps['num_gaps']} total "
                         f"({gaps['short_gaps']} short, {gaps['medium_gaps']} medium, "
                         f"{gaps['long_gaps']} long)")
        report.append("")
    
    return "\n".join(report)