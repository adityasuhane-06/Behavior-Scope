"""
Temporal aggregation of visual features.

Clinical rationale:
- Individual frames are noisy → aggregate over time windows
- Behavioral patterns emerge at temporal scales (3-10 seconds)
- Sliding windows capture transitions and sustained states

Engineering approach:
- Statistical aggregation (mean, std, max, percentiles)
- Sliding window analysis (overlap for smooth tracking)
- Robust to missing data (handle undetected frames)
- Temporal derivatives (rate of change indicators)

Statistical measures:
- Mean: Average behavior level
- Std: Variability (instability indicator)
- Max: Peak intensity (dysregulation spikes)
- 95th percentile: Sustained high levels (robust to outliers)
- Gradient: Rate of change (escalation/de-escalation)
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Callable, Union, Tuple
import warnings

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AggregatedFeatures:
    """
    Aggregated features over a time window.
    
    Attributes:
        window_start_time: Window start in seconds
        window_end_time: Window end in seconds
        num_frames: Number of frames in window
        face_features: Dict of aggregated face features
        pose_features: Dict of aggregated pose features
    """
    window_start_time: float
    window_end_time: float
    num_frames: int
    face_features: Dict[str, float]
    pose_features: Dict[str, float]


class TemporalAggregator:
    """
    Aggregate visual features over temporal windows.
    
    Features computed:
    - Mean, std, max, min, median
    - 95th percentile (robust peak)
    - Temporal gradient (rate of change)
    - Missing data ratio
    
    Usage:
        aggregator = TemporalAggregator(window_duration=5.0, hop_duration=2.5)
        aggregated = aggregator.aggregate(face_features, pose_features, fps=5.0)
    """
    
    def __init__(
        self,
        window_duration: float = 5.0,
        hop_duration: float = 2.5,
        aggregation_functions: List[str] = None
    ):
        """
        Initialize temporal aggregator.
        
        Args:
            window_duration: Window size in seconds
            hop_duration: Hop size in seconds (overlap = window - hop)
            aggregation_functions: List of aggregation functions to apply
                                  Options: 'mean', 'std', 'max', 'min', 'median',
                                          'percentile_95', 'gradient'
        """
        self.window_duration = window_duration
        self.hop_duration = hop_duration
        
        if aggregation_functions is None:
            self.aggregation_functions = ['mean', 'std', 'max', 'percentile_95']
        else:
            self.aggregation_functions = aggregation_functions
        
        logger.info(
            f"Temporal aggregator initialized: window={window_duration}s, "
            f"hop={hop_duration}s, functions={self.aggregation_functions}"
        )
    
    def aggregate(
        self,
        face_features: List,
        pose_features: List,
        fps: float
    ) -> List[AggregatedFeatures]:
        """
        Aggregate features over sliding windows.
        
        Args:
            face_features: List of FaceFeatures objects
            pose_features: List of PoseFeatures objects
            fps: Frames per second
            
        Returns:
            List of AggregatedFeatures (one per window)
        """
        if not face_features or not pose_features:
            logger.warning("Empty feature lists provided")
            return []
        
        # Compute number of windows
        duration = max(face_features[-1].timestamp, pose_features[-1].timestamp)
        num_windows = int((duration - self.window_duration) / self.hop_duration) + 1
        
        logger.info(f"Aggregating over {num_windows} windows ({duration:.1f}s total)")
        
        aggregated_list = []
        
        for i in range(num_windows):
            window_start = i * self.hop_duration
            window_end = window_start + self.window_duration
            
            # Extract features in window
            window_face_features = [
                f for f in face_features
                if window_start <= f.timestamp < window_end
            ]
            
            window_pose_features = [
                f for f in pose_features
                if window_start <= f.timestamp < window_end
            ]
            
            # Aggregate
            agg_face = self._aggregate_face_features(window_face_features)
            agg_pose = self._aggregate_pose_features(window_pose_features)
            
            aggregated = AggregatedFeatures(
                window_start_time=window_start,
                window_end_time=window_end,
                num_frames=len(window_face_features),
                face_features=agg_face,
                pose_features=agg_pose
            )
            
            aggregated_list.append(aggregated)
        
        return aggregated_list
    
    def _aggregate_face_features(self, features: List) -> Dict[str, float]:
        """
        Aggregate face features over window.
        
        Features to aggregate:
        - head_pose (yaw, pitch, roll separately)
        - facial_motion_energy
        - gaze_proxy
        - landmark_confidence
        """
        if not features:
            return self._empty_face_dict()
        
        # Filter to detected faces
        detected = [f for f in features if f.face_detected]
        
        if not detected:
            return self._empty_face_dict()
        
        # Extract feature arrays
        yaws = np.array([f.head_pose[0] for f in detected])
        pitches = np.array([f.head_pose[1] for f in detected])
        rolls = np.array([f.head_pose[2] for f in detected])
        motion_energies = np.array([f.facial_motion_energy for f in detected])
        gaze_proxies = np.array([f.gaze_proxy for f in detected])
        confidences = np.array([f.landmark_confidence for f in detected])
        
        # Aggregate each feature
        agg = {}
        
        for feature_name, values in [
            ('head_yaw', yaws),
            ('head_pitch', pitches),
            ('head_roll', rolls),
            ('facial_motion_energy', motion_energies),
            ('gaze_proxy', gaze_proxies),
            ('landmark_confidence', confidences)
        ]:
            agg.update(self._apply_aggregation_functions(feature_name, values))
        
        # Add detection rate
        agg['face_detection_rate'] = len(detected) / len(features)
        
        return agg
    
    def _aggregate_pose_features(self, features: List) -> Dict[str, float]:
        """
        Aggregate pose features over window.
        
        Features to aggregate:
        - upper_body_motion
        - hand_velocity (left, right, max)
        - shoulder_stability
        - posture_angle
        - visibility_score
        """
        if not features:
            return self._empty_pose_dict()
        
        # Filter to detected poses
        detected = [f for f in features if f.pose_detected]
        
        if not detected:
            return self._empty_pose_dict()
        
        # Extract feature arrays
        upper_body_motions = np.array([f.upper_body_motion for f in detected])
        hand_vels_left = np.array([f.hand_velocity_left for f in detected])
        hand_vels_right = np.array([f.hand_velocity_right for f in detected])
        hand_vels_max = np.maximum(hand_vels_left, hand_vels_right)
        shoulder_stabilities = np.array([f.shoulder_stability for f in detected])
        posture_angles = np.array([f.posture_angle for f in detected])
        visibilities = np.array([f.visibility_score for f in detected])
        
        # Aggregate each feature
        agg = {}
        
        for feature_name, values in [
            ('upper_body_motion', upper_body_motions),
            ('hand_velocity_left', hand_vels_left),
            ('hand_velocity_right', hand_vels_right),
            ('hand_velocity_max', hand_vels_max),
            ('shoulder_stability', shoulder_stabilities),
            ('posture_angle', posture_angles),
            ('visibility_score', visibilities)
        ]:
            agg.update(self._apply_aggregation_functions(feature_name, values))
        
        # Add detection rate
        agg['pose_detection_rate'] = len(detected) / len(features)
        
        return agg
    
    def _apply_aggregation_functions(
        self,
        feature_name: str,
        values: np.ndarray
    ) -> Dict[str, float]:
        """
        Apply all aggregation functions to a feature.
        
        Returns dict with keys like: 'feature_name_mean', 'feature_name_std', etc.
        """
        if len(values) == 0:
            # No data - return zeros
            return {f"{feature_name}_{func}": 0.0 for func in self.aggregation_functions}
        
        agg_dict = {}
        
        for func_name in self.aggregation_functions:
            if func_name == 'mean':
                value = np.mean(values)
            elif func_name == 'std':
                value = np.std(values)
            elif func_name == 'max':
                value = np.max(values)
            elif func_name == 'min':
                value = np.min(values)
            elif func_name == 'median':
                value = np.median(values)
            elif func_name == 'percentile_95':
                value = np.percentile(values, 95)
            elif func_name == 'gradient':
                # Rate of change (first - last) / duration
                value = (values[-1] - values[0]) / len(values) if len(values) > 1 else 0.0
            else:
                logger.warning(f"Unknown aggregation function: {func_name}")
                value = 0.0
            
            agg_dict[f"{feature_name}_{func_name}"] = float(value)
        
        return agg_dict
    
    def _empty_face_dict(self) -> Dict[str, float]:
        """Return empty face feature dict (all zeros)."""
        features = ['head_yaw', 'head_pitch', 'head_roll', 
                   'facial_motion_energy', 'gaze_proxy', 'landmark_confidence']
        
        agg_dict = {}
        for feature in features:
            for func in self.aggregation_functions:
                agg_dict[f"{feature}_{func}"] = 0.0
        agg_dict['face_detection_rate'] = 0.0
        
        return agg_dict
    
    def _empty_pose_dict(self) -> Dict[str, float]:
        """Return empty pose feature dict (all zeros)."""
        features = ['upper_body_motion', 'hand_velocity_left', 'hand_velocity_right',
                   'hand_velocity_max', 'shoulder_stability', 'posture_angle', 'visibility_score']
        
        agg_dict = {}
        for feature in features:
            for func in self.aggregation_functions:
                agg_dict[f"{feature}_{func}"] = 0.0
        agg_dict['pose_detection_rate'] = 0.0
        
        return agg_dict


def aggregate_features(
    face_features: List,
    pose_features: List,
    fps: float,
    window_duration: float = 5.0,
    hop_duration: float = 2.5
) -> List[AggregatedFeatures]:
    """
    Convenience function for feature aggregation.
    
    Args:
        face_features: List of FaceFeatures
        pose_features: List of PoseFeatures
        fps: Frames per second
        window_duration: Window size in seconds
        hop_duration: Hop size in seconds
        
    Returns:
        List of AggregatedFeatures
    """
    aggregator = TemporalAggregator(
        window_duration=window_duration,
        hop_duration=hop_duration
    )
    
    return aggregator.aggregate(face_features, pose_features, fps)


def compute_sliding_window_stats(
    values: np.ndarray,
    window_size: int,
    stat_func: Callable[[np.ndarray], float]
) -> np.ndarray:
    """
    Compute sliding window statistics.
    
    Args:
        values: 1D array of values
        window_size: Window size in samples
        stat_func: Function to compute statistic (e.g., np.mean, np.std)
        
    Returns:
        Array of windowed statistics
    """
    if len(values) < window_size:
        logger.warning(f"Values length ({len(values)}) < window size ({window_size})")
        return np.array([stat_func(values)])
    
    num_windows = len(values) - window_size + 1
    stats = []
    
    for i in range(num_windows):
        window = values[i:i+window_size]
        stat = stat_func(window)
        stats.append(stat)
    
    return np.array(stats)


def detect_motion_bursts(
    motion_values: np.ndarray,
    threshold_multiplier: float = 2.0,
    min_burst_duration: int = 3
) -> List[Tuple[int, int]]:
    """
    Detect bursts of high motion activity.
    
    Clinical interpretation: Sudden movement increases may indicate
    dysregulation episodes or agitation.
    
    Args:
        motion_values: Array of motion energy values
        threshold_multiplier: Multiplier for mean (threshold = mean * multiplier)
        min_burst_duration: Minimum consecutive frames to count as burst
        
    Returns:
        List of (start_idx, end_idx) tuples for detected bursts
    """
    if len(motion_values) == 0:
        return []
    
    # Compute threshold
    threshold = np.mean(motion_values) * threshold_multiplier
    
    # Find frames above threshold
    above_threshold = motion_values > threshold
    
    # Find continuous bursts
    bursts = []
    in_burst = False
    burst_start = 0
    
    for i, is_above in enumerate(above_threshold):
        if is_above and not in_burst:
            # Burst start
            burst_start = i
            in_burst = True
        elif not is_above and in_burst:
            # Burst end
            burst_duration = i - burst_start
            if burst_duration >= min_burst_duration:
                bursts.append((burst_start, i - 1))
            in_burst = False
    
    # Handle case where burst extends to end
    if in_burst:
        burst_duration = len(motion_values) - burst_start
        if burst_duration >= min_burst_duration:
            bursts.append((burst_start, len(motion_values) - 1))
    
    logger.info(f"Detected {len(bursts)} motion bursts (threshold={threshold:.3f})")
    
    return bursts
