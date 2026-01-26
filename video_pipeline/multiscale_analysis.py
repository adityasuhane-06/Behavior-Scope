"""
Multi-Scale Temporal Analysis for Behavioral Metrics.

Analyzes behavioral patterns across multiple temporal scales:
1. Short-term (1-3 seconds): Micro-expressions, quick movements, immediate reactions
2. Medium-term (5-10 seconds): Behavioral episodes, regulation patterns
3. Long-term (30-60 seconds): Sustained states, trends, overall patterns

CLINICAL RATIONALE:
- Different behavioral patterns emerge at different time scales
- Micro-expressions occur in milliseconds to seconds
- Behavioral episodes (dysregulation, engagement) last 5-15 seconds
- Sustained states and trends require longer observation periods
- Cross-scale consistency indicates robust behavioral patterns

TECHNICAL APPROACH:
- Multi-resolution temporal windows
- Scale-specific feature extraction
- Cross-scale correlation analysis
- Hierarchical pattern detection
- Adaptive baseline estimation across scales

IMPROVEMENTS OVER SINGLE-SCALE:
- Captures both transient and sustained patterns
- Reduces false positives from single-frame artifacts
- Provides richer behavioral characterization
- Enables detection of complex temporal dynamics
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
import warnings

import numpy as np
from scipy import signal
from scipy.stats import pearsonr, zscore
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ScaleAnalysisResult:
    """
    Analysis result for a single temporal scale.
    
    Attributes:
        scale_name: Name of the temporal scale
        window_duration: Duration of analysis window (seconds)
        num_windows: Number of windows analyzed
        features: Dictionary of scale-specific features
        patterns: Detected patterns at this scale
        stability_score: Temporal stability at this scale (0-1)
        variability_score: Variability/dynamics at this scale (0-1)
        quality_score: Data quality at this scale (0-1)
    """
    scale_name: str
    window_duration: float
    num_windows: int
    features: Dict[str, np.ndarray]
    patterns: List[Dict]
    stability_score: float
    variability_score: float
    quality_score: float


@dataclass
class MultiScaleAnalysisResult:
    """
    Complete multi-scale analysis result.
    
    Attributes:
        short_scale: Short-term analysis (1-3s)
        medium_scale: Medium-term analysis (5-10s)
        long_scale: Long-term analysis (30-60s)
        cross_scale_correlations: Correlations between scales
        adaptive_baseline: Adaptive baseline computed across scales
        consistency_features: Features for regulation consistency analysis
        dominant_patterns: Most prominent patterns across all scales
        overall_stability: Overall temporal stability score
    """
    short_scale: ScaleAnalysisResult
    medium_scale: ScaleAnalysisResult
    long_scale: ScaleAnalysisResult
    cross_scale_correlations: Dict[str, float] = field(default_factory=dict)
    adaptive_baseline: Dict[str, float] = field(default_factory=dict)
    consistency_features: Dict[str, float] = field(default_factory=dict)
    dominant_patterns: List[Dict] = field(default_factory=list)
    overall_stability: float = 0.0
    clinical_significance: str = ""


@dataclass
class MultiScaleFeatures:
    """
    Multi-scale features for a single time point.
    
    Attributes:
        timestamp: Time point
        short_term: Short-term features
        medium_term: Medium-term features
        long_term: Long-term features
        cross_scale_consistency: Consistency across scales
        dominant_scale: Scale with strongest signal
        confidence: Overall confidence in measurements
    """
    timestamp: float
    short_term: Dict[str, float] = field(default_factory=dict)
    medium_term: Dict[str, float] = field(default_factory=dict)
    long_term: Dict[str, float] = field(default_factory=dict)
    cross_scale_consistency: float = 0.0
    dominant_scale: str = ""
    confidence: float = 0.0


@dataclass
class BehavioralPattern:
    """
    Detected behavioral pattern.
    
    Attributes:
        start_time: Pattern start time
        end_time: Pattern end time
        pattern_type: Type of pattern
        primary_scale: Primary temporal scale
        intensity: Pattern intensity
        consistency: Pattern consistency
        contributing_features: Features that contributed to detection
        clinical_significance: Clinical interpretation
    """
    start_time: float
    end_time: float
    pattern_type: str
    primary_scale: str
    intensity: float
    consistency: float
    contributing_features: Dict[str, float] = field(default_factory=dict)
    clinical_significance: str = ""


class MultiScaleAnalyzer:
    """
    Multi-scale temporal analyzer for behavioral pattern detection.
    
    Analyzes behavioral features at three temporal scales:
    1. Short-term (1-3 seconds): Immediate reactions, micro-expressions
    2. Medium-term (5-10 seconds): Behavioral episodes, transitions
    3. Long-term (30-60 seconds): Sustained states, overall patterns
    
    Usage:
        analyzer = MultiScaleAnalyzer()
        patterns = analyzer.analyze(face_features, pose_features, fps=5.0)
    """
    
    def __init__(
        self,
        short_window_duration: float = 2.0,
        medium_window_duration: float = 7.5,
        long_window_duration: float = 45.0,
        consistency_threshold: float = 0.6,
        pattern_threshold: float = 0.7
    ):
        """
        Initialize multi-scale analyzer.
        
        Args:
            short_window_duration: Short-term window size in seconds
            medium_window_duration: Medium-term window size in seconds
            long_window_duration: Long-term window size in seconds
            consistency_threshold: Minimum cross-scale consistency
            pattern_threshold: Minimum intensity for pattern detection
        """
        self.short_window = short_window_duration
        self.medium_window = medium_window_duration
        self.long_window = long_window_duration
        self.consistency_threshold = consistency_threshold
        self.pattern_threshold = pattern_threshold
        
        # Feature extractors for each scale
        self.feature_extractors = {
            'short': self._extract_short_term_features,
            'medium': self._extract_medium_term_features,
            'long': self._extract_long_term_features
        }
        
        logger.info(
            f"Multi-scale analyzer initialized: "
            f"windows=({short_window_duration}s, {medium_window_duration}s, {long_window_duration}s)"
        )
    
    def analyze(
        self,
        face_features: List,
        pose_features: List,
        fps: float = 5.0
    ) -> Tuple[List[MultiScaleFeatures], List[BehavioralPattern]]:
        """
        Perform multi-scale temporal analysis.
        
        Args:
            face_features: List of face feature objects
            pose_features: List of pose feature objects
            fps: Frames per second
            
        Returns:
            Tuple of (multi_scale_features, detected_patterns)
        """
        if not face_features or not pose_features:
            logger.warning("Empty feature lists provided")
            return [], []
        
        logger.info(f"Starting multi-scale analysis of {len(face_features)} frames")
        
        # Extract time series for analysis
        time_series = self._extract_time_series(face_features, pose_features)
        
        # Analyze at each scale
        multi_scale_features = self._compute_multi_scale_features(time_series)
        
        # Detect behavioral patterns
        patterns = self._detect_behavioral_patterns(multi_scale_features)
        
        logger.info(
            f"Multi-scale analysis complete: "
            f"{len(multi_scale_features)} time points, {len(patterns)} patterns detected"
        )
        
        return multi_scale_features, patterns
    
    def _extract_time_series(self, face_features: List, pose_features: List) -> Dict:
        """
        Extract time series data from face and pose features.
        
        Returns unified time series with all relevant behavioral signals.
        """
        time_series = {
            'timestamps': [],
            'face_motion': [],
            'head_yaw': [],
            'head_pitch': [],
            'head_roll': [],
            'gaze_proxy': [],
            'upper_body_motion': [],
            'hand_velocity_left': [],
            'hand_velocity_right': [],
            'posture_angle': [],
            'face_detected': [],
            'pose_detected': []
        }
        
        # Align face and pose features by timestamp
        aligned_features = self._align_features(face_features, pose_features)
        
        for face_feat, pose_feat in aligned_features:
            if face_feat is not None:
                time_series['timestamps'].append(face_feat.timestamp)
                time_series['face_motion'].append(face_feat.facial_motion_energy)
                time_series['head_yaw'].append(face_feat.head_pose[0])
                time_series['head_pitch'].append(face_feat.head_pose[1])
                time_series['head_roll'].append(face_feat.head_pose[2])
                time_series['gaze_proxy'].append(face_feat.gaze_proxy)
                time_series['face_detected'].append(face_feat.face_detected)
            else:
                # Handle missing face data
                time_series['timestamps'].append(pose_feat.timestamp if pose_feat else 0.0)
                time_series['face_motion'].append(0.0)
                time_series['head_yaw'].append(0.0)
                time_series['head_pitch'].append(0.0)
                time_series['head_roll'].append(0.0)
                time_series['gaze_proxy'].append(0.0)
                time_series['face_detected'].append(False)
            
            if pose_feat is not None:
                time_series['upper_body_motion'].append(pose_feat.upper_body_motion)
                time_series['hand_velocity_left'].append(pose_feat.hand_velocity_left)
                time_series['hand_velocity_right'].append(pose_feat.hand_velocity_right)
                time_series['posture_angle'].append(pose_feat.posture_angle)
                time_series['pose_detected'].append(pose_feat.pose_detected)
            else:
                # Handle missing pose data
                time_series['upper_body_motion'].append(0.0)
                time_series['hand_velocity_left'].append(0.0)
                time_series['hand_velocity_right'].append(0.0)
                time_series['posture_angle'].append(0.0)
                time_series['pose_detected'].append(False)
        
        # Convert to numpy arrays
        for key in time_series:
            time_series[key] = np.array(time_series[key])
        
        return time_series
    
    def _align_features(self, face_features: List, pose_features: List) -> List[Tuple]:
        """
        Align face and pose features by timestamp.
        
        Returns list of (face_feat, pose_feat) tuples aligned by time.
        """
        aligned = []
        
        # Create timestamp-indexed dictionaries
        face_by_time = {f.timestamp: f for f in face_features}
        pose_by_time = {f.timestamp: f for f in pose_features}
        
        # Get all unique timestamps
        all_timestamps = sorted(set(face_by_time.keys()) | set(pose_by_time.keys()))
        
        for timestamp in all_timestamps:
            face_feat = face_by_time.get(timestamp)
            pose_feat = pose_by_time.get(timestamp)
            aligned.append((face_feat, pose_feat))
        
        return aligned
    
    def _compute_multi_scale_features(self, time_series: Dict) -> List[MultiScaleFeatures]:
        """
        Compute features at multiple temporal scales for each time point.
        """
        timestamps = time_series['timestamps']
        multi_scale_features = []
        
        for i, timestamp in enumerate(timestamps):
            # Extract features at each scale
            short_features = self._extract_features_at_scale(
                time_series, i, self.short_window, 'short'
            )
            medium_features = self._extract_features_at_scale(
                time_series, i, self.medium_window, 'medium'
            )
            long_features = self._extract_features_at_scale(
                time_series, i, self.long_window, 'long'
            )
            
            # Compute cross-scale consistency
            consistency = self._compute_cross_scale_consistency(
                short_features, medium_features, long_features
            )
            
            # Determine dominant scale
            dominant_scale = self._determine_dominant_scale(
                short_features, medium_features, long_features
            )
            
            # Compute overall confidence
            confidence = self._compute_confidence(
                time_series, i, short_features, medium_features, long_features
            )
            
            ms_features = MultiScaleFeatures(
                timestamp=timestamp,
                short_term=short_features,
                medium_term=medium_features,
                long_term=long_features,
                cross_scale_consistency=consistency,
                dominant_scale=dominant_scale,
                confidence=confidence
            )
            
            multi_scale_features.append(ms_features)
        
        return multi_scale_features
    
    def _extract_features_at_scale(
        self,
        time_series: Dict,
        center_idx: int,
        window_duration: float,
        scale: str
    ) -> Dict[str, float]:
        """
        Extract features within a temporal window around center_idx.
        """
        timestamps = time_series['timestamps']
        center_time = timestamps[center_idx]
        
        # Find window boundaries
        start_time = center_time - window_duration / 2
        end_time = center_time + window_duration / 2
        
        # Find indices within window
        window_mask = (timestamps >= start_time) & (timestamps <= end_time)
        window_indices = np.where(window_mask)[0]
        
        if len(window_indices) < 2:
            # Insufficient data for this scale
            return self._empty_features_dict()
        
        # Extract features using scale-specific method
        extractor = self.feature_extractors[scale]
        return extractor(time_series, window_indices)
    
    def _extract_short_term_features(
        self,
        time_series: Dict,
        window_indices: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract short-term features (1-3 seconds).
        
        Focus on:
        - Rapid changes (derivatives)
        - Peak detection
        - Micro-expressions
        - Immediate reactions
        """
        features = {}
        
        # Motion dynamics (focus on rapid changes)
        face_motion = time_series['face_motion'][window_indices]
        body_motion = time_series['upper_body_motion'][window_indices]
        
        # Rapid change indicators (high-frequency components)
        features['face_motion_peak'] = np.max(face_motion) if len(face_motion) > 0 else 0.0
        features['face_motion_derivative'] = np.std(np.diff(face_motion)) if len(face_motion) > 1 else 0.0
        features['body_motion_peak'] = np.max(body_motion) if len(body_motion) > 0 else 0.0
        features['body_motion_derivative'] = np.std(np.diff(body_motion)) if len(body_motion) > 1 else 0.0
        
        # Head movement dynamics (quick turns, nods)
        head_yaw = time_series['head_yaw'][window_indices]
        head_pitch = time_series['head_pitch'][window_indices]
        
        features['head_yaw_range'] = np.ptp(head_yaw) if len(head_yaw) > 0 else 0.0
        features['head_pitch_range'] = np.ptp(head_pitch) if len(head_pitch) > 0 else 0.0
        features['head_movement_speed'] = np.mean(np.abs(np.diff(head_yaw))) if len(head_yaw) > 1 else 0.0
        
        # Gaze shifts (rapid attention changes)
        gaze_proxy = time_series['gaze_proxy'][window_indices]
        features['gaze_shift_intensity'] = np.std(gaze_proxy) if len(gaze_proxy) > 0 else 0.0
        features['gaze_peak'] = np.max(gaze_proxy) if len(gaze_proxy) > 0 else 0.0
        
        # Hand movement bursts (fidgeting, gestures)
        hand_left = time_series['hand_velocity_left'][window_indices]
        hand_right = time_series['hand_velocity_right'][window_indices]
        
        features['hand_burst_left'] = np.max(hand_left) if len(hand_left) > 0 else 0.0
        features['hand_burst_right'] = np.max(hand_right) if len(hand_right) > 0 else 0.0
        features['hand_asymmetry'] = abs(np.mean(hand_left) - np.mean(hand_right)) if len(hand_left) > 0 else 0.0
        
        return features
    
    def _extract_medium_term_features(
        self,
        time_series: Dict,
        window_indices: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract medium-term features (5-10 seconds).
        
        Focus on:
        - Behavioral episodes
        - Regulation patterns
        - Sustained changes
        - Pattern consistency
        """
        features = {}
        
        # Sustained motion patterns
        face_motion = time_series['face_motion'][window_indices]
        body_motion = time_series['upper_body_motion'][window_indices]
        
        features['face_motion_sustained'] = np.mean(face_motion) if len(face_motion) > 0 else 0.0
        features['face_motion_consistency'] = 1.0 / (1.0 + np.std(face_motion)) if len(face_motion) > 0 else 0.0
        features['body_motion_sustained'] = np.mean(body_motion) if len(body_motion) > 0 else 0.0
        features['body_motion_trend'] = self._compute_trend(body_motion)
        
        # Head pose stability/instability
        head_yaw = time_series['head_yaw'][window_indices]
        head_pitch = time_series['head_pitch'][window_indices]
        head_roll = time_series['head_roll'][window_indices]
        
        features['head_stability'] = 1.0 / (1.0 + np.mean([
            np.std(head_yaw), np.std(head_pitch), np.std(head_roll)
        ])) if len(head_yaw) > 0 else 0.0
        
        features['head_pose_drift'] = self._compute_drift(head_yaw, head_pitch)
        
        # Attention patterns
        gaze_proxy = time_series['gaze_proxy'][window_indices]
        features['attention_stability'] = 1.0 / (1.0 + np.std(gaze_proxy)) if len(gaze_proxy) > 0 else 0.0
        features['attention_trend'] = self._compute_trend(gaze_proxy)
        
        # Motor regulation patterns
        hand_left = time_series['hand_velocity_left'][window_indices]
        hand_right = time_series['hand_velocity_right'][window_indices]
        
        features['motor_regulation'] = 1.0 / (1.0 + np.mean([
            np.std(hand_left), np.std(hand_right)
        ])) if len(hand_left) > 0 else 0.0
        
        # Rhythmic patterns (stereotypies)
        features['rhythmic_face'] = self._detect_rhythmic_pattern(face_motion)
        features['rhythmic_body'] = self._detect_rhythmic_pattern(body_motion)
        features['rhythmic_hands'] = self._detect_rhythmic_pattern(
            np.maximum(hand_left, hand_right) if len(hand_left) > 0 else np.array([0])
        )
        
        return features
    
    def _extract_long_term_features(
        self,
        time_series: Dict,
        window_indices: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract long-term features (30-60 seconds).
        
        Focus on:
        - Overall behavioral state
        - Sustained patterns
        - Gradual changes
        - Session-level trends
        """
        features = {}
        
        # Overall activity levels
        face_motion = time_series['face_motion'][window_indices]
        body_motion = time_series['upper_body_motion'][window_indices]
        
        features['overall_face_activity'] = np.mean(face_motion) if len(face_motion) > 0 else 0.0
        features['overall_body_activity'] = np.mean(body_motion) if len(body_motion) > 0 else 0.0
        
        # Long-term trends
        features['face_activity_trend'] = self._compute_trend(face_motion)
        features['body_activity_trend'] = self._compute_trend(body_motion)
        
        # Behavioral state consistency
        features['behavioral_consistency'] = self._compute_behavioral_consistency(
            time_series, window_indices
        )
        
        # Attention patterns over time
        gaze_proxy = time_series['gaze_proxy'][window_indices]
        features['sustained_attention'] = self._compute_sustained_attention(gaze_proxy)
        features['attention_variability'] = np.std(gaze_proxy) if len(gaze_proxy) > 0 else 0.0
        
        # Motor patterns
        hand_left = time_series['hand_velocity_left'][window_indices]
        hand_right = time_series['hand_velocity_right'][window_indices]
        
        features['sustained_motor_activity'] = np.mean([
            np.mean(hand_left), np.mean(hand_right)
        ]) if len(hand_left) > 0 else 0.0
        
        # Head pose patterns
        head_yaw = time_series['head_yaw'][window_indices]
        head_pitch = time_series['head_pitch'][window_indices]
        
        features['head_orientation_bias'] = np.mean(head_yaw) if len(head_yaw) > 0 else 0.0
        features['head_elevation_bias'] = np.mean(head_pitch) if len(head_pitch) > 0 else 0.0
        
        # Data quality over long term
        face_detected = time_series['face_detected'][window_indices]
        pose_detected = time_series['pose_detected'][window_indices]
        
        features['face_detection_rate'] = np.mean(face_detected) if len(face_detected) > 0 else 0.0
        features['pose_detection_rate'] = np.mean(pose_detected) if len(pose_detected) > 0 else 0.0
        
        return features
    
    def _compute_trend(self, signal: np.ndarray) -> float:
        """
        Compute linear trend in signal (positive = increasing, negative = decreasing).
        """
        if len(signal) < 2:
            return 0.0
        
        x = np.arange(len(signal))
        try:
            slope, _ = np.polyfit(x, signal, 1)
            return float(slope)
        except:
            return 0.0
    
    def _compute_drift(self, yaw: np.ndarray, pitch: np.ndarray) -> float:
        """
        Compute head pose drift (gradual change in orientation).
        """
        if len(yaw) < 2 or len(pitch) < 2:
            return 0.0
        
        yaw_drift = abs(yaw[-1] - yaw[0])
        pitch_drift = abs(pitch[-1] - pitch[0])
        
        return float(np.sqrt(yaw_drift**2 + pitch_drift**2))
    
    def _detect_rhythmic_pattern(self, signal: np.ndarray) -> float:
        """
        Detect rhythmic/repetitive patterns in signal using autocorrelation.
        """
        if len(signal) < 10:
            return 0.0
        
        # Remove DC component
        signal_centered = signal - np.mean(signal)
        
        # Compute autocorrelation
        autocorr = np.correlate(signal_centered, signal_centered, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Normalize
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        else:
            return 0.0
        
        # Find peaks in autocorrelation (indicating periodicity)
        if len(autocorr) > 5:
            peaks, _ = signal.find_peaks(autocorr[1:], height=0.3)
            if len(peaks) > 0:
                return float(np.max(autocorr[peaks + 1]))
        
        return 0.0
    
    def _compute_behavioral_consistency(
        self,
        time_series: Dict,
        window_indices: np.ndarray
    ) -> float:
        """
        Compute overall behavioral consistency across multiple features.
        """
        if len(window_indices) < 5:
            return 0.0
        
        # Extract key behavioral signals
        signals = [
            time_series['face_motion'][window_indices],
            time_series['upper_body_motion'][window_indices],
            time_series['gaze_proxy'][window_indices]
        ]
        
        # Compute coefficient of variation for each signal
        cvs = []
        for signal_data in signals:
            if len(signal_data) > 0 and np.mean(signal_data) > 0:
                cv = np.std(signal_data) / np.mean(signal_data)
                cvs.append(cv)
        
        if not cvs:
            return 0.0
        
        # Consistency = inverse of average coefficient of variation
        avg_cv = np.mean(cvs)
        consistency = 1.0 / (1.0 + avg_cv)
        
        return float(consistency)
    
    def _compute_sustained_attention(self, gaze_proxy: np.ndarray) -> float:
        """
        Compute sustained attention score from gaze proxy signal.
        """
        if len(gaze_proxy) < 5:
            return 0.0
        
        # Sustained attention = low variance + low mean (stable, forward gaze)
        variance_component = 1.0 / (1.0 + np.std(gaze_proxy))
        mean_component = 1.0 / (1.0 + np.mean(gaze_proxy))
        
        sustained_attention = (variance_component + mean_component) / 2.0
        
        return float(sustained_attention)
    
    def _compute_cross_scale_consistency(
        self,
        short_features: Dict,
        medium_features: Dict,
        long_features: Dict
    ) -> float:
        """
        Compute consistency of patterns across temporal scales.
        
        High consistency indicates genuine behavioral patterns.
        Low consistency suggests artifacts or noise.
        """
        # Compare similar features across scales
        comparisons = []
        
        # Motion consistency
        short_motion = short_features.get('face_motion_peak', 0.0)
        medium_motion = medium_features.get('face_motion_sustained', 0.0)
        long_motion = long_features.get('overall_face_activity', 0.0)
        
        if max(short_motion, medium_motion, long_motion) > 0:
            motion_consistency = 1.0 - np.std([short_motion, medium_motion, long_motion]) / np.mean([short_motion, medium_motion, long_motion])
            comparisons.append(max(0.0, motion_consistency))
        
        # Attention consistency
        short_gaze = short_features.get('gaze_shift_intensity', 0.0)
        medium_attention = medium_features.get('attention_stability', 0.0)
        long_attention = long_features.get('sustained_attention', 0.0)
        
        # Invert short_gaze for consistency (high shifts = low stability)
        short_attention = 1.0 - short_gaze
        
        if max(short_attention, medium_attention, long_attention) > 0:
            attention_consistency = 1.0 - np.std([short_attention, medium_attention, long_attention]) / np.mean([short_attention, medium_attention, long_attention])
            comparisons.append(max(0.0, attention_consistency))
        
        # Motor consistency
        short_motor = short_features.get('hand_burst_left', 0.0) + short_features.get('hand_burst_right', 0.0)
        medium_motor = 1.0 - medium_features.get('motor_regulation', 0.0)  # Invert regulation
        long_motor = long_features.get('sustained_motor_activity', 0.0)
        
        if max(short_motor, medium_motor, long_motor) > 0:
            motor_consistency = 1.0 - np.std([short_motor, medium_motor, long_motor]) / np.mean([short_motor, medium_motor, long_motor])
            comparisons.append(max(0.0, motor_consistency))
        
        if not comparisons:
            return 0.5  # Neutral consistency
        
        return float(np.mean(comparisons))
    
    def _determine_dominant_scale(
        self,
        short_features: Dict,
        medium_features: Dict,
        long_features: Dict
    ) -> str:
        """
        Determine which temporal scale shows the strongest behavioral signal.
        """
        # Compute signal strength for each scale
        short_strength = self._compute_signal_strength(short_features)
        medium_strength = self._compute_signal_strength(medium_features)
        long_strength = self._compute_signal_strength(long_features)
        
        strengths = {
            'short': short_strength,
            'medium': medium_strength,
            'long': long_strength
        }
        
        return max(strengths, key=strengths.get)
    
    def _compute_signal_strength(self, features: Dict) -> float:
        """
        Compute overall signal strength from feature dictionary.
        """
        if not features:
            return 0.0
        
        # Normalize and sum all feature values
        values = [abs(v) for v in features.values() if isinstance(v, (int, float))]
        
        if not values:
            return 0.0
        
        return float(np.mean(values))
    
    def _compute_confidence(
        self,
        time_series: Dict,
        center_idx: int,
        short_features: Dict,
        medium_features: Dict,
        long_features: Dict
    ) -> float:
        """
        Compute overall confidence in multi-scale measurements.
        """
        # Data quality component
        face_detected = time_series['face_detected'][center_idx]
        pose_detected = time_series['pose_detected'][center_idx]
        data_quality = (float(face_detected) + float(pose_detected)) / 2.0
        
        # Feature completeness component
        total_features = len(short_features) + len(medium_features) + len(long_features)
        expected_features = 12 + 10 + 12  # Expected number of features per scale
        completeness = min(1.0, total_features / expected_features)
        
        # Cross-scale consistency component (already computed)
        consistency = self._compute_cross_scale_consistency(
            short_features, medium_features, long_features
        )
        
        # Combined confidence
        confidence = (data_quality * 0.4 + completeness * 0.3 + consistency * 0.3)
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _detect_behavioral_patterns(
        self,
        multi_scale_features: List[MultiScaleFeatures]
    ) -> List[BehavioralPattern]:
        """
        Detect behavioral patterns from multi-scale features.
        """
        if not multi_scale_features:
            return []
        
        patterns = []
        
        # Pattern detection strategies
        patterns.extend(self._detect_agitation_patterns(multi_scale_features))
        patterns.extend(self._detect_attention_patterns(multi_scale_features))
        patterns.extend(self._detect_regulation_patterns(multi_scale_features))
        patterns.extend(self._detect_stereotypy_patterns(multi_scale_features))
        
        # Sort by intensity and consistency
        patterns.sort(key=lambda p: p.intensity * p.consistency, reverse=True)
        
        return patterns
    
    def _detect_agitation_patterns(
        self,
        multi_scale_features: List[MultiScaleFeatures]
    ) -> List[BehavioralPattern]:
        """
        Detect motor agitation patterns across scales.
        """
        patterns = []
        
        for i, ms_feat in enumerate(multi_scale_features):
            # Check for agitation indicators across scales
            short_agitation = (
                ms_feat.short_term.get('face_motion_peak', 0.0) +
                ms_feat.short_term.get('body_motion_peak', 0.0) +
                ms_feat.short_term.get('hand_burst_left', 0.0) +
                ms_feat.short_term.get('hand_burst_right', 0.0)
            ) / 4.0
            
            medium_agitation = 1.0 - ms_feat.medium_term.get('motor_regulation', 1.0)
            long_agitation = ms_feat.long_term.get('sustained_motor_activity', 0.0)
            
            # Combined agitation score
            agitation_intensity = np.mean([short_agitation, medium_agitation, long_agitation])
            
            if agitation_intensity > self.pattern_threshold:
                # Determine primary scale
                scale_scores = {
                    'short': short_agitation,
                    'medium': medium_agitation,
                    'long': long_agitation
                }
                primary_scale = max(scale_scores, key=scale_scores.get)
                
                pattern = BehavioralPattern(
                    start_time=ms_feat.timestamp,
                    end_time=ms_feat.timestamp + 1.0,  # Will be extended by merging
                    pattern_type='motor_agitation',
                    primary_scale=primary_scale,
                    intensity=agitation_intensity,
                    consistency=ms_feat.cross_scale_consistency,
                    contributing_features={
                        'short_agitation': short_agitation,
                        'medium_agitation': medium_agitation,
                        'long_agitation': long_agitation
                    },
                    clinical_significance=self._interpret_agitation_pattern(
                        agitation_intensity, primary_scale
                    )
                )
                
                patterns.append(pattern)
        
        # Merge adjacent patterns
        return self._merge_adjacent_patterns(patterns)
    
    def _detect_attention_patterns(
        self,
        multi_scale_features: List[MultiScaleFeatures]
    ) -> List[BehavioralPattern]:
        """
        Detect attention-related patterns across scales.
        """
        patterns = []
        
        for ms_feat in multi_scale_features:
            # Attention instability indicators
            short_instability = ms_feat.short_term.get('gaze_shift_intensity', 0.0)
            medium_instability = 1.0 - ms_feat.medium_term.get('attention_stability', 1.0)
            long_instability = 1.0 - ms_feat.long_term.get('sustained_attention', 1.0)
            
            attention_instability = np.mean([short_instability, medium_instability, long_instability])
            
            if attention_instability > self.pattern_threshold:
                scale_scores = {
                    'short': short_instability,
                    'medium': medium_instability,
                    'long': long_instability
                }
                primary_scale = max(scale_scores, key=scale_scores.get)
                
                pattern = BehavioralPattern(
                    start_time=ms_feat.timestamp,
                    end_time=ms_feat.timestamp + 1.0,
                    pattern_type='attention_instability',
                    primary_scale=primary_scale,
                    intensity=attention_instability,
                    consistency=ms_feat.cross_scale_consistency,
                    contributing_features={
                        'short_instability': short_instability,
                        'medium_instability': medium_instability,
                        'long_instability': long_instability
                    },
                    clinical_significance=self._interpret_attention_pattern(
                        attention_instability, primary_scale
                    )
                )
                
                patterns.append(pattern)
        
        return self._merge_adjacent_patterns(patterns)
    
    def _detect_regulation_patterns(
        self,
        multi_scale_features: List[MultiScaleFeatures]
    ) -> List[BehavioralPattern]:
        """
        Detect behavioral regulation patterns.
        """
        patterns = []
        
        for ms_feat in multi_scale_features:
            # Regulation indicators (higher = better regulation)
            short_regulation = 1.0 - ms_feat.short_term.get('face_motion_derivative', 0.0)
            medium_regulation = ms_feat.medium_term.get('behavioral_consistency', 0.0)
            long_regulation = ms_feat.long_term.get('behavioral_consistency', 0.0)
            
            # Look for dysregulation (low regulation scores)
            dysregulation = 1.0 - np.mean([short_regulation, medium_regulation, long_regulation])
            
            if dysregulation > self.pattern_threshold:
                scale_scores = {
                    'short': 1.0 - short_regulation,
                    'medium': 1.0 - medium_regulation,
                    'long': 1.0 - long_regulation
                }
                primary_scale = max(scale_scores, key=scale_scores.get)
                
                pattern = BehavioralPattern(
                    start_time=ms_feat.timestamp,
                    end_time=ms_feat.timestamp + 1.0,
                    pattern_type='behavioral_dysregulation',
                    primary_scale=primary_scale,
                    intensity=dysregulation,
                    consistency=ms_feat.cross_scale_consistency,
                    contributing_features={
                        'short_dysregulation': 1.0 - short_regulation,
                        'medium_dysregulation': 1.0 - medium_regulation,
                        'long_dysregulation': 1.0 - long_regulation
                    },
                    clinical_significance=self._interpret_regulation_pattern(
                        dysregulation, primary_scale
                    )
                )
                
                patterns.append(pattern)
        
        return self._merge_adjacent_patterns(patterns)
    
    def _detect_stereotypy_patterns(
        self,
        multi_scale_features: List[MultiScaleFeatures]
    ) -> List[BehavioralPattern]:
        """
        Detect stereotyped/repetitive movement patterns.
        """
        patterns = []
        
        for ms_feat in multi_scale_features:
            # Rhythmic pattern indicators
            rhythmic_face = ms_feat.medium_term.get('rhythmic_face', 0.0)
            rhythmic_body = ms_feat.medium_term.get('rhythmic_body', 0.0)
            rhythmic_hands = ms_feat.medium_term.get('rhythmic_hands', 0.0)
            
            stereotypy_intensity = np.max([rhythmic_face, rhythmic_body, rhythmic_hands])
            
            if stereotypy_intensity > self.pattern_threshold:
                # Determine which body part shows strongest pattern
                body_parts = {
                    'face': rhythmic_face,
                    'body': rhythmic_body,
                    'hands': rhythmic_hands
                }
                dominant_body_part = max(body_parts, key=body_parts.get)
                
                pattern = BehavioralPattern(
                    start_time=ms_feat.timestamp,
                    end_time=ms_feat.timestamp + 1.0,
                    pattern_type=f'stereotypy_{dominant_body_part}',
                    primary_scale='medium',  # Stereotypies are medium-term patterns
                    intensity=stereotypy_intensity,
                    consistency=ms_feat.cross_scale_consistency,
                    contributing_features={
                        'rhythmic_face': rhythmic_face,
                        'rhythmic_body': rhythmic_body,
                        'rhythmic_hands': rhythmic_hands
                    },
                    clinical_significance=self._interpret_stereotypy_pattern(
                        stereotypy_intensity, dominant_body_part
                    )
                )
                
                patterns.append(pattern)
        
        return self._merge_adjacent_patterns(patterns)
    
    def _merge_adjacent_patterns(
        self,
        patterns: List[BehavioralPattern]
    ) -> List[BehavioralPattern]:
        """
        Merge temporally adjacent patterns of the same type.
        """
        if not patterns:
            return []
        
        # Sort by start time
        patterns.sort(key=lambda p: p.start_time)
        
        merged = []
        current_pattern = patterns[0]
        
        for next_pattern in patterns[1:]:
            # Check if patterns are adjacent and of same type
            if (next_pattern.start_time - current_pattern.end_time < 2.0 and
                next_pattern.pattern_type == current_pattern.pattern_type):
                
                # Merge patterns
                current_pattern.end_time = next_pattern.end_time
                current_pattern.intensity = max(current_pattern.intensity, next_pattern.intensity)
                current_pattern.consistency = (current_pattern.consistency + next_pattern.consistency) / 2
                
                # Merge contributing features
                for key, value in next_pattern.contributing_features.items():
                    if key in current_pattern.contributing_features:
                        current_pattern.contributing_features[key] = max(
                            current_pattern.contributing_features[key], value
                        )
                    else:
                        current_pattern.contributing_features[key] = value
            else:
                # Start new pattern
                merged.append(current_pattern)
                current_pattern = next_pattern
        
        merged.append(current_pattern)
        
        return merged
    
    def _interpret_agitation_pattern(self, intensity: float, primary_scale: str) -> str:
        """Generate clinical interpretation for agitation patterns."""
        if primary_scale == 'short':
            return f"Acute motor agitation episode (intensity: {intensity:.2f}) - rapid, intense movements suggesting immediate dysregulation"
        elif primary_scale == 'medium':
            return f"Sustained motor agitation (intensity: {intensity:.2f}) - persistent elevated movement indicating ongoing dysregulation"
        else:
            return f"Chronic motor agitation (intensity: {intensity:.2f}) - prolonged elevated activity suggesting sustained arousal state"
    
    def _interpret_attention_pattern(self, intensity: float, primary_scale: str) -> str:
        """Generate clinical interpretation for attention patterns."""
        if primary_scale == 'short':
            return f"Attention instability (intensity: {intensity:.2f}) - rapid gaze shifts indicating difficulty maintaining focus"
        elif primary_scale == 'medium':
            return f"Sustained attention difficulties (intensity: {intensity:.2f}) - persistent attention instability over behavioral episodes"
        else:
            return f"Chronic attention dysregulation (intensity: {intensity:.2f}) - prolonged attention difficulties throughout session"
    
    def _interpret_regulation_pattern(self, intensity: float, primary_scale: str) -> str:
        """Generate clinical interpretation for regulation patterns."""
        if primary_scale == 'short':
            return f"Acute behavioral dysregulation (intensity: {intensity:.2f}) - rapid behavioral changes indicating immediate regulation challenges"
        elif primary_scale == 'medium':
            return f"Episode-level dysregulation (intensity: {intensity:.2f}) - inconsistent behavioral patterns during interaction episodes"
        else:
            return f"Sustained dysregulation (intensity: {intensity:.2f}) - persistent behavioral inconsistency throughout session"
    
    def _interpret_stereotypy_pattern(self, intensity: float, body_part: str) -> str:
        """Generate clinical interpretation for stereotypy patterns."""
        return f"Repetitive {body_part} movements detected (intensity: {intensity:.2f}) - rhythmic pattern consistent with stereotyped behavior"
    
    def _empty_features_dict(self) -> Dict[str, float]:
        """Return empty features dictionary."""
        return {
            'face_motion_peak': 0.0,
            'body_motion_peak': 0.0,
            'gaze_shift_intensity': 0.0,
            'hand_burst_left': 0.0,
            'hand_burst_right': 0.0
        }


# Convenience functions
def analyze_multi_scale_patterns(
    face_features: List,
    pose_features: List,
    fps: float = 5.0,
    consistency_threshold: float = 0.6
) -> Tuple[List[MultiScaleFeatures], List[BehavioralPattern]]:
    """
    Convenience function for multi-scale behavioral pattern analysis.
    
    Args:
        face_features: List of face feature objects
        pose_features: List of pose feature objects
        fps: Frames per second
        consistency_threshold: Minimum cross-scale consistency
        
    Returns:
        Tuple of (multi_scale_features, behavioral_patterns)
    """
    analyzer = MultiScaleAnalyzer(consistency_threshold=consistency_threshold)
    return analyzer.analyze(face_features, pose_features, fps)


def get_pattern_summary(patterns: List[BehavioralPattern]) -> Dict:
    """
    Generate summary statistics for detected patterns.
    
    Args:
        patterns: List of detected behavioral patterns
        
    Returns:
        Dictionary with pattern summary statistics
    """
    if not patterns:
        return {
            'total_patterns': 0,
            'pattern_types': {},
            'primary_scales': {},
            'mean_intensity': 0.0,
            'mean_consistency': 0.0,
            'total_duration': 0.0
        }
    
    # Count by type
    pattern_types = {}
    for pattern in patterns:
        ptype = pattern.pattern_type
        pattern_types[ptype] = pattern_types.get(ptype, 0) + 1
    
    # Count by primary scale
    primary_scales = {}
    for pattern in patterns:
        scale = pattern.primary_scale
        primary_scales[scale] = primary_scales.get(scale, 0) + 1
    
    # Compute statistics
    intensities = [p.intensity for p in patterns]
    consistencies = [p.consistency for p in patterns]
    durations = [p.end_time - p.start_time for p in patterns]
    
    return {
        'total_patterns': len(patterns),
        'pattern_types': pattern_types,
        'primary_scales': primary_scales,
        'mean_intensity': float(np.mean(intensities)),
        'mean_consistency': float(np.mean(consistencies)),
        'total_duration': float(np.sum(durations))
    }