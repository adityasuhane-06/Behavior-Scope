"""
Conservative multimodal fusion.

Fusion strategy:
- Conservative: Both modalities must show evidence for high confidence
- Explainable: Track which features contributed to detection
- Temporal alignment: Ensure audio and video evidence temporally overlap

Decision rules:
1. Strong signal: Both audio and video scores > high thresholds
2. Moderate signal: One high, one moderate
3. Weak signal: Only one modality slightly elevated
4. No signal: Both below thresholds

Clinical rationale:
- Audio-only signals may be artifacts (throat clearing, external noise)
- Video-only signals may be artifacts (camera shake, lighting changes)
- Multi-modal agreement indicates genuine behavioral dysregulation
- Conservative approach prioritizes precision over recall (clinical context)

Engineering approach:
- Weighted combination of modality scores
- Configurable thresholds for flexibility
- Metadata preservation for interpretability
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FusedEvidence:
    """
    Fused evidence from audio and video modalities.
    
    Attributes:
        start_time: Window start in seconds
        end_time: Window end in seconds
        audio_score: Audio instability score (0-1)
        video_score: Video dysregulation score (0-1)
        fused_confidence: Combined confidence (0-1)
        confidence_level: 'strong', 'moderate', 'weak', or 'none'
        contributing_features: Dict of features that contributed
        explanation: Human-readable explanation
    """
    start_time: float
    end_time: float
    audio_score: float
    video_score: float
    fused_confidence: float
    confidence_level: str
    contributing_features: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class FusionEngine:
    """
    Multimodal fusion engine for audio-video evidence.
    
    Fusion logic:
    - Weighted combination: audio_weight * audio_score + video_weight * video_score
    - Threshold-based confidence levels
    - Conservative approach: require agreement
    
    Usage:
        engine = FusionEngine(config)
        fused = engine.fuse(audio_windows, video_features)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize fusion engine.
        
        Args:
            config: Configuration dict with fusion parameters
        """
        self.config = config
        
        # Extract thresholds
        fusion_config = config.get('fusion', {})
        agreement_rules = fusion_config.get('agreement_rules', {})
        
        self.strong_audio_threshold = agreement_rules.get('strong_audio_threshold', 0.7)
        self.strong_video_threshold = agreement_rules.get('strong_video_threshold', 0.7)
        self.strong_confidence = agreement_rules.get('strong_confidence', 0.9)
        
        self.moderate_audio_threshold = agreement_rules.get('moderate_audio_threshold', 0.5)
        self.moderate_video_threshold = agreement_rules.get('moderate_video_threshold', 0.5)
        self.moderate_confidence = agreement_rules.get('moderate_confidence', 0.6)
        
        self.weak_confidence = agreement_rules.get('weak_confidence', 0.3)
        self.min_reportable_confidence = fusion_config.get('min_reportable_confidence', 0.5)
        
        # Modality weights (can be tuned based on validation)
        self.audio_weight = 0.5  # Equal weighting by default
        self.video_weight = 0.5
        
        logger.info(
            f"Fusion engine initialized: "
            f"strong_thresholds=({self.strong_audio_threshold}, {self.strong_video_threshold}), "
            f"min_confidence={self.min_reportable_confidence}"
        )
    
    def fuse(
        self,
        audio_windows: List,
        video_aggregated: List
    ) -> List[FusedEvidence]:
        """
        Fuse audio instability windows with video aggregated features.
        
        Algorithm:
        1. Align audio and video windows temporally
        2. For each time window:
           - Extract audio instability score
           - Compute video dysregulation score
           - Apply fusion decision rules
           - Generate fused evidence with confidence
        
        Args:
            audio_windows: List of InstabilityWindow objects from audio pipeline
            video_aggregated: List of AggregatedFeatures from video pipeline
            
        Returns:
            List of FusedEvidence objects (sorted by confidence)
        """
        if not audio_windows or not video_aggregated:
            logger.warning("Empty input to fusion engine")
            return []
        
        logger.info(
            f"Fusing {len(audio_windows)} audio windows with "
            f"{len(video_aggregated)} video windows"
        )
        
        fused_list = []
        
        # For each audio window, find overlapping video windows
        for audio_win in audio_windows:
            # Find video windows that overlap with audio window
            overlapping_video = [
                v for v in video_aggregated
                if self._windows_overlap(
                    audio_win.start_time, audio_win.end_time,
                    v.window_start_time, v.window_end_time
                )
            ]
            
            if not overlapping_video:
                logger.debug(
                    f"No video features for audio window "
                    f"[{audio_win.start_time:.1f}, {audio_win.end_time:.1f}]"
                )
                continue
            
            # Average video features across overlapping windows
            video_score = self._compute_video_score(overlapping_video)
            
            # Apply fusion decision rules
            fused = self._apply_fusion_rules(
                audio_win.start_time,
                audio_win.end_time,
                audio_win.instability_score,
                video_score,
                audio_win.contributing_features,
                overlapping_video
            )
            
            # Only include if above minimum confidence
            if fused.fused_confidence >= self.min_reportable_confidence:
                fused_list.append(fused)
        
        # Sort by confidence (descending)
        fused_list.sort(key=lambda f: f.fused_confidence, reverse=True)
        
        logger.info(
            f"Fused {len(fused_list)} windows with confidence >= "
            f"{self.min_reportable_confidence}"
        )
        
        return fused_list
    
    def _windows_overlap(
        self,
        start1: float, end1: float,
        start2: float, end2: float
    ) -> bool:
        """Check if two time windows overlap."""
        return not (end1 <= start2 or end2 <= start1)
    
    def _compute_video_score(self, video_windows: List) -> float:
        """
        Compute aggregate video dysregulation score.
        
        Combines multiple visual indicators:
        - Head movement variance (agitation)
        - Upper-body motion (motor dysregulation)
        - Hand velocity (fidgeting)
        - Gaze shifts (attention instability)
        
        Returns:
            Video score (0-1)
        """
        if not video_windows:
            return 0.0
        
        # Extract key features from video windows
        head_motion_values = []
        body_motion_values = []
        hand_velocity_values = []
        gaze_proxy_values = []
        
        for v_win in video_windows:
            face_feat = v_win.face_features
            pose_feat = v_win.pose_features
            
            # Head movement (std of yaw/pitch/roll)
            head_motion = np.mean([
                face_feat.get('head_yaw_std', 0.0),
                face_feat.get('head_pitch_std', 0.0),
                face_feat.get('head_roll_std', 0.0)
            ])
            head_motion_values.append(head_motion)
            
            # Body motion (upper body)
            body_motion = pose_feat.get('upper_body_motion_mean', 0.0)
            body_motion_values.append(body_motion)
            
            # Hand velocity (max across both hands)
            hand_vel = pose_feat.get('hand_velocity_max_max', 0.0)
            hand_velocity_values.append(hand_vel)
            
            # Gaze proxy
            gaze = face_feat.get('gaze_proxy_mean', 0.0)
            gaze_proxy_values.append(gaze)
        
        # Normalize and weight features
        # These weights can be tuned based on clinical validation
        feature_weights = {
            'head_motion': 0.3,
            'body_motion': 0.3,
            'hand_velocity': 0.2,
            'gaze_proxy': 0.2
        }
        
        # Normalize each feature (0-1 scale)
        # Using sigmoid-like normalization
        head_motion_norm = self._normalize_feature(np.mean(head_motion_values), scale=20.0)
        body_motion_norm = self._normalize_feature(np.mean(body_motion_values), scale=0.15)
        hand_vel_norm = self._normalize_feature(np.mean(hand_velocity_values), scale=30.0)
        gaze_norm = self._normalize_feature(np.mean(gaze_proxy_values), scale=0.1)
        
        # Weighted combination
        video_score = (
            feature_weights['head_motion'] * head_motion_norm +
            feature_weights['body_motion'] * body_motion_norm +
            feature_weights['hand_velocity'] * hand_vel_norm +
            feature_weights['gaze_proxy'] * gaze_norm
        )
        
        return float(np.clip(video_score, 0.0, 1.0))
    
    def _normalize_feature(self, value: float, scale: float) -> float:
        """
        Normalize feature using sigmoid-like function.
        
        Args:
            value: Raw feature value
            scale: Scale parameter (higher = slower saturation)
            
        Returns:
            Normalized value (0-1)
        """
        return 1.0 / (1.0 + np.exp(-value / scale + 3.0))
    
    def _apply_fusion_rules(
        self,
        start_time: float,
        end_time: float,
        audio_score: float,
        video_score: float,
        audio_features: Dict,
        video_windows: List
    ) -> FusedEvidence:
        """
        Apply fusion decision rules to determine confidence level.
        
        Decision tree:
        1. Both high (>= strong thresholds) → Strong confidence (0.9)
        2. One high, one moderate → Moderate confidence (0.6)
        3. Only one slightly elevated → Weak confidence (0.3)
        4. Both low → No signal (0.0)
        """
        # Check thresholds
        audio_high = audio_score >= self.strong_audio_threshold
        audio_moderate = audio_score >= self.moderate_audio_threshold
        
        video_high = video_score >= self.strong_video_threshold
        video_moderate = video_score >= self.moderate_video_threshold
        
        # Apply decision rules
        if audio_high and video_high:
            # Strong signal: both modalities agree at high level
            confidence = self.strong_confidence
            confidence_level = 'strong'
            explanation = (
                f"Strong multimodal agreement: "
                f"Audio instability={audio_score:.2f}, "
                f"Video dysregulation={video_score:.2f}"
            )
        
        elif (audio_high and video_moderate) or (audio_moderate and video_high):
            # Moderate signal: one high, one moderate
            confidence = self.moderate_confidence
            confidence_level = 'moderate'
            
            if audio_high:
                explanation = f"Audio-driven signal (score={audio_score:.2f}) with moderate video support ({video_score:.2f})"
            else:
                explanation = f"Video-driven signal (score={video_score:.2f}) with moderate audio support ({audio_score:.2f})"
        
        elif audio_moderate or video_moderate:
            # Weak signal: only one modality moderately elevated
            confidence = self.weak_confidence
            confidence_level = 'weak'
            
            if audio_moderate:
                explanation = f"Audio-only signal (score={audio_score:.2f}), minimal video evidence ({video_score:.2f})"
            else:
                explanation = f"Video-only signal (score={video_score:.2f}), minimal audio evidence ({audio_score:.2f})"
        
        else:
            # No signal
            confidence = 0.0
            confidence_level = 'none'
            explanation = "Both modalities below thresholds"
        
        # Collect contributing features
        contributing_features = {
            'audio_score': audio_score,
            'video_score': video_score,
            **{f"audio_{k}": v for k, v in audio_features.items()},
        }
        
        return FusedEvidence(
            start_time=start_time,
            end_time=end_time,
            audio_score=audio_score,
            video_score=video_score,
            fused_confidence=confidence,
            confidence_level=confidence_level,
            contributing_features=contributing_features,
            explanation=explanation
        )


def fuse_audio_video_evidence(
    audio_windows: List,
    video_aggregated: List,
    config: Dict
) -> List[FusedEvidence]:
    """
    Convenience function for multimodal fusion.
    
    Args:
        audio_windows: List of InstabilityWindow objects
        video_aggregated: List of AggregatedFeatures objects
        config: Configuration dictionary
        
    Returns:
        List of FusedEvidence objects
    """
    engine = FusionEngine(config)
    return engine.fuse(audio_windows, video_aggregated)


def compute_multimodal_confidence(
    audio_score: float,
    video_score: float,
    audio_weight: float = 0.5,
    video_weight: float = 0.5
) -> float:
    """
    Compute multimodal confidence using weighted combination.
    
    Simple alternative to rule-based fusion.
    
    Args:
        audio_score: Audio instability score (0-1)
        video_score: Video dysregulation score (0-1)
        audio_weight: Weight for audio modality
        video_weight: Weight for video modality
        
    Returns:
        Combined confidence score (0-1)
    """
    # Normalize weights
    total_weight = audio_weight + video_weight
    audio_weight /= total_weight
    video_weight /= total_weight
    
    # Weighted combination
    confidence = audio_weight * audio_score + video_weight * video_score
    
    return float(np.clip(confidence, 0.0, 1.0))


def filter_by_confidence(
    fused_evidence: List[FusedEvidence],
    min_confidence: float = 0.5
) -> List[FusedEvidence]:
    """
    Filter fused evidence by minimum confidence threshold.
    
    Args:
        fused_evidence: List of FusedEvidence objects
        min_confidence: Minimum confidence threshold
        
    Returns:
        Filtered list
    """
    filtered = [fe for fe in fused_evidence if fe.fused_confidence >= min_confidence]
    
    logger.info(
        f"Filtered {len(fused_evidence)} → {len(filtered)} windows "
        f"(confidence >= {min_confidence})"
    )
    
    return filtered


def get_top_dysregulation_windows(
    fused_evidence: List[FusedEvidence],
    n: int = 5
) -> List[FusedEvidence]:
    """
    Get top N highest-confidence dysregulation windows.
    
    Args:
        fused_evidence: List of FusedEvidence objects
        n: Number of top windows to return
        
    Returns:
        Top N windows sorted by confidence (descending)
    """
    sorted_evidence = sorted(
        fused_evidence,
        key=lambda fe: fe.fused_confidence,
        reverse=True
    )
    
    return sorted_evidence[:n]
