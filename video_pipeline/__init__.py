"""
Video analysis pipeline for behavioral verification.

This package implements MediaPipe-based visual analysis:
1. Face analysis (head pose, gaze proxy, facial motion)
2. Pose analysis (upper-body movement, hand tracking)
3. Temporal aggregation (sliding-window statistics)

Clinical rationale:
- Visual cues verify and contextualize audio-detected instability
- Head movement, gaze shifts, motor agitation are regulation markers
- Temporal aggregation captures behavioral patterns, not isolated frames
"""

from .face_analyzer import (
    FaceAnalyzer,
    FaceFeatures,
    analyze_face_segment
)
from .pose_analyzer import (
    PoseAnalyzer,
    PoseFeatures,
    analyze_pose_segment
)
from .improved_temporal_agg import (
    ImprovedTemporalAggregator as TemporalAggregator,
    aggregate_features_improved as aggregate_features,
    ImprovedAggregatedFeatures as AggregatedFeatures
)
# Import legacy functions for backward compatibility
from .temporal_agg import (
    compute_sliding_window_stats,
    detect_motion_bursts
)

__all__ = [
    'FaceAnalyzer',
    'FaceFeatures',
    'analyze_face_segment',
    'PoseAnalyzer',
    'PoseFeatures',
    'analyze_pose_segment',
    'TemporalAggregator',
    'aggregate_features',
    'AggregatedFeatures',
    'compute_sliding_window_stats',
    'detect_motion_bursts',
]
