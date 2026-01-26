"""
Core data models for the Enhanced Eye Contact & Attention Tracking System.
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from .enums import (
    DetectionApproach, AttentionType, ScanningPattern, ZoneType,
    AuditEventType, QualityFlag, GazeTarget, AgeGroup, ClinicalPopulation,
    ExportFormat, DataType
)


@dataclass
class GazeVector:
    """3D gaze direction vector with confidence and timestamp."""
    x: float  # Horizontal gaze direction (-1 to 1, left to right)
    y: float  # Vertical gaze direction (-1 to 1, down to up)  
    z: float  # Depth component (0 to 1, near to far)
    confidence: float  # Gaze estimation confidence (0-1)
    timestamp: float  # Frame timestamp in seconds
    
    def __post_init__(self):
        """Validate gaze vector components."""
        self.x = max(-1.0, min(1.0, self.x))
        self.y = max(-1.0, min(1.0, self.y))
        self.z = max(0.0, min(1.0, self.z))
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    def magnitude(self) -> float:
        """Calculate the magnitude of the gaze vector."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def angle_to(self, other: 'GazeVector') -> float:
        """Calculate angle between two gaze vectors in radians."""
        dot_product = self.x * other.x + self.y * other.y + self.z * other.z
        magnitudes = self.magnitude() * other.magnitude()
        if magnitudes == 0:
            return 0.0
        return np.arccos(np.clip(dot_product / magnitudes, -1.0, 1.0))


@dataclass
class FrameResult:
    """Complete analysis result for a single video frame."""
    timestamp: float
    confidence_score: float  # Eye contact confidence (0-1)
    binary_decision: bool  # Binary eye contact decision
    detection_approach: DetectionApproach
    gaze_vector: Optional[GazeVector] = None
    gaze_target: Optional[GazeTarget] = None
    quality_flags: List[QualityFlag] = field(default_factory=list)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_quality_flag(self, flag: QualityFlag) -> bool:
        """Check if frame has specific quality flag."""
        return flag in self.quality_flags
    
    def is_high_quality(self) -> bool:
        """Check if frame meets high quality standards."""
        return (QualityFlag.HIGH_QUALITY in self.quality_flags and
                QualityFlag.FACE_DETECTION_FAILED not in self.quality_flags)


@dataclass
class JointAttentionEvent:
    """Joint attention episode between subject and social partner."""
    start_time: float
    end_time: float
    duration: float
    attention_type: AttentionType
    target_object: Optional[str] = None
    alignment_score: float = 0.0  # How well aligned the attention is (0-1)
    confidence: float = 0.0  # Detection confidence (0-1)
    response_latency: Optional[float] = None  # Time to respond to attention cue
    
    @property
    def is_initiated(self) -> bool:
        """Check if this is subject-initiated joint attention."""
        return self.attention_type == AttentionType.INITIATED
    
    @property
    def is_responding(self) -> bool:
        """Check if this is responding joint attention."""
        return self.attention_type == AttentionType.RESPONDING


@dataclass
class VisualTrackingData:
    """Visual tracking and eye movement analysis for a time window."""
    timestamp: float
    window_duration: float
    eye_movement_velocity: float  # Average velocity in degrees/second
    saccade_count: int  # Number of saccadic eye movements
    fixation_duration: float  # Average fixation duration in seconds
    scanning_pattern: ScanningPattern
    repetitive_behavior_score: float  # Score for repetitive visual behaviors (0-1)
    attention_stability: float  # How stable attention is over time (0-1)
    
    def is_repetitive(self, threshold: float = 0.7) -> bool:
        """Check if repetitive behaviors exceed threshold."""
        return self.repetitive_behavior_score > threshold


@dataclass
class Coordinate:
    """2D coordinate with optional confidence."""
    x: float
    y: float
    confidence: float = 1.0


@dataclass
class ZoneCoordinates:
    """Coordinates defining an attention zone."""
    zone_type: str  # "bounding_box", "polygon", "circle"
    coordinates: List[Coordinate]
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is within the zone (simplified implementation)."""
        if self.zone_type == "bounding_box" and len(self.coordinates) >= 2:
            min_x = min(coord.x for coord in self.coordinates)
            max_x = max(coord.x for coord in self.coordinates)
            min_y = min(coord.y for coord in self.coordinates)
            max_y = max(coord.y for coord in self.coordinates)
            return min_x <= x <= max_x and min_y <= y <= max_y
        return False


@dataclass
class AttentionZoneEvent:
    """Attention event within a specific zone."""
    zone_id: str
    zone_name: str
    entry_time: float
    exit_time: float
    duration: float
    attention_intensity: float  # Average gaze confidence in zone (0-1)
    peak_intensity: float = 0.0  # Peak attention intensity
    
    @property
    def is_brief(self, threshold: float = 0.5) -> bool:
        """Check if attention duration is brief."""
        return self.duration < threshold


@dataclass
class EpisodeData:
    """Extended episode data with attention tracking information."""
    start_time: float
    end_time: float
    duration: float
    average_confidence: float
    peak_confidence: float
    quality_score: float
    associated_gaze_data: List[GazeVector] = field(default_factory=list)
    joint_attention_events: List[JointAttentionEvent] = field(default_factory=list)
    dominant_gaze_target: Optional[GazeTarget] = None
    
    @property
    def has_joint_attention(self) -> bool:
        """Check if episode contains joint attention events."""
        return len(self.joint_attention_events) > 0


@dataclass
class ComprehensiveMetrics:
    """Complete metrics combining all attention tracking domains."""
    window_start: float
    window_end: float
    
    # Eye contact metrics
    total_duration: float
    eye_contact_duration: float
    eye_contact_percentage: float
    episode_count: int
    average_episode_duration: float
    max_episode_duration: float
    inter_episode_intervals: List[float] = field(default_factory=list)
    
    # Gaze direction metrics
    gaze_direction_distribution: Dict[str, float] = field(default_factory=dict)
    average_gaze_stability: float = 0.0
    gaze_shift_frequency: float = 0.0
    
    # Joint attention metrics
    joint_attention_episodes: int = 0
    joint_attention_duration: float = 0.0
    initiated_attention_ratio: float = 0.0
    attention_response_latency: float = 0.0
    
    # Visual tracking metrics
    average_fixation_duration: float = 0.0
    saccade_frequency: float = 0.0
    scanning_pattern_complexity: float = 0.0
    repetitive_behavior_score: float = 0.0
    
    # Attention zone metrics
    zone_attention_distribution: Dict[str, float] = field(default_factory=dict)
    zone_transition_frequency: float = 0.0
    preferred_attention_zones: List[str] = field(default_factory=list)
    
    # Data quality
    data_completeness: float = 1.0
    gaze_tracking_quality: float = 1.0
    
    def get_overall_attention_score(self) -> float:
        """Calculate overall attention quality score (0-100)."""
        components = [
            self.eye_contact_percentage,
            self.average_gaze_stability * 100,
            (1.0 - self.repetitive_behavior_score) * 100,
            self.data_completeness * 100
        ]
        return sum(components) / len(components)


@dataclass
class ThresholdConfig:
    """Configuration parameters for detection and analysis."""
    detection_approach: DetectionApproach
    confidence_threshold: float = 0.5
    minimum_episode_duration: float = 0.5
    temporal_smoothing: bool = True
    
    # Gaze direction parameters
    gaze_stability_threshold: float = 0.1
    gaze_shift_sensitivity: float = 0.05
    
    # Joint attention parameters
    joint_attention_alignment_threshold: float = 0.7
    attention_response_latency_threshold: float = 3.0
    
    # Visual tracking parameters
    fixation_duration_threshold: float = 0.2
    saccade_velocity_threshold: float = 30.0
    
    # Attention zone parameters
    zone_entry_threshold: float = 0.6
    zone_transition_sensitivity: float = 0.1
    
    def validate(self) -> List[str]:
        """Validate configuration parameters and return any errors."""
        errors = []
        if not 0.0 <= self.confidence_threshold <= 1.0:
            errors.append("confidence_threshold must be between 0.0 and 1.0")
        if self.minimum_episode_duration < 0.0:
            errors.append("minimum_episode_duration must be non-negative")
        if not 0.0 <= self.gaze_stability_threshold <= 1.0:
            errors.append("gaze_stability_threshold must be between 0.0 and 1.0")
        return errors


@dataclass
class AuditEntry:
    """Single entry in the audit trail."""
    timestamp: float
    event_type: AuditEventType
    detection_approach: DetectionApproach
    threshold_config: Optional[ThresholdConfig] = None
    gaze_data_quality: float = 1.0
    attention_tracking_quality: float = 1.0
    data_quality: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit entry to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'event_type': self.event_type.value,
            'detection_approach': self.detection_approach.value,
            'gaze_data_quality': self.gaze_data_quality,
            'attention_tracking_quality': self.attention_tracking_quality,
            'data_quality': self.data_quality,
            'metadata': self.metadata
        }


# Additional supporting data structures

@dataclass
class TimeRange:
    """Time range specification."""
    start_time: float
    end_time: float
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def contains(self, timestamp: float) -> bool:
        return self.start_time <= timestamp <= self.end_time


@dataclass
class TimeWindow:
    """Time window for analysis."""
    start_time: float
    duration: float
    
    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


@dataclass
class StabilityMetrics:
    """Gaze stability measurements."""
    mean_deviation: float
    standard_deviation: float
    stability_score: float  # 0-1, higher = more stable
    
    def is_stable(self, threshold: float = 0.7) -> bool:
        return self.stability_score > threshold


@dataclass
class AlignmentScore:
    """Joint attention alignment measurement."""
    spatial_alignment: float  # How spatially aligned the gazes are (0-1)
    temporal_alignment: float  # How temporally synchronized (0-1)
    overall_score: float  # Combined alignment score (0-1)
    
    def is_well_aligned(self, threshold: float = 0.7) -> bool:
        return self.overall_score > threshold