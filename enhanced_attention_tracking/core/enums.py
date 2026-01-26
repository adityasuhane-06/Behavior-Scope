"""
Enumerations for the Enhanced Eye Contact & Attention Tracking System.
"""

from enum import Enum, auto
from typing import List


class DetectionApproach(Enum):
    """Different approaches for eye contact detection."""
    EPISODE_BASED = "episode_based"
    CONTINUOUS_SCORING = "continuous_scoring"
    FRAME_LEVEL_TRACKING = "frame_level_tracking"
    HYBRID = "hybrid"


class AttentionType(Enum):
    """Types of joint attention behaviors."""
    INITIATED = "initiated"  # Subject initiates joint attention
    RESPONDING = "responding"  # Subject responds to joint attention cue
    MUTUAL = "mutual"  # Simultaneous mutual attention
    NONE = "none"  # No joint attention detected


class ScanningPattern(Enum):
    """Visual scanning behavior patterns."""
    SYSTEMATIC = "systematic"  # Organized, methodical scanning
    RANDOM = "random"  # Disorganized, unpredictable scanning
    CENTRAL_BIAS = "central_bias"  # Preference for central visual field
    PERIPHERAL_BIAS = "peripheral_bias"  # Preference for peripheral areas
    REPETITIVE = "repetitive"  # Stereotyped, repetitive patterns
    AVOIDANT = "avoidant"  # Active avoidance of certain areas


class ZoneType(Enum):
    """Types of attention zones."""
    FACE = "face"  # Face/eye region
    OBJECT = "object"  # Specific object of interest
    BACKGROUND = "background"  # Background/environmental areas
    SOCIAL_PARTNER = "social_partner"  # Therapist/caregiver region
    CUSTOM = "custom"  # User-defined custom zone


class AuditEventType(Enum):
    """Types of events recorded in audit trail."""
    DETECTION_DECISION = "detection_decision"
    CONFIGURATION_CHANGE = "configuration_change"
    MISSING_DATA_EVENT = "missing_data_event"
    QUALITY_WARNING = "quality_warning"
    SYSTEM_ERROR = "system_error"
    ANALYSIS_START = "analysis_start"
    ANALYSIS_COMPLETE = "analysis_complete"


class QualityFlag(Enum):
    """Data quality indicators."""
    HIGH_QUALITY = "high_quality"
    MODERATE_QUALITY = "moderate_quality"
    LOW_QUALITY = "low_quality"
    FACE_DETECTION_FAILED = "face_detection_failed"
    GAZE_ESTIMATION_UNCERTAIN = "gaze_estimation_uncertain"
    RAPID_STATE_CHANGE = "rapid_state_change"
    MISSING_DATA = "missing_data"
    INTERPOLATED_DATA = "interpolated_data"


class AgeGroup(Enum):
    """Age groups for population-specific analysis."""
    YOUNG_CHILD = "young_child"  # 3-5 years
    SCHOOL_AGE = "school_age"  # 6-8 years
    PRE_TEEN = "pre_teen"  # 9-12 years
    TEEN = "teen"  # 13-17 years
    ADULT = "adult"  # 18+ years


class ClinicalPopulation(Enum):
    """Clinical population types."""
    NEUROTYPICAL = "neurotypical"
    AUTISM_SPECTRUM = "autism_spectrum"
    ADHD = "adhd"
    ANXIETY_DISORDER = "anxiety_disorder"
    DEVELOPMENTAL_DELAY = "developmental_delay"
    MIXED_CLINICAL = "mixed_clinical"


class ExportFormat(Enum):
    """Data export formats."""
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    PARQUET = "parquet"


class DataType(Enum):
    """Types of data for export."""
    FRAME_LEVEL = "frame_level"
    EPISODE_LEVEL = "episode_level"
    AGGREGATED_METRICS = "aggregated_metrics"
    AUDIT_TRAIL = "audit_trail"
    ATTENTION_HEATMAPS = "attention_heatmaps"
    COMPREHENSIVE_REPORT = "comprehensive_report"


class GazeTarget(Enum):
    """Gaze target classifications."""
    CAMERA_DIRECT = "camera_direct"  # Looking directly at camera
    CAMERA_PERIPHERAL = "camera_peripheral"  # Near-camera gaze
    FACE_REGION = "face_region"  # Looking at therapist's face
    OBJECT_OF_INTEREST = "object_of_interest"  # Specific object
    BACKGROUND = "background"  # Environmental background
    OFF_SCREEN = "off_screen"  # Looking outside video frame
    UNKNOWN = "unknown"  # Cannot determine target