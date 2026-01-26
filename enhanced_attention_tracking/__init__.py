"""
Enhanced Eye Contact & Attention Tracking System

This module provides comprehensive eye contact and attention analysis capabilities
for behavioral assessment videos, including:

- Multiple detection approaches (episode-based, continuous scoring, frame-level)
- Gaze direction analysis with 3D vector calculation
- Joint attention detection between subjects and therapists/caregivers
- Visual tracking pattern analysis (fixations, saccades, scanning behaviors)
- Attention zone configuration and tracking
- Clinical-grade reporting and audit trails

The system integrates seamlessly with existing autism analysis pipelines while
providing robust, auditable measurements for clinical and research applications.
"""

from .core.data_models import (
    FrameResult,
    GazeVector,
    JointAttentionEvent,
    VisualTrackingData,
    AttentionZoneEvent,
    EpisodeData,
    ComprehensiveMetrics,
    AuditEntry
)

from .core.interfaces import (
    DetectionEngine,
    GazeDirectionAnalyzer,
    JointAttentionDetector,
    VisualTrackingAnalyzer,
    AttentionZoneTracker,
    AttentionMetricsCollector,
    AuditTrail
)

from .core.enums import (
    DetectionApproach,
    AttentionType,
    ScanningPattern,
    ZoneType,
    AuditEventType,
    QualityFlag
)

__version__ = "1.0.0"
__author__ = "Behavioral Analysis Team"

# Main system components will be imported here as they're implemented
# from .detection import DetectionEngineImpl
# from .analysis import GazeDirectionAnalyzerImpl, JointAttentionDetectorImpl
# from .tracking import VisualTrackingAnalyzerImpl, AttentionZoneTrackerImpl
# from .metrics import AttentionMetricsCollectorImpl
# from .audit import AuditTrailImpl