"""
Interface definitions for the Enhanced Eye Contact & Attention Tracking System.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
from datetime import timedelta

from .data_models import (
    FrameResult, GazeVector, JointAttentionEvent, VisualTrackingData,
    AttentionZoneEvent, ComprehensiveMetrics, AuditEntry, ThresholdConfig,
    TimeRange, TimeWindow, StabilityMetrics, AlignmentScore
)
from .enums import DetectionApproach, GazeTarget, AttentionType, ScanningPattern


class DetectionEngine(ABC):
    """Interface for eye contact detection engines."""
    
    @abstractmethod
    def configure_approach(self, approach: DetectionApproach, config: ThresholdConfig) -> None:
        """Configure the detection approach and parameters."""
        pass
    
    @abstractmethod
    def process_frame(self, frame: Any, timestamp: float) -> FrameResult:
        """Process a single video frame and return detection result."""
        pass
    
    @abstractmethod
    def get_supported_approaches(self) -> List[DetectionApproach]:
        """Get list of supported detection approaches."""
        pass
    
    @abstractmethod
    def validate_configuration(self, config: ThresholdConfig) -> List[str]:
        """Validate configuration and return any errors."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the detection engine state."""
        pass


class GazeDirectionAnalyzer(ABC):
    """Interface for gaze direction analysis."""
    
    @abstractmethod
    def calculate_gaze_vector(self, face_landmarks: Any, head_pose: Tuple[float, float, float]) -> Optional[GazeVector]:
        """Calculate 3D gaze vector from face landmarks and head pose."""
        pass
    
    @abstractmethod
    def classify_gaze_target(self, gaze_vector: GazeVector, attention_zones: List[Any]) -> GazeTarget:
        """Classify what the subject is looking at."""
        pass
    
    @abstractmethod
    def get_gaze_confidence(self, face_quality: float, landmark_confidence: float) -> float:
        """Calculate confidence score for gaze estimation."""
        pass
    
    @abstractmethod
    def track_gaze_stability(self, gaze_history: List[GazeVector], window_size: int) -> StabilityMetrics:
        """Analyze gaze stability over a time window."""
        pass
    
    @abstractmethod
    def detect_gaze_shifts(self, gaze_sequence: List[GazeVector], sensitivity: float) -> List[float]:
        """Detect significant gaze shifts and return timestamps."""
        pass


class JointAttentionDetector(ABC):
    """Interface for joint attention detection."""
    
    @abstractmethod
    def detect_joint_attention(self, subject_gaze: GazeVector, partner_gaze: Optional[GazeVector], 
                              objects: List[Any]) -> Optional[JointAttentionEvent]:
        """Detect joint attention between subject and partner."""
        pass
    
    @abstractmethod
    def classify_attention_type(self, event: JointAttentionEvent, 
                               temporal_context: Dict[str, Any]) -> AttentionType:
        """Classify the type of joint attention (initiated, responding, mutual)."""
        pass
    
    @abstractmethod
    def calculate_attention_alignment(self, gaze_vectors: List[GazeVector]) -> AlignmentScore:
        """Calculate how well aligned multiple gaze vectors are."""
        pass
    
    @abstractmethod
    def track_attention_shifts(self, attention_history: List[Any]) -> Dict[str, Any]:
        """Track patterns in attention shifting behavior."""
        pass
    
    @abstractmethod
    def measure_response_latency(self, cue_timestamp: float, response_timestamp: float) -> float:
        """Measure latency between attention cue and response."""
        pass


class VisualTrackingAnalyzer(ABC):
    """Interface for visual tracking pattern analysis."""
    
    @abstractmethod
    def analyze_eye_movements(self, gaze_sequence: List[GazeVector], 
                             timestamps: List[float]) -> VisualTrackingData:
        """Analyze eye movement patterns from gaze sequence."""
        pass
    
    @abstractmethod
    def detect_scanning_patterns(self, gaze_path: List[GazeVector], 
                                attention_zones: List[Any]) -> ScanningPattern:
        """Identify the dominant scanning pattern."""
        pass
    
    @abstractmethod
    def calculate_fixation_metrics(self, gaze_sequence: List[GazeVector]) -> Dict[str, float]:
        """Calculate fixation duration and frequency metrics."""
        pass
    
    @abstractmethod
    def identify_repetitive_behaviors(self, visual_patterns: List[Any]) -> float:
        """Identify and score repetitive visual behaviors (0-1)."""
        pass
    
    @abstractmethod
    def calculate_saccade_metrics(self, gaze_sequence: List[GazeVector]) -> Dict[str, float]:
        """Calculate saccadic eye movement metrics."""
        pass
    
    @abstractmethod
    def assess_attention_stability(self, gaze_data: List[GazeVector], window_duration: float) -> float:
        """Assess how stable attention is over time (0-1)."""
        pass


class AttentionZoneTracker(ABC):
    """Interface for attention zone tracking."""
    
    @abstractmethod
    def configure_zones(self, zone_definitions: List[Dict[str, Any]]) -> None:
        """Configure attention zones for tracking."""
        pass
    
    @abstractmethod
    def track_zone_attention(self, gaze_vector: GazeVector, timestamp: float) -> Optional[AttentionZoneEvent]:
        """Track attention within configured zones."""
        pass
    
    @abstractmethod
    def calculate_zone_metrics(self, time_window: TimeWindow) -> Dict[str, Any]:
        """Calculate attention metrics for each zone."""
        pass
    
    @abstractmethod
    def generate_attention_heatmap(self, attention_data: List[AttentionZoneEvent]) -> Any:
        """Generate attention heatmap visualization data."""
        pass
    
    @abstractmethod
    def detect_zone_transitions(self, attention_sequence: List[AttentionZoneEvent]) -> List[Dict[str, Any]]:
        """Detect and analyze transitions between attention zones."""
        pass
    
    @abstractmethod
    def calculate_zone_preferences(self, zone_events: List[AttentionZoneEvent]) -> Dict[str, float]:
        """Calculate preference scores for each attention zone."""
        pass


class AttentionMetricsCollector(ABC):
    """Interface for comprehensive attention metrics collection."""
    
    @abstractmethod
    def add_frame_result(self, result: FrameResult) -> None:
        """Add a frame-level detection result."""
        pass
    
    @abstractmethod
    def add_gaze_data(self, gaze_vector: GazeVector, timestamp: float) -> None:
        """Add gaze direction data."""
        pass
    
    @abstractmethod
    def add_joint_attention_event(self, event: JointAttentionEvent) -> None:
        """Add a joint attention event."""
        pass
    
    @abstractmethod
    def add_visual_tracking_data(self, tracking_data: VisualTrackingData) -> None:
        """Add visual tracking analysis data."""
        pass
    
    @abstractmethod
    def calculate_comprehensive_metrics(self, window_size: timedelta) -> ComprehensiveMetrics:
        """Calculate comprehensive metrics for a time window."""
        pass
    
    @abstractmethod
    def get_attention_distribution(self) -> Dict[str, float]:
        """Get distribution of attention across different targets/zones."""
        pass
    
    @abstractmethod
    def export_attention_data(self, format_type: str, time_range: Optional[TimeRange] = None) -> Any:
        """Export attention data in specified format."""
        pass
    
    @abstractmethod
    def get_attention_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive data quality report."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the metrics collector state."""
        pass


class AuditTrail(ABC):
    """Interface for audit trail management."""
    
    @abstractmethod
    def log_detection_decision(self, timestamp: float, decision: bool, confidence: float, 
                              approach: DetectionApproach) -> None:
        """Log a detection decision."""
        pass
    
    @abstractmethod
    def log_configuration_change(self, config: ThresholdConfig, timestamp: float) -> None:
        """Log a configuration change."""
        pass
    
    @abstractmethod
    def log_missing_data_event(self, start_time: float, duration: float, cause: str) -> None:
        """Log a missing data event."""
        pass
    
    @abstractmethod
    def log_quality_warning(self, timestamp: float, warning_type: str, details: Dict[str, Any]) -> None:
        """Log a data quality warning."""
        pass
    
    @abstractmethod
    def generate_audit_report(self, time_range: TimeRange) -> Dict[str, Any]:
        """Generate comprehensive audit report for time range."""
        pass
    
    @abstractmethod
    def export_audit_data(self, format_type: str, time_range: Optional[TimeRange] = None) -> Any:
        """Export audit trail data."""
        pass
    
    @abstractmethod
    def get_audit_entries(self, time_range: Optional[TimeRange] = None) -> List[AuditEntry]:
        """Get audit entries for specified time range."""
        pass
    
    @abstractmethod
    def verify_integrity(self) -> Dict[str, Any]:
        """Verify audit trail integrity and completeness."""
        pass


class MissingDataHandler(ABC):
    """Interface for handling missing data and quality issues."""
    
    @abstractmethod
    def detect_missing_data(self, timestamps: List[float], expected_fps: float) -> List[Tuple[float, float]]:
        """Detect gaps in data and return (start_time, duration) tuples."""
        pass
    
    @abstractmethod
    def classify_missing_data_cause(self, gap_duration: float, context: Dict[str, Any]) -> str:
        """Classify the likely cause of missing data."""
        pass
    
    @abstractmethod
    def calculate_confidence_intervals(self, data_completeness: float) -> Tuple[float, float]:
        """Calculate confidence intervals based on data completeness."""
        pass
    
    @abstractmethod
    def generate_quality_warnings(self, metrics: ComprehensiveMetrics) -> List[str]:
        """Generate quality warnings based on metrics."""
        pass
    
    @abstractmethod
    def interpolate_missing_data(self, data_sequence: List[Any], gap_indices: List[int]) -> List[Any]:
        """Interpolate missing data points where appropriate."""
        pass


# Factory interface for creating system components
class AttentionTrackingSystemFactory(ABC):
    """Factory interface for creating attention tracking system components."""
    
    @abstractmethod
    def create_detection_engine(self, approach: DetectionApproach) -> DetectionEngine:
        """Create a detection engine for the specified approach."""
        pass
    
    @abstractmethod
    def create_gaze_analyzer(self) -> GazeDirectionAnalyzer:
        """Create a gaze direction analyzer."""
        pass
    
    @abstractmethod
    def create_joint_attention_detector(self) -> JointAttentionDetector:
        """Create a joint attention detector."""
        pass
    
    @abstractmethod
    def create_visual_tracking_analyzer(self) -> VisualTrackingAnalyzer:
        """Create a visual tracking analyzer."""
        pass
    
    @abstractmethod
    def create_attention_zone_tracker(self) -> AttentionZoneTracker:
        """Create an attention zone tracker."""
        pass
    
    @abstractmethod
    def create_metrics_collector(self) -> AttentionMetricsCollector:
        """Create an attention metrics collector."""
        pass
    
    @abstractmethod
    def create_audit_trail(self) -> AuditTrail:
        """Create an audit trail manager."""
        pass
    
    @abstractmethod
    def create_missing_data_handler(self) -> MissingDataHandler:
        """Create a missing data handler."""
        pass