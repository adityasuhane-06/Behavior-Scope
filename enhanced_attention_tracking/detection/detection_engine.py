"""
Detection Engine implementation for Enhanced Eye Contact & Attention Tracking System.

This module provides the main detection engine that orchestrates multiple detection
approaches and manages threshold configurations.
"""

import logging
from typing import List, Dict, Optional, Any
import numpy as np

from ..core.interfaces import DetectionEngine
from ..core.data_models import FrameResult, GazeVector, ThresholdConfig
from ..core.enums import DetectionApproach, QualityFlag, GazeTarget

logger = logging.getLogger(__name__)


class DetectionEngineImpl(DetectionEngine):
    """
    Main detection engine implementation.
    
    Orchestrates multiple detection approaches and manages configuration.
    Integrates with existing MediaPipe and AI-enhanced detection methods.
    """
    
    def __init__(self):
        """Initialize the detection engine."""
        self.current_approach = DetectionApproach.EPISODE_BASED
        self.config = ThresholdConfig(detection_approach=self.current_approach)
        self.frame_history: List[FrameResult] = []
        self.supported_approaches = [
            DetectionApproach.EPISODE_BASED,
            DetectionApproach.CONTINUOUS_SCORING,
            DetectionApproach.FRAME_LEVEL_TRACKING,
            DetectionApproach.HYBRID
        ]
        
        # Integration with existing system
        self._mediapipe_detector = None
        self._ai_enhanced_detector = None
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize integration with existing detection systems."""
        try:
            # Try to import existing eye contact analysis
            from autism_analysis.eye_contact import analyze_eye_contact
            from autism_analysis.enhanced_eye_contact import analyze_eye_contact_enhanced
            
            self._legacy_analyzer = analyze_eye_contact
            self._enhanced_analyzer = analyze_eye_contact_enhanced
            logger.info("Successfully integrated with existing eye contact analysis")
            
        except ImportError as e:
            logger.warning(f"Could not import existing analyzers: {e}")
            self._legacy_analyzer = None
            self._enhanced_analyzer = None
    
    def configure_approach(self, approach: DetectionApproach, config: ThresholdConfig) -> None:
        """Configure the detection approach and parameters."""
        if approach not in self.supported_approaches:
            raise ValueError(f"Unsupported detection approach: {approach}")
        
        # Validate configuration
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {errors}")
        
        self.current_approach = approach
        self.config = config
        
        logger.info(f"Configured detection engine with approach: {approach}")
        logger.debug(f"Configuration: confidence_threshold={config.confidence_threshold}, "
                    f"min_episode_duration={config.minimum_episode_duration}")
    
    def process_frame(self, frame: Any, timestamp: float) -> FrameResult:
        """Process a single video frame and return detection result."""
        try:
            # Extract face features from frame (integrate with existing pipeline)
            face_features = self._extract_face_features(frame)
            
            if face_features is None:
                return self._create_failed_frame_result(timestamp, "Face detection failed")
            
            # Calculate gaze vector
            gaze_vector = self._calculate_gaze_vector(face_features, timestamp)
            
            # Apply detection approach
            confidence_score = self._calculate_confidence_score(face_features, gaze_vector)
            binary_decision = self._apply_threshold(confidence_score)
            
            # Classify gaze target
            gaze_target = self._classify_gaze_target(gaze_vector) if gaze_vector else None
            
            # Determine quality flags
            quality_flags = self._assess_quality(face_features, gaze_vector, confidence_score)
            
            # Create frame result
            frame_result = FrameResult(
                timestamp=timestamp,
                confidence_score=confidence_score,
                binary_decision=binary_decision,
                detection_approach=self.current_approach,
                gaze_vector=gaze_vector,
                gaze_target=gaze_target,
                quality_flags=quality_flags,
                processing_metadata={
                    'face_detected': face_features is not None,
                    'gaze_estimated': gaze_vector is not None,
                    'approach': self.current_approach.value
                }
            )
            
            # Store in history for temporal analysis
            self.frame_history.append(frame_result)
            
            # Apply temporal smoothing if enabled
            if self.config.temporal_smoothing:
                frame_result = self._apply_temporal_smoothing(frame_result)
            
            return frame_result
            
        except Exception as e:
            logger.error(f"Error processing frame at {timestamp}: {e}")
            return self._create_failed_frame_result(timestamp, f"Processing error: {e}")
    
    def _extract_face_features(self, frame: Any) -> Optional[Dict[str, Any]]:
        """Extract face features from frame using existing pipeline."""
        try:
            # Get frame data from frame object (may be dict with 'data' key or raw array)
            if isinstance(frame, dict):
                frame_array = frame.get('data')
                frame_idx = frame.get('frame_id', 0)
                timestamp = frame.get('timestamp', 0.0)
            else:
                frame_array = frame
                frame_idx = 0
                timestamp = 0.0
            
            if frame_array is None:
                return None
            
            # Use real face analyzer from video_pipeline
            try:
                from video_pipeline.face_analyzer import FaceAnalyzer
                
                # Initialize face analyzer if not already done
                if not hasattr(self, '_face_analyzer') or self._face_analyzer is None:
                    self._face_analyzer = FaceAnalyzer(
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                        static_image_mode=False
                    )
                
                # Analyze the frame
                face_result = self._face_analyzer.analyze_frame(
                    frame_array, 
                    frame_idx=frame_idx, 
                    timestamp=timestamp
                )
                
                if face_result.face_detected:
                    return {
                        'face_detected': True,
                        'head_pose': face_result.head_pose,  # (yaw, pitch, roll)
                        'landmarks': None,  # Not directly available, but head_pose is
                        'confidence': face_result.landmark_confidence,
                        'gaze_proxy': face_result.gaze_proxy,
                        'facial_motion_energy': face_result.facial_motion_energy
                    }
                else:
                    return {
                        'face_detected': False,
                        'head_pose': (0.0, 0.0, 0.0),
                        'landmarks': None,
                        'confidence': 0.0,
                        'gaze_proxy': 0.0,
                        'facial_motion_energy': 0.0
                    }
                    
            except ImportError as ie:
                logger.warning(f"Could not import FaceAnalyzer: {ie}. Using fallback.")
                # Fallback to basic face detection if FaceAnalyzer not available
                return self._fallback_face_detection(frame_array)
            
        except Exception as e:
            logger.warning(f"Face feature extraction failed: {e}")
            return None
    
    def _fallback_face_detection(self, frame_array) -> Optional[Dict[str, Any]]:
        """Fallback face detection when FaceAnalyzer is not available."""
        try:
            import mediapipe as mp
            
            # Initialize MediaPipe Face Detection
            if not hasattr(self, '_mp_face_mesh') or self._mp_face_mesh is None:
                self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5
                )
            
            # Process frame
            results = self._mp_face_mesh.process(frame_array)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                h, w = frame_array.shape[:2]
                
                # Extract landmarks as numpy array
                landmarks = np.array([
                    [lm.x * w, lm.y * h, lm.z]
                    for lm in face_landmarks.landmark
                ])
                
                # Simple head pose estimation from landmarks
                # Using nose tip, chin, left/right eye outer corners
                nose_tip = landmarks[1]  # Nose tip
                chin = landmarks[152]  # Chin
                left_eye = landmarks[33]  # Left eye outer
                right_eye = landmarks[263]  # Right eye outer
                
                # Calculate yaw from eye positions
                eye_center = (left_eye + right_eye) / 2
                yaw = np.degrees(np.arctan2(right_eye[0] - left_eye[0], 
                                           right_eye[2] - left_eye[2])) - 90
                
                # Calculate pitch from nose-chin vector
                face_vector = chin - nose_tip
                pitch = np.degrees(np.arctan2(face_vector[1], face_vector[2])) - 90
                
                # Roll is approximated from eye line angle
                roll = np.degrees(np.arctan2(right_eye[1] - left_eye[1], 
                                            right_eye[0] - left_eye[0]))
                
                return {
                    'face_detected': True,
                    'head_pose': (float(yaw), float(pitch), float(roll)),
                    'landmarks': landmarks,
                    'confidence': 0.9  # MediaPipe typically has high confidence when detected
                }
            else:
                return {
                    'face_detected': False,
                    'head_pose': (0.0, 0.0, 0.0),
                    'landmarks': None,
                    'confidence': 0.0
                }
                
        except Exception as e:
            logger.warning(f"Fallback face detection failed: {e}")
            return None

    
    def _calculate_gaze_vector(self, face_features: Dict[str, Any], timestamp: float) -> Optional[GazeVector]:
        """Calculate 3D gaze vector from face features."""
        try:
            if not face_features.get('face_detected', False):
                return None
            
            head_pose = face_features.get('head_pose', (0.0, 0.0, 0.0))
            landmarks = face_features.get('landmarks')
            confidence = face_features.get('confidence', 0.0)
            
            if landmarks is None:
                return None
            
            # Simplified gaze estimation (in real implementation, this would use
            # proper 3D gaze estimation algorithms)
            yaw, pitch, roll = head_pose
            
            # Convert head pose to gaze direction (simplified)
            # In real implementation, this would use eye landmarks for precise gaze
            gaze_x = np.sin(np.radians(yaw))
            gaze_y = -np.sin(np.radians(pitch))  # Negative because screen Y is inverted
            gaze_z = np.cos(np.radians(yaw)) * np.cos(np.radians(pitch))
            
            # Normalize to unit vector
            magnitude = np.sqrt(gaze_x**2 + gaze_y**2 + gaze_z**2)
            if magnitude > 0:
                gaze_x /= magnitude
                gaze_y /= magnitude
                gaze_z /= magnitude
            
            return GazeVector(
                x=gaze_x,
                y=gaze_y,
                z=gaze_z,
                confidence=confidence,
                timestamp=timestamp
            )
            
        except Exception as e:
            logger.warning(f"Gaze vector calculation failed: {e}")
            return None
    
    def _calculate_confidence_score(self, face_features: Dict[str, Any], gaze_vector: Optional[GazeVector]) -> float:
        """Calculate eye contact confidence score based on detection approach."""
        if self.current_approach == DetectionApproach.EPISODE_BASED:
            return self._episode_based_confidence(face_features, gaze_vector)
        elif self.current_approach == DetectionApproach.CONTINUOUS_SCORING:
            return self._continuous_scoring_confidence(face_features, gaze_vector)
        elif self.current_approach == DetectionApproach.FRAME_LEVEL_TRACKING:
            return self._frame_level_confidence(face_features, gaze_vector)
        elif self.current_approach == DetectionApproach.HYBRID:
            return self._hybrid_confidence(face_features, gaze_vector)
        else:
            return 0.0
    
    def _episode_based_confidence(self, face_features: Dict[str, Any], gaze_vector: Optional[GazeVector]) -> float:
        """Calculate confidence for episode-based detection."""
        if not face_features.get('face_detected', False) or gaze_vector is None:
            return 0.0
        
        # Check if gaze is directed toward camera
        # Camera is typically at (0, 0, 1) in normalized coordinates
        camera_direction = np.array([0.0, 0.0, 1.0])
        gaze_direction = np.array([gaze_vector.x, gaze_vector.y, gaze_vector.z])
        
        # Calculate alignment with camera direction
        dot_product = np.dot(gaze_direction, camera_direction)
        alignment_score = max(0.0, dot_product)  # Only positive alignment
        
        # Combine with face detection confidence
        face_confidence = face_features.get('confidence', 0.0)
        
        return (alignment_score * 0.7 + face_confidence * 0.3) * gaze_vector.confidence
    
    def _continuous_scoring_confidence(self, face_features: Dict[str, Any], gaze_vector: Optional[GazeVector]) -> float:
        """Calculate confidence for continuous scoring approach."""
        if not face_features.get('face_detected', False):
            return 0.0
        
        base_confidence = face_features.get('confidence', 0.0)
        
        if gaze_vector is None:
            return base_confidence * 0.5  # Reduced confidence without gaze
        
        # Continuous scoring considers gaze stability over time
        if len(self.frame_history) > 0:
            recent_frames = self.frame_history[-5:]  # Last 5 frames
            gaze_stability = self._calculate_gaze_stability(recent_frames)
            return base_confidence * gaze_vector.confidence * gaze_stability
        
        return base_confidence * gaze_vector.confidence
    
    def _frame_level_confidence(self, face_features: Dict[str, Any], gaze_vector: Optional[GazeVector]) -> float:
        """Calculate confidence for frame-level tracking."""
        if not face_features.get('face_detected', False):
            return 0.0
        
        # Frame-level tracking focuses on immediate detection
        head_pose = face_features.get('head_pose', (0.0, 0.0, 0.0))
        yaw, pitch, roll = head_pose
        
        # Check if head is facing forward
        head_facing_score = max(0.0, 1.0 - abs(yaw) / 30.0)  # 30 degrees threshold
        
        if gaze_vector is None:
            return head_facing_score * face_features.get('confidence', 0.0)
        
        # Combine head pose and gaze direction
        gaze_forward_score = max(0.0, gaze_vector.z)  # Z component indicates forward gaze
        
        return (head_facing_score * 0.6 + gaze_forward_score * 0.4) * gaze_vector.confidence
    
    def _hybrid_confidence(self, face_features: Dict[str, Any], gaze_vector: Optional[GazeVector]) -> float:
        """Calculate confidence using hybrid approach."""
        # Combine multiple approaches
        episode_conf = self._episode_based_confidence(face_features, gaze_vector)
        continuous_conf = self._continuous_scoring_confidence(face_features, gaze_vector)
        frame_conf = self._frame_level_confidence(face_features, gaze_vector)
        
        # Weighted combination
        return (episode_conf * 0.4 + continuous_conf * 0.3 + frame_conf * 0.3)
    
    def _apply_threshold(self, confidence_score: float) -> bool:
        """Apply threshold to convert confidence to binary decision."""
        return confidence_score >= self.config.confidence_threshold
    
    def _classify_gaze_target(self, gaze_vector: GazeVector) -> GazeTarget:
        """Classify what the subject is looking at."""
        # Simplified classification based on gaze vector
        if gaze_vector.z > 0.8:  # Looking forward
            if abs(gaze_vector.x) < 0.2 and abs(gaze_vector.y) < 0.2:
                return GazeTarget.CAMERA_DIRECT
            else:
                return GazeTarget.CAMERA_PERIPHERAL
        elif gaze_vector.z > 0.3:
            return GazeTarget.FACE_REGION
        else:
            return GazeTarget.OFF_SCREEN
    
    def _assess_quality(self, face_features: Optional[Dict[str, Any]], 
                       gaze_vector: Optional[GazeVector], confidence_score: float) -> List[QualityFlag]:
        """Assess data quality and return appropriate flags."""
        flags = []
        
        if face_features is None:
            flags.append(QualityFlag.FACE_DETECTION_FAILED)
            flags.append(QualityFlag.LOW_QUALITY)
            return flags
        
        if not face_features.get('face_detected', False):
            flags.append(QualityFlag.FACE_DETECTION_FAILED)
        
        if gaze_vector is None:
            flags.append(QualityFlag.GAZE_ESTIMATION_UNCERTAIN)
        
        # Check for rapid state changes
        if len(self.frame_history) > 0:
            last_result = self.frame_history[-1]
            if last_result.binary_decision != (confidence_score >= self.config.confidence_threshold):
                flags.append(QualityFlag.RAPID_STATE_CHANGE)
        
        # Overall quality assessment
        if confidence_score > 0.8 and gaze_vector and gaze_vector.confidence > 0.8:
            flags.append(QualityFlag.HIGH_QUALITY)
        elif confidence_score > 0.5:
            flags.append(QualityFlag.MODERATE_QUALITY)
        else:
            flags.append(QualityFlag.LOW_QUALITY)
        
        return flags
    
    def _calculate_gaze_stability(self, recent_frames: List[FrameResult]) -> float:
        """Calculate gaze stability over recent frames."""
        if len(recent_frames) < 2:
            return 1.0
        
        gaze_vectors = [f.gaze_vector for f in recent_frames if f.gaze_vector is not None]
        if len(gaze_vectors) < 2:
            return 0.5
        
        # Calculate variance in gaze direction
        x_values = [gv.x for gv in gaze_vectors]
        y_values = [gv.y for gv in gaze_vectors]
        
        x_var = np.var(x_values)
        y_var = np.var(y_values)
        
        # Convert variance to stability score (lower variance = higher stability)
        stability = 1.0 / (1.0 + x_var + y_var)
        return min(1.0, stability)
    
    def _apply_temporal_smoothing(self, frame_result: FrameResult) -> FrameResult:
        """Apply temporal smoothing to reduce noise."""
        if len(self.frame_history) < 3:
            return frame_result
        
        # Simple moving average of confidence scores
        recent_confidences = [f.confidence_score for f in self.frame_history[-3:]]
        smoothed_confidence = np.mean(recent_confidences)
        
        # Update frame result with smoothed confidence
        frame_result.confidence_score = smoothed_confidence
        frame_result.binary_decision = smoothed_confidence >= self.config.confidence_threshold
        
        return frame_result
    
    def _create_failed_frame_result(self, timestamp: float, reason: str) -> FrameResult:
        """Create a frame result for failed processing."""
        return FrameResult(
            timestamp=timestamp,
            confidence_score=0.0,
            binary_decision=False,
            detection_approach=self.current_approach,
            gaze_vector=None,
            gaze_target=None,
            quality_flags=[QualityFlag.LOW_QUALITY, QualityFlag.MISSING_DATA],
            processing_metadata={'error': reason}
        )
    
    def get_supported_approaches(self) -> List[DetectionApproach]:
        """Get list of supported detection approaches."""
        return self.supported_approaches.copy()
    
    def validate_configuration(self, config: ThresholdConfig) -> List[str]:
        """Validate configuration and return any errors."""
        return config.validate()
    
    def reset(self) -> None:
        """Reset the detection engine state."""
        self.frame_history.clear()
        logger.info("Detection engine state reset")
    
    def get_frame_history(self) -> List[FrameResult]:
        """Get frame processing history (for debugging/analysis)."""
        return self.frame_history.copy()
    
    def get_current_config(self) -> ThresholdConfig:
        """Get current configuration."""
        return self.config