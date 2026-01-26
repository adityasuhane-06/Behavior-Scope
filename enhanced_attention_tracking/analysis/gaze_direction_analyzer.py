"""
Gaze Direction Analyzer implementation for Enhanced Eye Contact & Attention Tracking System.

This module provides 3D gaze vector calculation, gaze target classification,
and gaze stability analysis.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

from ..core.interfaces import GazeDirectionAnalyzer
from ..core.data_models import GazeVector, StabilityMetrics
from ..core.enums import GazeTarget

logger = logging.getLogger(__name__)


class GazeDirectionAnalyzerImpl(GazeDirectionAnalyzer):
    """
    Implementation of gaze direction analysis.
    
    Provides 3D gaze vector calculation from face landmarks and head pose,
    gaze target classification, and stability tracking.
    """
    
    def __init__(self):
        """Initialize the gaze direction analyzer."""
        self.gaze_history: List[GazeVector] = []
        self.stability_window_size = 10  # Number of frames for stability analysis
        
        # Camera parameters (can be configured)
        self.camera_position = np.array([0.0, 0.0, 1.0])  # Normalized camera direction
        self.face_region_threshold = 0.3  # Threshold for face region classification
        self.direct_gaze_threshold = 0.15  # Threshold for direct camera gaze
        
        logger.info("Gaze Direction Analyzer initialized")
    
    def calculate_gaze_vector(self, face_landmarks: Any, head_pose: Tuple[float, float, float]) -> Optional[GazeVector]:
        """
        Calculate 3D gaze vector from face landmarks and head pose.
        
        Args:
            face_landmarks: MediaPipe face landmarks or similar structure
            head_pose: (yaw, pitch, roll) in degrees
            
        Returns:
            GazeVector with 3D direction and confidence, or None if calculation fails
        """
        try:
            if face_landmarks is None or head_pose is None:
                return None
            
            yaw, pitch, roll = head_pose
            
            # Validate head pose values
            if abs(yaw) > 90 or abs(pitch) > 90 or abs(roll) > 90:
                logger.warning(f"Extreme head pose values: yaw={yaw}, pitch={pitch}, roll={roll}")
                return None
            
            # Method 1: Head pose-based gaze estimation (primary method)
            gaze_vector_head = self._calculate_gaze_from_head_pose(yaw, pitch, roll)
            
            # Method 2: Eye landmark-based refinement (if landmarks available)
            gaze_refinement = self._calculate_gaze_refinement_from_landmarks(face_landmarks)
            
            # Combine head pose and eye landmark information
            if gaze_refinement is not None:
                # Weighted combination: head pose (70%) + eye refinement (30%)
                final_x = gaze_vector_head[0] * 0.7 + gaze_refinement[0] * 0.3
                final_y = gaze_vector_head[1] * 0.7 + gaze_refinement[1] * 0.3
                final_z = gaze_vector_head[2] * 0.7 + gaze_refinement[2] * 0.3
                confidence = 0.9  # High confidence when both methods available
            else:
                # Use head pose only
                final_x, final_y, final_z = gaze_vector_head
                confidence = 0.7  # Moderate confidence with head pose only
            
            # Normalize the vector
            magnitude = np.sqrt(final_x**2 + final_y**2 + final_z**2)
            if magnitude > 0:
                final_x /= magnitude
                final_y /= magnitude
                final_z /= magnitude
            
            # Create gaze vector with current timestamp
            import time
            gaze_vector = GazeVector(
                x=final_x,
                y=final_y,
                z=final_z,
                confidence=confidence,
                timestamp=time.time()
            )
            
            # Store in history for stability analysis
            self.gaze_history.append(gaze_vector)
            if len(self.gaze_history) > self.stability_window_size * 2:
                self.gaze_history = self.gaze_history[-self.stability_window_size:]
            
            return gaze_vector
            
        except Exception as e:
            logger.error(f"Error calculating gaze vector: {e}")
            return None
    
    def _calculate_gaze_from_head_pose(self, yaw: float, pitch: float, roll: float) -> Tuple[float, float, float]:
        """Calculate gaze direction from head pose angles."""
        # Convert degrees to radians
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        roll_rad = np.radians(roll)
        
        # Calculate 3D gaze direction vector
        # Assuming forward gaze when head is straight (yaw=0, pitch=0)
        gaze_x = np.sin(yaw_rad) * np.cos(pitch_rad)
        gaze_y = -np.sin(pitch_rad)  # Negative because screen Y is typically inverted
        gaze_z = np.cos(yaw_rad) * np.cos(pitch_rad)
        
        return (gaze_x, gaze_y, gaze_z)
    
    def _calculate_gaze_refinement_from_landmarks(self, face_landmarks: Any) -> Optional[Tuple[float, float, float]]:
        """
        Calculate gaze refinement from eye landmarks.
        
        This is a simplified implementation. In a full system, this would use
        detailed eye landmark analysis for precise gaze estimation.
        """
        try:
            if face_landmarks is None:
                return None
            
            # For MediaPipe landmarks, we would extract eye region landmarks
            # and calculate iris position relative to eye corners
            
            # Simplified implementation: assume landmarks is a numpy array
            if hasattr(face_landmarks, 'shape') and len(face_landmarks.shape) == 2:
                # Extract eye region landmarks (simplified indices)
                # In real MediaPipe, these would be specific landmark indices for eyes
                left_eye_landmarks = face_landmarks[33:42]  # Approximate left eye region
                right_eye_landmarks = face_landmarks[362:371]  # Approximate right eye region
                
                # Calculate average eye gaze direction (simplified)
                left_center = np.mean(left_eye_landmarks, axis=0)
                right_center = np.mean(right_eye_landmarks, axis=0)
                
                # Simple gaze estimation based on eye center positions
                eye_center = (left_center + right_center) / 2
                
                # Convert to normalized gaze direction (simplified)
                gaze_x = (eye_center[0] - 0.5) * 2  # Normalize to [-1, 1]
                gaze_y = (eye_center[1] - 0.5) * 2
                gaze_z = 0.8  # Assume mostly forward gaze
                
                return (gaze_x, gaze_y, gaze_z)
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not calculate gaze refinement: {e}")
            return None
    
    def classify_gaze_target(self, gaze_vector: GazeVector, attention_zones: List[Any]) -> GazeTarget:
        """
        Classify what the subject is looking at based on gaze vector.
        
        Args:
            gaze_vector: 3D gaze direction vector
            attention_zones: List of defined attention zones (optional)
            
        Returns:
            GazeTarget classification
        """
        try:
            # Calculate alignment with camera direction
            camera_alignment = np.dot(
                [gaze_vector.x, gaze_vector.y, gaze_vector.z],
                self.camera_position
            )
            
            # Direct camera gaze (high Z component, low X/Y deviation)
            if (gaze_vector.z > 0.8 and 
                abs(gaze_vector.x) < self.direct_gaze_threshold and 
                abs(gaze_vector.y) < self.direct_gaze_threshold):
                return GazeTarget.CAMERA_DIRECT
            
            # Peripheral camera gaze (moderate Z component)
            elif (gaze_vector.z > 0.5 and camera_alignment > 0.6):
                return GazeTarget.CAMERA_PERIPHERAL
            
            # Face region (moderate forward gaze with some deviation)
            elif (gaze_vector.z > self.face_region_threshold and 
                  camera_alignment > 0.3):
                return GazeTarget.FACE_REGION
            
            # Check attention zones if provided
            if attention_zones:
                zone_target = self._classify_attention_zone_target(gaze_vector, attention_zones)
                if zone_target != GazeTarget.UNKNOWN:
                    return zone_target
            
            # Off-screen gaze (low Z component or extreme X/Y)
            if (gaze_vector.z < 0.2 or 
                abs(gaze_vector.x) > 0.8 or 
                abs(gaze_vector.y) > 0.8):
                return GazeTarget.OFF_SCREEN
            
            # Background/environmental gaze
            elif gaze_vector.z > 0.1:
                return GazeTarget.BACKGROUND
            
            return GazeTarget.UNKNOWN
            
        except Exception as e:
            logger.error(f"Error classifying gaze target: {e}")
            return GazeTarget.UNKNOWN
    
    def _classify_attention_zone_target(self, gaze_vector: GazeVector, attention_zones: List[Any]) -> GazeTarget:
        """Classify gaze target based on attention zones."""
        try:
            if not attention_zones:
                return GazeTarget.UNKNOWN
            
            # Check if gaze vector is in any of the configured zones
            for zone in attention_zones:
                if isinstance(zone, dict):
                    zone_type = zone.get('zone_type', 'custom')
                    coordinates = zone.get('coordinates', {})
                    
                    # Check based on zone type
                    if zone_type == 'face' or zone_type == 'social_partner':
                        if (gaze_vector.z > 0.5 and 
                            abs(gaze_vector.x) < 0.4 and 
                            abs(gaze_vector.y) < 0.4):
                            return GazeTarget.FACE_REGION
                    
                    elif zone_type == 'object':
                        # Check bounding box
                        coord_list = coordinates.get('coordinates', [])
                        if len(coord_list) >= 2:
                            x_coords = [c.get('x', 0) for c in coord_list if isinstance(c, dict)]
                            y_coords = [c.get('y', 0) for c in coord_list if isinstance(c, dict)]
                            if x_coords and y_coords:
                                min_x, max_x = min(x_coords), max(x_coords)
                                min_y, max_y = min(y_coords), max(y_coords)
                                if (min_x <= gaze_vector.x <= max_x and 
                                    min_y <= gaze_vector.y <= max_y and
                                    gaze_vector.z > 0.3):
                                    return GazeTarget.OBJECT
                    
                    elif zone_type == 'background':
                        if (gaze_vector.z > 0.2 and 
                            (abs(gaze_vector.x) > 0.5 or abs(gaze_vector.y) > 0.5)):
                            return GazeTarget.BACKGROUND
            
            return GazeTarget.UNKNOWN
            
        except Exception as e:
            logger.debug(f"Error classifying attention zone target: {e}")
            return GazeTarget.UNKNOWN

    
    def get_gaze_confidence(self, face_quality: float, landmark_confidence: float) -> float:
        """
        Calculate confidence score for gaze estimation.
        
        Args:
            face_quality: Quality of face detection (0-1)
            landmark_confidence: Confidence of landmark detection (0-1)
            
        Returns:
            Combined confidence score (0-1)
        """
        try:
            # Base confidence from face detection quality
            base_confidence = face_quality * 0.6
            
            # Landmark contribution
            landmark_contribution = landmark_confidence * 0.3
            
            # Stability contribution (based on recent gaze history)
            stability_contribution = 0.0
            if len(self.gaze_history) >= 3:
                recent_vectors = self.gaze_history[-3:]
                stability_metrics = self.track_gaze_stability(recent_vectors, len(recent_vectors))
                stability_contribution = stability_metrics.stability_score * 0.1
            
            total_confidence = base_confidence + landmark_contribution + stability_contribution
            return min(1.0, max(0.0, total_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating gaze confidence: {e}")
            return 0.0
    
    def track_gaze_stability(self, gaze_history: List[GazeVector], window_size: int) -> StabilityMetrics:
        """
        Analyze gaze stability over a time window.
        
        Args:
            gaze_history: List of recent gaze vectors
            window_size: Number of vectors to analyze
            
        Returns:
            StabilityMetrics with deviation and stability scores
        """
        try:
            if len(gaze_history) < 2:
                return StabilityMetrics(
                    mean_deviation=0.0,
                    standard_deviation=0.0,
                    stability_score=1.0
                )
            
            # Use the most recent vectors up to window_size
            vectors_to_analyze = gaze_history[-window_size:] if len(gaze_history) >= window_size else gaze_history
            
            # Calculate angular deviations between consecutive vectors
            angular_deviations = []
            for i in range(1, len(vectors_to_analyze)):
                angle = vectors_to_analyze[i-1].angle_to(vectors_to_analyze[i])
                angular_deviations.append(angle)
            
            if not angular_deviations:
                return StabilityMetrics(
                    mean_deviation=0.0,
                    standard_deviation=0.0,
                    stability_score=1.0
                )
            
            # Calculate statistics
            mean_deviation = np.mean(angular_deviations)
            std_deviation = np.std(angular_deviations)
            
            # Convert to stability score (lower deviation = higher stability)
            # Use exponential decay: stability = exp(-k * mean_deviation)
            stability_score = np.exp(-5.0 * mean_deviation)  # k=5.0 for reasonable sensitivity
            stability_score = min(1.0, max(0.0, stability_score))
            
            return StabilityMetrics(
                mean_deviation=float(mean_deviation),
                standard_deviation=float(std_deviation),
                stability_score=float(stability_score)
            )
            
        except Exception as e:
            logger.error(f"Error tracking gaze stability: {e}")
            return StabilityMetrics(
                mean_deviation=1.0,
                standard_deviation=1.0,
                stability_score=0.0
            )
    
    def detect_gaze_shifts(self, gaze_sequence: List[GazeVector], sensitivity: float) -> List[float]:
        """
        Detect significant gaze shifts and return timestamps.
        
        Args:
            gaze_sequence: Sequence of gaze vectors
            sensitivity: Sensitivity threshold for shift detection (0-1)
            
        Returns:
            List of timestamps where significant gaze shifts occurred
        """
        try:
            if len(gaze_sequence) < 2:
                return []
            
            shift_timestamps = []
            
            # Calculate threshold based on sensitivity
            # Lower sensitivity = higher threshold (fewer shifts detected)
            angle_threshold = (1.0 - sensitivity) * np.pi / 4  # Max threshold: π/4 radians (45 degrees)
            
            for i in range(1, len(gaze_sequence)):
                prev_vector = gaze_sequence[i-1]
                curr_vector = gaze_sequence[i]
                
                # Calculate angular difference
                angle_diff = prev_vector.angle_to(curr_vector)
                
                # Check if shift exceeds threshold
                if angle_diff > angle_threshold:
                    shift_timestamps.append(curr_vector.timestamp)
                    logger.debug(f"Gaze shift detected at {curr_vector.timestamp:.3f}s, "
                               f"angle: {np.degrees(angle_diff):.1f}°")
            
            return shift_timestamps
            
        except Exception as e:
            logger.error(f"Error detecting gaze shifts: {e}")
            return []
    
    def get_gaze_distribution(self, time_window: float = 10.0) -> Dict[str, float]:
        """
        Get distribution of gaze targets over recent time window.
        
        Args:
            time_window: Time window in seconds
            
        Returns:
            Dictionary with gaze target distribution percentages
        """
        try:
            if not self.gaze_history:
                return {}
            
            # Filter vectors within time window
            current_time = self.gaze_history[-1].timestamp
            recent_vectors = [
                gv for gv in self.gaze_history 
                if current_time - gv.timestamp <= time_window
            ]
            
            if not recent_vectors:
                return {}
            
            # Classify each vector and count occurrences
            target_counts = {}
            for gv in recent_vectors:
                target = self.classify_gaze_target(gv, [])  # No attention zones for now
                target_name = target.value
                target_counts[target_name] = target_counts.get(target_name, 0) + 1
            
            # Convert to percentages
            total_count = len(recent_vectors)
            distribution = {
                target: (count / total_count) * 100 
                for target, count in target_counts.items()
            }
            
            return distribution
            
        except Exception as e:
            logger.error(f"Error calculating gaze distribution: {e}")
            return {}
    
    def reset(self) -> None:
        """Reset the analyzer state."""
        self.gaze_history.clear()
        logger.info("Gaze Direction Analyzer reset")
    
    def get_current_stability(self) -> float:
        """Get current gaze stability score."""
        if len(self.gaze_history) < 3:
            return 1.0
        
        stability_metrics = self.track_gaze_stability(self.gaze_history, min(10, len(self.gaze_history)))
        return stability_metrics.stability_score