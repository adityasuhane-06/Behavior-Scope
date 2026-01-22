"""
Face analysis using MediaPipe Face Mesh.

Clinical markers extracted:
1. Head pose (yaw, pitch, roll) - attention orientation
2. Facial motion energy - overall movement intensity
3. Gaze proxy (iris position variance) - visual attention shifts
4. Facial landmark stability - restlessness indicator

Engineering decisions:
- MediaPipe Face Mesh: 468 3D landmarks, efficient on CPU
- Focus on head-level features (not fine-grained facial expressions)
- Avoid emotion classification (not clinical goal)
- Robust to partial occlusions and head rotations

Clinical rationale:
- Excessive head movement → motor agitation, restlessness
- Frequent gaze shifts → attention instability
- Sustained orientation away → potential avoidance behavior
- High facial motion → general motor dysregulation
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings

import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Suppress MediaPipe warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not installed. Face analysis unavailable.")


@dataclass
class FaceFeatures:
    """
    Facial features extracted from a single frame.
    
    Attributes:
        frame_idx: Frame index
        timestamp: Time in seconds
        face_detected: Whether face was detected
        head_pose: (yaw, pitch, roll) in degrees
        facial_motion_energy: Normalized motion energy (0-1)
        gaze_proxy: Iris position variance (higher = more shifts)
        landmark_confidence: Detection confidence (0-1)
        num_landmarks: Number of detected landmarks
    """
    frame_idx: int
    timestamp: float
    face_detected: bool
    head_pose: Tuple[float, float, float]  # (yaw, pitch, roll)
    facial_motion_energy: float
    gaze_proxy: float
    landmark_confidence: float
    num_landmarks: int


class FaceAnalyzer:
    """
    Analyze facial features using MediaPipe Face Mesh.
    
    Features extracted:
    - Head pose estimation (3D orientation)
    - Facial motion energy (frame-to-frame landmark displacement)
    - Gaze proxy (iris landmark variance)
    - Landmark stability (detection confidence)
    
    Clinical interpretation:
    - High head_pose variance → motor agitation
    - High gaze_proxy → attention instability
    - High motion_energy → general restlessness
    
    Usage:
        analyzer = FaceAnalyzer()
        features = analyzer.analyze_frame(frame)
    """
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False
    ):
        """
        Initialize face analyzer.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
            static_image_mode: If True, treat each frame independently
                             If False, use temporal tracking (more efficient)
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not installed. Install with: pip install mediapipe")
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize Face Mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,  # Analyze only primary face (patient)
            refine_landmarks=True,  # Include iris landmarks for gaze
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Store previous landmarks for motion computation
        self.prev_landmarks = None
        
        logger.info("Face analyzer initialized (MediaPipe Face Mesh)")
    
    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float
    ) -> FaceFeatures:
        """
        Analyze single frame for facial features.
        
        Args:
            frame: RGB frame (H, W, 3)
            frame_idx: Frame index
            timestamp: Frame timestamp in seconds
            
        Returns:
            FaceFeatures object
        """
        # Process frame
        results = self.face_mesh.process(frame)
        
        if not results.multi_face_landmarks or len(results.multi_face_landmarks) == 0:
            # No face detected
            logger.debug(f"No face detected in frame {frame_idx}")
            
            # Reset previous landmarks
            self.prev_landmarks = None
            
            return FaceFeatures(
                frame_idx=frame_idx,
                timestamp=timestamp,
                face_detected=False,
                head_pose=(0.0, 0.0, 0.0),
                facial_motion_energy=0.0,
                gaze_proxy=0.0,
                landmark_confidence=0.0,
                num_landmarks=0
            )
        
        # Get primary face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to numpy array for computation
        h, w = frame.shape[:2]
        landmarks = []
        
        for lm in face_landmarks.landmark:
            landmarks.append([lm.x * w, lm.y * h, lm.z * w])  # Denormalize
        
        landmarks = np.array(landmarks)  # Shape: (468, 3)
        
        # Compute head pose
        head_pose = self._estimate_head_pose(landmarks, w, h)
        
        # Compute facial motion energy
        if self.prev_landmarks is not None:
            motion_energy = self._compute_motion_energy(landmarks, self.prev_landmarks)
        else:
            motion_energy = 0.0
        
        # Update previous landmarks
        self.prev_landmarks = landmarks.copy()
        
        # Compute gaze proxy (iris variance)
        gaze_proxy = self._compute_gaze_proxy(landmarks)
        
        # Estimate confidence (based on landmark visibility)
        confidence = self._estimate_confidence(face_landmarks)
        
        return FaceFeatures(
            frame_idx=frame_idx,
            timestamp=timestamp,
            face_detected=True,
            head_pose=head_pose,
            facial_motion_energy=motion_energy,
            gaze_proxy=gaze_proxy,
            landmark_confidence=confidence,
            num_landmarks=len(landmarks)
        )
    
    def _estimate_head_pose(
        self,
        landmarks: np.ndarray,
        img_w: int,
        img_h: int
    ) -> Tuple[float, float, float]:
        """
        Estimate head pose (yaw, pitch, roll) from landmarks.
        
        Method: PnP (Perspective-n-Point) algorithm
        Uses 6 key facial landmarks (nose, eyes, mouth corners)
        
        Returns:
            (yaw, pitch, roll) in degrees
        """
        try:
            # Define 3D model points (canonical face)
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -330.0, -65.0),        # Chin
                (-225.0, 170.0, -135.0),     # Left eye corner
                (225.0, 170.0, -135.0),      # Right eye corner
                (-150.0, -150.0, -125.0),    # Left mouth corner
                (150.0, -150.0, -125.0)      # Right mouth corner
            ], dtype=np.float64)
            
            # Corresponding 2D image points (MediaPipe landmark indices)
            landmark_indices = [1, 152, 33, 263, 61, 291]  # Nose, chin, eyes, mouth
            
            image_points = np.array([
                [landmarks[idx, 0], landmarks[idx, 1]]
                for idx in landmark_indices
            ], dtype=np.float64)
            
            # Camera internals (approximation)
            focal_length = img_w
            center = (img_w / 2, img_h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            
            # Assume no lens distortion
            dist_coeffs = np.zeros((4, 1))
            
            # Solve PnP
            success, rotation_vec, translation_vec = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return (0.0, 0.0, 0.0)
            
            # Convert rotation vector to Euler angles
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            
            # Extract yaw, pitch, roll
            yaw = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
            pitch = np.arctan2(-rotation_mat[2, 0],
                             np.sqrt(rotation_mat[2, 1]**2 + rotation_mat[2, 2]**2))
            roll = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
            
            # Convert to degrees
            yaw = np.degrees(yaw)
            pitch = np.degrees(pitch)
            roll = np.degrees(roll)
            
            return (yaw, pitch, roll)
        
        except Exception as e:
            logger.debug(f"Head pose estimation failed: {e}")
            return (0.0, 0.0, 0.0)
    
    def _compute_motion_energy(
        self,
        current_landmarks: np.ndarray,
        prev_landmarks: np.ndarray
    ) -> float:
        """
        Compute facial motion energy (normalized displacement).
        
        Method: Mean Euclidean distance between corresponding landmarks
        
        Returns:
            Motion energy (0-1 scale, normalized)
        """
        # Compute displacement vectors
        displacement = np.linalg.norm(current_landmarks - prev_landmarks, axis=1)
        
        # Mean displacement
        mean_displacement = np.mean(displacement)
        
        # Normalize by image size (assuming 640x480 typical)
        normalized_energy = mean_displacement / 640.0
        
        # Clip to [0, 1]
        normalized_energy = np.clip(normalized_energy, 0.0, 1.0)
        
        return float(normalized_energy)
    
    def _compute_gaze_proxy(self, landmarks: np.ndarray) -> float:
        """
        Compute gaze proxy from iris landmarks.
        
        MediaPipe Face Mesh includes iris landmarks (indices 468-477).
        Variance in iris position indicates gaze shifts.
        
        Note: This is a proxy, not true gaze tracking.
        True gaze requires eye tracking hardware.
        
        Returns:
            Gaze variance (0-1 scale)
        """
        # Iris landmark indices (left: 468-472, right: 473-477)
        try:
            # Get iris centers
            left_iris = landmarks[468:473, :2]  # x, y only
            right_iris = landmarks[473:478, :2]
            
            # Compute variance
            left_var = np.var(left_iris, axis=0).sum()
            right_var = np.var(right_iris, axis=0).sum()
            
            # Average variance
            gaze_variance = (left_var + right_var) / 2.0
            
            # Normalize (empirical scaling)
            normalized_variance = np.clip(gaze_variance / 1000.0, 0.0, 1.0)
            
            return float(normalized_variance)
        
        except Exception as e:
            logger.debug(f"Gaze proxy computation failed: {e}")
            return 0.0
    
    def _estimate_confidence(self, face_landmarks) -> float:
        """
        Estimate detection confidence.
        
        Based on landmark visibility scores (if available).
        """
        try:
            # MediaPipe provides visibility score for some landmarks
            visibilities = [lm.visibility for lm in face_landmarks.landmark if hasattr(lm, 'visibility')]
            
            if visibilities:
                return float(np.mean(visibilities))
            else:
                # Fallback: assume high confidence if detected
                return 0.9
        except:
            return 0.9
    
    def reset(self):
        """Reset temporal tracking state."""
        self.prev_landmarks = None
    
    def close(self):
        """Release resources."""
        if self.face_mesh is not None:
            self.face_mesh.close()


def analyze_face_segment(
    frames: List[np.ndarray],
    start_frame_idx: int = 0,
    fps: float = 30.0
) -> List[FaceFeatures]:
    """
    Convenience function to analyze a segment of frames.
    
    Args:
        frames: List of RGB frames
        start_frame_idx: Starting frame index (for labeling)
        fps: Frames per second (for timestamps)
        
    Returns:
        List of FaceFeatures
    """
    analyzer = FaceAnalyzer(static_image_mode=False)  # Use tracking
    
    features_list = []
    
    try:
        for i, frame in enumerate(frames):
            frame_idx = start_frame_idx + i
            timestamp = frame_idx / fps
            
            features = analyzer.analyze_frame(frame, frame_idx, timestamp)
            features_list.append(features)
    
    finally:
        analyzer.close()
    
    logger.info(
        f"Analyzed {len(frames)} frames, "
        f"detected face in {sum(f.face_detected for f in features_list)} frames "
        f"({sum(f.face_detected for f in features_list)/len(frames)*100:.1f}%)"
    )
    
    return features_list
