"""
Pose analysis using MediaPipe Pose.

Clinical markers extracted:
1. Upper-body motion (shoulders, elbows, wrists) - motor agitation
2. Hand movement velocity - restlessness indicator
3. Postural stability - sustained orientation
4. Body landmark visibility - engagement/avoidance proxy

Engineering decisions:
- MediaPipe Pose: 33 3D keypoints, efficient on CPU
- Focus on upper body (relevant for seated therapy sessions)
- Track hand/arm movement (fidgeting, self-soothing behaviors)
- Normalize by frame size (handle varying video resolutions)

Clinical rationale:
- Excessive upper-body movement → motor dysregulation
- High hand velocity → fidgeting, restlessness
- Postural shifts → attention changes, discomfort
- Low visibility → avoidance postures (turned away)
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings

import numpy as np

logger = logging.getLogger(__name__)

# Suppress MediaPipe warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not installed. Pose analysis unavailable.")


@dataclass
class PoseFeatures:
    """
    Pose features extracted from a single frame.
    
    Attributes:
        frame_idx: Frame index
        timestamp: Time in seconds
        pose_detected: Whether pose was detected
        upper_body_motion: Normalized upper-body displacement (0-1)
        hand_velocity_left: Left hand velocity (pixels/frame)
        hand_velocity_right: Right hand velocity (pixels/frame)
        shoulder_stability: Shoulder position variance (lower = more stable)
        posture_angle: Torso angle from vertical (degrees)
        visibility_score: Average landmark visibility (0-1)
        num_landmarks: Number of detected landmarks
    """
    frame_idx: int
    timestamp: float
    pose_detected: bool
    upper_body_motion: float
    hand_velocity_left: float
    hand_velocity_right: float
    shoulder_stability: float
    posture_angle: float
    visibility_score: float
    num_landmarks: int


class PoseAnalyzer:
    """
    Analyze body pose using MediaPipe Pose.
    
    Features extracted:
    - Upper-body motion (shoulders, elbows, wrists)
    - Hand movement velocity (fidgeting indicator)
    - Postural stability (sustained orientation)
    - Visibility scores (engagement proxy)
    
    Clinical interpretation:
    - High upper_body_motion → motor agitation
    - High hand_velocity → fidgeting, self-soothing
    - Low shoulder_stability → postural shifts
    - Low visibility → avoidance posture
    
    Usage:
        analyzer = PoseAnalyzer()
        features = analyzer.analyze_frame(frame)
    """
    
    # MediaPipe Pose landmark indices
    # See: https://google.github.io/mediapipe/solutions/pose.html
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    
    # Upper body landmark indices (for motion computation)
    UPPER_BODY_INDICES = [11, 12, 13, 14, 15, 16]  # Shoulders, elbows, wrists
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        static_image_mode: bool = False
    ):
        """
        Initialize pose analyzer.
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for landmark tracking
            model_complexity: Model complexity (0=lite, 1=full, 2=heavy)
            static_image_mode: If True, treat each frame independently
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not installed. Install with: pip install mediapipe")
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize Pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Store previous landmarks for motion/velocity computation
        self.prev_landmarks = None
        
        logger.info(f"Pose analyzer initialized (MediaPipe Pose, complexity={model_complexity})")
    
    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float
    ) -> PoseFeatures:
        """
        Analyze single frame for pose features.
        
        Args:
            frame: RGB frame (H, W, 3)
            frame_idx: Frame index
            timestamp: Frame timestamp in seconds
            
        Returns:
            PoseFeatures object
        """
        # Process frame
        results = self.pose.process(frame)
        
        if not results.pose_landmarks:
            # No pose detected
            logger.debug(f"No pose detected in frame {frame_idx}")
            
            # Reset previous landmarks
            self.prev_landmarks = None
            
            return PoseFeatures(
                frame_idx=frame_idx,
                timestamp=timestamp,
                pose_detected=False,
                upper_body_motion=0.0,
                hand_velocity_left=0.0,
                hand_velocity_right=0.0,
                shoulder_stability=0.0,
                posture_angle=0.0,
                visibility_score=0.0,
                num_landmarks=0
            )
        
        # Get pose landmarks
        pose_landmarks = results.pose_landmarks
        
        # Convert to numpy array
        h, w = frame.shape[:2]
        landmarks = []
        visibilities = []
        
        for lm in pose_landmarks.landmark:
            landmarks.append([lm.x * w, lm.y * h, lm.z * w])  # Denormalize
            visibilities.append(lm.visibility)
        
        landmarks = np.array(landmarks)  # Shape: (33, 3)
        visibilities = np.array(visibilities)
        
        # Compute upper-body motion
        if self.prev_landmarks is not None:
            upper_body_motion = self._compute_upper_body_motion(landmarks, self.prev_landmarks)
            hand_vel_left = self._compute_hand_velocity(landmarks, self.prev_landmarks, 'left')
            hand_vel_right = self._compute_hand_velocity(landmarks, self.prev_landmarks, 'right')
        else:
            upper_body_motion = 0.0
            hand_vel_left = 0.0
            hand_vel_right = 0.0
        
        # Update previous landmarks
        self.prev_landmarks = landmarks.copy()
        
        # Compute shoulder stability
        shoulder_stability = self._compute_shoulder_stability(landmarks)
        
        # Compute posture angle
        posture_angle = self._compute_posture_angle(landmarks)
        
        # Average visibility
        visibility_score = float(np.mean(visibilities))
        
        return PoseFeatures(
            frame_idx=frame_idx,
            timestamp=timestamp,
            pose_detected=True,
            upper_body_motion=upper_body_motion,
            hand_velocity_left=hand_vel_left,
            hand_velocity_right=hand_vel_right,
            shoulder_stability=shoulder_stability,
            posture_angle=posture_angle,
            visibility_score=visibility_score,
            num_landmarks=len(landmarks)
        )
    
    def _compute_upper_body_motion(
        self,
        current_landmarks: np.ndarray,
        prev_landmarks: np.ndarray
    ) -> float:
        """
        Compute upper-body motion energy.
        
        Method: Mean displacement of upper-body landmarks
        (shoulders, elbows, wrists)
        
        Returns:
            Normalized motion energy (0-1)
        """
        # Extract upper-body landmarks
        current_upper = current_landmarks[self.UPPER_BODY_INDICES]
        prev_upper = prev_landmarks[self.UPPER_BODY_INDICES]
        
        # Compute displacements
        displacements = np.linalg.norm(current_upper - prev_upper, axis=1)
        
        # Mean displacement
        mean_displacement = np.mean(displacements)
        
        # Normalize by image size (assuming 640px width)
        normalized_motion = mean_displacement / 640.0
        
        # Clip to [0, 1]
        normalized_motion = np.clip(normalized_motion, 0.0, 1.0)
        
        return float(normalized_motion)
    
    def _compute_hand_velocity(
        self,
        current_landmarks: np.ndarray,
        prev_landmarks: np.ndarray,
        hand: str
    ) -> float:
        """
        Compute hand velocity (pixels per frame).
        
        Args:
            current_landmarks: Current frame landmarks
            prev_landmarks: Previous frame landmarks
            hand: 'left' or 'right'
            
        Returns:
            Hand velocity in pixels/frame
        """
        if hand == 'left':
            wrist_idx = self.LEFT_WRIST
        else:
            wrist_idx = self.RIGHT_WRIST
        
        # Compute displacement
        displacement = np.linalg.norm(
            current_landmarks[wrist_idx] - prev_landmarks[wrist_idx]
        )
        
        return float(displacement)
    
    def _compute_shoulder_stability(self, landmarks: np.ndarray) -> float:
        """
        Compute shoulder position stability.
        
        Method: Distance between shoulders (width variance proxy)
        Lower values = more stable posture
        
        Returns:
            Shoulder distance (normalized)
        """
        left_shoulder = landmarks[self.LEFT_SHOULDER]
        right_shoulder = landmarks[self.RIGHT_SHOULDER]
        
        # Compute distance
        distance = np.linalg.norm(left_shoulder - right_shoulder)
        
        # Normalize by image size
        normalized_distance = distance / 640.0
        
        return float(normalized_distance)
    
    def _compute_posture_angle(self, landmarks: np.ndarray) -> float:
        """
        Compute torso angle from vertical.
        
        Method: Angle between shoulder midpoint and hip midpoint
        
        Returns:
            Angle in degrees (0 = perfectly vertical)
        """
        try:
            # Compute midpoints
            shoulder_mid = (landmarks[self.LEFT_SHOULDER] + landmarks[self.RIGHT_SHOULDER]) / 2
            hip_mid = (landmarks[self.LEFT_HIP] + landmarks[self.RIGHT_HIP]) / 2
            
            # Compute vector from hips to shoulders
            torso_vector = shoulder_mid - hip_mid
            
            # Compute angle from vertical (y-axis)
            vertical = np.array([0, -1, 0])  # Negative y is up in image coordinates
            
            # Project to 2D (x, y)
            torso_2d = torso_vector[:2]
            vertical_2d = vertical[:2]
            
            # Compute angle
            cos_angle = np.dot(torso_2d, vertical_2d) / (
                np.linalg.norm(torso_2d) * np.linalg.norm(vertical_2d) + 1e-6
            )
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            
            # Convert to degrees
            angle_deg = np.degrees(angle)
            
            return float(angle_deg)
        
        except Exception as e:
            logger.debug(f"Posture angle computation failed: {e}")
            return 0.0
    
    def reset(self):
        """Reset temporal tracking state."""
        self.prev_landmarks = None
    
    def close(self):
        """Release resources."""
        if self.pose is not None:
            self.pose.close()


def analyze_pose_segment(
    frames: List[np.ndarray],
    start_frame_idx: int = 0,
    fps: float = 30.0
) -> List[PoseFeatures]:
    """
    Convenience function to analyze a segment of frames.
    
    Args:
        frames: List of RGB frames
        start_frame_idx: Starting frame index (for labeling)
        fps: Frames per second (for timestamps)
        
    Returns:
        List of PoseFeatures
    """
    analyzer = PoseAnalyzer(static_image_mode=False)  # Use tracking
    
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
        f"detected pose in {sum(f.pose_detected for f in features_list)} frames "
        f"({sum(f.pose_detected for f in features_list)/len(frames)*100:.1f}%)"
    )
    
    return features_list


def compute_body_motion_statistics(features_list: List[PoseFeatures]) -> dict:
    """
    Compute aggregate statistics for body motion features.
    
    Args:
        features_list: List of PoseFeatures
        
    Returns:
        Dictionary with statistics
    """
    # Filter to detected poses only
    detected = [f for f in features_list if f.pose_detected]
    
    if not detected:
        return {
            'detection_rate': 0.0,
            'mean_upper_body_motion': 0.0,
            'std_upper_body_motion': 0.0,
            'mean_hand_velocity': 0.0,
            'max_hand_velocity': 0.0,
            'mean_posture_angle': 0.0,
            'std_posture_angle': 0.0,
        }
    
    upper_body_motions = [f.upper_body_motion for f in detected]
    hand_velocities = [max(f.hand_velocity_left, f.hand_velocity_right) for f in detected]
    posture_angles = [f.posture_angle for f in detected]
    
    stats = {
        'detection_rate': len(detected) / len(features_list),
        'mean_upper_body_motion': np.mean(upper_body_motions),
        'std_upper_body_motion': np.std(upper_body_motions),
        'max_upper_body_motion': np.max(upper_body_motions),
        'mean_hand_velocity': np.mean(hand_velocities),
        'std_hand_velocity': np.std(hand_velocities),
        'max_hand_velocity': np.max(hand_velocities),
        'mean_posture_angle': np.mean(posture_angles),
        'std_posture_angle': np.std(posture_angles),
        'mean_visibility': np.mean([f.visibility_score for f in detected]),
    }
    
    return stats
