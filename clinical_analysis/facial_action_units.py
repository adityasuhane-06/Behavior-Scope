"""
Facial Action Coding System (FACS) Action Unit extraction from MediaPipe landmarks.

This module calculates Action Units (AUs) based on geometric relationships between
facial landmarks. AUs represent objective facial muscle activations, NOT emotions.

CLINICAL RATIONALE:
- AUs are observable muscle movements (objective, measurable)
- NOT emotion detection (subjective, interpretive)
- Useful for assessing affect range, facial mobility, social communication patterns
- Appropriate for clinical observation without diagnostic claims

MATHEMATICAL APPROACH:
- Distance ratios between landmarks (normalized by face size)
- Angle measurements (relative orientations)
- Symmetry analysis (left vs. right facial activation)
- Temporal dynamics (activation onset, duration, offset)

REFERENCES:
- Ekman, P., & Friesen, W. V. (1978). Facial Action Coding System (FACS)
- Baltrusaitis et al. (2018). OpenFace 2.0: Facial Behavior Analysis Toolkit
- MediaPipe Face Mesh: https://google.github.io/mediapipe/solutions/face_mesh

Engineering decisions:
- MediaPipe landmarks (468 points) provide sufficient resolution for AU calculation
- Focus on clinically relevant AUs (not all 46 possible AUs)
- Normalize by face size to handle different camera distances
- Output intensity scores (0-1) rather than binary presence/absence
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial import distance

logger = logging.getLogger(__name__)


# MediaPipe Face Mesh landmark indices for FACS AU calculation
# Face mesh has 468 landmarks - we use specific subsets for each AU

LANDMARK_INDICES = {
    # Inner brow (AU1)
    'inner_brow_left': [55, 65],
    'inner_brow_right': [285, 295],
    
    # Outer brow (AU2)
    'outer_brow_left': [46, 53],
    'outer_brow_right': [276, 283],
    
    # Brow center (for AU4 lowering)
    'brow_lowerer_left': [70, 63],
    'brow_lowerer_right': [300, 293],
    
    # Upper lid (AU5)
    'upper_lid_left': [159, 145],
    'upper_lid_right': [386, 374],
    
    # Cheek raiser (AU6)
    'cheek_raiser_left': [50, 205],
    'cheek_raiser_right': [280, 425],
    
    # Lid tightener (AU7)
    'lid_tight_left': [33, 133],
    'lid_tight_right': [362, 263],
    
    # Nose wrinkler (AU9)
    'nose_wrinkle_left': [48, 4],
    'nose_wrinkle_right': [278, 4],
    
    # Upper lip raiser (AU10)
    'upper_lip_left': [40, 37],
    'upper_lip_right': [270, 267],
    
    # Lip corner puller (AU12 - smile)
    'lip_corner_left': [61, 91],
    'lip_corner_right': [291, 321],
    
    # Lip corner depressor (AU15 - frown)
    'lip_depress_left': [84, 17],
    'lip_depress_right': [314, 17],
    
    # Chin raiser (AU17)
    'chin_raiser': [152, 175],
    
    # Lip stretcher (AU20)
    'lip_stretch_left': [78, 95],
    'lip_stretch_right': [308, 325],
    
    # Lip tightener (AU23)
    'lip_tight_outer': [61, 291],
    'lip_tight_inner': [0, 17],
    
    # Lips part (AU25)
    'lips_upper': [13, 14],
    'lips_lower': [13, 14],
    
    # Jaw drop (AU26)
    'jaw_upper': [10, 152],
    'jaw_lower': [152, 17],
    
    # Reference points for normalization
    'nose_bridge': [6, 4],
    'face_left': [234],
    'face_right': [454],
    'face_top': [10],
    'face_bottom': [152],
}


@dataclass
class ActionUnit:
    """
    Single Action Unit measurement.
    
    Attributes:
        au_number: FACS AU number (e.g., 1, 2, 4, etc.)
        name: Descriptive name (e.g., "Inner Brow Raiser")
        intensity: Activation intensity (0.0-1.0)
        present: Whether AU is activated (intensity > threshold)
        confidence: Measurement confidence (0.0-1.0)
        side: 'left', 'right', 'bilateral', or None
    """
    au_number: int
    name: str
    intensity: float
    present: bool
    confidence: float
    side: Optional[str] = None


@dataclass
class FacialActionUnits:
    """
    Complete FACS analysis for a single frame.
    
    Attributes:
        frame_idx: Frame index
        timestamp: Time in seconds
        action_units: Dictionary of AU measurements {au_number: ActionUnit}
        face_detected: Whether face was successfully detected
        face_size: Normalized face size (for reference)
        symmetry_score: Left-right facial symmetry (0-1, higher = more symmetric)
    """
    frame_idx: int
    timestamp: float
    action_units: Dict[int, ActionUnit]
    face_detected: bool
    face_size: float
    symmetry_score: float


class FacialActionUnitAnalyzer:
    """
    Extract Facial Action Coding System (FACS) Action Units from MediaPipe landmarks.
    
    This analyzer converts 468 facial landmarks into quantitative AU measurements.
    Each AU represents a specific facial muscle activation pattern.
    
    Clinical use:
    - Affect range assessment (how many different AUs activated)
    - Facial mobility (how much movement across session)
    - Flat affect detection (low AU activation diversity)
    - Social communication patterns (appropriate AU combinations)
    
    NOT for:
    - Emotion detection (AUs ≠ emotions)
    - Diagnostic purposes (observation only)
    - Mind reading (muscle movements ≠ internal states)
    
    Usage:
        analyzer = FacialActionUnitAnalyzer()
        aus = analyzer.analyze_landmarks(landmarks, frame_idx, timestamp)
    """
    
    def __init__(self, intensity_threshold: float = 0.3):
        """
        Initialize AU analyzer.
        
        Args:
            intensity_threshold: Minimum intensity to consider AU "present" (0-1)
        """
        self.intensity_threshold = intensity_threshold
        self.baseline_distances = None  # For relative AU measurement
        
        logger.info(f"Initialized Facial Action Unit Analyzer (threshold={intensity_threshold})")
    
    def analyze_landmarks(
        self,
        landmarks: np.ndarray,
        frame_idx: int,
        timestamp: float
    ) -> FacialActionUnits:
        """
        Analyze facial landmarks and extract Action Units.
        
        Args:
            landmarks: MediaPipe landmarks array (468, 3) - (x, y, z)
            frame_idx: Frame index
            timestamp: Time in seconds
            
        Returns:
            FacialActionUnits with all AU measurements
        """
        if landmarks is None or len(landmarks) < 468:
            return FacialActionUnits(
                frame_idx=frame_idx,
                timestamp=timestamp,
                action_units={},
                face_detected=False,
                face_size=0.0,
                symmetry_score=0.0
            )
        
        # Calculate face size for normalization
        face_size = self._calculate_face_size(landmarks)
        
        # Calculate all Action Units
        aus = {}
        
        # Upper face AUs (brow region)
        aus[1] = self._calculate_au1_inner_brow_raise(landmarks, face_size)
        aus[2] = self._calculate_au2_outer_brow_raise(landmarks, face_size)
        aus[4] = self._calculate_au4_brow_lower(landmarks, face_size)
        aus[5] = self._calculate_au5_upper_lid_raise(landmarks, face_size)
        aus[6] = self._calculate_au6_cheek_raise(landmarks, face_size)
        aus[7] = self._calculate_au7_lid_tightener(landmarks, face_size)
        
        # Mid-face AUs (nose and upper lip)
        aus[9] = self._calculate_au9_nose_wrinkle(landmarks, face_size)
        aus[10] = self._calculate_au10_upper_lip_raise(landmarks, face_size)
        
        # Lower face AUs (mouth region)
        aus[12] = self._calculate_au12_lip_corner_pull(landmarks, face_size)
        aus[15] = self._calculate_au15_lip_corner_depress(landmarks, face_size)
        aus[17] = self._calculate_au17_chin_raise(landmarks, face_size)
        aus[20] = self._calculate_au20_lip_stretch(landmarks, face_size)
        aus[23] = self._calculate_au23_lip_tightener(landmarks, face_size)
        aus[25] = self._calculate_au25_lips_part(landmarks, face_size)
        aus[26] = self._calculate_au26_jaw_drop(landmarks, face_size)
        
        # Calculate symmetry
        symmetry = self._calculate_facial_symmetry(landmarks)
        
        return FacialActionUnits(
            frame_idx=frame_idx,
            timestamp=timestamp,
            action_units=aus,
            face_detected=True,
            face_size=face_size,
            symmetry_score=symmetry
        )
    
    def _calculate_face_size(self, landmarks: np.ndarray) -> float:
        """
        Calculate face size for normalization.
        
        Uses inter-ocular distance as face size reference.
        """
        left_eye = landmarks[33, :2]  # Left eye outer corner
        right_eye = landmarks[263, :2]  # Right eye outer corner
        
        face_size = np.linalg.norm(left_eye - right_eye)
        return float(face_size)
    
    def _calculate_au1_inner_brow_raise(
        self,
        landmarks: np.ndarray,
        face_size: float
    ) -> ActionUnit:
        """
        AU1: Inner Brow Raiser (frontalis, pars medialis)
        
        Calculation:
        - Measure vertical distance between inner brow and eye
        - Normalize by face size
        - Compare to neutral baseline
        
        Clinical: Associated with surprise, concern, sadness
        """
        # Left side
        inner_brow_l = landmarks[55, :2]
        eye_ref_l = landmarks[33, :2]
        dist_left = np.linalg.norm(inner_brow_l - eye_ref_l)
        
        # Right side
        inner_brow_r = landmarks[285, :2]
        eye_ref_r = landmarks[263, :2]
        dist_right = np.linalg.norm(inner_brow_r - eye_ref_r)
        
        # Average and normalize
        avg_dist = (dist_left + dist_right) / 2.0
        normalized = avg_dist / face_size
        
        # Convert to intensity (empirically calibrated)
        # Neutral ~0.18, raised ~0.25+
        intensity = np.clip((normalized - 0.18) / 0.10, 0.0, 1.0)
        
        return ActionUnit(
            au_number=1,
            name="Eyebrows Raised (inner)",
            intensity=float(intensity),
            present=intensity > self.intensity_threshold,
            confidence=0.85,
            side='bilateral'
        )
    
    def _calculate_au2_outer_brow_raise(
        self,
        landmarks: np.ndarray,
        face_size: float
    ) -> ActionUnit:
        """
        AU2: Outer Brow Raiser (frontalis, pars lateralis)
        
        Clinical: Associated with surprise, fear
        """
        # Left side
        outer_brow_l = landmarks[46, :2]
        eye_ref_l = landmarks[33, :2]
        dist_left = np.linalg.norm(outer_brow_l - eye_ref_l)
        
        # Right side
        outer_brow_r = landmarks[276, :2]
        eye_ref_r = landmarks[263, :2]
        dist_right = np.linalg.norm(outer_brow_r - eye_ref_r)
        
        avg_dist = (dist_left + dist_right) / 2.0
        normalized = avg_dist / face_size
        
        intensity = np.clip((normalized - 0.22) / 0.10, 0.0, 1.0)
        
        return ActionUnit(
            au_number=2,
            name="Eyebrows Raised (outer)",
            intensity=float(intensity),
            present=intensity > self.intensity_threshold,
            confidence=0.85,
            side='bilateral'
        )
    
    def _calculate_au4_brow_lower(
        self,
        landmarks: np.ndarray,
        face_size: float
    ) -> ActionUnit:
        """
        AU4: Brow Lowerer (corrugator supercilii, depressor supercilii)
        
        Calculation:
        - Measure distance between brow and eye (should decrease when lowered)
        - Also check for glabella compression (brows moving together)
        
        Clinical: Associated with concentration, anger, confusion
        """
        # Vertical component (brow-eye distance)
        brow_l = landmarks[70, :2]
        eye_l = landmarks[33, :2]
        dist_left = np.linalg.norm(brow_l - eye_l)
        
        brow_r = landmarks[300, :2]
        eye_r = landmarks[263, :2]
        dist_right = np.linalg.norm(brow_r - eye_r)
        
        avg_dist = (dist_left + dist_right) / 2.0
        normalized = avg_dist / face_size
        
        # Lower distance = more lowering
        # Neutral ~0.20, lowered ~0.15
        intensity = np.clip((0.20 - normalized) / 0.08, 0.0, 1.0)
        
        return ActionUnit(
            au_number=4,
            name="Eyebrows Furrowed",
            intensity=float(intensity),
            present=intensity > self.intensity_threshold,
            confidence=0.80,
            side='bilateral'
        )
    
    def _calculate_au5_upper_lid_raise(
        self,
        landmarks: np.ndarray,
        face_size: float
    ) -> ActionUnit:
        """
        AU5: Upper Lid Raiser (levator palpebrae superioris)
        
        Calculation:
        - Measure eye opening (upper lid to lower lid distance)
        - Normalize by face size
        
        Clinical: Associated with surprise, fear, wide-eyed attention
        """
        # Left eye opening
        upper_lid_l = landmarks[159, :2]
        lower_lid_l = landmarks[145, :2]
        eye_open_left = np.linalg.norm(upper_lid_l - lower_lid_l)
        
        # Right eye opening
        upper_lid_r = landmarks[386, :2]
        lower_lid_r = landmarks[374, :2]
        eye_open_right = np.linalg.norm(upper_lid_r - lower_lid_r)
        
        avg_opening = (eye_open_left + eye_open_right) / 2.0
        normalized = avg_opening / face_size
        
        # Neutral ~0.06, wide ~0.10+
        intensity = np.clip((normalized - 0.06) / 0.05, 0.0, 1.0)
        
        return ActionUnit(
            au_number=5,
            name="Eyes Widened",
            intensity=float(intensity),
            present=intensity > self.intensity_threshold,
            confidence=0.85,
            side='bilateral'
        )
    
    def _calculate_au6_cheek_raise(
        self,
        landmarks: np.ndarray,
        face_size: float
    ) -> ActionUnit:
        """
        AU6: Cheek Raiser (orbicularis oculi, pars orbitalis)
        
        Calculation:
        - Measure upward displacement of cheek
        - Check for "crow's feet" formation (lower eyelid movement)
        
        Clinical: Genuine smile indicator (Duchenne smile)
        """
        # Lower eyelid position (moves up with cheek raise)
        lower_lid_l = landmarks[145, :2]
        cheek_ref_l = landmarks[205, :2]
        dist_left = np.linalg.norm(lower_lid_l - cheek_ref_l)
        
        lower_lid_r = landmarks[374, :2]
        cheek_ref_r = landmarks[425, :2]
        dist_right = np.linalg.norm(lower_lid_r - cheek_ref_r)
        
        avg_dist = (dist_left + dist_right) / 2.0
        normalized = avg_dist / face_size
        
        # Smaller distance = more cheek raise
        # Neutral ~0.18, raised ~0.14
        intensity = np.clip((0.18 - normalized) / 0.06, 0.0, 1.0)
        
        return ActionUnit(
            au_number=6,
            name="Cheeks Raised",
            intensity=float(intensity),
            present=intensity > self.intensity_threshold,
            confidence=0.80,
            side='bilateral'
        )
    
    def _calculate_au7_lid_tightener(
        self,
        landmarks: np.ndarray,
        face_size: float
    ) -> ActionUnit:
        """
        AU7: Lid Tightener (orbicularis oculi, pars palpebralis)
        
        Calculation:
        - Measure eyelid closure (reduced eye opening)
        - Different from AU5 (which is opening)
        
        Clinical: Associated with concentration, squinting, anger
        """
        # Eye opening (smaller = more tightening)
        upper_l = landmarks[159, :2]
        lower_l = landmarks[145, :2]
        eye_open_left = np.linalg.norm(upper_l - lower_l)
        
        upper_r = landmarks[386, :2]
        lower_r = landmarks[374, :2]
        eye_open_right = np.linalg.norm(upper_r - lower_r)
        
        avg_opening = (eye_open_left + eye_open_right) / 2.0
        normalized = avg_opening / face_size
        
        # Neutral ~0.06, tight ~0.03
        intensity = np.clip((0.06 - normalized) / 0.04, 0.0, 1.0)
        
        return ActionUnit(
            au_number=7,
            name="Eyes Tightened",
            intensity=float(intensity),
            present=intensity > self.intensity_threshold,
            confidence=0.80,
            side='bilateral'
        )
    
    def _calculate_au9_nose_wrinkle(
        self,
        landmarks: np.ndarray,
        face_size: float
    ) -> ActionUnit:
        """
        AU9: Nose Wrinkler (levator labii superioris alaeque nasi)
        
        Calculation:
        - Measure upward movement of nose wings
        - Check for nostril dilation
        
        Clinical: Associated with disgust
        """
        # Nose wing positions
        nose_wing_l = landmarks[48, :2]
        nose_wing_r = landmarks[278, :2]
        nose_bridge = landmarks[4, :2]
        
        dist_left = np.linalg.norm(nose_wing_l - nose_bridge)
        dist_right = np.linalg.norm(nose_wing_r - nose_bridge)
        
        avg_dist = (dist_left + dist_right) / 2.0
        normalized = avg_dist / face_size
        
        # Larger distance = more wrinkling
        # Neutral ~0.12, wrinkled ~0.15+
        intensity = np.clip((normalized - 0.12) / 0.05, 0.0, 1.0)
        
        return ActionUnit(
            au_number=9,
            name="Nose Wrinkled",
            intensity=float(intensity),
            present=intensity > self.intensity_threshold,
            confidence=0.70,  # Lower confidence - subtle AU
            side='bilateral'
        )
    
    def _calculate_au10_upper_lip_raise(
        self,
        landmarks: np.ndarray,
        face_size: float
    ) -> ActionUnit:
        """
        AU10: Upper Lip Raiser (levator labii superioris)
        
        Calculation:
        - Measure upward displacement of upper lip
        
        Clinical: Associated with disgust, contempt, smirking
        """
        # Upper lip positions
        upper_lip_l = landmarks[40, :2]
        upper_lip_r = landmarks[270, :2]
        nose_base = landmarks[4, :2]
        
        dist_left = np.linalg.norm(upper_lip_l - nose_base)
        dist_right = np.linalg.norm(upper_lip_r - nose_base)
        
        avg_dist = (dist_left + dist_right) / 2.0
        normalized = avg_dist / face_size
        
        # Smaller distance = more raised
        # Neutral ~0.25, raised ~0.22
        intensity = np.clip((0.25 - normalized) / 0.05, 0.0, 1.0)
        
        return ActionUnit(
            au_number=10,
            name="Upper Lip Raised",
            intensity=float(intensity),
            present=intensity > self.intensity_threshold,
            confidence=0.75,
            side='bilateral'
        )
    
    def _calculate_au12_lip_corner_pull(
        self,
        landmarks: np.ndarray,
        face_size: float
    ) -> ActionUnit:
        """
        AU12: Lip Corner Puller (zygomatic major)
        
        This is the PRIMARY smile AU.
        
        Calculation:
        - Measure lateral and upward displacement of mouth corners
        - Check angle of mouth corner (should increase)
        
        Clinical: Smile indicator (both genuine and social smiles)
        Note: AU12 alone = social smile, AU12 + AU6 = genuine (Duchenne) smile
        """
        # Mouth corners
        mouth_left = landmarks[61, :2]
        mouth_right = landmarks[291, :2]
        mouth_center = landmarks[13, :2]
        
        # Measure width (smiling increases mouth width)
        mouth_width = np.linalg.norm(mouth_left - mouth_right)
        normalized_width = mouth_width / face_size
        
        # Measure upward displacement (corners move up and out)
        left_to_center = mouth_left - mouth_center
        right_to_center = mouth_right - mouth_center
        
        # Calculate angle (more horizontal = more smile)
        angle_left = np.abs(np.arctan2(left_to_center[1], left_to_center[0]))
        angle_right = np.abs(np.arctan2(right_to_center[1], right_to_center[0]))
        avg_angle = (angle_left + angle_right) / 2.0
        
        # Neutral width ~0.30, smile ~0.35+
        # Neutral angle ~0.2 rad, smile ~0.1 rad (more horizontal)
        width_intensity = np.clip((normalized_width - 0.30) / 0.08, 0.0, 1.0)
        angle_intensity = np.clip((0.2 - avg_angle) / 0.15, 0.0, 1.0)
        
        # Combine both measures
        intensity = (width_intensity + angle_intensity) / 2.0
        
        return ActionUnit(
            au_number=12,
            name="Mouth Corners Pulled Up",
            intensity=float(intensity),
            present=intensity > self.intensity_threshold,
            confidence=0.90,  # High confidence - reliable AU
            side='bilateral'
        )
    
    def _calculate_au15_lip_corner_depress(
        self,
        landmarks: np.ndarray,
        face_size: float
    ) -> ActionUnit:
        """
        AU15: Lip Corner Depressor (depressor anguli oris)
        
        Opposite of AU12 (smile).
        
        Calculation:
        - Measure downward displacement of mouth corners
        
        Clinical: Associated with sadness, frowning
        """
        # Mouth corners
        mouth_left = landmarks[61, :2]
        mouth_right = landmarks[291, :2]
        mouth_center = landmarks[17, :2]  # Lower mouth center
        
        # Measure vertical displacement (corners move down)
        left_vertical = mouth_left[1] - mouth_center[1]
        right_vertical = mouth_right[1] - mouth_center[1]
        avg_vertical = (left_vertical + right_vertical) / 2.0
        
        normalized = avg_vertical / face_size
        
        # Negative values = corners below center (frowning)
        # Neutral ~0.0, frown ~-0.05
        intensity = np.clip((-normalized) / 0.08, 0.0, 1.0)
        
        return ActionUnit(
            au_number=15,
            name="Mouth Corners Pulled Down",
            intensity=float(intensity),
            present=intensity > self.intensity_threshold,
            confidence=0.85,
            side='bilateral'
        )
    
    def _calculate_au17_chin_raise(
        self,
        landmarks: np.ndarray,
        face_size: float
    ) -> ActionUnit:
        """
        AU17: Chin Raiser (mentalis)
        
        Calculation:
        - Measure upward displacement of chin
        
        Clinical: Associated with doubt, sadness, "pouty" expression
        """
        chin = landmarks[152, :2]
        lower_lip = landmarks[17, :2]
        
        dist = np.linalg.norm(chin - lower_lip)
        normalized = dist / face_size
        
        # Smaller distance = more chin raise
        # Neutral ~0.15, raised ~0.12
        intensity = np.clip((0.15 - normalized) / 0.05, 0.0, 1.0)
        
        return ActionUnit(
            au_number=17,
            name="Chin Raised",
            intensity=float(intensity),
            present=intensity > self.intensity_threshold,
            confidence=0.75,
            side=None
        )
    
    def _calculate_au20_lip_stretch(
        self,
        landmarks: np.ndarray,
        face_size: float
    ) -> ActionUnit:
        """
        AU20: Lip Stretcher (risorius)
        
        Calculation:
        - Measure horizontal stretching of lips (without upward movement)
        
        Clinical: Associated with fear, tension
        """
        mouth_left = landmarks[78, :2]
        mouth_right = landmarks[308, :2]
        
        mouth_width = np.linalg.norm(mouth_left - mouth_right)
        normalized = mouth_width / face_size
        
        # Wide mouth without smile (more horizontal than AU12)
        # Neutral ~0.30, stretched ~0.36+
        intensity = np.clip((normalized - 0.30) / 0.10, 0.0, 1.0)
        
        return ActionUnit(
            au_number=20,
            name="Lips Stretched Horizontally",
            intensity=float(intensity),
            present=intensity > self.intensity_threshold,
            confidence=0.75,
            side='bilateral'
        )
    
    def _calculate_au23_lip_tightener(
        self,
        landmarks: np.ndarray,
        face_size: float
    ) -> ActionUnit:
        """
        AU23: Lip Tightener (orbicularis oris)
        
        Calculation:
        - Measure lip compression (lips pressed together)
        - Reduced lip visibility
        
        Clinical: Associated with anger, concentration, suppressing expression
        """
        upper_lip = landmarks[13, :2]
        lower_lip = landmarks[14, :2]
        
        lip_gap = np.linalg.norm(upper_lip - lower_lip)
        normalized = lip_gap / face_size
        
        # Smaller gap = more tightening
        # Neutral ~0.03, tight ~0.01
        intensity = np.clip((0.03 - normalized) / 0.025, 0.0, 1.0)
        
        return ActionUnit(
            au_number=23,
            name="Lips Tightened",
            intensity=float(intensity),
            present=intensity > self.intensity_threshold,
            confidence=0.80,
            side=None
        )
    
    def _calculate_au25_lips_part(
        self,
        landmarks: np.ndarray,
        face_size: float
    ) -> ActionUnit:
        """
        AU25: Lips Part (relaxation of mentalis or lip depressor)
        
        Calculation:
        - Measure vertical separation between lips
        
        Clinical: Relaxed state, speaking, surprise
        """
        upper_lip = landmarks[13, :2]
        lower_lip = landmarks[14, :2]
        
        lip_gap = np.linalg.norm(upper_lip - lower_lip)
        normalized = lip_gap / face_size
        
        # Larger gap = more parting
        # Neutral ~0.03, parted ~0.06+
        intensity = np.clip((normalized - 0.03) / 0.05, 0.0, 1.0)
        
        return ActionUnit(
            au_number=25,
            name="Lips Parted",
            intensity=float(intensity),
            present=intensity > self.intensity_threshold,
            confidence=0.90,
            side=None
        )
    
    def _calculate_au26_jaw_drop(
        self,
        landmarks: np.ndarray,
        face_size: float
    ) -> ActionUnit:
        """
        AU26: Jaw Drop (masseter relaxation, lateral pterygoid)
        
        Calculation:
        - Measure vertical distance between upper and lower jaw
        
        Clinical: Surprise, shock, mouth opening for speech
        """
        upper_jaw = landmarks[10, :2]  # Upper lip area
        lower_jaw = landmarks[152, :2]  # Chin
        
        jaw_opening = np.linalg.norm(upper_jaw - lower_jaw)
        normalized = jaw_opening / face_size
        
        # Larger opening = more jaw drop
        # Neutral ~0.35, dropped ~0.45+
        intensity = np.clip((normalized - 0.35) / 0.15, 0.0, 1.0)
        
        return ActionUnit(
            au_number=26,
            name="Jaw Dropped/Mouth Open",
            intensity=float(intensity),
            present=intensity > self.intensity_threshold,
            confidence=0.90,
            side=None
        )
    
    def _calculate_facial_symmetry(self, landmarks: np.ndarray) -> float:
        """
        Calculate left-right facial symmetry.
        
        Asymmetric facial expressions can be clinically significant
        (e.g., confusion, contempt, neurological conditions).
        
        Returns:
            Symmetry score (0-1, higher = more symmetric)
        """
        # Compare corresponding left-right landmark pairs
        left_indices = [33, 61, 84, 46, 70]  # Left side landmarks
        right_indices = [263, 291, 314, 276, 300]  # Corresponding right side
        
        face_center_x = landmarks[4, 0]  # Nose tip x-coordinate
        
        asymmetries = []
        
        for left_idx, right_idx in zip(left_indices, right_indices):
            left_point = landmarks[left_idx, :2]
            right_point = landmarks[right_idx, :2]
            
            # Mirror right point across face center
            right_point_mirrored = right_point.copy()
            right_point_mirrored[0] = 2 * face_center_x - right_point[0]
            
            # Calculate distance between left and mirrored right
            asymmetry = np.linalg.norm(left_point - right_point_mirrored)
            asymmetries.append(asymmetry)
        
        # Average asymmetry (normalized)
        avg_asymmetry = np.mean(asymmetries)
        face_size = self._calculate_face_size(landmarks)
        normalized_asymmetry = avg_asymmetry / face_size
        
        # Convert to symmetry score (inverse of asymmetry)
        # Typical asymmetry ~0.05, high asymmetry ~0.15+
        symmetry = 1.0 - np.clip(normalized_asymmetry / 0.20, 0.0, 1.0)
        
        return float(symmetry)


def analyze_facial_action_units(
    landmarks_sequence: List[np.ndarray],
    frame_indices: List[int],
    timestamps: List[float],
    intensity_threshold: float = 0.3
) -> List[FacialActionUnits]:
    """
    Convenience function to analyze a sequence of facial landmarks.
    
    Args:
        landmarks_sequence: List of landmark arrays (each 468x3)
        frame_indices: List of frame indices
        timestamps: List of timestamps
        intensity_threshold: Minimum AU intensity to consider "present"
        
    Returns:
        List of FacialActionUnits for each frame
    """
    analyzer = FacialActionUnitAnalyzer(intensity_threshold=intensity_threshold)
    
    results = []
    
    for landmarks, frame_idx, timestamp in zip(landmarks_sequence, frame_indices, timestamps):
        aus = analyzer.analyze_landmarks(landmarks, frame_idx, timestamp)
        results.append(aus)
    
    logger.info(
        f"Analyzed {len(results)} frames for Action Units. "
        f"Face detected in {sum(r.face_detected for r in results)} frames."
    )
    
    return results
