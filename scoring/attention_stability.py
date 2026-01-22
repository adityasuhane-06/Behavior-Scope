"""
Attention Stability Score computation.

Measures sustained attention and engagement based on:
- Head pose consistency (sustained orientation)
- Gaze proxy stability (reduced visual scanning)
- Detection consistency (remained in frame)

Score interpretation:
- 80-100: Highly stable attention (sustained engagement)
- 60-79: Moderately stable attention
- 40-59: Variable attention (frequent shifts)
- 0-39: Unstable attention (difficulty maintaining focus)

Clinical rationale:
- Sustained head orientation → maintained attention
- Reduced gaze shifts → focused processing
- Consistent detection → engagement with session
- Frequent orientation changes → attention dysregulation

Engineering approach:
- Head pose variance (lower = more stable)
- Gaze proxy variance (lower = more stable)
- Detection rate (higher = more present)
- Inverted and normalized to 0-100 scale
"""

import logging
from typing import List, Dict

import numpy as np

logger = logging.getLogger(__name__)


def compute_attention_stability_score(
    video_aggregated: List,
    config: Dict
) -> Dict:
    """
    Compute Attention Stability Score from video features.
    
    Formula:
        ASS = 100 * weighted_avg(head_pose_stability, gaze_stability, presence)
    
    Args:
        video_aggregated: List of AggregatedFeatures from video pipeline
        config: Configuration dict with weights
        
    Returns:
        Dictionary with:
        - score: Overall ASS (0-100)
        - head_pose_stability: Head orientation consistency (0-100)
        - gaze_stability: Gaze steadiness (0-100)
        - presence_score: Detection consistency (0-100)
        - explanation: Human-readable interpretation
    """
    if not video_aggregated:
        logger.warning("Empty video features")
        return {
            'score': 50.0,
            'head_pose_stability': 50.0,
            'gaze_stability': 50.0,
            'presence_score': 50.0,
            'explanation': "Insufficient data for attention stability assessment"
        }
    
    logger.info(f"Computing Attention Stability Score from {len(video_aggregated)} windows")
    
    # Get weights from config
    scoring_config = config.get('scoring', {}).get('attention_stability', {})
    weight_head = scoring_config.get('head_pose_weight', 0.5)
    weight_gaze = scoring_config.get('gaze_proxy_weight', 0.5)
    
    # Compute subcomponents
    head_stability = _compute_head_pose_stability(video_aggregated)
    gaze_stability = _compute_gaze_stability(video_aggregated)
    presence = _compute_presence_score(video_aggregated)
    
    # Weighted combination (include presence as modifier)
    base_score = (
        weight_head * head_stability +
        weight_gaze * gaze_stability
    )
    
    # Adjust by presence (penalize if frequently out of frame)
    overall_score = base_score * (0.5 + 0.5 * presence / 100.0)
    
    # Generate explanation
    explanation = _generate_ass_explanation(
        overall_score,
        head_stability,
        gaze_stability,
        presence
    )
    
    logger.info(f"Attention Stability Score: {overall_score:.1f}/100")
    
    return {
        'score': float(overall_score),
        'head_pose_stability': float(head_stability),
        'gaze_stability': float(gaze_stability),
        'presence_score': float(presence),
        'explanation': explanation
    }


def _compute_head_pose_stability(video_aggregated: List) -> float:
    """
    Compute head pose stability score.
    
    Method:
    - Extract temporal variance in head pose angles
    - Lower variance = more stable = higher score
    - Convert to 0-100 scale (inverted)
    """
    # Collect head pose variances across windows
    yaw_vars = []
    pitch_vars = []
    roll_vars = []
    
    for window in video_aggregated:
        face_feat = window.face_features
        
        # Variance of head pose angles within each window
        yaw_vars.append(face_feat.get('head_yaw_std', 0.0))
        pitch_vars.append(face_feat.get('head_pitch_std', 0.0))
        roll_vars.append(face_feat.get('head_roll_std', 0.0))
    
    if not yaw_vars:
        return 50.0
    
    # Compute overall variance (std of std across windows)
    yaw_instability = np.std(yaw_vars)
    pitch_instability = np.std(pitch_vars)
    roll_instability = np.std(roll_vars)
    
    # Combined instability
    total_instability = np.mean([yaw_instability, pitch_instability, roll_instability])
    
    # Convert to stability score
    # Empirical: 0° instability = 100, 20° instability = 0
    stability_score = 100.0 * (1.0 - np.clip(total_instability / 20.0, 0.0, 1.0))
    
    return stability_score


def _compute_gaze_stability(video_aggregated: List) -> float:
    """
    Compute gaze stability score.
    
    Method:
    - Extract gaze proxy variance
    - Lower variance = more stable = higher score
    """
    gaze_values = []
    
    for window in video_aggregated:
        face_feat = window.face_features
        gaze = face_feat.get('gaze_proxy_mean', 0.0)
        gaze_values.append(gaze)
    
    if not gaze_values:
        return 50.0
    
    # Compute variance in gaze proxy
    gaze_variance = np.var(gaze_values)
    
    # Convert to stability score
    # Empirical: 0 variance = 100, 0.05 variance = 0
    stability_score = 100.0 * (1.0 - np.clip(gaze_variance / 0.05, 0.0, 1.0))
    
    return stability_score


def _compute_presence_score(video_aggregated: List) -> float:
    """
    Compute presence/detection consistency score.
    
    Method:
    - Calculate face and pose detection rates
    - High detection = engaged and present
    """
    face_detection_rates = []
    pose_detection_rates = []
    
    for window in video_aggregated:
        face_feat = window.face_features
        pose_feat = window.pose_features
        
        face_rate = face_feat.get('face_detection_rate', 0.0)
        pose_rate = pose_feat.get('pose_detection_rate', 0.0)
        
        face_detection_rates.append(face_rate)
        pose_detection_rates.append(pose_rate)
    
    if not face_detection_rates:
        return 50.0
    
    # Average detection rates
    mean_face_detection = np.mean(face_detection_rates)
    mean_pose_detection = np.mean(pose_detection_rates)
    
    # Combined presence score (average)
    presence_score = 100.0 * (mean_face_detection + mean_pose_detection) / 2.0
    
    return presence_score


def _generate_ass_explanation(
    overall: float,
    head: float,
    gaze: float,
    presence: float
) -> str:
    """Generate human-readable explanation of ASS score."""
    
    # Overall interpretation
    if overall >= 80:
        level = "highly stable attention"
    elif overall >= 60:
        level = "moderately stable attention"
    elif overall >= 40:
        level = "variable attention"
    else:
        level = "unstable attention"
    
    explanation = f"Attention patterns show {level} (score: {overall:.1f}/100). "
    
    # Component breakdown
    factors = []
    
    if head < 60:
        factors.append(f"frequent head orientation changes (stability: {head:.1f})")
    
    if gaze < 60:
        factors.append(f"variable gaze patterns (stability: {gaze:.1f})")
    
    if presence < 70:
        factors.append(f"inconsistent presence in frame (detection: {presence:.1f}%)")
    
    if factors:
        explanation += "Contributing factors: " + ", ".join(factors) + "."
    else:
        explanation += "All attention indicators show good stability."
    
    return explanation
