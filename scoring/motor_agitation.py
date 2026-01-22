"""
Motor Agitation Index computation.

Measures movement intensity based on visual features:
- Head movement (yaw, pitch, roll variability)
- Upper-body motion (shoulder, arm, torso)
- Hand movement velocity (fidgeting indicator)

Score interpretation:
- 0-20: Minimal movement (very calm or withdrawn)
- 21-40: Low movement (calm, controlled)
- 41-60: Moderate movement (typical engagement)
- 61-80: Elevated movement (restlessness)
- 81-100: High agitation (excessive motor activity)

Clinical rationale:
- Motor agitation → arousal, anxiety, inability to sit still
- Fidgeting (hand movements) → self-soothing, discomfort
- Head movements → attention shifts, restlessness
- Postural shifts → discomfort, avoidance

Engineering approach:
- Aggregate movement across body regions
- Normalize by baseline activity level
- Weighted combination (head > body > hands)
- 0-100 scale (higher = more agitation)
"""

import logging
from typing import List, Dict

import numpy as np

logger = logging.getLogger(__name__)


def compute_motor_agitation_index(
    video_aggregated: List,
    config: Dict
) -> Dict:
    """
    Compute Motor Agitation Index from video features.
    
    Formula:
        MAI = 100 * weighted_avg(head_motion, body_motion, hand_motion)
    
    Each component normalized to [0, 1] scale.
    
    Args:
        video_aggregated: List of AggregatedFeatures from video pipeline
        config: Configuration dict with weights
        
    Returns:
        Dictionary with:
        - score: Overall MAI (0-100)
        - head_motion_score: Head movement component (0-100)
        - body_motion_score: Body movement component (0-100)
        - hand_motion_score: Hand movement component (0-100)
        - explanation: Human-readable interpretation
    """
    if not video_aggregated:
        logger.warning("Empty video features")
        return {
            'score': 30.0,  # Assume low baseline
            'head_motion_score': 30.0,
            'body_motion_score': 30.0,
            'hand_motion_score': 30.0,
            'explanation': "Insufficient data for motor agitation assessment"
        }
    
    logger.info(f"Computing Motor Agitation Index from {len(video_aggregated)} windows")
    
    # Get weights from config
    scoring_config = config.get('scoring', {}).get('motor_agitation', {})
    weight_head = scoring_config.get('head_motion_weight', 0.4)
    weight_body = scoring_config.get('body_motion_weight', 0.4)
    weight_hand = scoring_config.get('hand_motion_weight', 0.2)
    
    # Compute subcomponents
    head_score = _compute_head_motion_score(video_aggregated)
    body_score = _compute_body_motion_score(video_aggregated)
    hand_score = _compute_hand_motion_score(video_aggregated)
    
    # Weighted combination
    overall_score = (
        weight_head * head_score +
        weight_body * body_score +
        weight_hand * hand_score
    )
    
    # Generate explanation
    explanation = _generate_mai_explanation(
        overall_score,
        head_score,
        body_score,
        hand_score
    )
    
    logger.info(f"Motor Agitation Index: {overall_score:.1f}/100")
    
    return {
        'score': float(overall_score),
        'head_motion_score': float(head_score),
        'body_motion_score': float(body_score),
        'hand_motion_score': float(hand_score),
        'explanation': explanation
    }


def _compute_head_motion_score(video_aggregated: List) -> float:
    """
    Compute head movement agitation score.
    
    Method:
    - Extract head pose variance (yaw, pitch, roll std)
    - Normalize to 0-100 scale
    - Higher = more head movement
    """
    head_motion_values = []
    
    for window in video_aggregated:
        face_feat = window.face_features
        
        # Average std across head pose angles
        yaw_std = face_feat.get('head_yaw_std', 0.0)
        pitch_std = face_feat.get('head_pitch_std', 0.0)
        roll_std = face_feat.get('head_roll_std', 0.0)
        
        # Combined head motion (average of angular variances)
        head_motion = np.mean([yaw_std, pitch_std, roll_std])
        head_motion_values.append(head_motion)
    
    if not head_motion_values:
        return 30.0
    
    # Compute mean head motion
    mean_motion = np.mean(head_motion_values)
    
    # Normalize to 0-100 scale
    # Empirical scaling: 0° = 0, 30° = 100
    normalized_score = np.clip((mean_motion / 30.0) * 100.0, 0.0, 100.0)
    
    return normalized_score


def _compute_body_motion_score(video_aggregated: List) -> float:
    """
    Compute body movement agitation score.
    
    Method:
    - Extract upper-body motion energy
    - Normalize to 0-100 scale
    """
    body_motion_values = []
    
    for window in video_aggregated:
        pose_feat = window.pose_features
        
        # Upper body motion (mean and max)
        body_motion_mean = pose_feat.get('upper_body_motion_mean', 0.0)
        body_motion_max = pose_feat.get('upper_body_motion_max', 0.0)
        
        # Use 95th percentile for robustness
        body_motion = pose_feat.get('upper_body_motion_percentile_95', body_motion_mean)
        body_motion_values.append(body_motion)
    
    if not body_motion_values:
        return 30.0
    
    # Compute mean body motion
    mean_motion = np.mean(body_motion_values)
    
    # Normalize to 0-100 scale
    # Empirical scaling: 0.0 = 0, 0.2 = 100
    normalized_score = np.clip((mean_motion / 0.2) * 100.0, 0.0, 100.0)
    
    return normalized_score


def _compute_hand_motion_score(video_aggregated: List) -> float:
    """
    Compute hand movement (fidgeting) score.
    
    Method:
    - Extract hand velocity (both hands)
    - Normalize to 0-100 scale
    """
    hand_velocity_values = []
    
    for window in video_aggregated:
        pose_feat = window.pose_features
        
        # Max hand velocity across both hands
        hand_vel = pose_feat.get('hand_velocity_max_mean', 0.0)
        hand_velocity_values.append(hand_vel)
    
    if not hand_velocity_values:
        return 30.0
    
    # Compute mean hand velocity
    mean_velocity = np.mean(hand_velocity_values)
    
    # Normalize to 0-100 scale
    # Empirical scaling: 0 px/frame = 0, 40 px/frame = 100
    normalized_score = np.clip((mean_velocity / 40.0) * 100.0, 0.0, 100.0)
    
    return normalized_score


def _generate_mai_explanation(
    overall: float,
    head: float,
    body: float,
    hand: float
) -> str:
    """Generate human-readable explanation of MAI score."""
    
    # Overall interpretation
    if overall >= 80:
        level = "high motor agitation"
    elif overall >= 60:
        level = "elevated motor activity"
    elif overall >= 40:
        level = "moderate motor activity"
    elif overall >= 20:
        level = "low motor activity"
    else:
        level = "minimal motor activity"
    
    explanation = f"Motor patterns show {level} (score: {overall:.1f}/100). "
    
    # Component breakdown
    components = []
    
    if head > 60:
        components.append(f"significant head movement ({head:.1f})")
    
    if body > 60:
        components.append(f"elevated body motion ({body:.1f})")
    
    if hand > 60:
        components.append(f"frequent hand movements ({hand:.1f})")
    
    if components:
        explanation += "Contributing factors: " + ", ".join(components) + "."
    else:
        if overall < 30:
            explanation += "Movement levels are very low, which may indicate withdrawal or low engagement."
        else:
            explanation += "Movement levels are within typical range."
    
    return explanation
