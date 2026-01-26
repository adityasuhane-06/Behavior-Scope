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
"""

import logging
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


def _extract_val(feat_dict: Dict, key: str, metric: str) -> float:
    """
    Safely extract statistical value from feature dictionary.
    Handles both nested (ImprovedAggregator) and flat (Legacy) structures.
    """
    if not isinstance(feat_dict, dict):
        return 0.0
    
    # Try nested structure (ImprovedAggregator style)
    # e.g., feat_dict['head_yaw']['mean']
    if key in feat_dict:
        val = feat_dict[key]
        if isinstance(val, dict):
            v = val.get(metric, 0.0)
            return float(v) if v is not None else 0.0
        
        # Fallback for scalar legacy values if metric is 'mean'
        if metric == 'mean':
            try:
                return float(val)
            except (ValueError, TypeError):
                pass

    # Try flat key (Legacy style)
    # e.g., feat_dict['head_yaw_mean']
    flat_key = f"{key}_{metric}"
    if flat_key in feat_dict:
        v = feat_dict[flat_key]
        return float(v) if v is not None else 0.0
        
    return 0.0


def compute_attention_stability_score(
    video_aggregated: List,
    config: Dict
) -> Dict:
    """
    Compute Attention Stability Score from video features.
    
    Formula:
        ASS = Base Stability * Presence Factor
        Base Stability = 0.5 * HeadStability + 0.5 * GazeStability
    
    Args:
        video_aggregated: List of AggregatedFeatures (windows)
        config: Configuration dict with weights
        
    Returns:
        Dictionary with scores and raw metrics.
    """
    if not video_aggregated:
        logger.warning("Empty video features")
        return {
            'score': 50.0,
            'head_pose_stability': 50.0,
            'gaze_stability': 50.0,
            'presence_score': 50.0,
            'raw_head_variance': 0.0,
            'raw_gaze_variance': 0.0,
            'explanation': "Insufficient data for attention stability assessment"
        }
    
    logger.info(f"Computing Attention Stability Score from {len(video_aggregated)} windows")
    
    # Get weights from config
    scoring_config = config.get('scoring', {}).get('attention_stability', {})
    weight_head = scoring_config.get('head_pose_weight', 0.5)
    weight_gaze = scoring_config.get('gaze_proxy_weight', 0.5)
    
    # Compute subcomponents
    head_stability, raw_head_var = _compute_head_pose_stability(video_aggregated)
    gaze_stability, raw_gaze_var = _compute_gaze_stability(video_aggregated)
    presence = _compute_presence_score(video_aggregated)
    
    # Weighted combination for base stability
    base_score = (
        weight_head * head_stability +
        weight_gaze * gaze_stability
    )
    
    # Adjust by presence (penalize if frequently out of frame)
    # Formula: Score scales down to 50% if presence is 0
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
        'raw_head_variance': float(raw_head_var),
        'raw_gaze_variance': float(raw_gaze_var),
        'explanation': explanation
    }


def _compute_head_pose_stability(video_aggregated: List) -> Tuple[float, float]:
    """
    Compute head pose stability score.
    
    Method:
    - Intra-window instability: Mean of std deviations (shaking)
    - Inter-window instability: Std deviation of means (drift)
    - Total instability = Intra + Inter
    
    Returns: (score, raw_instability_degrees)
    """
    yaw_means, yaw_stds = [], []
    pitch_means, pitch_stds = [], []
    roll_means, roll_stds = [], []
    
    for window in video_aggregated:
        ff = window.face_features
        # Extract means and stds for all angles
        yaw_means.append(_extract_val(ff, 'head_yaw', 'mean'))
        yaw_stds.append(_extract_val(ff, 'head_yaw', 'std'))
        
        pitch_means.append(_extract_val(ff, 'head_pitch', 'mean'))
        pitch_stds.append(_extract_val(ff, 'head_pitch', 'std'))
        
        roll_means.append(_extract_val(ff, 'head_roll', 'mean'))
        roll_stds.append(_extract_val(ff, 'head_roll', 'std'))
    
    if not yaw_means:
        return 50.0, 0.0
    
    # Calculate drift (inter-window) and shake (intra-window)
    # Yaw
    yaw_drift = np.std(yaw_means)
    yaw_shake = np.mean(yaw_stds)
    yaw_instability = yaw_drift + yaw_shake
    
    # Pitch
    pitch_drift = np.std(pitch_means)
    pitch_shake = np.mean(pitch_stds)
    pitch_instability = pitch_drift + pitch_shake
    
    # Roll
    roll_drift = np.std(roll_means)
    roll_shake = np.mean(roll_stds)
    roll_instability = roll_drift + roll_shake
    
    # Average total instability across axes
    total_instability = np.mean([yaw_instability, pitch_instability, roll_instability])
    
    # Convert to stability score (0-100)
    # Threshold: 20.0 degrees
    # 0 deg = 100 score, 20 deg = 0 score
    stability_score = 100.0 * (1.0 - np.clip(total_instability / 20.0, 0.0, 1.0))
    
    return stability_score, total_instability


def _compute_gaze_stability(video_aggregated: List) -> Tuple[float, float]:
    """
    Compute gaze stability score.
    
    Method:
    - Intra-window var + Inter-window drift
    
    Returns: (score, raw_variance_equivalent)
    """
    gaze_means = []
    gaze_stds = []
    
    for window in video_aggregated:
        ff = window.face_features
        gaze_means.append(_extract_val(ff, 'gaze_proxy', 'mean'))
        gaze_stds.append(_extract_val(ff, 'gaze_proxy', 'std'))
    
    if not gaze_means:
        return 50.0, 0.0
    
    # Calculate drift and shake
    # Note: inputs are std devs, so we sum std devs
    gaze_drift = np.std(gaze_means)
    gaze_shake = np.mean(gaze_stds)
    
    # Total instability in std-dev space
    total_instability = gaze_drift + gaze_shake
    
    # Convert to Variance-equivalent for scoring
    # Threshold is "0.05 Variance"
    total_instability_sq = total_instability ** 2
    
    # Convert to stability score
    stability_score = 100.0 * (1.0 - np.clip(total_instability_sq / 0.05, 0.0, 1.0))
    
    return stability_score, total_instability_sq


def _compute_presence_score(video_aggregated: List) -> float:
    """
    Compute presence/detection consistency score.
    """
    face_rates = []
    pose_rates = []
    
    for window in video_aggregated:
        ff = window.face_features
        pf = window.pose_features
        
        face_rates.append(_extract_val(ff, 'face_detection_rate', 'mean'))
        pose_rates.append(_extract_val(pf, 'pose_detection_rate', 'mean'))
    
    if not face_rates:
        return 50.0
    
    mean_face = np.mean(face_rates)
    mean_pose = np.mean(pose_rates)
    
    # Combined presence score (0-100)
    presence_score = 100.0 * (mean_face + mean_pose) / 2.0
    
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
