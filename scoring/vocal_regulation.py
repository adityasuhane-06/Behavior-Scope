"""
Vocal Regulation Index computation.

Measures speech pattern stability based on prosodic features:
- Speech rate consistency (avoiding rapid/slow extremes)
- Pause regularity (appropriate pause patterns)
- Prosodic stability (pitch and energy consistency)

Score interpretation:
- 80-100: Highly regulated speech patterns
- 60-79: Moderately regulated with some variability
- 40-59: Noticeable instability in speech patterns
- 0-39: Significant vocal dysregulation

Clinical rationale:
- Regulated speech → cognitive control, emotional stability
- Speech rate extremes → agitation (fast) or withdrawal (slow)
- Irregular pauses → processing difficulty, uncertainty
- Prosodic instability → emotional dysregulation

Engineering approach:
- Z-score based deviation from baseline
- Weighted combination of subcomponents
- Inverted scale (instability → regulation)
- Normalized to 0-100 range
"""

import logging
from typing import List, Dict

import numpy as np

logger = logging.getLogger(__name__)


def compute_vocal_regulation_index(
    prosodic_features: List,
    baseline: Dict,
    config: Dict
) -> Dict:
    """
    Compute Vocal Regulation Index from prosodic features.
    
    Formula:
        VRI = 100 * (1 - weighted_avg(speech_rate_dev, pause_irreg, prosody_instability))
    
    Where each component is normalized to [0, 1] scale.
    
    Args:
        prosodic_features: List of ProsodicFeatures from audio pipeline
        baseline: Baseline statistics (mean/std for each feature)
        config: Configuration dict with weights
        
    Returns:
        Dictionary with:
        - score: Overall VRI (0-100)
        - speech_rate_score: Speech rate subcomponent (0-100)
        - pause_score: Pause regularity subcomponent (0-100)
        - prosody_score: Prosodic stability subcomponent (0-100)
        - explanation: Human-readable interpretation
    """
    if not prosodic_features or not baseline:
        logger.warning("Empty prosodic features or baseline")
        return {
            'score': 50.0,  # Neutral score
            'speech_rate_score': 50.0,
            'pause_score': 50.0,
            'prosody_score': 50.0,
            'explanation': "Insufficient data for vocal regulation assessment"
        }
    
    logger.info(f"Computing Vocal Regulation Index from {len(prosodic_features)} windows")
    
    # Get weights from config
    scoring_config = config.get('scoring', {}).get('vocal_regulation', {})
    weight_baseline = scoring_config.get('baseline_weight', 0.4)
    weight_pause = scoring_config.get('pause_weight', 0.3)
    weight_prosody = scoring_config.get('prosody_weight', 0.3)
    
    # Compute subcomponents
    speech_rate_score = _compute_speech_rate_score(prosodic_features, baseline)
    pause_score = _compute_pause_score(prosodic_features, baseline)
    prosody_score = _compute_prosody_score(prosodic_features, baseline)
    
    # Weighted combination
    overall_score = (
        weight_baseline * speech_rate_score +
        weight_pause * pause_score +
        weight_prosody * prosody_score
    )
    
    # Generate explanation
    explanation = _generate_vri_explanation(
        overall_score,
        speech_rate_score,
        pause_score,
        prosody_score
    )
    
    logger.info(f"Vocal Regulation Index: {overall_score:.1f}/100")
    
    return {
        'score': float(overall_score),
        'speech_rate_score': float(speech_rate_score),
        'pause_score': float(pause_score),
        'prosody_score': float(prosody_score),
        'explanation': explanation
    }


def _compute_speech_rate_score(features: List, baseline: Dict) -> float:
    """
    Compute speech rate consistency score.
    
    Method:
    - Extract speech rate variability (std dev from baseline)
    - Convert to deviation score (0 = baseline, 1 = extreme deviation)
    - Invert to regulation score (100 = consistent, 0 = highly variable)
    """
    speech_rates = [f.speech_rate for f in features if f.speech_rate > 0]
    
    if not speech_rates:
        return 50.0
    
    # Compute statistics
    mean_rate = np.mean(speech_rates)
    std_rate = np.std(speech_rates)
    
    # Z-score deviation from baseline
    baseline_mean = baseline.get('speech_rate_mean', mean_rate)
    baseline_std = baseline.get('speech_rate_std', 1.0)
    
    if baseline_std < 0.01:
        baseline_std = 1.0
    
    # Mean z-score across windows
    z_scores = [(rate - baseline_mean) / baseline_std for rate in speech_rates]
    mean_abs_zscore = np.mean(np.abs(z_scores))
    
    # Convert to regulation score
    # z=0 → score=100, z=3 → score=0
    deviation_score = np.clip(mean_abs_zscore / 3.0, 0.0, 1.0)
    regulation_score = 100.0 * (1.0 - deviation_score)
    
    return regulation_score


def _compute_pause_score(features: List, baseline: Dict) -> float:
    """
    Compute pause regularity score.
    
    Method:
    - Evaluate pause duration consistency
    - Penalize excessive or minimal pausing
    - Convert to regulation score
    """
    pause_durations = [f.pause_duration_mean for f in features]
    pause_counts = [f.pause_count for f in features]
    
    if not pause_durations:
        return 50.0
    
    # Compute pause irregularity
    mean_pause = np.mean(pause_durations)
    std_pause = np.std(pause_durations)
    
    baseline_mean = baseline.get('pause_duration_mean', mean_pause)
    baseline_std = baseline.get('pause_duration_std', 0.5)
    
    if baseline_std < 0.01:
        baseline_std = 0.5
    
    # Z-score deviation
    z_score = abs(mean_pause - baseline_mean) / baseline_std
    
    # Pause count regularity (normalized)
    mean_count = np.mean(pause_counts)
    std_count = np.std(pause_counts)
    count_irregularity = std_count / (mean_count + 1.0)  # Coefficient of variation
    
    # Combined pause score
    pause_deviation = np.clip((z_score + count_irregularity) / 4.0, 0.0, 1.0)
    pause_score = 100.0 * (1.0 - pause_deviation)
    
    return pause_score


def _compute_prosody_score(features: List, baseline: Dict) -> float:
    """
    Compute prosodic stability score.
    
    Method:
    - Evaluate pitch variance (excessive = unstable)
    - Evaluate energy variance (excessive = unstable)
    - Convert to stability score
    """
    pitch_stds = [f.pitch_std for f in features if f.pitch_std > 0]
    energy_stds = [f.energy_std for f in features if f.energy_std > 0]
    
    if not pitch_stds or not energy_stds:
        return 50.0
    
    # Pitch variability
    mean_pitch_std = np.mean(pitch_stds)
    baseline_pitch_var = baseline.get('pitch_variability_mean', mean_pitch_std)
    baseline_pitch_std = baseline.get('pitch_variability_std', 10.0)
    
    if baseline_pitch_std < 0.01:
        baseline_pitch_std = 10.0
    
    pitch_z = abs(mean_pitch_std - baseline_pitch_var) / baseline_pitch_std
    
    # Energy variability
    mean_energy_std = np.mean(energy_stds)
    baseline_energy_var = baseline.get('energy_variability_mean', mean_energy_std)
    baseline_energy_std = baseline.get('energy_variability_std', 2.0)
    
    if baseline_energy_std < 0.01:
        baseline_energy_std = 2.0
    
    energy_z = abs(mean_energy_std - baseline_energy_var) / baseline_energy_std
    
    # Combined prosody instability
    prosody_instability = np.clip((pitch_z + energy_z) / 4.0, 0.0, 1.0)
    prosody_score = 100.0 * (1.0 - prosody_instability)
    
    return prosody_score


def _generate_vri_explanation(
    overall: float,
    speech_rate: float,
    pause: float,
    prosody: float
) -> str:
    """Generate human-readable explanation of VRI score."""
    
    # Overall interpretation
    if overall >= 80:
        level = "highly regulated"
    elif overall >= 60:
        level = "moderately regulated"
    elif overall >= 40:
        level = "showing instability"
    else:
        level = "significantly dysregulated"
    
    explanation = f"Vocal patterns are {level} (score: {overall:.1f}/100). "
    
    # Component breakdown
    components = []
    
    if speech_rate < 60:
        components.append(f"speech rate variability (score: {speech_rate:.1f})")
    
    if pause < 60:
        components.append(f"pause irregularity (score: {pause:.1f})")
    
    if prosody < 60:
        components.append(f"prosodic instability (score: {prosody:.1f})")
    
    if components:
        explanation += "Contributing factors: " + ", ".join(components) + "."
    else:
        explanation += "All vocal components show good stability."
    
    return explanation
