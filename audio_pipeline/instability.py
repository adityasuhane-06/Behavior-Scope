"""
Rule-based vocal instability detection.

This is the CORE of the audio-first approach: using interpretable rules to flag
dysregulation windows based on prosodic features.

Clinical rationale:
Vocal instability manifests as:
1. Speech rate deviations (too fast or too slow vs baseline)
2. Pause irregularity (unusual frequency or duration)
3. Pitch variance spikes (prosodic instability)
4. Energy fluctuations (arousal changes)

Engineering approach:
- Z-score normalization: compare to individual baseline (handles inter-subject variability)
- Composite scoring: multiple features must agree (reduce false positives)
- Tunable thresholds: clinicians can adjust sensitivity via config
- Explainable output: store which features triggered each flag

NOT EMOTION CLASSIFICATION:
- We detect INSTABILITY (change from baseline), not emotions
- No labels like "angry", "sad" - only "unstable", "stable"
- This is regulation assessment, not affective state detection
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InstabilityWindow:
    """
    Represents a detected vocal instability window.
    
    Attributes:
        start_time: Window start in seconds
        end_time: Window end in seconds
        instability_score: Composite score (0-1, higher = more unstable)
        contributing_features: Dict mapping feature names to z-scores
        explanation: Human-readable explanation of why window was flagged
    """
    start_time: float
    end_time: float
    instability_score: float
    contributing_features: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end_time - self.start_time


def detect_vocal_instability(
    prosodic_features: List,
    config: Dict,
    baseline: Dict = None
) -> List[InstabilityWindow]:
    """
    Detect vocal instability windows using rule-based analysis.
    
    Algorithm:
    1. Compute z-scores for each feature relative to baseline
    2. Flag features exceeding thresholds
    3. Compute composite instability score
    4. Filter windows below minimum score
    5. Merge adjacent windows
    
    Args:
        prosodic_features: List of ProsodicFeatures from prosody module
        config: Configuration dict with thresholds (from thresholds.yaml)
        baseline: Optional pre-computed baseline statistics
                 If None, computed from prosodic_features
        
    Returns:
        List of InstabilityWindow objects (sorted by score, descending)
        
    Clinical interpretation:
        High scores indicate periods where vocal patterns deviate significantly
        from the individual's baseline, suggesting potential dysregulation.
    """
    if not prosodic_features:
        logger.warning("No prosodic features provided")
        return []
    
    logger.info(f"Detecting instability across {len(prosodic_features)} windows")
    
    # Compute baseline if not provided
    if baseline is None:
        from .prosody import compute_baseline_statistics
        baseline = compute_baseline_statistics(prosodic_features)
    
    # Get thresholds from config
    speech_rate_threshold = config.get('speech_rate_zscore_threshold', 2.0)
    pause_threshold = config.get('pause_irregularity_zscore_threshold', 1.8)
    pitch_threshold = config.get('pitch_variance_zscore_threshold', 2.2)
    energy_threshold = config.get('energy_variance_zscore_threshold', 2.0)
    min_score = config.get('instability_score_threshold', 0.6)
    min_duration = config.get('min_instability_duration_sec', 3.0)
    
    # Analyze each window
    instability_windows = []
    
    for features in prosodic_features:
        # Compute z-scores for each feature
        z_scores = _compute_feature_zscores(features, baseline)
        
        # Check which features exceed thresholds
        flags = {
            'speech_rate': abs(z_scores['speech_rate']) > speech_rate_threshold,
            'pause_irregularity': abs(z_scores['pause_irregularity']) > pause_threshold,
            'pitch_variance': z_scores['pitch_variability'] > pitch_threshold,
            'energy_variance': z_scores['energy_variability'] > energy_threshold,
        }
        
        # Compute composite instability score
        # Weighted average of flagged features
        weights = {
            'speech_rate': 0.3,
            'pause_irregularity': 0.25,
            'pitch_variance': 0.25,
            'energy_variance': 0.2,
        }
        
        score = 0.0
        for feature, flagged in flags.items():
            if flagged:
                # Use z-score magnitude (capped at 5.0 for numerical stability)
                z_magnitude = min(abs(z_scores[feature.replace('_irregularity', '_duration').replace('_variance', '_variability')]), 5.0)
                score += weights[feature] * (z_magnitude / 5.0)
        
        # Only flag windows above threshold
        if score >= min_score:
            # Generate explanation
            explanation = _generate_explanation(features, z_scores, flags, baseline)
            
            window = InstabilityWindow(
                start_time=features.start_time,
                end_time=features.end_time,
                instability_score=score,
                contributing_features=z_scores,
                explanation=explanation
            )
            
            instability_windows.append(window)
    
    # Filter by minimum duration
    instability_windows = [
        w for w in instability_windows
        if w.duration >= min_duration
    ]
    
    # Merge adjacent windows
    if instability_windows:
        instability_windows = _merge_adjacent_windows(
            instability_windows,
            max_gap=config.get('merge_gap_threshold_sec', 2.0)
        )
    
    # Sort by score (highest first)
    instability_windows.sort(key=lambda w: w.instability_score, reverse=True)
    
    logger.info(
        f"Detected {len(instability_windows)} instability windows "
        f"(total duration: {sum(w.duration for w in instability_windows):.1f}s)"
    )
    
    return instability_windows


def _compute_feature_zscores(features, baseline: Dict) -> Dict[str, float]:
    """
    Compute z-scores for prosodic features relative to baseline.
    
    Z-score = (value - baseline_mean) / baseline_std
    
    Returns dict with z-score for each feature.
    """
    z_scores = {}
    
    # Speech rate z-score
    z_scores['speech_rate'] = _compute_zscore(
        features.speech_rate,
        baseline['speech_rate_mean'],
        baseline['speech_rate_std']
    )
    
    # Pause irregularity (deviation from expected pause duration)
    z_scores['pause_irregularity'] = _compute_zscore(
        features.pause_duration_mean,
        baseline['pause_duration_mean'],
        baseline['pause_duration_std']
    )
    
    # Pitch variance (instability)
    z_scores['pitch_variability'] = _compute_zscore(
        features.pitch_std,
        baseline['pitch_variability_mean'],
        baseline['pitch_variability_std']
    )
    
    # Energy variance
    z_scores['energy_variability'] = _compute_zscore(
        features.energy_std,
        baseline['energy_variability_mean'],
        baseline['energy_variability_std']
    )
    
    return z_scores


def _compute_zscore(value: float, mean: float, std: float) -> float:
    """Compute z-score with protection against zero std."""
    if std < 1e-6:
        return 0.0
    return (value - mean) / std


def _generate_explanation(features, z_scores: Dict, flags: Dict, baseline: Dict) -> str:
    """
    Generate human-readable explanation for why window was flagged.
    
    This is CRITICAL for clinical interpretability and trust.
    """
    explanations = []
    
    if flags['speech_rate']:
        direction = "faster" if features.speech_rate > baseline['speech_rate_mean'] else "slower"
        explanations.append(
            f"Speech rate {direction} than baseline "
            f"({features.speech_rate:.2f} vs {baseline['speech_rate_mean']:.2f} syll/s, "
            f"z={z_scores['speech_rate']:.2f})"
        )
    
    if flags['pause_irregularity']:
        explanations.append(
            f"Unusual pause duration "
            f"({features.pause_duration_mean:.2f}s vs {baseline['pause_duration_mean']:.2f}s, "
            f"z={z_scores['pause_irregularity']:.2f})"
        )
    
    if flags['pitch_variance']:
        explanations.append(
            f"High pitch variability "
            f"(std={features.pitch_std:.1f}Hz vs baseline {baseline['pitch_variability_mean']:.1f}Hz, "
            f"z={z_scores['pitch_variability']:.2f})"
        )
    
    if flags['energy_variance']:
        explanations.append(
            f"High energy fluctuation "
            f"(std={features.energy_std:.1f}dB vs baseline {baseline['energy_variability_mean']:.1f}dB, "
            f"z={z_scores['energy_variability']:.2f})"
        )
    
    if explanations:
        return "; ".join(explanations)
    else:
        return "No significant deviations detected"


def _merge_adjacent_windows(
    windows: List[InstabilityWindow],
    max_gap: float = 2.0
) -> List[InstabilityWindow]:
    """
    Merge adjacent instability windows separated by small gaps.
    
    Clinical rationale:
    - Brief stable periods between unstable periods likely represent
      continuous dysregulation episode
    - Merging reduces fragmentation and improves segment prioritization
    
    Args:
        windows: List of InstabilityWindow objects (must be sorted by start_time)
        max_gap: Maximum gap duration to merge (seconds)
        
    Returns:
        List of merged windows
    """
    if not windows:
        return []
    
    # Sort by start time
    windows = sorted(windows, key=lambda w: w.start_time)
    
    merged = []
    current = windows[0]
    
    for next_window in windows[1:]:
        gap = next_window.start_time - current.end_time
        
        if gap <= max_gap:
            # Merge windows
            current = InstabilityWindow(
                start_time=current.start_time,
                end_time=next_window.end_time,
                instability_score=max(current.instability_score, next_window.instability_score),
                contributing_features=current.contributing_features,  # Keep first window's features
                explanation=f"{current.explanation} [merged with adjacent window]"
            )
        else:
            # Gap too large, save current and start new
            merged.append(current)
            current = next_window
    
    # Add final window
    merged.append(current)
    
    logger.debug(f"Merged {len(windows)} windows into {len(merged)} windows")
    
    return merged


def get_top_instability_windows(
    windows: List[InstabilityWindow],
    n: int = 5
) -> List[InstabilityWindow]:
    """
    Get top N highest-scoring instability windows.
    
    Used to prioritize which video segments to analyze.
    
    Args:
        windows: List of InstabilityWindow objects
        n: Number of top windows to return
        
    Returns:
        Top N windows sorted by score (descending)
    """
    sorted_windows = sorted(windows, key=lambda w: w.instability_score, reverse=True)
    return sorted_windows[:n]
