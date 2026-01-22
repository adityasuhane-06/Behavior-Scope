"""
Regulation Consistency Index computation.

Measures temporal consistency of regulatory states:
- Autocorrelation of instability scores
- Variability in behavioral indices over time
- Pattern stability vs erratic shifts

Score interpretation:
- 80-100: Highly consistent regulation (stable state)
- 60-79: Moderately consistent (gradual changes)
- 40-59: Variable regulation (fluctuating states)
- 0-39: Inconsistent regulation (erratic patterns)

Clinical rationale:
- Consistent regulation → sustained coping, stable state
- Gradual changes → natural arousal/recovery cycles
- Erratic shifts → difficulty maintaining regulatory control
- Pattern analysis reveals regulatory capacity

Engineering approach:
- Temporal autocorrelation (lag-based similarity)
- Coefficient of variation (relative variability)
- Smoothing window analysis
- Normalized to 0-100 scale
"""

import logging
from typing import List, Dict

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


def compute_regulation_consistency_index(
    fused_evidence: List,
    config: Dict
) -> Dict:
    """
    Compute Regulation Consistency Index from fused evidence timeline.
    
    Formula:
        RCI = 100 * (1 - variability_score) * autocorrelation_score
    
    Where:
    - variability_score: Coefficient of variation in confidence scores
    - autocorrelation_score: Temporal autocorrelation at lag window
    
    Args:
        fused_evidence: List of FusedEvidence objects (temporal timeline)
        config: Configuration dict
        
    Returns:
        Dictionary with:
        - score: Overall RCI (0-100)
        - autocorrelation: Temporal autocorrelation (0-1)
        - variability: Coefficient of variation (0-1)
        - trend: 'improving', 'stable', or 'worsening'
        - explanation: Human-readable interpretation
    """
    if not fused_evidence or len(fused_evidence) < 3:
        logger.warning("Insufficient fused evidence for consistency analysis")
        return {
            'score': 50.0,
            'autocorrelation': 0.5,
            'variability': 0.5,
            'trend': 'unknown',
            'explanation': "Insufficient data for consistency assessment (need at least 3 windows)"
        }
    
    logger.info(f"Computing Regulation Consistency Index from {len(fused_evidence)} windows")
    
    # Get config parameters
    consistency_config = config.get('scoring', {}).get('consistency', {})
    lag_window_sec = consistency_config.get('lag_window_sec', 10.0)
    smoothing_window = consistency_config.get('smoothing_window', 5)
    
    # Extract confidence scores timeline
    sorted_evidence = sorted(fused_evidence, key=lambda fe: fe.start_time)
    confidence_scores = np.array([fe.fused_confidence for fe in sorted_evidence])
    timestamps = np.array([fe.start_time for fe in sorted_evidence])
    
    # Compute subcomponents
    autocorr = _compute_temporal_autocorrelation(
        confidence_scores,
        timestamps,
        lag_window_sec
    )
    
    variability = _compute_relative_variability(confidence_scores)
    
    trend = _compute_temporal_trend(confidence_scores, smoothing_window)
    
    # Compute overall consistency score
    # High autocorrelation + low variability = high consistency
    consistency_score = 100.0 * (1.0 - variability) * autocorr
    
    # Generate explanation
    explanation = _generate_rci_explanation(
        consistency_score,
        autocorr,
        variability,
        trend
    )
    
    logger.info(f"Regulation Consistency Index: {consistency_score:.1f}/100")
    
    return {
        'score': float(consistency_score),
        'autocorrelation': float(autocorr),
        'variability': float(variability),
        'trend': trend,
        'explanation': explanation
    }


def _compute_temporal_autocorrelation(
    scores: np.ndarray,
    timestamps: np.ndarray,
    lag_sec: float
) -> float:
    """
    Compute temporal autocorrelation at specified lag.
    
    Method:
    - Compare each window with window ~lag_sec earlier
    - Compute correlation coefficient
    - Higher correlation = more consistent pattern
    
    Returns:
        Autocorrelation coefficient (0-1, normalized)
    """
    if len(scores) < 2:
        return 0.5
    
    # Compute mean window duration
    if len(timestamps) > 1:
        mean_duration = np.mean(np.diff(timestamps))
        lag_windows = int(lag_sec / mean_duration)
    else:
        lag_windows = 1
    
    lag_windows = max(1, min(lag_windows, len(scores) - 1))
    
    # Compute autocorrelation at lag
    if len(scores) > lag_windows:
        # Pearson correlation between series and lagged series
        series1 = scores[:-lag_windows]
        series2 = scores[lag_windows:]
        
        if len(series1) > 1 and np.std(series1) > 0 and np.std(series2) > 0:
            correlation = np.corrcoef(series1, series2)[0, 1]
            # Convert to 0-1 scale (correlation ranges from -1 to 1)
            normalized_corr = (correlation + 1.0) / 2.0
        else:
            normalized_corr = 0.5
    else:
        # Not enough data for lag
        normalized_corr = 0.5
    
    return float(np.clip(normalized_corr, 0.0, 1.0))


def _compute_relative_variability(scores: np.ndarray) -> float:
    """
    Compute coefficient of variation (relative variability).
    
    Method:
    - CV = std / mean
    - Normalized to 0-1 scale
    
    Returns:
        Variability score (0-1, higher = more variable)
    """
    if len(scores) == 0:
        return 0.5
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    if mean_score < 0.01:
        # Very low scores - high relative variability
        return 0.8
    
    # Coefficient of variation
    cv = std_score / mean_score
    
    # Normalize: CV of 0 = 0, CV of 1.0 = 1
    variability = np.clip(cv, 0.0, 1.0)
    
    return float(variability)


def _compute_temporal_trend(
    scores: np.ndarray,
    smoothing_window: int
) -> str:
    """
    Compute temporal trend (improving, stable, or worsening).
    
    Method:
    - Apply smoothing (moving average)
    - Compare first half to second half
    - Determine trend direction
    
    Returns:
        'improving', 'stable', or 'worsening'
    """
    if len(scores) < 4:
        return 'unknown'
    
    # Apply smoothing
    if len(scores) >= smoothing_window:
        smoothed = np.convolve(
            scores,
            np.ones(smoothing_window) / smoothing_window,
            mode='valid'
        )
    else:
        smoothed = scores
    
    # Split into first and second half
    mid = len(smoothed) // 2
    first_half = smoothed[:mid]
    second_half = smoothed[mid:]
    
    if len(first_half) == 0 or len(second_half) == 0:
        return 'stable'
    
    # Compare means
    mean_first = np.mean(first_half)
    mean_second = np.mean(second_half)
    
    # Compute relative change
    if mean_first < 0.01:
        return 'stable'
    
    relative_change = (mean_second - mean_first) / mean_first
    
    # Classify trend
    if relative_change < -0.15:
        # Confidence decreased > 15% → worsening regulation
        return 'worsening'
    elif relative_change > 0.15:
        # Confidence decreased > 15% → improving regulation
        # (lower confidence = less dysregulation = improvement)
        return 'improving'
    else:
        return 'stable'


def _generate_rci_explanation(
    overall: float,
    autocorr: float,
    variability: float,
    trend: str
) -> str:
    """Generate human-readable explanation of RCI score."""
    
    # Overall interpretation
    if overall >= 80:
        level = "highly consistent regulation"
    elif overall >= 60:
        level = "moderately consistent regulation"
    elif overall >= 40:
        level = "variable regulation patterns"
    else:
        level = "inconsistent regulation patterns"
    
    explanation = f"Behavioral regulation shows {level} (score: {overall:.1f}/100). "
    
    # Trend interpretation
    if trend == 'improving':
        explanation += "Patterns show improvement over time (regulation strengthening). "
    elif trend == 'worsening':
        explanation += "Patterns show decline over time (regulation weakening). "
    else:
        explanation += "Patterns remain relatively stable over time. "
    
    # Factor breakdown
    factors = []
    
    if autocorr < 0.4:
        factors.append(f"low temporal consistency (autocorr: {autocorr:.2f})")
    
    if variability > 0.6:
        factors.append(f"high variability in states (CV: {variability:.2f})")
    
    if factors:
        explanation += "Contributing factors: " + ", ".join(factors) + "."
    else:
        explanation += "Both consistency and stability indicators are good."
    
    return explanation
