"""
Facial Affect Index - Clinical behavioral scoring based on Action Units.

This module computes clinical metrics from FACS Action Units WITHOUT
emotion labeling or diagnostic claims.

Metrics computed:
1. Affect Range Score: How diverse are facial expressions?
2. Facial Mobility Index: How much facial movement occurs?
3. Flat Affect Indicator: Is there reduced facial expressiveness?
4. Congruence Score: Do facial patterns match context?
5. Symmetry Index: Left-right facial balance

IMPORTANT CLINICAL CONSTRAINTS:
- NO emotion detection (AUs ≠ emotions)
- NO diagnostic claims (observation only)
- Objective measurements of observable muscle activations
- Results require professional clinical interpretation

Usage:
    from scoring.facial_affect_index import compute_facial_affect_index
    
    index = compute_facial_affect_index(
        au_sequence=facial_aus,
        duration=session_duration
    )
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from scipy import stats

from clinical_analysis.facial_action_units import FacialActionUnits

logger = logging.getLogger(__name__)


@dataclass
class FacialAffectIndex:
    """
    Clinical facial pattern metrics.
    
    Attributes:
        affect_range_score: Diversity of AU activations (0-100, higher = more diverse)
        facial_mobility_index: Overall facial movement (0-100, higher = more mobile)
        flat_affect_indicator: Reduced expressiveness (0-100, higher = flatter)
        congruence_score: Context-appropriate patterns (0-100, higher = more appropriate)
        symmetry_index: Left-right facial balance (0-100, higher = more symmetric)
        au_activation_frequency: AUs activated per minute
        dominant_aus: Most frequently activated Action Units
        facial_affect_index: Composite score (0-100)
    """
    affect_range_score: float
    facial_mobility_index: float
    flat_affect_indicator: float
    congruence_score: float
    symmetry_index: float
    au_activation_frequency: float
    dominant_aus: List[int]
    facial_affect_index: float


def compute_facial_affect_index(
    au_sequence: List[FacialActionUnits],
    duration: float,
    context: Optional[str] = None
) -> FacialAffectIndex:
    """
    Compute Facial Affect Index from Action Unit sequence.
    
    Args:
        au_sequence: List of FacialActionUnits from video analysis
        duration: Session duration in seconds
        context: Optional context ('therapy', 'conversation', etc.)
        
    Returns:
        FacialAffectIndex with all computed metrics
    """
    if not au_sequence or len(au_sequence) == 0:
        logger.warning("Empty AU sequence - returning null index")
        return _null_facial_affect_index()
    
    # Filter out frames where no face was detected
    valid_frames = [au for au in au_sequence if au.face_detected]
    
    if len(valid_frames) == 0:
        logger.warning("No valid face detections - returning null index")
        return _null_facial_affect_index()
    
    logger.info(f"Computing Facial Affect Index from {len(valid_frames)} frames over {duration:.1f}s")
    
    # Calculate each component
    affect_range = _calculate_affect_range(valid_frames)
    mobility_index = _calculate_facial_mobility(valid_frames)
    flat_affect = _calculate_flat_affect_indicator(valid_frames)
    congruence = _calculate_congruence_score(valid_frames, context)
    symmetry = _calculate_symmetry_index(valid_frames)
    activation_freq = _calculate_au_frequency(valid_frames, duration)
    dominant_aus = _identify_dominant_aus(valid_frames)
    
    # Compute composite Facial Affect Index
    # Weighted combination of components
    composite_index = (
        affect_range * 0.30 +      # Diversity of expressions
        mobility_index * 0.25 +    # Amount of movement
        (100 - flat_affect) * 0.25 +  # Inverse of flat affect
        congruence * 0.10 +        # Context appropriateness
        symmetry * 0.10            # Facial symmetry
    )
    
    return FacialAffectIndex(
        affect_range_score=affect_range,
        facial_mobility_index=mobility_index,
        flat_affect_indicator=flat_affect,
        congruence_score=congruence,
        symmetry_index=symmetry,
        au_activation_frequency=activation_freq,
        dominant_aus=dominant_aus,
        facial_affect_index=composite_index
    )


def _calculate_affect_range(frames: List[FacialActionUnits]) -> float:
    """
    Calculate Affect Range Score.
    
    Measures: How many different AUs are activated across the session?
    Higher score = more diverse facial expressions
    Lower score = limited facial repertoire
    
    Clinical significance:
    - Reduced range may indicate flat affect, depression, or social communication differences
    - Very high range is typically healthy
    
    Returns:
        Score 0-100
    """
    # Collect all activated AUs across session
    activated_aus = set()
    
    for frame in frames:
        for au_num, au in frame.action_units.items():
            if au.present:  # AU intensity > threshold
                activated_aus.add(au_num)
    
    # Calculate diversity
    # Maximum possible AUs = 15 (tracking 15 different AUs)
    max_aus = 15
    unique_aus = len(activated_aus)
    
    # Normalize to 0-100
    range_score = (unique_aus / max_aus) * 100
    
    logger.debug(f"Affect range: {unique_aus}/{max_aus} unique AUs activated = {range_score:.1f}/100")
    
    return float(range_score)


def _calculate_facial_mobility(frames: List[FacialActionUnits]) -> float:
    """
    Calculate Facial Mobility Index.
    
    Measures: How much facial movement occurs across the session?
    Higher score = more facial activity
    Lower score = reduced facial movement
    
    Clinical significance:
    - Low mobility may indicate flat affect, motor inhibition
    - Very high mobility may indicate agitation, tic disorders
    
    Returns:
        Score 0-100
    """
    # Calculate average AU intensity across all frames
    total_intensity = 0.0
    total_au_measurements = 0
    
    for frame in frames:
        for au_num, au in frame.action_units.items():
            total_intensity += au.intensity
            total_au_measurements += 1
    
    if total_au_measurements == 0:
        return 0.0
    
    # Average intensity per AU per frame
    avg_intensity = total_intensity / total_au_measurements
    
    # Also consider activation frequency (how often any AU is present)
    activation_counts = sum(
        sum(au.present for au in frame.action_units.values())
        for frame in frames
    )
    avg_activations_per_frame = activation_counts / len(frames)
    
    # Combine both measures
    # High intensity + high activation count = high mobility
    intensity_score = avg_intensity * 100  # Already 0-1 scale
    activation_score = min(avg_activations_per_frame / 3.0, 1.0) * 100  # Normalize by expected ~3 AUs per frame
    
    mobility_index = (intensity_score * 0.6 + activation_score * 0.4)
    
    logger.debug(
        f"Facial mobility: avg_intensity={avg_intensity:.3f}, "
        f"avg_activations={avg_activations_per_frame:.2f} → {mobility_index:.1f}/100"
    )
    
    return float(mobility_index)


def _calculate_flat_affect_indicator(frames: List[FacialActionUnits]) -> float:
    """
    Calculate Flat Affect Indicator.
    
    Measures: Is there reduced facial expressiveness?
    Higher score = flatter affect (less expression)
    Lower score = normal/varied expression
    
    Clinical significance:
    - High flat affect may be observed in depression, schizophrenia, autism
    - Normal variability shows healthy affective expression
    
    Returns:
        Score 0-100
    """
    # Multiple indicators of flat affect:
    # 1. Low AU activation diversity (few different AUs)
    # 2. Low AU intensities (weak activations)
    # 3. Long periods with no AU activations
    
    # 1. AU diversity (inverse of affect range)
    activated_aus = set()
    for frame in frames:
        for au_num, au in frame.action_units.items():
            if au.present:
                activated_aus.add(au_num)
    
    diversity_score = len(activated_aus) / 15.0  # 0-1
    
    # 2. Average intensity
    total_intensity = sum(
        au.intensity
        for frame in frames
        for au in frame.action_units.values()
    )
    total_measurements = sum(len(frame.action_units) for frame in frames)
    avg_intensity = total_intensity / total_measurements if total_measurements > 0 else 0.0
    
    # 3. Proportion of "inactive" frames (no AUs activated)
    inactive_frames = sum(
        1 for frame in frames
        if not any(au.present for au in frame.action_units.values())
    )
    inactive_ratio = inactive_frames / len(frames)
    
    # Combine indicators (higher = flatter affect)
    flat_affect_score = (
        (1.0 - diversity_score) * 40 +  # Low diversity
        (1.0 - avg_intensity) * 35 +    # Low intensity
        inactive_ratio * 25             # Many inactive frames
    )
    
    logger.debug(
        f"Flat affect indicators: diversity={diversity_score:.2f}, "
        f"intensity={avg_intensity:.3f}, inactive_ratio={inactive_ratio:.2f} "
        f"→ {flat_affect_score:.1f}/100"
    )
    
    return float(flat_affect_score)


def _calculate_congruence_score(
    frames: List[FacialActionUnits],
    context: Optional[str]
) -> float:
    """
    Calculate Congruence Score.
    
    Measures: Do facial patterns match the expected context?
    Higher score = patterns appropriate for context
    Lower score = incongruent patterns
    
    Clinical significance:
    - Incongruent affect may indicate emotional dysregulation
    - Context-dependent (therapy vs. casual conversation)
    
    Note: Without speech/context analysis, this is limited.
    For now, returns a moderate default score.
    
    Returns:
        Score 0-100
    """
    # TODO: This requires multimodal analysis (speech + facial patterns)
    # For now, return a neutral score
    
    # Basic heuristic: Check for unusual AU combinations
    # (e.g., smile + brow lower = potentially incongruent)
    
    incongruent_frames = 0
    
    for frame in frames:
        aus = frame.action_units
        
        # Check for contradictory patterns
        # Example: AU12 (smile) + AU4 (brow lower) = mixed signal
        if (aus.get(12, None) and aus[12].present and
            aus.get(4, None) and aus[4].present):
            incongruent_frames += 1
        
        # Example: AU15 (frown) + AU6 (cheek raise) = mixed
        if (aus.get(15, None) and aus[15].present and
            aus.get(6, None) and aus[6].present):
            incongruent_frames += 1
    
    congruence_ratio = 1.0 - (incongruent_frames / len(frames))
    congruence_score = congruence_ratio * 100
    
    logger.debug(
        f"Congruence: {incongruent_frames}/{len(frames)} incongruent frames "
        f"→ {congruence_score:.1f}/100"
    )
    
    return float(congruence_score)


def _calculate_symmetry_index(frames: List[FacialActionUnits]) -> float:
    """
    Calculate Symmetry Index.
    
    Measures: Left-right facial symmetry
    Higher score = more symmetric
    Lower score = asymmetric (may be clinically significant)
    
    Clinical significance:
    - Asymmetry may indicate neurological conditions
    - Some emotions naturally produce asymmetry (contempt)
    
    Returns:
        Score 0-100
    """
    symmetry_scores = [frame.symmetry_score for frame in frames if frame.face_detected]
    
    if not symmetry_scores:
        return 50.0  # Neutral if no data
    
    avg_symmetry = np.mean(symmetry_scores)
    symmetry_index = avg_symmetry * 100  # Already 0-1 scale
    
    logger.debug(f"Symmetry: avg={avg_symmetry:.3f} → {symmetry_index:.1f}/100")
    
    return float(symmetry_index)


def _calculate_au_frequency(frames: List[FacialActionUnits], duration: float) -> float:
    """
    Calculate AU activation frequency (activations per minute).
    
    Returns:
        Frequency (activations/minute)
    """
    total_activations = sum(
        sum(au.present for au in frame.action_units.values())
        for frame in frames
    )
    
    frequency = (total_activations / duration) * 60.0 if duration > 0 else 0.0
    
    logger.debug(f"AU frequency: {total_activations} activations / {duration:.1f}s = {frequency:.1f}/min")
    
    return float(frequency)


def _identify_dominant_aus(frames: List[FacialActionUnits], top_n: int = 5) -> List[int]:
    """
    Identify most frequently activated Action Units.
    
    Returns:
        List of AU numbers (most frequent first)
    """
    au_counts = {}
    
    for frame in frames:
        for au_num, au in frame.action_units.items():
            if au.present:
                au_counts[au_num] = au_counts.get(au_num, 0) + 1
    
    # Sort by frequency
    sorted_aus = sorted(au_counts.items(), key=lambda x: x[1], reverse=True)
    dominant_aus = [au_num for au_num, count in sorted_aus[:top_n]]
    
    logger.debug(f"Dominant AUs: {dominant_aus}")
    
    return dominant_aus


def _null_facial_affect_index() -> FacialAffectIndex:
    """Return null/default index when no valid data."""
    return FacialAffectIndex(
        affect_range_score=0.0,
        facial_mobility_index=0.0,
        flat_affect_indicator=100.0,  # Maximum flat affect (no data)
        congruence_score=50.0,
        symmetry_index=50.0,
        au_activation_frequency=0.0,
        dominant_aus=[],
        facial_affect_index=0.0
    )


# AU clinical descriptions (for reporting)
AU_DESCRIPTIONS = {
    1: "Inner Brow Raiser",
    2: "Outer Brow Raiser",
    4: "Brow Lowerer",
    5: "Upper Lid Raiser",
    6: "Cheek Raiser",
    7: "Lid Tightener",
    9: "Nose Wrinkler",
    10: "Upper Lip Raiser",
    12: "Lip Corner Puller",
    15: "Lip Corner Depressor",
    17: "Chin Raiser",
    20: "Lip Stretcher",
    23: "Lip Tightener",
    25: "Lips Part",
    26: "Jaw Drop",
}

# Common AU pattern descriptions (NOT emotion labels)
AU_PATTERN_DESCRIPTIONS = {
    (12, 6): "Bilateral smile pattern (AU12+AU6)",
    (1, 2): "Bilateral brow raise (AU1+AU2)",
    (4, 7): "Brow lower with lid tighten (AU4+AU7)",
    (15,): "Lip corner depression (AU15)",
    (12,): "Lip corner pull without cheek raise (AU12 only)",
    (25, 26): "Mouth opening pattern (AU25+AU26)",
}
