"""
Social engagement index for autism assessment.

Integrates multiple social communication markers:
- Eye contact patterns
- Turn-taking reciprocity
- Attention to social partner
- Response to social bids

Clinical rationale (Autism):
- Social communication deficits are core ASD feature
- Multiple markers provide holistic view
- Integration reduces false positives
- Clinically relevant composite score

Engineering approach:
- Weighted combination of component indices
- Evidence-based weighting from clinical literature
- Normalized 0-100 scale
- Interpretable subscores
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SocialEngagementMetrics:
    """
    Social engagement composite metrics.
    
    Attributes:
        social_engagement_index: Overall index (0-100, higher=better engagement)
        eye_contact_component: Eye contact contribution (0-100)
        turn_taking_component: Turn-taking contribution (0-100)
        responsiveness_component: Response latency contribution (0-100)
        attention_component: Attention stability contribution (0-100)
        component_weights: Weights used for each component
        interpretation: Clinical interpretation
        confidence: Analysis confidence (0-1)
    """
    social_engagement_index: float
    eye_contact_component: float
    turn_taking_component: float
    responsiveness_component: float
    attention_component: float
    component_weights: Dict[str, float]
    interpretation: str
    confidence: float = 0.0


def compute_social_engagement_index(
    eye_contact_analysis=None,
    turn_taking_analysis=None,
    attention_stability_score: Optional[float] = None,
    config: Dict = None
) -> SocialEngagementMetrics:
    """
    Compute composite social engagement index from multiple sources.
    
    Args:
        eye_contact_analysis: EyeContactAnalysis object
        turn_taking_analysis: TurnTakingAnalysis object
        attention_stability_score: ASS score from scoring module
        config: Configuration dict with weights
        
    Returns:
        SocialEngagementMetrics object
    """
    logger.info("Computing social engagement index")
    
    # Get component weights from config
    if config is None:
        config = {}
    
    weights = config.get('autism_analysis', {}).get('social_engagement_weights', {
        'eye_contact': 0.35,
        'turn_taking': 0.30,
        'responsiveness': 0.20,
        'attention': 0.15
    })
    
    # Extract component scores
    eye_contact_score = 0.0
    if eye_contact_analysis:
        # Handle both original and enhanced eye contact analysis
        if hasattr(eye_contact_analysis, 'final_eye_contact_score'):
            # Enhanced eye contact analysis
            eye_contact_score = eye_contact_analysis.final_eye_contact_score
        else:
            # Original eye contact analysis
            eye_contact_score = eye_contact_analysis.eye_contact_score
    
    turn_taking_score = 0.0
    if turn_taking_analysis:
        turn_taking_score = turn_taking_analysis.reciprocity_score
    
    responsiveness_score = 0.0
    if turn_taking_analysis:
        # Convert response latency to score (lower latency = higher score)
        mean_latency = turn_taking_analysis.mean_response_latency
        responsiveness_score = _latency_to_score(mean_latency)
    
    attention_score = attention_stability_score if attention_stability_score is not None else 50.0
    
    # Compute weighted composite
    sei = (
        eye_contact_score * weights['eye_contact'] +
        turn_taking_score * weights['turn_taking'] +
        responsiveness_score * weights['responsiveness'] +
        attention_score * weights['attention']
    )
    
    # Interpretation
    interpretation = _generate_sei_interpretation(
        sei,
        eye_contact_score,
        turn_taking_score,
        responsiveness_score,
        attention_score
    )
    
    # Confidence (based on data availability)
    confidence = _compute_sei_confidence(
        eye_contact_analysis,
        turn_taking_analysis,
        attention_stability_score
    )
    
    return SocialEngagementMetrics(
        social_engagement_index=float(sei),
        eye_contact_component=float(eye_contact_score),
        turn_taking_component=float(turn_taking_score),
        responsiveness_component=float(responsiveness_score),
        attention_component=float(attention_score),
        component_weights=weights,
        interpretation=interpretation,
        confidence=float(confidence)
    )


def _latency_to_score(latency_sec: float) -> float:
    """
    Convert response latency to score.
    
    Typical latency ~1s = 100 points
    Elevated latency >3s = lower score
    """
    if latency_sec < 0.5:
        # Very fast (may indicate interruption/impulsivity)
        return 70.0
    elif latency_sec < 1.5:
        # Typical range
        return 100.0 - (abs(latency_sec - 1.0) * 20)
    elif latency_sec < 3.0:
        # Somewhat elevated
        return 80.0 - ((latency_sec - 1.5) * 20)
    else:
        # Highly elevated
        return max(20.0, 50.0 - (latency_sec - 3.0) * 10)


def _generate_sei_interpretation(
    sei: float,
    eye_contact: float,
    turn_taking: float,
    responsiveness: float,
    attention: float
) -> str:
    """Generate clinical interpretation of SEI."""
    
    interpretation = []
    
    # Overall level
    if sei < 40:
        interpretation.append("Significant social engagement challenges present.")
    elif sei < 60:
        interpretation.append("Below-typical social engagement, consistent with social communication difficulties.")
    elif sei < 75:
        interpretation.append("Mild social engagement challenges observed.")
    else:
        interpretation.append("Social engagement within typical range.")
    
    # Identify weakest component
    components = {
        'eye contact': eye_contact,
        'turn-taking': turn_taking,
        'responsiveness': responsiveness,
        'attention': attention
    }
    
    weakest = min(components, key=components.get)
    weakest_score = components[weakest]
    
    if weakest_score < 50:
        interpretation.append(f"Primary area of concern: {weakest} (score: {weakest_score:.1f}).")
    
    # Identify strengths
    strongest = max(components, key=components.get)
    strongest_score = components[strongest]
    
    if strongest_score > 70:
        interpretation.append(f"Relative strength: {strongest} (score: {strongest_score:.1f}).")
    
    return " ".join(interpretation)


def _compute_sei_confidence(
    eye_contact_analysis,
    turn_taking_analysis,
    attention_score: Optional[float]
) -> float:
    """
    Compute confidence in SEI based on data availability.
    
    Returns value 0-1 (1 = all components available with good data)
    """
    confidence = 0.0
    
    # Eye contact component - handle both old and enhanced analysis
    if eye_contact_analysis:
        # Check if it's enhanced analysis (has analysis_method attribute)
        if hasattr(eye_contact_analysis, 'analysis_method'):
            # Enhanced analysis - use final score and confidence
            if eye_contact_analysis.final_eye_contact_score > 50:
                confidence += 0.35
            elif eye_contact_analysis.final_eye_contact_score > 25:
                confidence += 0.25
            else:
                confidence += 0.15
        else:
            # Original analysis - use episode count
            if eye_contact_analysis.episode_count > 5:
                confidence += 0.35
            else:
                confidence += 0.20
    
    # Turn-taking component
    if turn_taking_analysis and turn_taking_analysis.total_turns > 10:
        confidence += 0.30
    elif turn_taking_analysis:
        confidence += 0.15
    
    # Responsiveness (included in turn-taking)
    if turn_taking_analysis and turn_taking_analysis.child_turns > 5:
        confidence += 0.20
    
    # Attention component
    if attention_score is not None:
        confidence += 0.15
    
    return confidence
