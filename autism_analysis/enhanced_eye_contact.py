"""
Enhanced Eye Contact Analysis with Gemini AI + MediaPipe.

This module provides a hybrid approach combining:
1. MediaPipe: Technical face detection and pose estimation
2. Gemini Vision: Contextual understanding and clinical assessment

Benefits:
- More accurate eye contact detection
- Better handling of edge cases (peripheral gaze, brief contacts)
- Clinical-grade behavioral assessment
- Context-aware analysis (social engagement vs. avoidance)
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

from video_pipeline.gemini_eye_contact import GeminiEyeContactDetector, integrate_with_existing_pipeline
from autism_analysis.eye_contact import EyeContactAnalysis, EyeContactEvent

logger = logging.getLogger(__name__)


@dataclass
class EnhancedEyeContactAnalysis:
    """Enhanced eye contact analysis combining MediaPipe + Gemini."""
    
    # Original MediaPipe analysis
    mediapipe_analysis: EyeContactAnalysis
    
    # Gemini AI analysis
    gemini_percentage: float
    gemini_confidence: float
    gemini_gaze_pattern: str
    gemini_social_engagement: str
    gemini_observations: List[str]
    
    # Combined/hybrid results
    final_eye_contact_score: float
    final_percentage: float
    percentage_of_session: float  # Add this attribute for compatibility
    confidence_level: str  # "high", "medium", "low"
    clinical_interpretation: str
    analysis_method: str  # "hybrid", "mediapipe_only", "gemini_only"
    
    # Additional compatibility attributes (with defaults)
    episode_count: int = 0
    eye_contact_score: float = 0.0
    explanation: str = ""
    total_duration: float = 0.0
    mean_episode_duration: float = 0.0
    during_speaking_percentage: float = 0.0
    during_listening_percentage: float = 0.0
    longest_episode: float = 0.0
    avoidance_score: float = 0.0
    frequency_per_minute: float = 0.0


def analyze_eye_contact_enhanced(
    video_aggregated: List,
    video_path: str,
    turn_analysis=None,
    config: Dict = None,
    clinical_transcript = None
) -> EnhancedEyeContactAnalysis:
    """
    Enhanced eye contact analysis using both MediaPipe and Gemini AI.
    Falls back to local pretrained models if Gemini is unavailable.
    
    Args:
        video_aggregated: Video analysis results
        video_path: Path to video file
        turn_analysis: Turn-taking analysis for context
        config: Configuration dictionary
        clinical_transcript: Clinical transcript with key frames for optimal analysis
        
    Returns:
        EnhancedEyeContactAnalysis with hybrid results
    """
    logger.info("Starting enhanced eye contact analysis")
    
    # Step 1: Run original MediaPipe analysis
    from autism_analysis.eye_contact import analyze_eye_contact
    mediapipe_analysis = analyze_eye_contact(video_aggregated, turn_analysis, config)
    
    logger.info(f"MediaPipe analysis: {mediapipe_analysis.eye_contact_score:.1f}/100")
    
    # Step 2: Try Gemini analysis if enabled
    gemini_result = None
    gemini_config = config.get('autism_analysis', {}).get('gemini_eye_contact', {})
    
    if gemini_config.get('enabled', False):
        logger.info("Attempting Gemini AI eye contact analysis...")
        try:
            gemini_result = integrate_with_existing_pipeline(
                video_aggregated, video_path, config, clinical_transcript
            )
            if gemini_result:
                logger.info(f"Gemini analysis: {gemini_result['eye_contact_percentage']:.1f}% eye contact")
        except Exception as e:
            logger.warning(f"Gemini analysis failed: {e}")
            logger.info("Falling back to local pretrained model analysis...")
            gemini_result = None
    
    # Step 3: Try local pretrained model if Gemini failed or disabled
    local_result = None
    if not gemini_result:
        logger.info("Running local pretrained model analysis...")
        try:
            from video_pipeline.local_eye_contact_detector import integrate_with_transcript
            local_result = integrate_with_transcript(
                video_aggregated, video_path, clinical_transcript, config
            )
            if local_result:
                logger.info(f"Local analysis: {local_result['eye_contact_percentage']:.1f}% eye contact")
        except Exception as e:
            logger.warning(f"Local analysis failed: {e}")
            local_result = None
    
    # Step 4: Combine results based on what's available
    if gemini_result:
        return _create_hybrid_analysis(mediapipe_analysis, gemini_result, config)
    elif local_result:
        return _create_local_hybrid_analysis(mediapipe_analysis, local_result, config)
    else:
        return _create_mediapipe_only_analysis(mediapipe_analysis)


def _create_local_hybrid_analysis(
    mp_analysis: EyeContactAnalysis,
    local_result: Dict,
    config: Dict
) -> EnhancedEyeContactAnalysis:
    """Create hybrid analysis combining MediaPipe + Local pretrained model."""
    
    # Combine scores using weighted average
    # MediaPipe is good for technical detection
    # Local model is good for contextual understanding without API dependency
    mp_weight = 0.5
    local_weight = 0.5
    
    # Convert local percentage to 0-100 score
    local_score = min(local_result['eye_contact_percentage'] * 1.5, 100)  # Scale appropriately
    
    final_score = (mp_analysis.eye_contact_score * mp_weight + 
                   local_score * local_weight)
    
    # Final percentage (favor local model for percentage)
    final_percentage = (mp_analysis.percentage_of_session * 0.4 + 
                       local_result['eye_contact_percentage'] * 0.6)
    
    # Confidence level based on agreement
    score_diff = abs(mp_analysis.eye_contact_score - local_score)
    if score_diff < 20:
        confidence = "high"
    elif score_diff < 35:
        confidence = "medium"
    else:
        confidence = "low"
    
    # Clinical interpretation
    interpretation = _generate_local_hybrid_interpretation(
        mp_analysis, local_result, final_score, final_percentage
    )
    
    return EnhancedEyeContactAnalysis(
        mediapipe_analysis=mp_analysis,
        gemini_percentage=local_result['eye_contact_percentage'],  # Use local result
        gemini_confidence=local_result['average_confidence'],
        gemini_gaze_pattern=local_result['dominant_gaze_pattern'],
        gemini_social_engagement=local_result['social_engagement_level'],
        gemini_observations=local_result['clinical_observations'],
        final_eye_contact_score=final_score,
        final_percentage=final_percentage,
        percentage_of_session=final_percentage,  # Add this line
        confidence_level=confidence,
        clinical_interpretation=interpretation,
        episode_count=mp_analysis.episode_count,  # Add compatibility attributes
        eye_contact_score=final_score,
        explanation=interpretation,
        total_duration=mp_analysis.total_duration,
        mean_episode_duration=mp_analysis.mean_episode_duration,
        during_speaking_percentage=mp_analysis.during_speaking_percentage,
        during_listening_percentage=mp_analysis.during_listening_percentage,
        longest_episode=mp_analysis.longest_episode,
        avoidance_score=mp_analysis.avoidance_score,
        frequency_per_minute=mp_analysis.frequency_per_minute,
        analysis_method="local_hybrid"
    )


def _create_hybrid_analysis(
    mp_analysis: EyeContactAnalysis,
    gemini_result: Dict,
    config: Dict
) -> EnhancedEyeContactAnalysis:
    """Create hybrid analysis combining both methods."""
    
    # Extract Gemini data
    gemini_analysis = gemini_result['gemini_analysis']
    
    # Combine scores using weighted average
    # MediaPipe is good for technical detection
    # Gemini is good for contextual understanding
    mp_weight = 0.4
    gemini_weight = 0.6
    
    # Convert Gemini percentage to 0-100 score
    gemini_score = min(gemini_analysis.eye_contact_percentage * 2, 100)  # Scale up
    
    final_score = (mp_analysis.eye_contact_score * mp_weight + 
                   gemini_score * gemini_weight)
    
    # Final percentage (favor Gemini for percentage)
    final_percentage = (mp_analysis.percentage_of_session * 0.3 + 
                       gemini_analysis.eye_contact_percentage * 0.7)
    
    # Confidence level based on agreement
    score_diff = abs(mp_analysis.eye_contact_score - gemini_score)
    if score_diff < 15:
        confidence = "high"
    elif score_diff < 30:
        confidence = "medium"
    else:
        confidence = "low"
    
    # Clinical interpretation
    interpretation = _generate_hybrid_interpretation(
        mp_analysis, gemini_analysis, final_score, final_percentage
    )
    
    return EnhancedEyeContactAnalysis(
        mediapipe_analysis=mp_analysis,
        gemini_percentage=gemini_analysis.eye_contact_percentage,
        gemini_confidence=gemini_analysis.average_confidence,
        gemini_gaze_pattern=gemini_analysis.dominant_gaze_pattern,
        gemini_social_engagement=gemini_analysis.social_engagement_level,
        gemini_observations=gemini_analysis.clinical_observations,
        final_eye_contact_score=final_score,
        final_percentage=final_percentage,
        percentage_of_session=final_percentage,  # Add this line
        confidence_level=confidence,
        clinical_interpretation=interpretation,
        episode_count=mp_analysis.episode_count,  # Add compatibility attributes
        eye_contact_score=final_score,
        explanation=interpretation,
        total_duration=mp_analysis.total_duration,
        mean_episode_duration=mp_analysis.mean_episode_duration,
        during_speaking_percentage=mp_analysis.during_speaking_percentage,
        during_listening_percentage=mp_analysis.during_listening_percentage,
        longest_episode=mp_analysis.longest_episode,
        avoidance_score=mp_analysis.avoidance_score,
        frequency_per_minute=mp_analysis.frequency_per_minute,
        analysis_method="hybrid"
    )


def _create_mediapipe_only_analysis(
    mp_analysis: EyeContactAnalysis
) -> EnhancedEyeContactAnalysis:
    """Create analysis using only MediaPipe (fallback)."""
    
    return EnhancedEyeContactAnalysis(
        mediapipe_analysis=mp_analysis,
        gemini_percentage=0.0,
        gemini_confidence=0.0,
        gemini_gaze_pattern="unavailable",
        gemini_social_engagement="unavailable",
        gemini_observations=["Gemini analysis unavailable"],
        final_eye_contact_score=mp_analysis.eye_contact_score,
        final_percentage=mp_analysis.percentage_of_session,
        percentage_of_session=mp_analysis.percentage_of_session,  # Add this line
        confidence_level="medium",
        clinical_interpretation=mp_analysis.explanation + " (MediaPipe only)",
        episode_count=mp_analysis.episode_count,  # Add compatibility attributes
        eye_contact_score=mp_analysis.eye_contact_score,
        explanation=mp_analysis.explanation + " (MediaPipe only)",
        total_duration=mp_analysis.total_duration,
        mean_episode_duration=mp_analysis.mean_episode_duration,
        during_speaking_percentage=mp_analysis.during_speaking_percentage,
        during_listening_percentage=mp_analysis.during_listening_percentage,
        longest_episode=mp_analysis.longest_episode,
        avoidance_score=mp_analysis.avoidance_score,
        frequency_per_minute=mp_analysis.frequency_per_minute,
        analysis_method="mediapipe_only"
    )


def _generate_local_hybrid_interpretation(
    mp_analysis: EyeContactAnalysis,
    local_result: Dict,
    final_score: float,
    final_percentage: float
) -> str:
    """Generate clinical interpretation from local hybrid analysis."""
    
    interpretation = []
    
    # Overall assessment
    if final_score < 25:
        interpretation.append(f"Significant eye contact challenges identified (score: {final_score:.1f}/100).")
    elif final_score < 50:
        interpretation.append(f"Below-typical eye contact patterns observed (score: {final_score:.1f}/100).")
    elif final_score < 75:
        interpretation.append(f"Moderate eye contact engagement (score: {final_score:.1f}/100).")
    else:
        interpretation.append(f"Good eye contact maintenance (score: {final_score:.1f}/100).")
    
    # Percentage context
    interpretation.append(f"Eye contact present in {final_percentage:.1f}% of analyzed frames.")
    
    # Method agreement
    local_score = min(local_result['eye_contact_percentage'] * 1.5, 100)
    score_diff = abs(mp_analysis.eye_contact_score - local_score)
    if score_diff < 20:
        interpretation.append("High agreement between technical and local model analysis methods.")
    elif score_diff < 35:
        interpretation.append("Moderate agreement between analysis methods.")
    else:
        interpretation.append("Significant differences between technical and contextual assessments noted.")
    
    # Local model insights
    if local_result['dominant_gaze_pattern'] != "direct":
        interpretation.append(f"Predominant gaze pattern: {local_result['dominant_gaze_pattern']}.")
    
    if local_result['social_engagement_level'] == "avoidant":
        interpretation.append("Social avoidance behaviors identified through local model analysis.")
    elif local_result['social_engagement_level'] == "engaged":
        interpretation.append("Positive social engagement indicators observed.")
    
    # Clinical observations from local model
    if local_result['clinical_observations']:
        key_observations = local_result['clinical_observations'][:2]  # Top 2
        interpretation.extend(key_observations)
    
    # Add method note
    interpretation.append("Analysis performed using local pretrained models (no API dependency).")
    
    return " ".join(interpretation)


def _generate_hybrid_interpretation(
    mp_analysis: EyeContactAnalysis,
    gemini_analysis,
    final_score: float,
    final_percentage: float
) -> str:
    """Generate clinical interpretation from hybrid analysis."""
    
    interpretation = []
    
    # Overall assessment
    if final_score < 25:
        interpretation.append(f"Significant eye contact challenges identified (score: {final_score:.1f}/100).")
    elif final_score < 50:
        interpretation.append(f"Below-typical eye contact patterns observed (score: {final_score:.1f}/100).")
    elif final_score < 75:
        interpretation.append(f"Moderate eye contact engagement (score: {final_score:.1f}/100).")
    else:
        interpretation.append(f"Good eye contact maintenance (score: {final_score:.1f}/100).")
    
    # Percentage context
    interpretation.append(f"Eye contact present in {final_percentage:.1f}% of analyzed frames.")
    
    # Method agreement
    score_diff = abs(mp_analysis.eye_contact_score - (gemini_analysis.eye_contact_percentage * 2))
    if score_diff < 15:
        interpretation.append("High agreement between technical and contextual analysis methods.")
    elif score_diff < 30:
        interpretation.append("Moderate agreement between analysis methods.")
    else:
        interpretation.append("Significant differences between technical and contextual assessments noted.")
    
    # Gemini insights
    if gemini_analysis.dominant_gaze_pattern != "direct":
        interpretation.append(f"Predominant gaze pattern: {gemini_analysis.dominant_gaze_pattern}.")
    
    if gemini_analysis.social_engagement_level == "avoidant":
        interpretation.append("Social avoidance behaviors identified through contextual analysis.")
    elif gemini_analysis.social_engagement_level == "engaged":
        interpretation.append("Positive social engagement indicators observed.")
    
    # Clinical observations from Gemini
    if gemini_analysis.clinical_observations:
        key_observations = gemini_analysis.clinical_observations[:2]  # Top 2
        interpretation.extend(key_observations)
    
    return " ".join(interpretation)


def get_enhanced_eye_contact_summary(analysis: EnhancedEyeContactAnalysis) -> Dict:
    """Get summary for dashboard display."""
    
    return {
        'final_score': analysis.final_eye_contact_score,
        'final_percentage': analysis.final_percentage,
        'confidence_level': analysis.confidence_level,
        'analysis_method': analysis.analysis_method,
        'mediapipe_score': analysis.mediapipe_analysis.eye_contact_score,
        'gemini_percentage': analysis.gemini_percentage,
        'gemini_gaze_pattern': analysis.gemini_gaze_pattern,
        'gemini_social_engagement': analysis.gemini_social_engagement,
        'clinical_interpretation': analysis.clinical_interpretation,
        'key_observations': analysis.gemini_observations[:3] if analysis.gemini_observations else [],
        # New metrics
        'frequency_per_min': analysis.frequency_per_minute,
        'mean_duration': analysis.mean_episode_duration,
        'total_duration': analysis.total_duration,
        'longest_episode': analysis.longest_episode
    }


# Integration with existing autism analysis
def replace_eye_contact_analysis():
    """
    Instructions for integrating enhanced eye contact analysis.
    
    In autism_analysis/__init__.py, replace:
        from .eye_contact import analyze_eye_contact
    
    With:
        from .enhanced_eye_contact import analyze_eye_contact_enhanced as analyze_eye_contact
    
    This will seamlessly upgrade the eye contact analysis while maintaining
    the same interface for the rest of the system.
    """
    pass


if __name__ == "__main__":
    print("Enhanced Eye Contact Analysis module loaded")
    print("Features:")
    print("- MediaPipe technical detection")
    print("- Gemini AI contextual understanding")
    print("- Hybrid scoring and confidence assessment")
    print("- Clinical-grade behavioral interpretation")