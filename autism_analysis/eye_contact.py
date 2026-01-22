"""
Eye contact analysis for autism assessment.

Measures direct eye contact patterns:
- Eye contact duration and frequency
- Eye contact during speaking vs. listening
- Eye contact avoidance patterns
- Social referencing behaviors

Clinical rationale (Autism):
- Reduced eye contact is a hallmark ASD feature
- Eye contact during conversation indicates social engagement
- Developmentally appropriate eye contact emerges early
- ADOS-2 includes eye contact rating

Engineering approach:
- Combines face direction (head pose) + gaze estimation
- Direct eye contact = facing camera + eyes directed forward
- Temporal analysis of contact episodes
- Context-aware (during speaking vs. listening)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EyeContactEvent:
    """
    Single eye contact episode.
    
    Attributes:
        start_time: Episode start (seconds)
        end_time: Episode end (seconds)
        duration: Episode duration (seconds)
        during_speaking: Whether child was speaking
        during_listening: Whether therapist was speaking
        confidence: Detection confidence (0-1)
    """
    start_time: float
    end_time: float
    duration: float
    during_speaking: bool = False
    during_listening: bool = False
    confidence: float = 0.0


@dataclass
class EyeContactAnalysis:
    """
    Complete eye contact analysis.
    
    Attributes:
        total_duration: Total eye contact time (seconds)
        episode_count: Number of eye contact episodes
        mean_episode_duration: Average episode length
        frequency_per_minute: Episodes per minute
        percentage_of_session: % of session with eye contact
        during_speaking_percentage: % eye contact while child speaking
        during_listening_percentage: % eye contact while therapist speaking
        longest_episode: Longest continuous eye contact
        eye_contact_score: Overall score (0-100, higher=more eye contact)
        avoidance_score: Eye avoidance indicator (0-100, higher=more avoidance)
        events: List of eye contact episodes
        explanation: Clinical interpretation
    """
    total_duration: float
    episode_count: int
    mean_episode_duration: float
    frequency_per_minute: float
    percentage_of_session: float
    during_speaking_percentage: float
    during_listening_percentage: float
    longest_episode: float
    eye_contact_score: float
    avoidance_score: float
    events: List[EyeContactEvent] = field(default_factory=list)
    explanation: str = ""


def analyze_eye_contact(
    video_aggregated: List,
    turn_analysis=None,
    config: Dict = None
) -> EyeContactAnalysis:
    """
    Analyze eye contact patterns from video features.
    
    Args:
        video_aggregated: List of AggregatedFeatures from video pipeline
        turn_analysis: Optional TurnTakingAnalysis for context
        config: Configuration dict with thresholds
        
    Returns:
        EyeContactAnalysis object
    """
    if not video_aggregated:
        logger.warning("No video features provided")
        return _create_empty_eye_contact_analysis()
    
    logger.info(f"Analyzing eye contact from {len(video_aggregated)} windows")
    
    # Get config thresholds
    if config is None:
        config = {}
    autism_config = config.get('autism_analysis', {}).get('eye_contact', {})
    
    head_facing_threshold = autism_config.get('head_facing_threshold_deg', 30.0)
    gaze_forward_threshold = autism_config.get('gaze_forward_threshold', 0.05)
    min_episode_duration = autism_config.get('min_episode_duration_sec', 0.5)
    
    # Extract eye contact episodes
    events = _extract_eye_contact_events(
        video_aggregated,
        head_facing_threshold,
        gaze_forward_threshold,
        min_episode_duration,
        turn_analysis
    )
    
    # Compute statistics
    total_duration = sum(e.duration for e in events)
    episode_count = len(events)
    
    mean_duration = total_duration / episode_count if episode_count > 0 else 0.0
    
    # Session duration (from first to last window)
    session_duration = 0.0
    if video_aggregated:
        session_duration = video_aggregated[-1].window_end_time - video_aggregated[0].window_start_time
    
    frequency = (episode_count / session_duration * 60) if session_duration > 0 else 0.0
    percentage = (total_duration / session_duration * 100) if session_duration > 0 else 0.0
    
    # Context-specific percentages
    speaking_events = [e for e in events if e.during_speaking]
    listening_events = [e for e in events if e.during_listening]
    
    speaking_duration = sum(e.duration for e in speaking_events)
    listening_duration = sum(e.duration for e in listening_events)
    
    # Estimate speaking/listening time from turn analysis if available
    if turn_analysis:
        total_speaking = turn_analysis.child_speaking_time
        total_listening = turn_analysis.therapist_speaking_time
    else:
        # Rough estimate: 50/50
        total_speaking = session_duration / 2
        total_listening = session_duration / 2
    
    speaking_pct = (speaking_duration / total_speaking * 100) if total_speaking > 0 else 0.0
    listening_pct = (listening_duration / total_listening * 100) if total_listening > 0 else 0.0
    
    longest = max([e.duration for e in events]) if events else 0.0
    
    # Scores
    eye_contact_score = _compute_eye_contact_score(percentage, mean_duration, frequency)
    avoidance_score = 100 - eye_contact_score  # Inverse
    
    # Explanation
    explanation = _generate_eye_contact_explanation(
        percentage,
        episode_count,
        mean_duration,
        speaking_pct,
        listening_pct
    )
    
    return EyeContactAnalysis(
        total_duration=float(total_duration),
        episode_count=episode_count,
        mean_episode_duration=float(mean_duration),
        frequency_per_minute=float(frequency),
        percentage_of_session=float(percentage),
        during_speaking_percentage=float(speaking_pct),
        during_listening_percentage=float(listening_pct),
        longest_episode=float(longest),
        eye_contact_score=float(eye_contact_score),
        avoidance_score=float(avoidance_score),
        events=events,
        explanation=explanation
    )


def compute_eye_contact_during_speaking(
    eye_contact_analysis: EyeContactAnalysis
) -> Dict:
    """
    Detailed metrics for eye contact during child's speaking.
    
    Critical for autism: eye contact while speaking is often more challenging.
    """
    speaking_events = [e for e in eye_contact_analysis.events if e.during_speaking]
    
    if not speaking_events:
        return {
            'episode_count': 0,
            'total_duration': 0.0,
            'mean_duration': 0.0,
            'percentage': 0.0
        }
    
    total_dur = sum(e.duration for e in speaking_events)
    mean_dur = total_dur / len(speaking_events)
    
    return {
        'episode_count': len(speaking_events),
        'total_duration': float(total_dur),
        'mean_duration': float(mean_dur),
        'percentage': eye_contact_analysis.during_speaking_percentage
    }


def _extract_eye_contact_events(
    video_aggregated: List,
    head_threshold: float,
    gaze_threshold: float,
    min_duration: float,
    turn_analysis
) -> List[EyeContactEvent]:
    """
    Extract eye contact episodes from video features.
    
    Eye contact criteria:
    1. Head facing forward (yaw < threshold)
    2. Gaze directed forward (gaze_proxy < threshold)
    3. Sustained for minimum duration
    """
    events = []
    
    in_contact = False
    contact_start = 0.0
    
    for window in video_aggregated:
        face_feat = window.face_features
        
        # Check head pose (yaw close to 0 = facing forward)
        head_yaw = abs(face_feat.get('head_yaw_mean', 90.0))
        
        # Check gaze proxy (low value = looking forward)
        gaze = face_feat.get('gaze_proxy_mean', 1.0)
        
        # Eye contact if both criteria met
        is_eye_contact = (head_yaw < head_threshold) and (gaze < gaze_threshold)
        
        window_start = window.window_start_time
        window_end = window.window_end_time
        
        if is_eye_contact and not in_contact:
            # Start new episode
            in_contact = True
            contact_start = window_start
        
        elif not is_eye_contact and in_contact:
            # End episode
            in_contact = False
            duration = window_start - contact_start
            
            if duration >= min_duration:
                # Determine context (speaking/listening)
                during_speaking, during_listening = _get_speaking_context(
                    contact_start,
                    window_start,
                    turn_analysis
                )
                
                event = EyeContactEvent(
                    start_time=contact_start,
                    end_time=window_start,
                    duration=duration,
                    during_speaking=during_speaking,
                    during_listening=during_listening,
                    confidence=0.8  # TODO: Compute from detection confidence
                )
                events.append(event)
    
    # Handle ongoing episode at end
    if in_contact:
        duration = video_aggregated[-1].window_end_time - contact_start
        if duration >= min_duration:
            during_speaking, during_listening = _get_speaking_context(
                contact_start,
                video_aggregated[-1].window_end_time,
                turn_analysis
            )
            
            event = EyeContactEvent(
                start_time=contact_start,
                end_time=video_aggregated[-1].window_end_time,
                duration=duration,
                during_speaking=during_speaking,
                during_listening=during_listening,
                confidence=0.8
            )
            events.append(event)
    
    return events


def _get_speaking_context(
    start_time: float,
    end_time: float,
    turn_analysis
) -> tuple:
    """
    Determine if eye contact occurred during child speaking or therapist speaking.
    """
    if turn_analysis is None:
        return False, False
    
    # Check overlap with turns
    child_speaking = False
    therapist_speaking = False
    
    for turn in turn_analysis.turn_events:
        # Check overlap
        if not (turn.end_time <= start_time or turn.start_time >= end_time):
            if turn.speaker == 'child':
                child_speaking = True
            else:
                therapist_speaking = True
    
    return child_speaking, therapist_speaking


def _compute_eye_contact_score(
    percentage: float,
    mean_duration: float,
    frequency: float
) -> float:
    """
    Compute overall eye contact quality score.
    
    Components:
    - Percentage of session (0-50 points)
    - Episode duration (0-30 points)
    - Frequency (0-20 points)
    """
    # Component 1: Session percentage (20-40% is typical)
    if percentage < 10:
        pct_score = percentage * 2  # 0-20 points
    elif percentage < 40:
        pct_score = 20 + (percentage - 10) * 1  # 20-50 points
    else:
        pct_score = 50  # Cap at 50
    
    # Component 2: Mean duration (1-3s is good)
    if mean_duration < 0.5:
        dur_score = mean_duration * 20  # 0-10 points
    elif mean_duration < 3:
        dur_score = 10 + (mean_duration - 0.5) * 8  # 10-30 points
    else:
        dur_score = 30  # Cap
    
    # Component 3: Frequency (3-6 per minute is good)
    if frequency < 3:
        freq_score = frequency * 3.3  # 0-10 points
    elif frequency < 6:
        freq_score = 10 + (frequency - 3) * 3.3  # 10-20 points
    else:
        freq_score = 20  # Cap
    
    total = pct_score + dur_score + freq_score
    return float(np.clip(total, 0.0, 100.0))


def _generate_eye_contact_explanation(
    percentage: float,
    count: int,
    mean_duration: float,
    speaking_pct: float,
    listening_pct: float
) -> str:
    """Generate clinical interpretation."""
    
    explanation = []
    
    # Overall level
    if percentage < 10:
        explanation.append(f"Minimal eye contact observed ({percentage:.1f}% of session), indicating significant avoidance.")
    elif percentage < 25:
        explanation.append(f"Below-typical eye contact ({percentage:.1f}% of session), consistent with ASD presentation.")
    elif percentage < 40:
        explanation.append(f"Moderate eye contact ({percentage:.1f}% of session), approaching typical range.")
    else:
        explanation.append(f"Good eye contact ({percentage:.1f}% of session), within typical range.")
    
    # Episode characteristics
    explanation.append(f"{count} episodes detected (mean duration: {mean_duration:.2f}s).")
    
    # Context comparison
    if speaking_pct < listening_pct / 2:
        explanation.append("Eye contact significantly reduced during child's speaking compared to listening.")
    elif speaking_pct > listening_pct * 1.5:
        explanation.append("Eye contact more frequent during child's speaking than listening.")
    
    return " ".join(explanation)


def _create_empty_eye_contact_analysis() -> EyeContactAnalysis:
    """Create empty analysis for error cases."""
    return EyeContactAnalysis(
        total_duration=0.0,
        episode_count=0,
        mean_episode_duration=0.0,
        frequency_per_minute=0.0,
        percentage_of_session=0.0,
        during_speaking_percentage=0.0,
        during_listening_percentage=0.0,
        longest_episode=0.0,
        eye_contact_score=0.0,
        avoidance_score=100.0,
        events=[],
        explanation="Insufficient data for eye contact analysis"
    )
