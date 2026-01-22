"""
Turn-taking analysis module for autism speech therapy.

Analyzes conversational dynamics between child and therapist:
- Speaker balance (who talks more)
- Response latency (time to respond)
- Turn transitions (smooth vs. interrupted)
- Conversational reciprocity

Clinical rationale (Autism):
- Children with autism often show delayed responses
- Turn-taking deficits are core social communication markers
- Imbalanced conversations indicate engagement issues
- Response latency correlates with processing/social challenges

Engineering approach:
- Uses enhanced speaker diarization (child vs. therapist labels)
- Temporal analysis of speaker transitions
- Statistical modeling of conversational patterns
- Clinical threshold-based scoring
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TurnTakingEvent:
    """
    Single turn-taking event in conversation.
    
    Attributes:
        speaker: 'child' or 'therapist'
        start_time: Turn start in seconds
        end_time: Turn end in seconds
        duration: Turn duration in seconds
        response_latency: Gap since previous turn (None if first turn)
        interruption: Whether this turn interrupted previous speaker
        previous_speaker: Previous speaker label
    """
    speaker: str
    start_time: float
    end_time: float
    duration: float
    response_latency: Optional[float] = None
    interruption: bool = False
    previous_speaker: Optional[str] = None


@dataclass
class TurnTakingAnalysis:
    """
    Complete turn-taking analysis for a session.
    
    Attributes:
        total_turns: Total number of turns
        child_turns: Number of child turns
        therapist_turns: Number of therapist turns
        child_speaking_time: Total child speaking time (seconds)
        therapist_speaking_time: Total therapist speaking time (seconds)
        child_percentage: Percentage of speaking time (child)
        mean_response_latency: Average response time (seconds)
        median_response_latency: Median response time
        max_response_latency: Longest response gap
        interruption_count: Number of interruptions
        conversational_balance_score: Balance metric (0-100, 50=ideal)
        reciprocity_score: Reciprocity quality (0-100)
        turn_events: List of individual turn events
        explanation: Clinical interpretation
    """
    total_turns: int
    child_turns: int
    therapist_turns: int
    child_speaking_time: float
    therapist_speaking_time: float
    child_percentage: float
    mean_response_latency: float
    median_response_latency: float
    max_response_latency: float
    interruption_count: int
    conversational_balance_score: float
    reciprocity_score: float
    turn_events: List[TurnTakingEvent] = field(default_factory=list)
    explanation: str = ""


def analyze_turn_taking(
    speaker_segments: List,
    child_label: str = "SPEAKER_00",
    config: Dict = None
) -> TurnTakingAnalysis:
    """
    Analyze turn-taking dynamics from speaker segments.
    
    Args:
        speaker_segments: List of SpeakerSegment objects from diarization
        child_label: Speaker label for child (auto-detected if not specified)
        config: Configuration dict with thresholds
        
    Returns:
        TurnTakingAnalysis object with metrics
    """
    if not speaker_segments:
        logger.warning("No speaker segments provided")
        return _create_empty_analysis()
    
    logger.info(f"Analyzing turn-taking for {len(speaker_segments)} segments")
    
    # Get config thresholds
    if config is None:
        config = {}
    autism_config = config.get('autism_analysis', {}).get('turn_taking', {})
    
    interruption_threshold = autism_config.get('interruption_threshold_sec', 0.5)
    latency_threshold_typical = autism_config.get('typical_latency_sec', 1.0)
    latency_threshold_elevated = autism_config.get('elevated_latency_sec', 3.0)
    
    # Auto-detect child speaker if not specified (typically speaks less)
    if child_label == "SPEAKER_00":
        child_label = _identify_child_speaker(speaker_segments)
        logger.info(f"Auto-detected child speaker: {child_label}")
    
    # Build turn events
    turn_events = _build_turn_events(
        speaker_segments,
        child_label,
        interruption_threshold
    )
    
    # Compute statistics
    child_turns = [t for t in turn_events if t.speaker == 'child']
    therapist_turns = [t for t in turn_events if t.speaker == 'therapist']
    
    child_speaking_time = sum(t.duration for t in child_turns)
    therapist_speaking_time = sum(t.duration for t in therapist_turns)
    total_speaking_time = child_speaking_time + therapist_speaking_time
    
    child_percentage = (child_speaking_time / total_speaking_time * 100) if total_speaking_time > 0 else 0
    
    # Response latencies (excluding None values)
    child_latencies = [t.response_latency for t in child_turns if t.response_latency is not None]
    therapist_latencies = [t.response_latency for t in therapist_turns if t.response_latency is not None]
    all_latencies = child_latencies + therapist_latencies
    
    mean_latency = np.mean(all_latencies) if all_latencies else 0.0
    median_latency = np.median(all_latencies) if all_latencies else 0.0
    max_latency = np.max(all_latencies) if all_latencies else 0.0
    
    # Interruptions
    interruption_count = sum(1 for t in turn_events if t.interruption)
    
    # Conversational balance score (50 = ideal, 0 or 100 = highly imbalanced)
    balance_score = 100 - abs(50 - child_percentage)
    
    # Reciprocity score (based on turn-taking smoothness)
    reciprocity_score = _compute_reciprocity_score(
        turn_events,
        mean_latency,
        interruption_count,
        latency_threshold_typical
    )
    
    # Generate explanation
    explanation = _generate_turn_taking_explanation(
        child_percentage,
        len(child_turns),
        len(therapist_turns),
        mean_latency,
        interruption_count,
        latency_threshold_typical,
        latency_threshold_elevated
    )
    
    return TurnTakingAnalysis(
        total_turns=len(turn_events),
        child_turns=len(child_turns),
        therapist_turns=len(therapist_turns),
        child_speaking_time=child_speaking_time,
        therapist_speaking_time=therapist_speaking_time,
        child_percentage=child_percentage,
        mean_response_latency=float(mean_latency),
        median_response_latency=float(median_latency),
        max_response_latency=float(max_latency),
        interruption_count=interruption_count,
        conversational_balance_score=float(balance_score),
        reciprocity_score=float(reciprocity_score),
        turn_events=turn_events,
        explanation=explanation
    )


def compute_response_latency_child(
    turn_analysis: TurnTakingAnalysis
) -> Dict:
    """
    Compute child-specific response latency metrics.
    
    Focus on child's response times (critical for autism assessment).
    
    Args:
        turn_analysis: TurnTakingAnalysis object
        
    Returns:
        Dictionary with child latency statistics
    """
    child_turns = [t for t in turn_analysis.turn_events if t.speaker == 'child']
    child_latencies = [t.response_latency for t in child_turns if t.response_latency is not None]
    
    if not child_latencies:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'count': 0,
            'percentage_elevated': 0.0
        }
    
    return {
        'mean': float(np.mean(child_latencies)),
        'median': float(np.median(child_latencies)),
        'std': float(np.std(child_latencies)),
        'min': float(np.min(child_latencies)),
        'max': float(np.max(child_latencies)),
        'count': len(child_latencies),
        'percentage_elevated': sum(1 for lat in child_latencies if lat > 3.0) / len(child_latencies) * 100
    }


def _identify_child_speaker(speaker_segments: List) -> str:
    """
    Identify which speaker is the child (typically speaks less).
    
    Heuristic: Child usually has less total speaking time than therapist.
    """
    speaker_times = {}
    
    for segment in speaker_segments:
        label = segment.speaker_id
        duration = segment.end_time - segment.start_time
        
        if label not in speaker_times:
            speaker_times[label] = 0.0
        speaker_times[label] += duration
    
    # Speaker with less time is likely the child
    if speaker_times:
        child_speaker = min(speaker_times, key=speaker_times.get)
        return child_speaker
    
    return "SPEAKER_00"


def _build_turn_events(
    speaker_segments: List,
    child_label: str,
    interruption_threshold: float
) -> List[TurnTakingEvent]:
    """
    Build turn-taking events from speaker segments.
    
    Handles overlapping segments and interruptions.
    """
    # Sort by start time
    sorted_segments = sorted(speaker_segments, key=lambda s: s.start_time)
    
    turn_events = []
    previous_speaker = None
    previous_end_time = 0.0
    
    for segment in sorted_segments:
        # Map to child/therapist
        speaker = 'child' if segment.speaker_id == child_label else 'therapist'
        
        start_time = segment.start_time
        end_time = segment.end_time
        duration = end_time - start_time
        
        # Calculate response latency (gap since previous turn)
        if previous_end_time > 0:
            gap = start_time - previous_end_time
            
            # Negative gap = overlap/interruption
            if gap < -interruption_threshold:
                interruption = True
                response_latency = 0.0  # Immediate/overlapping
            else:
                interruption = False
                response_latency = max(0.0, gap)
        else:
            # First turn
            response_latency = None
            interruption = False
        
        turn_event = TurnTakingEvent(
            speaker=speaker,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            response_latency=response_latency,
            interruption=interruption,
            previous_speaker=previous_speaker
        )
        
        turn_events.append(turn_event)
        
        previous_speaker = speaker
        previous_end_time = end_time
    
    return turn_events


def _compute_reciprocity_score(
    turn_events: List[TurnTakingEvent],
    mean_latency: float,
    interruption_count: int,
    typical_latency: float
) -> float:
    """
    Compute conversational reciprocity score.
    
    High reciprocity = smooth turn-taking, appropriate latencies
    Low reciprocity = long gaps, many interruptions, one-sided
    """
    if not turn_events:
        return 50.0
    
    # Component 1: Latency appropriateness (0-40 points)
    # Typical latency ~1s is ideal
    latency_deviation = abs(mean_latency - typical_latency)
    latency_score = 40 * (1.0 - np.clip(latency_deviation / 5.0, 0.0, 1.0))
    
    # Component 2: Low interruptions (0-30 points)
    interruption_rate = interruption_count / len(turn_events)
    interruption_score = 30 * (1.0 - np.clip(interruption_rate * 10, 0.0, 1.0))
    
    # Component 3: Turn alternation (0-30 points)
    alternations = 0
    for i in range(1, len(turn_events)):
        if turn_events[i].speaker != turn_events[i-1].speaker:
            alternations += 1
    
    alternation_rate = alternations / (len(turn_events) - 1) if len(turn_events) > 1 else 0
    alternation_score = 30 * alternation_rate
    
    total_score = latency_score + interruption_score + alternation_score
    
    return float(np.clip(total_score, 0.0, 100.0))


def _generate_turn_taking_explanation(
    child_percentage: float,
    child_turns: int,
    therapist_turns: int,
    mean_latency: float,
    interruption_count: int,
    typical_latency: float,
    elevated_latency: float
) -> str:
    """Generate clinical interpretation of turn-taking patterns."""
    
    explanation = []
    
    # Balance interpretation
    if child_percentage < 30:
        explanation.append(f"Child participation is low ({child_percentage:.1f}% of speaking time), indicating limited verbal engagement.")
    elif child_percentage > 60:
        explanation.append(f"Child dominates conversation ({child_percentage:.1f}% of speaking time), may indicate perseverative speech.")
    else:
        explanation.append(f"Conversational balance is appropriate ({child_percentage:.1f}% child, {100-child_percentage:.1f}% therapist).")
    
    # Turn count
    explanation.append(f"Total turns: {child_turns} child, {therapist_turns} therapist.")
    
    # Response latency interpretation
    if mean_latency < typical_latency:
        explanation.append(f"Response latency is quick (mean: {mean_latency:.2f}s), showing good responsiveness.")
    elif mean_latency > elevated_latency:
        explanation.append(f"Response latency is elevated (mean: {mean_latency:.2f}s), suggesting processing delays or reduced engagement.")
    else:
        explanation.append(f"Response latency is within typical range (mean: {mean_latency:.2f}s).")
    
    # Interruptions
    if interruption_count > 3:
        explanation.append(f"Frequent interruptions detected ({interruption_count}), indicating turn-taking difficulties.")
    elif interruption_count == 0:
        explanation.append("No interruptions observed, showing good turn-taking awareness.")
    
    return " ".join(explanation)


def _create_empty_analysis() -> TurnTakingAnalysis:
    """Create empty analysis for error cases."""
    return TurnTakingAnalysis(
        total_turns=0,
        child_turns=0,
        therapist_turns=0,
        child_speaking_time=0.0,
        therapist_speaking_time=0.0,
        child_percentage=0.0,
        mean_response_latency=0.0,
        median_response_latency=0.0,
        max_response_latency=0.0,
        interruption_count=0,
        conversational_balance_score=50.0,
        reciprocity_score=50.0,
        turn_events=[],
        explanation="Insufficient data for turn-taking analysis"
    )


def get_response_latency_distribution(
    turn_analysis: TurnTakingAnalysis,
    bins: List[float] = None
) -> Dict:
    """
    Get distribution of response latencies for histogram plotting.
    
    Args:
        turn_analysis: TurnTakingAnalysis object
        bins: Custom bin edges (default: [0, 1, 2, 3, 5, 10, inf])
        
    Returns:
        Dictionary with bin counts and labels
    """
    if bins is None:
        bins = [0, 1, 2, 3, 5, 10, float('inf')]
    
    # Get child latencies
    child_turns = [t for t in turn_analysis.turn_events if t.speaker == 'child']
    child_latencies = [t.response_latency for t in child_turns if t.response_latency is not None]
    
    if not child_latencies:
        return {'bins': [], 'counts': [], 'labels': []}
    
    # Count in each bin
    counts = []
    labels = []
    
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]
        
        if upper == float('inf'):
            count = sum(1 for lat in child_latencies if lat >= lower)
            label = f"{lower}+ sec"
        else:
            count = sum(1 for lat in child_latencies if lower <= lat < upper)
            label = f"{lower}-{upper} sec"
        
        counts.append(count)
        labels.append(label)
    
    return {
        'bins': bins[:-1],
        'counts': counts,
        'labels': labels
    }
