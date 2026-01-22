"""
Question-response ability analysis module.

Analyzes patient's ability to respond to clinician questions:
- Question detection (from clinician speech)
- Response detection (from patient speech)
- Response latency (time to start responding)
- Response completeness (duration, turn-taking)
- Response appropriateness (context-based heuristics)

Clinical rationale:
- Core assessment of comprehension and expressive ability
- Social communication competence
- Cognitive processing speed
- Language pragmatics evaluation

Engineering approach:
- Turn-taking based question-response pairing
- Prosodic cues for question detection (rising intonation)
- Temporal analysis of response patterns
- Statistical modeling of response quality
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QuestionEvent:
    """
    Detected question from clinician.
    
    Attributes:
        start_time: Question start (seconds)
        end_time: Question end (seconds)
        duration: Question duration (seconds)
        has_response: Whether patient responded
        response_latency: Time to response (None if no response)
        response_duration: Duration of response (None if no response)
        response_event: Paired ResponseEvent object
    """
    start_time: float
    end_time: float
    duration: float
    has_response: bool = False
    response_latency: Optional[float] = None
    response_duration: Optional[float] = None
    response_event: Optional['ResponseEvent'] = None


@dataclass
class ResponseEvent:
    """
    Patient response to question.
    
    Attributes:
        start_time: Response start (seconds)
        end_time: Response end (seconds)
        duration: Response duration (seconds)
        latency: Time since question end (seconds)
        completeness: Completeness rating (incomplete, brief, adequate, elaborate)
        appropriateness: Appropriateness score (0-100)
    """
    start_time: float
    end_time: float
    duration: float
    latency: float
    completeness: str
    appropriateness: float


@dataclass
class QuestionResponseAnalysis:
    """
    Complete question-response analysis.
    
    Attributes:
        total_questions: Number of questions detected
        answered_questions: Number of questions with responses
        response_rate: % of questions answered
        mean_response_latency: Average time to respond (seconds)
        median_response_latency: Median response time
        mean_response_duration: Average response length (seconds)
        appropriate_responses: Number of contextually appropriate responses
        appropriateness_rate: % of appropriate responses
        responsiveness_index: Overall responsiveness (0-100, higher=better)
        questions: List of QuestionEvent objects
        interpretation: Clinical interpretation
    """
    total_questions: int
    answered_questions: int
    response_rate: float
    mean_response_latency: float
    median_response_latency: float
    mean_response_duration: float
    appropriate_responses: int
    appropriateness_rate: float
    responsiveness_index: float
    questions: List[QuestionEvent] = field(default_factory=list)
    interpretation: str = ""


def analyze_question_response_ability(
    speaker_segments: List,
    prosodic_features: List,
    clinician_label: str = None,
    config: Dict = None
) -> QuestionResponseAnalysis:
    """
    Analyze patient's ability to respond to questions.
    
    Args:
        speaker_segments: List of speaker segments
        prosodic_features: List of prosodic feature dicts
        clinician_label: Speaker label for clinician/doctor
        config: Configuration dict with thresholds
        
    Returns:
        QuestionResponseAnalysis object
    """
    if not speaker_segments:
        logger.warning("No speaker segments provided")
        return _create_empty_qr_analysis()
    
    logger.info(f"Analyzing question-response ability from {len(speaker_segments)} segments")
    
    # Get config thresholds
    if config is None:
        config = {}
    qr_config = config.get('clinical_analysis', {}).get('question_response', {})
    
    max_response_latency = qr_config.get('max_response_latency_sec', 10.0)
    min_response_duration = qr_config.get('min_response_duration_sec', 0.5)
    pitch_rise_threshold = qr_config.get('pitch_rise_threshold_hz', 30.0)
    
    # Identify clinician (typically speaks more)
    if clinician_label is None:
        clinician_label = _identify_clinician(speaker_segments)
    
    # Identify patient
    all_speakers = set(s.speaker_id for s in speaker_segments)
    patient_label = next((s for s in all_speakers if s != clinician_label), "SPEAKER_00")
    
    logger.info(f"Identified clinician: {clinician_label}, patient: {patient_label}")
    
    # Detect questions from clinician
    question_events = _detect_questions(
        speaker_segments,
        prosodic_features,
        clinician_label,
        pitch_rise_threshold
    )
    
    logger.info(f"Detected {len(question_events)} potential questions")
    
    # Match responses from patient
    _match_responses(
        question_events,
        speaker_segments,
        patient_label,
        max_response_latency,
        min_response_duration
    )
    
    # Compute statistics
    total_questions = len(question_events)
    answered = [q for q in question_events if q.has_response]
    answered_count = len(answered)
    
    response_rate = (answered_count / total_questions * 100) if total_questions > 0 else 0.0
    
    # Response latencies
    latencies = [q.response_latency for q in answered if q.response_latency is not None]
    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    median_latency = float(np.median(latencies)) if latencies else 0.0
    
    # Response durations
    durations = [q.response_duration for q in answered if q.response_duration is not None]
    mean_duration = float(np.mean(durations)) if durations else 0.0
    
    # Appropriateness (based on latency + duration heuristics)
    appropriate = _assess_appropriateness(answered)
    appropriate_count = sum(1 for a in appropriate if a > 50)
    appropriateness_rate = (appropriate_count / answered_count * 100) if answered_count > 0 else 0.0
    
    # Responsiveness Index (composite score)
    responsiveness_index = _compute_responsiveness_index(
        response_rate,
        mean_latency,
        mean_duration,
        appropriateness_rate
    )
    
    # Interpretation
    interpretation = _generate_qr_interpretation(
        total_questions,
        response_rate,
        mean_latency,
        responsiveness_index
    )
    
    return QuestionResponseAnalysis(
        total_questions=total_questions,
        answered_questions=answered_count,
        response_rate=float(response_rate),
        mean_response_latency=mean_latency,
        median_response_latency=median_latency,
        mean_response_duration=mean_duration,
        appropriate_responses=appropriate_count,
        appropriateness_rate=float(appropriateness_rate),
        responsiveness_index=float(responsiveness_index),
        questions=question_events,
        interpretation=interpretation
    )


def _identify_clinician(speaker_segments: List) -> str:
    """Identify clinician speaker (typically speaks more)."""
    speaker_times = {}
    
    for segment in speaker_segments:
        label = segment.speaker_id
        duration = segment.end_time - segment.start_time
        
        if label not in speaker_times:
            speaker_times[label] = 0.0
        speaker_times[label] += duration
    
    # Clinician typically speaks more
    if speaker_times:
        clinician = max(speaker_times, key=speaker_times.get)
        return clinician
    
    return "SPEAKER_01"


def _detect_questions(
    speaker_segments: List,
    prosodic_features: List,
    clinician_label: str,
    pitch_rise_threshold: float
) -> List[QuestionEvent]:
    """
    Detect questions based on prosodic cues.
    
    Questions typically have:
    1. Rising pitch at end
    2. Clinician speaker
    3. Appropriate duration (not too short)
    """
    questions = []
    
    # Get clinician segments
    clinician_segments = [s for s in speaker_segments if s.speaker_id == clinician_label]
    
    for segment in clinician_segments:
        # Find matching prosodic features
        segment_features = _get_features_for_segment(prosodic_features, segment)
        
        if not segment_features:
            continue
        
        # Check for rising pitch (question intonation)
        has_rising_pitch = _detect_rising_pitch(segment_features, pitch_rise_threshold)
        
        # Duration check (questions typically 0.5-10 seconds)
        duration = segment.end_time - segment.start_time
        
        # Detect question if: rising pitch OR reasonable clinician utterance duration
        if has_rising_pitch or (0.5 < duration < 10.0):
            question = QuestionEvent(
                start_time=segment.start_time,
                end_time=segment.end_time,
                duration=duration
            )
            questions.append(question)
    
    return questions


def _get_features_for_segment(prosodic_features: List, segment) -> List:
    """Get prosodic features that overlap with segment."""
    matching = []
    
    for feature in prosodic_features:
        feat_start = feature.start_time
        feat_end = feature.end_time
        
        # Check overlap
        if not (feat_end <= segment.start_time or feat_start >= segment.end_time):
            matching.append(feature)
    
    return matching


def _detect_rising_pitch(features: List, threshold: float) -> bool:
    """
    Detect rising pitch contour (question intonation).
    
    Compare pitch in first vs. last portion of segment.
    """
    if len(features) < 2:
        return False
    
    # First third vs. last third
    third = len(features) // 3
    
    first_third = features[:third] if third > 0 else features[:1]
    last_third = features[-third:] if third > 0 else features[-1:]
    
    # Average pitch
    first_pitch = np.mean([f.pitch_mean for f in first_third])
    last_pitch = np.mean([f.pitch_mean for f in last_third])
    
    pitch_rise = last_pitch - first_pitch
    
    return pitch_rise > threshold


def _match_responses(
    questions: List[QuestionEvent],
    speaker_segments: List,
    patient_label: str,
    max_latency: float,
    min_duration: float
):
    """
    Match patient responses to questions.
    
    Modifies question events in-place.
    """
    # Get patient segments
    patient_segments = [s for s in speaker_segments if s.speaker_id == patient_label]
    
    for question in questions:
        # Find next patient segment after question ends
        question_end = question.end_time
        
        # Look for response within max_latency window
        for segment in patient_segments:
            if segment.start_time < question_end:
                continue  # Before question ends
            
            latency = segment.start_time - question_end
            
            if latency > max_latency:
                break  # Too late, no response
            
            duration = segment.end_time - segment.start_time
            
            if duration < min_duration:
                continue  # Too brief
            
            # Found response!
            question.has_response = True
            question.response_latency = latency
            question.response_duration = duration
            
            # Create response event
            completeness = _assess_completeness(duration)
            appropriateness = _estimate_appropriateness(latency, duration)
            
            question.response_event = ResponseEvent(
                start_time=segment.start_time,
                end_time=segment.end_time,
                duration=duration,
                latency=latency,
                completeness=completeness,
                appropriateness=appropriateness
            )
            
            break  # Found response for this question


def _assess_completeness(duration: float) -> str:
    """
    Assess response completeness based on duration.
    
    Heuristic:
    - < 1s: incomplete
    - 1-3s: brief
    - 3-10s: adequate
    - > 10s: elaborate
    """
    if duration < 1.0:
        return 'incomplete'
    elif duration < 3.0:
        return 'brief'
    elif duration < 10.0:
        return 'adequate'
    else:
        return 'elaborate'


def _estimate_appropriateness(latency: float, duration: float) -> float:
    """
    Estimate response appropriateness (0-100).
    
    Based on latency and duration heuristics.
    """
    # Latency component (0-50 points)
    if latency < 0.5:
        latency_score = 40  # Quick but maybe interrupted
    elif latency < 2.0:
        latency_score = 50  # Ideal
    elif latency < 5.0:
        latency_score = 35  # Somewhat delayed
    else:
        latency_score = 15  # Very delayed
    
    # Duration component (0-50 points)
    if duration < 1.0:
        duration_score = 20  # Too brief
    elif duration < 10.0:
        duration_score = 50  # Good
    else:
        duration_score = 30  # Very long (may be off-topic)
    
    total = latency_score + duration_score
    return float(np.clip(total, 0.0, 100.0))


def _assess_appropriateness(answered_questions: List[QuestionEvent]) -> List[float]:
    """Get appropriateness scores for all answered questions."""
    scores = []
    
    for q in answered_questions:
        if q.response_event:
            scores.append(q.response_event.appropriateness)
        else:
            scores.append(0.0)
    
    return scores


def _compute_responsiveness_index(
    response_rate: float,
    mean_latency: float,
    mean_duration: float,
    appropriateness_rate: float
) -> float:
    """
    Compute overall responsiveness index (0-100).
    
    Components:
    - Response rate (40%)
    - Latency quality (30%)
    - Appropriateness rate (30%)
    """
    # Component 1: Response rate (0-40 points)
    rate_score = response_rate * 0.4
    
    # Component 2: Latency quality (0-30 points)
    if mean_latency < 1.0:
        latency_score = 30
    elif mean_latency < 2.0:
        latency_score = 25
    elif mean_latency < 3.0:
        latency_score = 20
    elif mean_latency < 5.0:
        latency_score = 10
    else:
        latency_score = 5
    
    # Component 3: Appropriateness (0-30 points)
    approp_score = appropriateness_rate * 0.3
    
    total = rate_score + latency_score + approp_score
    
    return float(np.clip(total, 0.0, 100.0))


def _generate_qr_interpretation(
    total_questions: int,
    response_rate: float,
    mean_latency: float,
    responsiveness_index: float
) -> str:
    """Generate clinical interpretation."""
    
    if total_questions == 0:
        return "No questions detected in session. Unable to assess question-response ability."
    
    interpretation = []
    
    # Response rate
    if response_rate < 50:
        interpretation.append(f"Low response rate ({response_rate:.1f}%), indicating difficulty with question comprehension or expressive challenges.")
    elif response_rate < 80:
        interpretation.append(f"Moderate response rate ({response_rate:.1f}%), suggesting some difficulty with consistent responding.")
    else:
        interpretation.append(f"Good response rate ({response_rate:.1f}%), indicating appropriate question comprehension.")
    
    # Latency
    if mean_latency > 3.0:
        interpretation.append(f"Delayed responses (mean: {mean_latency:.2f}s), suggesting processing difficulties or reduced engagement.")
    elif mean_latency > 2.0:
        interpretation.append(f"Somewhat slow responses (mean: {mean_latency:.2f}s).")
    else:
        interpretation.append(f"Appropriate response timing (mean: {mean_latency:.2f}s).")
    
    # Overall
    interpretation.append(f"{total_questions} questions detected during session.")
    
    # Index interpretation
    if responsiveness_index < 40:
        interpretation.append("Overall responsiveness significantly below expected level.")
    elif responsiveness_index < 60:
        interpretation.append("Overall responsiveness below typical range.")
    else:
        interpretation.append("Overall responsiveness within functional range.")
    
    return " ".join(interpretation)


def _create_empty_qr_analysis() -> QuestionResponseAnalysis:
    """Create empty analysis for error cases."""
    return QuestionResponseAnalysis(
        total_questions=0,
        answered_questions=0,
        response_rate=0.0,
        mean_response_latency=0.0,
        median_response_latency=0.0,
        mean_response_duration=0.0,
        appropriate_responses=0,
        appropriateness_rate=0.0,
        responsiveness_index=0.0,
        questions=[],
        interpretation="Insufficient data for question-response analysis"
    )
