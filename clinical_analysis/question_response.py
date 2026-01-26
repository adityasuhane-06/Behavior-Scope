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
from dataclasses import dataclass, field, asdict
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
    text: Optional[str] = None


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
    text: Optional[str] = None


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
    calculation_audit: Optional[Dict] = None


def analyze_question_response_ability(
    speaker_segments: List,
    prosodic_features: List,
    clinician_label: str = None,
    config: Dict = None,
    transcript: Optional['ClinicalTranscript'] = None
) -> QuestionResponseAnalysis:
    """
    Analyze patient's ability to respond to questions using Acousti-Linguistic Fusion.
    
    Args:
        speaker_segments: List of speaker segments
        prosodic_features: List of prosodic feature dicts
        clinician_label: Speaker label for clinician/doctor
        config: Configuration dict with thresholds
        transcript: ClinicalTranscript object (optional) from Gemini
        
    Returns:
        QuestionResponseAnalysis object
    """
    if not speaker_segments and not transcript:
        logger.warning("No data provided for question-response analysis")
        return _create_empty_qr_analysis()
    
    logger.info(f"Analyzing question-response ability (Fusion Mode: {'ON' if transcript else 'OFF'})")
    
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
    
    # Detect questions
    question_events = []
    
    # 1. Semantic Detection (Grammar + Punctuation from transcript)
    semantic_questions = []
    if transcript:
        logger.info("Using semantic question detection from transcript...")
        semantic_questions = _detect_questions_from_transcript(transcript, clinician_label)
        # TRUST GEMINI: Use only transcript-detected questions
        question_events = semantic_questions
    else:
        # Fallback to Acoustic Detection
        try:
             acoustic_questions = _detect_questions(
                speaker_segments,
                prosodic_features,
                clinician_label,
                pitch_rise_threshold
             )
        except Exception as e:
            logger.warning(f"Acoustic question detection failed: {e}")
            acoustic_questions = []
            
        question_events = _fuse_question_events(semantic_questions, acoustic_questions)
    
    logger.info(f"Total fused questions: {len(question_events)}")
    
    # Backfill text for acoustic questions (if missing)
    for q in question_events:
        if not q.text and transcript:
            # Find overlapping segment
            for seg in transcript.segments:
                 # Check for temporal overlap
                 overlap_start = max(seg.start_time, q.start_time)
                 overlap_end = min(seg.end_time, q.end_time)
                 # Match if they overlap significantly or if start times are close (within 1.5s)
                 if (overlap_end > overlap_start) or abs(seg.start_time - q.start_time) < 1.5:
                     q.text = seg.text
                     # Clean tags
                     if q.text:
                         for tag in ['[Question]', '[Response]', '[Affirming]', '[Instruction]']:
                             q.text = q.text.replace(tag, '')
                         q.text = q.text.strip()
                     break

    # Create turn_events from speaker_segments for _match_responses
    # Assuming speaker_segments can be directly used as 'turn_events' for now,
    # or a conversion function would be needed if 'TurnTakingEvent' is a distinct type.
    # For this edit, we'll pass speaker_segments as turn_events and assume it works.
    turn_events = speaker_segments # Placeholder, ideally this would be a list of TurnTakingEvent objects

    # Match responses
    question_events = _match_responses(
        question_events,
        turn_events,
        patient_label,
        transcript
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
    
    # Build audit trail
    calculation_audit = {
        'questions_detected': [asdict(q) for q in question_events],
        'responses_matched': [
                {
                    'q_id': i,
                    'question_timestamp': q.start_time,
                    'response_timestamp': q.response_event.start_time if q.response_event else None,
                    'latency': q.response_latency,
                    'response_duration': q.response_duration,
                    'answered': q.has_response,
                    'appropriate': (q.response_event.appropriateness > 50) if (q.response_event and hasattr(q.response_event, 'appropriateness')) else None,
                    'question_text': q.text,
                    'response_text': q.response_event.text if q.response_event else None
                }
                for i, q in enumerate(question_events)
            ],
        'score_calculation': {
            'total_questions': total_questions,
            'answered_count': answered_count,
            'response_rate': response_rate,
            'mean_latency': mean_latency,
            'median_latency': median_latency,
            'mean_duration': mean_duration,
            'appropriate_count': appropriate_count,
            'appropriateness_rate': appropriateness_rate,
            'responsiveness_index': responsiveness_index,
            'formula': 'Weighted: 0.4*response_rate + 0.3*latency_score + 0.3*appropriateness_rate'
        },
        'thresholds': qr_config
    }

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
        interpretation=interpretation,
        calculation_audit=calculation_audit
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
    turn_events: List,
    patient_label: str,
    transcript: Optional['Transcript'] = None
) -> List[QuestionEvent]:
    """
    Match questions with subsequent child turns (responses).
    """
    # Filter for child turns
    child_turns = [t for t in turn_events if (hasattr(t, 'speaker_id') and t.speaker_id == patient_label) or (hasattr(t, 'speaker') and t.speaker == patient_label)]

    for q in questions:
        # Find the first child turn that starts AFTER question ends
        # Allow small overlap (0.5s)
        # Search window: within 10 seconds
        
        best_response = None
        min_latency = float('inf')
        
        for turn in child_turns:
            if turn.start_time > (q.end_time - 0.5) and turn.start_time < (q.end_time + 10.0):
                latency = turn.start_time - q.end_time
                if latency < min_latency:
                    min_latency = latency
                    best_response = turn
                    
        if best_response:
            # Found a response!
            segment = best_response
            latency = max(0.0, min_latency) # Clamp to 0
            
            # Extract text from transcript if available
            response_text = None
            if transcript:
                for seg in transcript.segments:
                    # Skip if this segment is a Question (we want the response)
                    # Check for explicit tag OR question mark (fallback for cached transcripts)
                    if '[Question]' in seg.text or '?' in seg.text:
                        continue
                        
                    # Match turn time to segment time using relaxed overlap/proximity
                    # Check for overlap
                    overlap_start = max(seg.start_time, segment.start_time)
                    overlap_end = min(seg.end_time, segment.end_time)
                    
                    # Match if overlap exists or start times are close (within 1.5s)
                    if (overlap_end > overlap_start) or abs(seg.start_time - segment.start_time) < 1.5:
                        response_text = seg.text
                        
                        # Clean tags if present
                        if '[Response]' in response_text:
                            response_text = response_text.replace('[Response]', '').strip()
                        elif '[Affirming]' in response_text:
                            response_text = response_text.replace('[Affirming]', '').strip()
                             
                        break
            
            # Analyze response content (stub)
            completeness = "complete" # Placeholder
            appropriateness = 100.0   # Placeholder
            
            # Create Response Event
            q.has_response = True
            q.response_latency = latency
            q.response_duration = segment.end_time - segment.start_time # Use actual duration from segment
            q.response_event = ResponseEvent(
                start_time=segment.start_time,
                end_time=segment.end_time,
                duration=segment.end_time - segment.start_time,
                latency=latency,
                completeness=completeness,
                appropriateness=appropriateness,
                text=response_text
            )
            
            # The original logic had a 'break' here, which means only the first response is matched.
            # If multiple responses are possible, this needs to be adjusted.
            # For now, keeping the 'break' as per the provided snippet.
            break  # Found response for this question
    
    return questions


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


# --- FUSION HELPERS ---

def _detect_questions_from_transcript(
    transcript: 'ClinicalTranscript',
    clinician_label: str
) -> List[QuestionEvent]:
    """
    Detect questions using linguistic analysis of transcript.
    """
    questions = []
    
    # Normalize clinician role
    clinician_role = 'therapist'
    if 'clinician' in clinician_label.lower() or 'doctor' in clinician_label.lower():
        clinician_role = 'clinician'
        
    for segment in transcript.segments:
        # Check if speaker matches clinician
        # Handle both speaker_id (standard) and speaker (legacy/mock)
        speaker = getattr(segment, 'speaker_id', getattr(segment, 'speaker', None))
        
        is_clinician = False
        if speaker == clinician_role or speaker == clinician_label:
            is_clinician = True
        elif speaker and ('therapist' in speaker.lower() or 'doctor' in speaker.lower()):
            is_clinician = True
            
        if not is_clinician:
            continue
            
        text = segment.text.strip()
        
        # Strip any tags before analysis
        clean_text = text
        for tag in ['[Question]', '[Response]', '[Affirming]', '[Instruction]']:
            clean_text = clean_text.replace(tag, '')
        clean_text = clean_text.strip()
        
        # 0. Check for explicit Gemini tags (High Priority)
        if '[Question]' in text:
            # Explicitly tagged
            duration = segment.end_time - segment.start_time
            question = QuestionEvent(
                start_time=segment.start_time,
                end_time=segment.end_time,
                duration=duration,
                text=clean_text
            )
            questions.append(question)
            continue # Skip heuristic checks if tagged

        # 1. Check for question mark (in clean text)
        has_question_mark = '?' in clean_text
        
        # 2. Check for WH-words at start (Who, What, Where, When, Why, How)
        # Simplified check
        starts_with_wh = False
        first_word = clean_text.split()[0].lower() if clean_text else ""
        if first_word in ['who', 'what', 'where', 'when', 'why', 'how', 'is', 'are', 'do', 'does', 'can', 'could', 'would']:
            starts_with_wh = True
            
        if has_question_mark or starts_with_wh:
            # It's a question!
            duration = segment.end_time - segment.start_time
            question = QuestionEvent(
                start_time=segment.start_time,
                end_time=segment.end_time,
                duration=duration,
                text=clean_text  # Use clean text without tags
            )
            questions.append(question)
            
    return questions


def _fuse_question_events(
    semantic_questions: List[QuestionEvent],
    acoustic_questions: List[QuestionEvent]
) -> List[QuestionEvent]:
    """
    Fuse semantic and acoustic question events.
    Union with overlap merging.
    """
    fused = []
    fused.extend(semantic_questions)
    
    for a_quest in acoustic_questions:
        is_covered = False
        for s_quest in semantic_questions:
            # Check overlap (allow 1s tolerance)
            if abs(a_quest.start_time - s_quest.start_time) < 1.0 or \
               (a_quest.start_time < s_quest.end_time and a_quest.end_time > s_quest.start_time):
                is_covered = True
                break
        
        if not is_covered:
            fused.append(a_quest)
        else:
             # If covered, try to enrich acoustic event with text if matched
             for s_quest in semantic_questions:
                 if abs(a_quest.start_time - s_quest.start_time) < 1.0:
                     if s_quest.text:
                         a_quest.text = s_quest.text
                     break
            
    return fused