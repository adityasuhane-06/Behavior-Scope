"""
Voice stuttering/disfluency analysis module.

Detects and quantifies speech disfluencies:
- Repetitions (sound, syllable, word)
- Prolongations (extended sounds)
- Blocks (silent pauses mid-word)
- Interjections (um, uh, like)

Clinical rationale:
- Disfluency rate indicates speech fluency level
- Pattern analysis distinguishes developmental vs. persistent stuttering
- Severity scoring guides therapy planning
- Progress monitoring for stuttering intervention

Engineering approach:
- Audio signal analysis for repetitions/prolongations
- Pause pattern analysis for blocks
- Transcript-based detection (if available)
- Temporal clustering of disfluencies
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


@dataclass
class DisfluencyEvent:
    """
    Single disfluency/stutter event.
    
    Attributes:
        start_time: Event start (seconds)
        end_time: Event end (seconds)
        duration: Event duration (seconds)
        disfluency_type: Type (repetition, prolongation, block, interjection)
        severity: Severity rating (mild, moderate, severe)
        audio_pattern: Audio characteristic (e.g., "sound_repetition")
        confidence: Detection confidence (0-1)
    """
    start_time: float
    end_time: float
    duration: float
    disfluency_type: str
    severity: str
    audio_pattern: str
    confidence: float = 0.0
    associated_word: Optional[str] = None


@dataclass
class StutteringAnalysis:
    """
    Complete stuttering/disfluency analysis.
    
    Attributes:
        total_disfluencies: Total number of disfluency events
        disfluency_rate: Disfluencies per 100 syllables (%)
        disfluency_types: Count by type
        severity_distribution: Count by severity
        total_speaking_time: Total speech duration (seconds)
        stuttering_severity_index: Overall severity (0-100, higher=more severe)
        longest_block: Longest block duration (seconds)
        repetition_units: Average repetition units (e.g., "ba-ba-ba" = 3)
        events: List of disfluency events
        interpretation: Clinical interpretation
    """
    total_disfluencies: int
    disfluency_rate: float
    disfluency_types: Dict[str, int]
    severity_distribution: Dict[str, int]
    total_speaking_time: float
    stuttering_severity_index: float
    longest_block: float
    repetition_units: float
    events: List[DisfluencyEvent] = field(default_factory=list)
    interpretation: str = ""
    calculation_audit: Optional[Dict] = None


def analyze_stuttering(
    prosodic_features: List,
    speaker_segments: List,
    target_speaker: str = None,
    config: Dict = None,
    transcript: Optional['ClinicalTranscript'] = None
) -> StutteringAnalysis:
    """
    Analyze stuttering/disfluency patterns from audio features AND clinical transcript.
    
    Args:
        prosodic_features: List of prosodic feature dicts
        speaker_segments: List of speaker segments
        target_speaker: Speaker to analyze (auto-detect if None)
        config: Configuration dict with thresholds
        transcript: ClinicalTranscript object (optional) from Gemini
        
    Returns:
        StutteringAnalysis object
    """
    if not prosodic_features or not speaker_segments:
        logger.warning("Insufficient data for stuttering analysis")
        return _create_empty_stuttering_analysis()
    
    logger.info(f"Analyzing stuttering from {len(prosodic_features)} segments")
    if transcript:
        logger.info("Using clinical transcript for linguistic disfluency detection (Fusion)")
    
    # Get config thresholds
    if config is None:
        config = {}
    stutter_config = config.get('clinical_analysis', {}).get('stuttering', {})
    
    repetition_threshold = stutter_config.get('repetition_cycle_threshold', 0.15)
    prolongation_threshold = stutter_config.get('prolongation_duration_sec', 0.5)
    block_threshold = stutter_config.get('block_silence_sec', 0.3)
    
    # Identify target speaker (patient/child)
    if target_speaker is None:
        target_speaker = _identify_target_speaker(speaker_segments)
    
    # Filter to target speaker
    target_segments = [s for s in speaker_segments if s.speaker_id == target_speaker]
    target_features = _match_features_to_segments(prosodic_features, target_segments)
    
    # Detect disfluencies
    events = []
    
    # 1. Sound/syllable repetitions (Acoustic)
    repetition_events = _detect_repetitions(
        target_features,
        target_segments,
        repetition_threshold
    )
    events.extend(repetition_events)
    
    # 2. Prolongations (Acoustic)
    prolongation_events = _detect_prolongations(
        target_features,
        target_segments,
        prolongation_threshold
    )
    events.extend(prolongation_events)
    
    # 3. Blocks (Acoustic)
    block_events = _detect_blocks(
        target_features,
        target_segments,
        block_threshold
    )
    events.extend(block_events)
    
    # 4. Linguistic Disfluencies (Transcript)
    if transcript:
        transcript_events = _detect_transcript_disfluencies(transcript, target_speaker)
        events = _fuse_disfluency_events(events, transcript_events)
    
    # Sort by time
    events.sort(key=lambda e: e.start_time)
    
    # Compute statistics
    total_disfluencies = len(events)
    
    # Count by type
    type_counts = {}
    for event in events:
        dtype = event.disfluency_type
        type_counts[dtype] = type_counts.get(dtype, 0) + 1
    
    # Count by severity
    severity_counts = {}
    for event in events:
        sev = event.severity
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    # Total speaking time
    total_speaking_time = sum(s.end_time - s.start_time for s in target_segments)
    
    # Estimate syllable count (rough: 3 syllables/second average)
    estimated_syllables = total_speaking_time * 3.0
    
    # Disfluency rate (per 100 syllables)
    disfluency_rate = (total_disfluencies / estimated_syllables * 100) if estimated_syllables > 0 else 0.0
    
    # Stuttering Severity Index (SSI-4 inspired)
    ssi, ssi_breakdown = _compute_stuttering_severity_index(
        disfluency_rate,
        events,
        total_speaking_time
    )
    
    # Longest block
    blocks = [e for e in events if e.disfluency_type == 'block']
    longest_block = max([e.duration for e in blocks]) if blocks else 0.0
    
    # Average repetition units
    repetitions = [e for e in events if 'repetition' in e.disfluency_type]
    repetition_units = _estimate_repetition_units(repetitions)
    
    # Interpretation
    interpretation = _generate_stuttering_interpretation(
        disfluency_rate,
        ssi,
        type_counts,
        total_disfluencies
    )
    
    # Build audit trail
    calculation_audit = {
        'events_detected': [asdict(e) for e in events],
        'score_calculation': {
            'total_speaking_time': total_speaking_time,
            'estimated_syllables': estimated_syllables,
            'total_disfluencies': total_disfluencies,
            'disfluency_rate_per_100': disfluency_rate,
            'longest_block_duration': longest_block,
            'average_repetition_units': repetition_units,
            'final_ssi': ssi,
            # SSI Component Breakdown
            'frequency_score': ssi_breakdown['frequency_score'],
            'frequency_max': ssi_breakdown['frequency_max'],
            'duration_score': ssi_breakdown['duration_score'],
            'duration_max': ssi_breakdown['duration_max'],
            'mean_event_duration': ssi_breakdown['mean_duration'],
            'severity_score': ssi_breakdown['severity_score'],
            'severity_max': ssi_breakdown['severity_max'],
            'formula': ssi_breakdown['formula']
        },
        'type_distribution': type_counts,
        'severity_distribution': severity_counts,
        'thresholds': stutter_config
    }
    
    return StutteringAnalysis(
        total_disfluencies=total_disfluencies,
        disfluency_rate=float(disfluency_rate),
        disfluency_types=type_counts,
        severity_distribution=severity_counts,
        total_speaking_time=float(total_speaking_time),
        stuttering_severity_index=float(ssi),
        longest_block=float(longest_block),
        repetition_units=float(repetition_units),
        events=events,
        interpretation=interpretation,
        calculation_audit=calculation_audit
    )


def _identify_target_speaker(speaker_segments: List) -> str:
    """Identify target speaker (patient/child - typically speaks less)."""
    speaker_times = {}
    
    for segment in speaker_segments:
        label = segment.speaker_id
        duration = segment.end_time - segment.start_time
        
        if label not in speaker_times:
            speaker_times[label] = 0.0
        speaker_times[label] += duration
    
    # Patient typically speaks less
    if speaker_times:
        target = min(speaker_times, key=speaker_times.get)
        return target
    
    return "SPEAKER_00"


def _match_features_to_segments(prosodic_features: List, segments: List) -> List:
    """Match prosodic features to speaker segments."""
    matched = []
    
    for feature in prosodic_features:
        feat_time = feature.start_time
        
        # Find matching segment
        for segment in segments:
            if segment.start_time <= feat_time <= segment.end_time:
                matched.append(feature)
                break
    
    return matched


def _detect_repetitions(
    features: List,
    segments: List,
    threshold: float
) -> List[DisfluencyEvent]:
    """
    Detect sound/syllable repetitions.
    
    Method: Look for rapid pitch/energy oscillations indicating repeated attempts.
    """
    events = []
    
    for i, feature in enumerate(features):
        # Check pitch variance (high variance in short window = repetition)
        pitch_std = feature.pitch_std
        pitch_mean = feature.pitch_mean
        
        # Normalized pitch variance
        pitch_cv = pitch_std / pitch_mean if pitch_mean > 0 else 0.0
        
        # High variability in short segment suggests repetition
        if pitch_cv > threshold:
            start_time = feature.start_time
            end_time = feature.end_time
            raw_duration = end_time - start_time
            
            # Cap duration at realistic values for stuttering events
            # Sound repetitions typically last 0.2-1.5 seconds, not full segments
            duration = min(raw_duration, 1.5)  # Cap at 1.5 seconds max
            
            # Severity based on pitch variability, not duration
            if pitch_cv > 0.3:
                severity = 'severe'
            elif pitch_cv > 0.2:
                severity = 'moderate'
            else:
                severity = 'mild'
            
            event = DisfluencyEvent(
                start_time=start_time,
                end_time=start_time + duration,  # Use capped duration
                duration=duration,
                disfluency_type='sound_repetition',
                severity=severity,
                audio_pattern=f'pitch_cv={pitch_cv:.3f}',
                confidence=0.7
            )
            events.append(event)
    
    return events


def _detect_prolongations(
    features: List,
    segments: List,
    min_duration: float
) -> List[DisfluencyEvent]:
    """
    Detect prolonged sounds.
    
    Method: Sustained energy with minimal pitch variation.
    """
    events = []
    
    for i, feature in enumerate(features):
        # Low pitch variance + sustained energy = prolongation
        pitch_std = feature.pitch_std
        energy_mean = feature.energy_mean
        
        raw_duration = feature.end_time - feature.start_time
        
        # Prolongation: low pitch variance, high energy, long duration
        if pitch_std < 10.0 and energy_mean > -30.0 and raw_duration > min_duration:
            start_time = feature.start_time
            
            # Cap duration at realistic values (prolongations rarely exceed 3s)
            duration = min(raw_duration, 3.0)
            
            # Severity based on duration
            if duration > 2.0:
                severity = 'severe'
            elif duration > 1.0:
                severity = 'moderate'
            else:
                severity = 'mild'
            
            event = DisfluencyEvent(
                start_time=start_time,
                end_time=start_time + duration,
                duration=duration,
                disfluency_type='prolongation',
                severity=severity,
                audio_pattern=f'sustained_{duration:.2f}s',
                confidence=0.75
            )
            events.append(event)
    
    return events


def _detect_blocks(
    features: List,
    segments: List,
    min_silence: float
) -> List[DisfluencyEvent]:
    """
    Detect blocks (unusual mid-speech pauses).
    
    Method: Silent gaps within utterances.
    """
    events = []
    
    for i in range(len(features) - 1):
        current = features[i]
        next_feat = features[i + 1]
        
        current_end = current.end_time
        next_start = next_feat.start_time
        
        gap = next_start - current_end
        
        # Block: unusual pause within speech segment
        if gap > min_silence and gap < 3.0:  # Max 3s (longer = natural pause)
            # Severity based on duration
            if gap > 1.5:
                severity = 'severe'
            elif gap > 0.8:
                severity = 'moderate'
            else:
                severity = 'mild'
            
            event = DisfluencyEvent(
                start_time=current_end,
                end_time=next_start,
                duration=gap,
                disfluency_type='block',
                severity=severity,
                audio_pattern=f'silence_{gap:.2f}s',
                confidence=0.65
            )
            events.append(event)
    
    return events


def _compute_stuttering_severity_index(
    disfluency_rate: float,
    events: List[DisfluencyEvent],
    speaking_time: float
) -> Tuple[float, Dict]:
    """
    Compute Stuttering Severity Index (0-100).
    
    Inspired by SSI-4 (Stuttering Severity Instrument).
    Components:
    - Frequency (disfluency rate) -> 0-40 points
    - Duration (mean event duration) -> 0-40 points  
    - Physical concomitants (severity distribution) -> 0-20 points
    
    Returns:
        Tuple of (total_score, breakdown_dict)
    """
    # Component 1: Frequency score (0-40 points)
    if disfluency_rate < 1.0:
        freq_score = 0
    elif disfluency_rate < 3.0:
        freq_score = 10
    elif disfluency_rate < 5.0:
        freq_score = 20
    elif disfluency_rate < 10.0:
        freq_score = 30
    else:
        freq_score = 40
    
    # Component 2: Duration score (0-40 points)
    durations = [e.duration for e in events]
    mean_duration = np.mean(durations) if durations else 0.0
    
    if mean_duration < 0.3:
        dur_score = 0
    elif mean_duration < 0.6:
        dur_score = 10
    elif mean_duration < 1.0:
        dur_score = 20
    elif mean_duration < 2.0:
        dur_score = 30
    else:
        dur_score = 40
    
    # Component 3: Severity distribution (0-20 points)
    severe_count = len([e for e in events if e.severity == 'severe'])
    severe_ratio = severe_count / len(events) if events else 0.0
    
    sev_score = min(severe_ratio * 40, 20)
    
    total = freq_score + dur_score + sev_score
    
    breakdown = {
        'frequency_score': freq_score,
        'frequency_max': 40,
        'frequency_rate': disfluency_rate,
        'duration_score': dur_score,
        'duration_max': 40,
        'mean_duration': mean_duration,
        'severity_score': sev_score,
        'severity_max': 20,
        'severe_ratio': severe_ratio,
        'formula': 'SSI = Frequency Score (0-40) + Duration Score (0-40) + Severity Score (0-20)'
    }
    
    return float(np.clip(total, 0.0, 100.0)), breakdown


def _estimate_repetition_units(repetitions: List[DisfluencyEvent]) -> float:
    """
    Estimate average repetition units (e.g., "ba-ba-ba" = 3 units).
    
    Rough estimate based on duration.
    """
    if not repetitions:
        return 0.0
    
    # Assume ~0.15s per repetition unit
    units = []
    for rep in repetitions:
        unit_count = max(2, int(rep.duration / 0.15))  # Minimum 2 units
        units.append(unit_count)
    
    return float(np.mean(units))


def _generate_stuttering_interpretation(
    disfluency_rate: float,
    ssi: float,
    type_counts: Dict[str, int],
    total_disfluencies: int
) -> str:
    """Generate clinical interpretation."""
    
    if total_disfluencies == 0:
        return "No disfluencies detected. Speech fluency appears typical."
    
    interpretation = []
    
    # Overall severity
    if ssi < 20:
        interpretation.append(f"Mild disfluency observed ({disfluency_rate:.1f}% disfluency rate).")
    elif ssi < 40:
        interpretation.append(f"Moderate stuttering present ({disfluency_rate:.1f}% disfluency rate).")
    elif ssi < 60:
        interpretation.append(f"Moderately severe stuttering ({disfluency_rate:.1f}% disfluency rate).")
    else:
        interpretation.append(f"Severe stuttering detected ({disfluency_rate:.1f}% disfluency rate).")
    
    # Disfluency count
    interpretation.append(f"{total_disfluencies} disfluency events identified.")
    
    # Type distribution
    if type_counts:
        most_common = max(type_counts, key=type_counts.get)
        interpretation.append(f"Predominant type: {most_common} ({type_counts[most_common]} occurrences).")
    
    # Clinical recommendation
    if disfluency_rate > 5.0:
        interpretation.append("Disfluency rate exceeds 5% - consider speech-language pathology referral.")
    elif disfluency_rate > 3.0:
        interpretation.append("Disfluency rate warrants monitoring and possible intervention.")
    
    return " ".join(interpretation)


def _create_empty_stuttering_analysis() -> StutteringAnalysis:
    """Create empty analysis for error cases."""
    return StutteringAnalysis(
        total_disfluencies=0,
        disfluency_rate=0.0,
        disfluency_types={},
        severity_distribution={},
        total_speaking_time=0.0,
        stuttering_severity_index=0.0,
        longest_block=0.0,
        repetition_units=0.0,
        events=[],
        interpretation="Insufficient data for stuttering analysis"
    )


def get_disfluency_timeline(
    stutter_analysis: StutteringAnalysis,
    bin_duration: float = 10.0
) -> Dict:
    """
    Get disfluency distribution over time for visualization.
    
    Args:
        stutter_analysis: StutteringAnalysis object
        bin_duration: Time bin size in seconds
        
    Returns:
        Dictionary with time bins and disfluency counts
    """
    if not stutter_analysis.events:
        return {'bins': [], 'counts': [], 'times': []}
    
    # Find time range
    max_time = max(e.end_time for e in stutter_analysis.events)
    num_bins = int(np.ceil(max_time / bin_duration))
    
    # Count disfluencies in each bin
    counts = [0] * num_bins
    
    for event in stutter_analysis.events:
        bin_idx = int(event.start_time / bin_duration)
        if bin_idx < num_bins:
            counts[bin_idx] += 1
    
    times = [i * bin_duration for i in range(num_bins)]
    
    return {
        'bins': list(range(num_bins)),
        'counts': counts,
        'times': times
    }


# --- FUSION HELPERS ---

def _detect_transcript_disfluencies(
    transcript: 'ClinicalTranscript',
    target_speaker: str
) -> List[DisfluencyEvent]:
    """
    Detect disfluencies from linguistic transcript tags.
    """
    events = []
    
    # Map speaker label (e.g., 'SPEAKER_00') to transcript roles ('child', 'therapist')
    # This is a simplification; ideally we pass explicit mapping
    target_role = 'child' # Default assumption for patient
    
    for segment in transcript.segments:
        # Check if segment speaker matches target
        # Handle both speaker_id (standard) and speaker (legacy)
        speaker = getattr(segment, 'speaker_id', getattr(segment, 'speaker', None))
        
        # More flexible target matching
        is_target = False
        
        # Explicit exclusions first
        if speaker and any(role in speaker.lower() for role in ['therapist', 'doctor', 'clinician', 'adult']):
            is_target = False
        
        elif speaker == target_role or speaker == target_speaker:
            is_target = True
        elif speaker and 'patient' in speaker.lower():
            is_target = True
        elif speaker and 'child' in speaker.lower():
            is_target = True
        # Also check for [Response] tag - responses are from the patient/child
        elif '[Response]' in segment.text or '[response]' in segment.text.lower():
            is_target = True
        # If no speaker info and it's not a question, assume it might be target
        elif speaker is None or speaker == '':
            # Check if it's NOT a therapist segment (no [Question] tag)
            if '[Question]' not in segment.text and '[question]' not in segment.text.lower():
                is_target = True
            
        if not is_target:
            continue
            
        # 1. Check strict textual repetitions (e.g., "I I I want")
        # This catches whole-word repetitions that acoustic signal often misses
        text = segment.text.lower()
        
        # Strip any tags before analysis
        for tag in ['[question]', '[response]', '[affirming]', '[instruction]']:
            text = text.replace(tag, '')
        text = text.strip()
        
        words = text.split()
        for i in range(len(words) - 1):
            if words[i] == words[i+1]:
                # Found word repetition
                # Estimate time (linear interpolation within segment)
                seg_duration = segment.end_time - segment.start_time
                word_duration = seg_duration / len(words) if len(words) > 0 else 0.1
                start = segment.start_time + (i * word_duration)
                
                event = DisfluencyEvent(
                    start_time=start,
                    end_time=start + (word_duration * 2),
                    duration=word_duration * 2,
                    disfluency_type='word_repetition',
                    severity='mild',
                    audio_pattern='text_repetition',
                    confidence=0.9,
                    associated_word=words[i]
                )
                events.append(event)
        
        # 2. Check for sound/syllable repetitions (e.g., "p-p-park", "k-k-k-kids")
        import re
        # Fixed regex: ([a-z]) captures first letter, (-\1)+ matches one or more "-letter" repetitions
        sound_repetition_pattern = r'\b([a-z])(-\1)+-[a-z]+\b'  # Matches p-p-park, k-k-k-kids, etc.
        matches = re.finditer(sound_repetition_pattern, text)
        for match in matches:
            word = match.group(0)
            # Estimate position
            word_start_idx = text[:match.start()].count(' ')
            seg_duration = segment.end_time - segment.start_time
            word_duration = seg_duration / len(words) if len(words) > 0 else 0.1
            start = segment.start_time + (word_start_idx * word_duration)
            
            event = DisfluencyEvent(
                start_time=start,
                end_time=start + word_duration,
                duration=word_duration,
                disfluency_type='sound_repetition',
                severity='moderate',
                audio_pattern='text_sound_repetition',
                confidence=0.95,
                associated_word=word
            )
            events.append(event)

        # 3. Check behavioral tags (Gemini's analysis) - safely access attributes
        behavioral_tags = getattr(segment, 'behavioral_tags', [])
        annotations = getattr(segment, 'annotations', [])
        
        if behavioral_tags and 'disfluency' in behavioral_tags:
             # Find specific annotations
             for ann in annotations:
                  if hasattr(ann, 'type') and ann.type == 'disfluency':
                      event = DisfluencyEvent(
                          start_time=getattr(ann, 'start_time', segment.start_time),
                          end_time=getattr(ann, 'end_time', segment.end_time),
                          duration=getattr(ann, 'end_time', segment.end_time) - getattr(ann, 'start_time', segment.start_time),
                          disfluency_type=ann.description.lower().split()[0] if getattr(ann, 'description', None) else 'stutter',
                          severity=getattr(ann, 'severity', 'mild') or 'mild',
                          audio_pattern='linguistic_tag',
                          confidence=0.85,
                          associated_word=segment.text
                      )
                      events.append(event)
                     
    return events


def _fuse_disfluency_events(
    acoustic_events: List[DisfluencyEvent],
    transcript_events: List[DisfluencyEvent]
) -> List[DisfluencyEvent]:
    """
    Fuse acoustic and linguistic disfluency events.
    Strategy: Union of events, merging overlaps.
    """
    fused = []
    
    # Start with all acoustic events
    fused.extend(acoustic_events)
    
    # Add non-overlapping transcript events
    for t_event in transcript_events:
        is_covered = False
        for a_event in acoustic_events:
            # Check overlap
            overlap_start = max(t_event.start_time, a_event.start_time)
            overlap_end = min(t_event.end_time, a_event.end_time)
            
            if overlap_end > overlap_start:
                # Overlap found! 
                # Linguistic detection usually more specific about TYPE, acoustic about TIMING
                # Update type if generic
                if a_event.disfluency_type in ['speech_anomaly', 'stutter']:
                    a_event.disfluency_type = t_event.disfluency_type
                
                # Boost confidence
                a_event.confidence = max(a_event.confidence, t_event.confidence)
                # Copy word context if available
                if t_event.associated_word:
                    a_event.associated_word = t_event.associated_word
                is_covered = True
                break
        
        if not is_covered:
            fused.append(t_event)
            
    return fused


# Helper for estimating repetition units for text-based events
def _estimate_text_repetition_units(events: List[DisfluencyEvent]) -> float:
    return 0.0