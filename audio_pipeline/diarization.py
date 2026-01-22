"""
Speaker Diarization module (patient  separation).

Engineering decision: Use pyannote.audio
- State-of-the-art open-source diarization (SOTA as of 2024)
- Pre-trained on diverse conversational datasets
- Handles overlapping speech
- Returns precise timestamps per speaker

- pyannote.audio requires HuggingFace authentication for pre-trained models
- Users must accept model license and set HF_TOKEN environment variable
- See: https://huggingface.co/pyannote/speaker-diarization
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """
    Represents a speaker-attributed segment.
    
    Attributes:
        start_time: Segment start in seconds
        end_time: Segment end in seconds
        speaker_id: Speaker label (e.g., 'SPEAKER_00', 'SPEAKER_01')
        is_patient: True if identified as patient, False if clinician, None if unknown
    """
    start_time: float
    end_time: float
    speaker_id: str
    is_patient: Optional[bool] = None
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end_time - self.start_time


def diarize_speakers(
    audio_path: str,
    config: dict,
    num_speakers: int = 2,
    min_speakers: int = 2,
    max_speakers: int = 2
) -> List[SpeakerSegment]:
    """
    Separate patient speech using speaker diarization.
    
    context:
    - Two-speaker scenario: patient + clinician
    - Patient speaking time ratio may indicate engagement
    - Turn-taking frequency can suggest interaction quality
    - Overlapping speech may indicate agitation or interruption patterns
    
    Args:
        audio_path: Path to audio file
        config: Configuration dictionary
        num_speakers: Expected number of speakers (2 for patient+clinician)
        min_speakers: Minimum speakers to detect
        max_speakers: Maximum speakers to detect
        
    Returns:
        List of SpeakerSegment objects
        
    Notes:
        - Requires pyannote.audio 3.0+ and HuggingFace token
        - First speaker is often the one speaking first (not necessarily patient)
        - Use heuristics to identify patient vs clinician (see _identify_patient_speaker)
    """
    logger.info(f"Running speaker diarization (expecting {num_speakers} speakers)")
    
    # Load audio
    import librosa
    audio_data, sample_rate = librosa.load(audio_path, sr=None)
    
    # Check for HuggingFace authentication
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if not hf_token:
        logger.warning(
            "HF_TOKEN not set. Speaker diarization requires HuggingFace authentication.\n"
            "Visit: https://huggingface.co/pyannote/speaker-diarization\n"
            "Falling back to single-speaker mode."
        )
        return _fallback_single_speaker(audio_path, config)
    
    try:
        from pyannote.audio import Pipeline
        
        # Load pre-trained diarization pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
            logger.info("Using GPU for diarization")
        
        # Run diarization with audio file path
        diarization = pipeline(
            audio_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        # Convert to SpeakerSegment objects
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(SpeakerSegment(
                start_time=turn.start,
                end_time=turn.end,
                speaker_id=speaker,
                is_patient=None  # Will be assigned by heuristics
            ))
        
        # Identify which speaker is the patient
        segments = _identify_patient_speaker(segments, audio_data, sample_rate)
        
        # Log statistics
        speaker_stats = _compute_speaker_stats(segments)
        logger.info(f"Diarization complete: {len(segments)} segments")
        for speaker_id, stats in speaker_stats.items():
            role = "PATIENT" if stats['is_patient'] else "CLINICIAN" if stats['is_patient'] is False else "UNKNOWN"
            logger.info(
                f"  {speaker_id} ({role}): {stats['duration']:.1f}s "
                f"({stats['num_segments']} segments, {stats['speaking_ratio']*100:.1f}% of total)"
            )
        
        return segments
        
    except ImportError:
        logger.error("pyannote.audio not installed. Install with: pip install pyannote.audio")
        return _fallback_single_speaker(audio_path, config)
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        return _fallback_single_speaker(audio_path, config)


def _fallback_single_speaker(audio_path, config) -> List[SpeakerSegment]:
    """
    Fallback when diarization fails: detect turn-taking based on silence gaps.
    Alternates between two speakers (clinician/patient) at each turn.
    
    Args:
        audio_path: Path to audio file
        config: Configuration dict
        
    Returns:
        List of speaker segments with alternating speakers
    """
    logger.warning("Using turn-based fallback (detecting speaker changes from silence gaps)")
    
    # Load audio
    import librosa
    audio_data, sample_rate = librosa.load(audio_path, sr=16000)
    duration = len(audio_data) / sample_rate
    
    # Detect speech segments using energy-based VAD
    frame_length = int(0.03 * sample_rate)  # 30ms frames
    hop_length = int(0.01 * sample_rate)    # 10ms hop
    
    # Compute frame energy
    energy = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
    energy_threshold = np.percentile(energy, 30)  # Lower threshold for speech
    
    # Find speech frames
    speech_frames = energy > energy_threshold
    
    # Convert frames to time
    frame_times = librosa.frames_to_time(np.arange(len(speech_frames)), sr=sample_rate, hop_length=hop_length)
    
    # Group consecutive speech frames into segments
    segments = []
    in_speech = False
    current_start = 0.0
    min_silence_duration = 0.3  # 300ms silence = speaker change
    
    for i, is_speech in enumerate(speech_frames):
        time = frame_times[i]
        
        if is_speech and not in_speech:
            # Start of speech
            current_start = time
            in_speech = True
        elif not is_speech and in_speech:
            # End of speech (silence detected)
            if time - current_start > 0.2:  # Minimum 200ms utterance
                segments.append((current_start, time))
            in_speech = False
    
    # Add final segment if still in speech
    if in_speech and duration - current_start > 0.2:
        segments.append((current_start, duration))
    
    # If no segments detected, return single speaker
    if not segments:
        logger.warning("No speech segments detected in fallback, using single speaker")
        return [
            SpeakerSegment(
                start_time=0.0,
                end_time=duration,
                speaker_id='SPEAKER_00',
                is_patient=True
            )
        ]
    
    # Alternate speakers based on turn-taking pattern
    # First speaker = clinician, second = patient, then alternates
    speaker_segments = []
    for i, (start, end) in enumerate(segments):
        speaker_id = f'SPEAKER_{i % 2}'  # Alternate between SPEAKER_0 and SPEAKER_1
        is_patient = (i % 2 == 1)  # Odd turns = patient, even turns = clinician
        
        speaker_segments.append(
            SpeakerSegment(
                start_time=start,
                end_time=end,
                speaker_id=speaker_id,
                is_patient=is_patient
            )
        )
    
    logger.info(f"Fallback detected {len(speaker_segments)} turns from {len(set(s.speaker_id for s in speaker_segments))} speakers")
    return speaker_segments


def _identify_patient_speaker(
    segments: List[SpeakerSegment],
    audio_data: np.ndarray,
    sample_rate: int
) -> List[SpeakerSegment]:
    """
    Heuristic to identify which speaker is the patient.
    
    Assumptions (typical clinical scenarios):
    1. Patient speaks more overall (longer total duration)
    2. Clinician asks shorter questions, patient gives longer responses
    3. Patient may have more variable speech patterns (we'll check in prosody module)
    
    This is a weak heuristic - in production, you'd:
    - Use manual annotation for training data
    - Train a speaker recognition model on known patient/clinician samples
    - Use microphone channel info (if available)
    
    For MVP: assume speaker with longer total speaking time is patient.
    """
    # Group by speaker
    speaker_durations = {}
    for seg in segments:
        if seg.speaker_id not in speaker_durations:
            speaker_durations[seg.speaker_id] = 0.0
        speaker_durations[seg.speaker_id] += seg.duration
    
    # Speaker with most speaking time is assumed to be patient
    if speaker_durations:
        patient_id = max(speaker_durations, key=speaker_durations.get)
        logger.info(
            f"Identified {patient_id} as patient "
            f"(speaking time: {speaker_durations[patient_id]:.1f}s)"
        )
        
        # Assign patient flag
        for seg in segments:
            seg.is_patient = (seg.speaker_id == patient_id)
    
    return segments


def _compute_speaker_stats(segments: List[SpeakerSegment]) -> Dict:
    """Compute per-speaker statistics."""
    stats = {}
    total_duration = max(seg.end_time for seg in segments) if segments else 0
    
    for seg in segments:
        if seg.speaker_id not in stats:
            stats[seg.speaker_id] = {
                'duration': 0.0,
                'num_segments': 0,
                'is_patient': seg.is_patient
            }
        
        stats[seg.speaker_id]['duration'] += seg.duration
        stats[seg.speaker_id]['num_segments'] += 1
    
    # Add speaking ratios
    for speaker_id in stats:
        stats[speaker_id]['speaking_ratio'] = stats[speaker_id]['duration'] / total_duration if total_duration > 0 else 0
    
    return stats


def get_patient_segments(segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
    """
    Filter segments to only include patient speech.
    
    Args:
        segments: All speaker segments
        
    Returns:
        Only segments where is_patient=True
    """
    patient_segments = [seg for seg in segments if seg.is_patient is True]
    total_duration = sum(seg.duration for seg in patient_segments)
    
    logger.info(
        f"Extracted {len(patient_segments)} patient segments "
        f"({total_duration:.1f}s total)"
    )
    
    return patient_segments
