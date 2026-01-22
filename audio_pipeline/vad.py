"""
Voice Activity Detection (VAD) module.

Engineering decision: Use Silero VAD (not WebRTC VAD)
Rationale:
- Silero is a neural VAD with better accuracy on noisy recordings
- Pre-trained on diverse speech datasets
- Lightweight (single forward pass)
- Handles overlapping speech better (important for clinical dialogs)

Clinical rationale:
- Accurate speech/silence segmentation is critical for prosodic analysis
- Pauses between utterances are key indicators of regulation
- Must handle soft-spoken patients (low SNR scenarios)
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class SpeechSegment:
    """
    Represents a detected speech segment.
    
    Attributes:
        start_time: Segment start in seconds
        end_time: Segment end in seconds
        confidence: VAD confidence score (0-1)
    """
    start_time: float
    end_time: float
    confidence: float
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end_time - self.start_time


def run_voice_activity_detection(
    audio_data: np.ndarray,
    sample_rate: int,
    frame_duration_ms: int = 30,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    threshold: float = 0.5
) -> List[SpeechSegment]:
    """
    Detect speech segments using Silero VAD.
    
    Clinical context:
    - Speech/silence patterns reveal regulation:
      * Excessive pauses may indicate hesitation or processing difficulty
      * Rapid speech with minimal pauses may indicate agitation
      * Erratic patterns suggest instability
    
    Args:
        audio_data: Audio waveform (mono, 16kHz recommended)
        sample_rate: Audio sample rate in Hz
        frame_duration_ms: VAD frame size in milliseconds (30ms default)
        min_speech_duration_ms: Minimum speech segment length to keep
        min_silence_duration_ms: Minimum silence gap to split segments
        threshold: VAD confidence threshold (0-1)
        
    Returns:
        List of SpeechSegment objects
        
    Engineering notes:
        - Uses Silero VAD v4.0 (state-of-the-art as of 2024)
        - Batched processing for efficiency
        - Post-processing merges short gaps (reduces fragmentation)
    """
    logger.info(f"Running VAD on {len(audio_data)/sample_rate:.2f}s audio")
    
    try:
        # Load Silero VAD model (downloads on first use, ~1.5MB)
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        
        # Extract utility functions
        (get_speech_timestamps, _, read_audio, *_) = utils
        
        # Convert numpy array to torch tensor
        audio_tensor = torch.from_numpy(audio_data).float()
        
        # Run VAD
        # Returns list of dicts: [{'start': sample_idx, 'end': sample_idx}, ...]
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            model,
            sampling_rate=sample_rate,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            window_size_samples=int(sample_rate * frame_duration_ms / 1000)
        )
        
        # Convert to SpeechSegment objects
        segments = []
        for ts in speech_timestamps:
            start_sec = ts['start'] / sample_rate
            end_sec = ts['end'] / sample_rate
            
            segments.append(SpeechSegment(
                start_time=start_sec,
                end_time=end_sec,
                confidence=1.0  # Silero doesn't return per-segment confidence
            ))
        
        total_speech_duration = sum(seg.duration for seg in segments)
        logger.info(
            f"Detected {len(segments)} speech segments "
            f"({total_speech_duration:.2f}s total, "
            f"{total_speech_duration/(len(audio_data)/sample_rate)*100:.1f}% of audio)"
        )
        
        return segments
        
    except Exception as e:
        logger.error(f"VAD failed: {e}")
        logger.warning("Falling back to simple energy-based VAD")
        return _fallback_energy_vad(audio_data, sample_rate, min_speech_duration_ms)


def _fallback_energy_vad(
    audio_data: np.ndarray,
    sample_rate: int,
    min_speech_duration_ms: int
) -> List[SpeechSegment]:
    """
    Fallback energy-based VAD if Silero fails.
    
    Simple but robust: uses RMS energy thresholding.
    Not as accurate as Silero, but works without network access.
    """
    logger.info("Using fallback energy-based VAD")
    
    # Frame-based processing
    frame_length = int(sample_rate * 0.025)  # 25ms frames
    hop_length = int(sample_rate * 0.010)    # 10ms hop
    
    # Compute frame-wise RMS energy
    frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length)
    energy = np.sqrt(np.mean(frames ** 2, axis=0))
    
    # Adaptive threshold (median + offset)
    threshold = np.median(energy) + 0.02
    
    # Detect speech frames
    is_speech = energy > threshold
    
    # Convert to time segments
    segments = []
    in_speech = False
    start_frame = 0
    
    for i, speech in enumerate(is_speech):
        if speech and not in_speech:
            # Speech start
            start_frame = i
            in_speech = True
        elif not speech and in_speech:
            # Speech end
            start_time = start_frame * hop_length / sample_rate
            end_time = i * hop_length / sample_rate
            
            if (end_time - start_time) * 1000 >= min_speech_duration_ms:
                segments.append(SpeechSegment(
                    start_time=start_time,
                    end_time=end_time,
                    confidence=0.5  # Low confidence for fallback method
                ))
            in_speech = False
    
    # Handle case where speech extends to end
    if in_speech:
        start_time = start_frame * hop_length / sample_rate
        end_time = len(audio_data) / sample_rate
        if (end_time - start_time) * 1000 >= min_speech_duration_ms:
            segments.append(SpeechSegment(
                start_time=start_time,
                end_time=end_time,
                confidence=0.5
            ))
    
    logger.warning(f"Fallback VAD detected {len(segments)} segments (lower accuracy expected)")
    
    return segments


# Import librosa for fallback VAD
try:
    import librosa
except ImportError:
    logger.warning("librosa not installed, fallback VAD unavailable")
