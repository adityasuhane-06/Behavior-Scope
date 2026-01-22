"""
Prosodic feature extraction for vocal dysregulation analysis.

Prosody = suprasegmental speech characteristics:
- Speech rate (syllables/second, words/minute)
- Pause patterns (duration, frequency, regularity)
- Pitch (fundamental frequency f0)
- Energy/intensity (volume)

Clinical rationale:
These features are established markers of behavioral regulation:
- Rapid speech rate → potential agitation or anxiety
- Irregular pauses → difficulty processing or formulating thoughts
- Pitch instability → emotional dysregulation
- Energy fluctuations → arousal state changes

Engineering approach:
- Use librosa for signal processing (robust, well-tested)
- Compute features over sliding windows (capture temporal dynamics)
- Normalize by speaker baseline (individual differences)
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import librosa

logger = logging.getLogger(__name__)


@dataclass
class ProsodicFeatures:
    """
    Container for prosodic features over a time window.
    
    Attributes:
        start_time: Window start in seconds
        end_time: Window end in seconds
        speech_rate: Estimated syllables per second
        pause_count: Number of pauses in window
        pause_duration_mean: Average pause duration (seconds)
        pause_duration_std: Pause duration variability
        pitch_mean: Mean fundamental frequency (Hz)
        pitch_std: Pitch variability (Hz)
        pitch_range: Pitch range (max - min) in Hz
        energy_mean: Mean RMS energy (dB)
        energy_std: Energy variability (dB)
        speaking_ratio: Proportion of window with speech (0-1)
    """
    start_time: float
    end_time: float
    speech_rate: float
    pause_count: int
    pause_duration_mean: float
    pause_duration_std: float
    pitch_mean: float
    pitch_std: float
    pitch_range: float
    energy_mean: float
    energy_std: float
    speaking_ratio: float


def compute_prosodic_features(
    audio_data: np.ndarray,
    sample_rate: int,
    speech_segments: List,
    window_duration: float = 5.0,
    hop_duration: float = 2.5,
    f0_min: float = 75.0,
    f0_max: float = 500.0
) -> List[ProsodicFeatures]:
    """
    Compute prosodic features over sliding windows.
    
    Clinical context:
    - 5-second windows: long enough to capture prosodic patterns,
      short enough to detect rapid changes
    - 50% overlap (2.5s hop): smooth temporal tracking
    
    Args:
        audio_data: Audio waveform (mono, 16kHz)
        sample_rate: Sample rate in Hz
        speech_segments: List of speech segments from VAD (with start_time, end_time)
        window_duration: Analysis window size in seconds (default 5s)
        hop_duration: Window hop size in seconds (default 2.5s)
        f0_min: Minimum f0 for pitch tracking (75Hz = low male voice)
        f0_max: Maximum f0 for pitch tracking (500Hz = high female voice)
        
    Returns:
        List of ProsodicFeatures for each window
        
    Engineering notes:
        - Uses librosa.pyin for pitch tracking (robust to noise)
        - Speech rate estimated from spectral flux (syllable nuclei proxy)
        - Pauses computed from speech segment gaps
    """
    duration = len(audio_data) / sample_rate
    logger.info(
        f"Computing prosodic features: {duration:.1f}s audio, "
        f"{window_duration}s windows, {hop_duration}s hop"
    )
    
    # Compute global features (for efficiency)
    pitch, voiced_flag, voiced_probs = librosa.pyin(
        audio_data,
        fmin=f0_min,
        fmax=f0_max,
        sr=sample_rate,
        frame_length=2048,
        hop_length=512
    )
    
    # Compute RMS energy
    rms = librosa.feature.rms(
        y=audio_data,
        frame_length=2048,
        hop_length=512
    )[0]
    
    # Convert to dB
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # Time axis for pitch and energy
    times_pitch = librosa.times_like(pitch, sr=sample_rate, hop_length=512)
    times_rms = librosa.times_like(rms, sr=sample_rate, hop_length=512)
    
    # Sliding window analysis
    features_list = []
    num_windows = int((duration - window_duration) / hop_duration) + 1
    
    for i in range(num_windows):
        window_start = i * hop_duration
        window_end = window_start + window_duration
        
        # Extract features for this window
        features = _extract_window_features(
            window_start, window_end,
            pitch, times_pitch,
            rms_db, times_rms,
            speech_segments,
            audio_data, sample_rate
        )
        
        features_list.append(features)
    
    logger.info(f"Extracted prosodic features for {len(features_list)} windows")
    
    return features_list


def _extract_window_features(
    start_time: float,
    end_time: float,
    pitch: np.ndarray,
    times_pitch: np.ndarray,
    rms_db: np.ndarray,
    times_rms: np.ndarray,
    speech_segments: List,
    audio_data: np.ndarray,
    sample_rate: int
) -> ProsodicFeatures:
    """Extract prosodic features for a single time window."""
    
    # Select pitch values in window
    pitch_mask = (times_pitch >= start_time) & (times_pitch < end_time)
    window_pitch = pitch[pitch_mask]
    # Remove NaN (unvoiced frames)
    window_pitch = window_pitch[~np.isnan(window_pitch)]
    
    # Pitch statistics
    if len(window_pitch) > 0:
        pitch_mean = np.mean(window_pitch)
        pitch_std = np.std(window_pitch)
        pitch_range = np.max(window_pitch) - np.min(window_pitch)
    else:
        pitch_mean = 0.0
        pitch_std = 0.0
        pitch_range = 0.0
    
    # Energy statistics
    rms_mask = (times_rms >= start_time) & (times_rms < end_time)
    window_rms = rms_db[rms_mask]
    
    if len(window_rms) > 0:
        energy_mean = np.mean(window_rms)
        energy_std = np.std(window_rms)
    else:
        energy_mean = 0.0
        energy_std = 0.0
    
    # Speech segments in window
    window_segments = [
        seg for seg in speech_segments
        if seg.start_time < end_time and seg.end_time > start_time
    ]
    
    # Speaking ratio
    total_speech_duration = 0.0
    for seg in window_segments:
        # Clip segment to window boundaries
        seg_start = max(seg.start_time, start_time)
        seg_end = min(seg.end_time, end_time)
        total_speech_duration += (seg_end - seg_start)
    
    window_duration = end_time - start_time
    speaking_ratio = total_speech_duration / window_duration if window_duration > 0 else 0.0
    
    # Pause analysis
    pauses = _compute_pauses(window_segments, start_time, end_time)
    pause_count = len(pauses)
    
    if pause_count > 0:
        pause_durations = [p[1] - p[0] for p in pauses]
        pause_duration_mean = np.mean(pause_durations)
        pause_duration_std = np.std(pause_durations) if len(pause_durations) > 1 else 0.0
    else:
        pause_duration_mean = 0.0
        pause_duration_std = 0.0
    
    # Speech rate estimation
    # Use spectral flux to detect syllable nuclei (vowels)
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    window_audio = audio_data[start_sample:end_sample]
    
    speech_rate = _estimate_speech_rate(window_audio, sample_rate, speaking_ratio)
    
    return ProsodicFeatures(
        start_time=start_time,
        end_time=end_time,
        speech_rate=speech_rate,
        pause_count=pause_count,
        pause_duration_mean=pause_duration_mean,
        pause_duration_std=pause_duration_std,
        pitch_mean=pitch_mean,
        pitch_std=pitch_std,
        pitch_range=pitch_range,
        energy_mean=energy_mean,
        energy_std=energy_std,
        speaking_ratio=speaking_ratio
    )


def _compute_pauses(
    segments: List,
    window_start: float,
    window_end: float,
    min_pause_duration: float = 0.2
) -> List[Tuple[float, float]]:
    """
    Detect pauses (gaps between speech segments).
    
    Args:
        segments: Speech segments
        window_start: Window start time
        window_end: Window end time
        min_pause_duration: Minimum pause duration to count (seconds)
        
    Returns:
        List of (pause_start, pause_end) tuples
    """
    if not segments:
        return []
    
    # Sort segments by start time
    segments = sorted(segments, key=lambda s: s.start_time)
    
    pauses = []
    
    for i in range(len(segments) - 1):
        pause_start = segments[i].end_time
        pause_end = segments[i + 1].start_time
        pause_duration = pause_end - pause_start
        
        # Check if pause is within window and meets minimum duration
        if (pause_start >= window_start and pause_end <= window_end and
            pause_duration >= min_pause_duration):
            pauses.append((pause_start, pause_end))
    
    return pauses


def _estimate_speech_rate(
    audio_segment: np.ndarray,
    sample_rate: int,
    speaking_ratio: float
) -> float:
    """
    Estimate speech rate in syllables per second.
    
    Method: Spectral flux peaks correlate with syllable nuclei (vowels).
    
    This is an approximation - true syllable counting requires phonetic analysis.
    But for detecting CHANGES in speech rate (our goal), this is sufficient.
    
    Args:
        audio_segment: Audio segment
        sample_rate: Sample rate
        speaking_ratio: Proportion of segment with speech (0-1)
        
    Returns:
        Estimated syllables per second
    """
    if len(audio_segment) == 0 or speaking_ratio < 0.1:
        return 0.0
    
    # Compute spectral flux (energy change between frames)
    spec = np.abs(librosa.stft(audio_segment))
    flux = np.sqrt(np.sum(np.diff(spec, axis=1) ** 2, axis=0))
    
    # Find peaks (potential syllable nuclei)
    from scipy.signal import find_peaks
    
    # Adaptive threshold
    threshold = np.mean(flux) + 0.5 * np.std(flux)
    peaks, _ = find_peaks(flux, height=threshold, distance=int(sample_rate / 512 * 0.1))
    
    # Speech rate = syllables / speaking time
    num_syllables = len(peaks)
    speaking_time = len(audio_segment) / sample_rate * speaking_ratio
    
    if speaking_time > 0:
        speech_rate = num_syllables / speaking_time
    else:
        speech_rate = 0.0
    
    # Clamp to reasonable range (0-10 syllables/sec)
    speech_rate = np.clip(speech_rate, 0.0, 10.0)
    
    return speech_rate


def compute_baseline_statistics(features_list: List[ProsodicFeatures]) -> dict:
    """
    Compute baseline statistics across all windows.
    
    Used for z-score normalization in instability detection.
    
    Args:
        features_list: List of ProsodicFeatures from all windows
        
    Returns:
        Dictionary with mean and std for each feature
    """
    if not features_list:
        return {}
    
    # Extract each feature type
    speech_rates = [f.speech_rate for f in features_list]
    pause_durations = [f.pause_duration_mean for f in features_list]
    pitch_stds = [f.pitch_std for f in features_list]
    energy_stds = [f.energy_std for f in features_list]
    
    baseline = {
        'speech_rate_mean': np.mean(speech_rates),
        'speech_rate_std': np.std(speech_rates),
        'pause_duration_mean': np.mean(pause_durations),
        'pause_duration_std': np.std(pause_durations),
        'pitch_variability_mean': np.mean(pitch_stds),
        'pitch_variability_std': np.std(pitch_stds),
        'energy_variability_mean': np.mean(energy_stds),
        'energy_variability_std': np.std(energy_stds),
    }
    
    logger.debug(f"Baseline statistics: speech_rate={baseline['speech_rate_mean']:.2f}±{baseline['speech_rate_std']:.2f} syll/s")
    
    return baseline
