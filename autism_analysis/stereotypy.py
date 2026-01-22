"""
Stereotyped movement detection for autism assessment.

Detects repetitive motor behaviors:
- Hand flapping
- Body rocking
- Head movements
- Repetitive gestures

Clinical rationale (Autism):
- Restricted/repetitive behaviors (RRBs) are core ASD criterion
- Stereotypies often increase during stress or excitement
- ADOS-2 includes stereotypy rating
- Movement patterns correlate with regulation challenges

Engineering approach:
- Frequency analysis of pose landmarks
- Cyclic motion detection (FFT)
- Pattern matching for known stereotypies
- Amplitude and tempo analysis
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


@dataclass
class StereotypyEvent:
    """
    Single stereotyped movement episode.
    
    Attributes:
        start_time: Episode start (seconds)
        end_time: Episode end (seconds)
        duration: Episode duration (seconds)
        stereotypy_type: Type classification (flapping, rocking, head_movement, other)
        frequency_hz: Repetition frequency
        amplitude: Movement amplitude (normalized)
        body_part: Body part involved
        confidence: Detection confidence (0-1)
    """
    start_time: float
    end_time: float
    duration: float
    stereotypy_type: str
    frequency_hz: float
    amplitude: float
    body_part: str
    confidence: float = 0.0


@dataclass
class StereotypyAnalysis:
    """
    Complete stereotypy analysis.
    
    Attributes:
        total_duration: Total stereotypy time (seconds)
        episode_count: Number of episodes
        stereotypy_types: Count by type
        mean_episode_duration: Average episode length
        mean_frequency: Average repetition rate (Hz)
        mean_amplitude: Average movement amplitude
        percentage_of_session: % of session with stereotypies
        intensity_score: Overall intensity (0-100)
        events: List of stereotypy episodes
        explanation: Clinical interpretation
    """
    total_duration: float
    episode_count: int
    stereotypy_types: Dict[str, int]
    mean_episode_duration: float
    mean_frequency: float
    mean_amplitude: float
    percentage_of_session: float
    intensity_score: float
    events: List[StereotypyEvent] = field(default_factory=list)
    explanation: str = ""


def detect_stereotyped_movements(
    video_aggregated: List,
    config: Dict = None
) -> StereotypyAnalysis:
    """
    Detect stereotyped movements from video features.
    
    Args:
        video_aggregated: List of AggregatedFeatures from video pipeline
        config: Configuration dict with thresholds
        
    Returns:
        StereotypyAnalysis object
    """
    if not video_aggregated:
        logger.warning("No video features provided")
        return _create_empty_stereotypy_analysis()
    
    logger.info(f"Detecting stereotypies from {len(video_aggregated)} windows")
    
    # Get config thresholds
    if config is None:
        config = {}
    autism_config = config.get('autism_analysis', {}).get('stereotypy', {})
    
    frequency_min = autism_config.get('frequency_min_hz', 0.5)
    frequency_max = autism_config.get('frequency_max_hz', 5.0)
    amplitude_threshold = autism_config.get('amplitude_threshold', 0.15)
    min_cycles = autism_config.get('min_cycles', 3)
    
    # Extract pose motion time series
    time_series = _extract_pose_time_series(video_aggregated)
    
    # Detect stereotypies for different body parts
    events = []
    
    # Hand stereotypies (flapping, wringing)
    hand_events = _detect_hand_stereotypies(
        time_series,
        frequency_min,
        frequency_max,
        amplitude_threshold,
        min_cycles
    )
    events.extend(hand_events)
    
    # Body rocking
    body_events = _detect_body_rocking(
        time_series,
        frequency_min,
        frequency_max,
        amplitude_threshold,
        min_cycles
    )
    events.extend(body_events)
    
    # Head movements
    head_events = _detect_head_stereotypies(
        time_series,
        frequency_min,
        frequency_max,
        amplitude_threshold,
        min_cycles
    )
    events.extend(head_events)
    
    # Sort by time
    events.sort(key=lambda e: e.start_time)
    
    # Compute statistics
    total_duration = sum(e.duration for e in events)
    episode_count = len(events)
    
    # Count by type
    type_counts = {}
    for event in events:
        stype = event.stereotypy_type
        type_counts[stype] = type_counts.get(stype, 0) + 1
    
    mean_duration = total_duration / episode_count if episode_count > 0 else 0.0
    mean_frequency = np.mean([e.frequency_hz for e in events]) if events else 0.0
    mean_amplitude = np.mean([e.amplitude for e in events]) if events else 0.0
    
    # Session duration
    session_duration = 0.0
    if video_aggregated:
        session_duration = video_aggregated[-1].window_end_time - video_aggregated[0].window_start_time
    
    percentage = (total_duration / session_duration * 100) if session_duration > 0 else 0.0
    
    # Intensity score
    intensity_score = _compute_stereotypy_intensity(
        percentage,
        episode_count,
        mean_frequency,
        mean_amplitude,
        session_duration
    )
    
    # Explanation
    explanation = _generate_stereotypy_explanation(
        episode_count,
        type_counts,
        percentage,
        mean_frequency
    )
    
    return StereotypyAnalysis(
        total_duration=float(total_duration),
        episode_count=episode_count,
        stereotypy_types=type_counts,
        mean_episode_duration=float(mean_duration),
        mean_frequency=float(mean_frequency),
        mean_amplitude=float(mean_amplitude),
        percentage_of_session=float(percentage),
        intensity_score=float(intensity_score),
        events=events,
        explanation=explanation
    )


def classify_stereotypy_type(
    body_part: str,
    frequency: float,
    amplitude: float
) -> str:
    """
    Classify stereotypy type based on characteristics.
    
    Common types:
    - flapping: Fast hand movements (>2 Hz, high amplitude)
    - rocking: Body movement (0.5-2 Hz)
    - head_nodding: Head movement (1-3 Hz, moderate amplitude)
    - hand_wringing: Hand movement (slow, lower amplitude)
    """
    if body_part == 'hand':
        if frequency > 2.0 and amplitude > 0.2:
            return 'flapping'
        else:
            return 'hand_wringing'
    
    elif body_part == 'body':
        return 'rocking'
    
    elif body_part == 'head':
        if 1.0 <= frequency <= 3.0:
            return 'head_nodding'
        else:
            return 'head_movement'
    
    else:
        return 'other'


def _extract_pose_time_series(video_aggregated: List) -> Dict:
    """
    Extract pose landmark time series for stereotypy detection.
    
    Returns dict with time series for each body part.
    """
    time_series = {
        'timestamps': [],
        'left_hand_x': [],
        'left_hand_y': [],
        'right_hand_x': [],
        'right_hand_y': [],
        'torso_center_x': [],
        'torso_center_y': [],
        'head_x': [],
        'head_y': []
    }
    
    for window in video_aggregated:
        pose_feat = window.pose_features
        
        time_series['timestamps'].append(window.window_start_time)
        
        # Extract key landmarks (use means from window)
        time_series['left_hand_x'].append(pose_feat.get('left_wrist_x_mean', 0.0))
        time_series['left_hand_y'].append(pose_feat.get('left_wrist_y_mean', 0.0))
        time_series['right_hand_x'].append(pose_feat.get('right_wrist_x_mean', 0.0))
        time_series['right_hand_y'].append(pose_feat.get('right_wrist_y_mean', 0.0))
        
        # Torso (shoulders center)
        time_series['torso_center_x'].append(pose_feat.get('shoulder_center_x_mean', 0.0))
        time_series['torso_center_y'].append(pose_feat.get('shoulder_center_y_mean', 0.0))
        
        # Head (nose landmark)
        time_series['head_x'].append(pose_feat.get('nose_x_mean', 0.0))
        time_series['head_y'].append(pose_feat.get('nose_y_mean', 0.0))
    
    # Convert to numpy arrays
    for key in time_series:
        time_series[key] = np.array(time_series[key])
    
    return time_series


def _detect_hand_stereotypies(
    time_series: Dict,
    freq_min: float,
    freq_max: float,
    amp_threshold: float,
    min_cycles: int
) -> List[StereotypyEvent]:
    """Detect hand flapping and wringing movements."""
    
    events = []
    
    # Analyze left hand
    left_events = _detect_cyclic_motion(
        time_series['timestamps'],
        time_series['left_hand_x'],
        time_series['left_hand_y'],
        freq_min,
        freq_max,
        amp_threshold,
        min_cycles,
        body_part='hand',
        label_prefix='left_'
    )
    events.extend(left_events)
    
    # Analyze right hand
    right_events = _detect_cyclic_motion(
        time_series['timestamps'],
        time_series['right_hand_x'],
        time_series['right_hand_y'],
        freq_min,
        freq_max,
        amp_threshold,
        min_cycles,
        body_part='hand',
        label_prefix='right_'
    )
    events.extend(right_events)
    
    return events


def _detect_body_rocking(
    time_series: Dict,
    freq_min: float,
    freq_max: float,
    amp_threshold: float,
    min_cycles: int
) -> List[StereotypyEvent]:
    """Detect body rocking movements."""
    
    return _detect_cyclic_motion(
        time_series['timestamps'],
        time_series['torso_center_x'],
        time_series['torso_center_y'],
        freq_min,
        freq_max,
        amp_threshold,
        min_cycles,
        body_part='body',
        label_prefix=''
    )


def _detect_head_stereotypies(
    time_series: Dict,
    freq_min: float,
    freq_max: float,
    amp_threshold: float,
    min_cycles: int
) -> List[StereotypyEvent]:
    """Detect repetitive head movements."""
    
    return _detect_cyclic_motion(
        time_series['timestamps'],
        time_series['head_x'],
        time_series['head_y'],
        freq_min,
        freq_max,
        amp_threshold,
        min_cycles,
        body_part='head',
        label_prefix=''
    )


def _detect_cyclic_motion(
    timestamps: np.ndarray,
    x_series: np.ndarray,
    y_series: np.ndarray,
    freq_min: float,
    freq_max: float,
    amp_threshold: float,
    min_cycles: int,
    body_part: str,
    label_prefix: str
) -> List[StereotypyEvent]:
    """
    Detect cyclic/repetitive motion using frequency analysis.
    
    Uses FFT to find dominant frequencies in motion signal.
    """
    if len(timestamps) < 10:
        return []
    
    # Compute magnitude of motion (combined x-y)
    motion_magnitude = np.sqrt(np.diff(x_series)**2 + np.diff(y_series)**2)
    
    # Pad to match timestamps (use mean for first point)
    motion_magnitude = np.concatenate([[motion_magnitude.mean()], motion_magnitude])
    
    # Sliding window analysis
    window_size = 30  # ~3 seconds at 10fps
    step_size = 10
    
    events = []
    
    for i in range(0, len(motion_magnitude) - window_size, step_size):
        window = motion_magnitude[i:i+window_size]
        window_times = timestamps[i:i+window_size]
        
        # Check if sufficient amplitude
        if window.std() < amp_threshold:
            continue
        
        # Frequency analysis
        freqs, powers = _compute_frequency_spectrum(window, window_times)
        
        # Find dominant frequency in target range
        valid_mask = (freqs >= freq_min) & (freqs <= freq_max)
        if not valid_mask.any():
            continue
        
        valid_freqs = freqs[valid_mask]
        valid_powers = powers[valid_mask]
        
        if len(valid_powers) == 0:
            continue
        
        peak_idx = np.argmax(valid_powers)
        dominant_freq = valid_freqs[peak_idx]
        dominant_power = valid_powers[peak_idx]
        
        # Check if significant peak
        if dominant_power < 2 * np.mean(powers):
            continue
        
        # Check minimum cycles
        duration = window_times[-1] - window_times[0]
        num_cycles = dominant_freq * duration
        
        if num_cycles < min_cycles:
            continue
        
        # Create event
        stereotypy_type = classify_stereotypy_type(
            body_part,
            dominant_freq,
            window.std()
        )
        
        event = StereotypyEvent(
            start_time=float(window_times[0]),
            end_time=float(window_times[-1]),
            duration=float(duration),
            stereotypy_type=label_prefix + stereotypy_type,
            frequency_hz=float(dominant_freq),
            amplitude=float(window.std()),
            body_part=body_part,
            confidence=0.75
        )
        
        events.append(event)
    
    # Merge overlapping events
    events = _merge_overlapping_events(events)
    
    return events


def _compute_frequency_spectrum(
    signal_data: np.ndarray,
    timestamps: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute frequency spectrum using FFT.
    
    Returns frequencies and power spectral density.
    """
    # Estimate sampling rate
    dt = np.mean(np.diff(timestamps))
    fs = 1.0 / dt if dt > 0 else 10.0  # Default 10 Hz
    
    # Remove DC component
    signal_data = signal_data - signal_data.mean()
    
    # Apply window to reduce spectral leakage
    window = np.hanning(len(signal_data))
    signal_windowed = signal_data * window
    
    # FFT
    fft_vals = np.fft.rfft(signal_windowed)
    fft_freqs = np.fft.rfftfreq(len(signal_windowed), d=1/fs)
    
    # Power spectrum
    power = np.abs(fft_vals) ** 2
    
    return fft_freqs, power


def _merge_overlapping_events(events: List[StereotypyEvent]) -> List[StereotypyEvent]:
    """Merge temporally overlapping stereotypy events."""
    
    if not events:
        return []
    
    # Sort by start time
    events.sort(key=lambda e: e.start_time)
    
    merged = [events[0]]
    
    for event in events[1:]:
        last = merged[-1]
        
        # Check overlap
        if event.start_time <= last.end_time:
            # Merge
            merged[-1] = StereotypyEvent(
                start_time=last.start_time,
                end_time=max(last.end_time, event.end_time),
                duration=max(last.end_time, event.end_time) - last.start_time,
                stereotypy_type=last.stereotypy_type,  # Keep first type
                frequency_hz=(last.frequency_hz + event.frequency_hz) / 2,
                amplitude=max(last.amplitude, event.amplitude),
                body_part=last.body_part,
                confidence=(last.confidence + event.confidence) / 2
            )
        else:
            merged.append(event)
    
    return merged


def _compute_stereotypy_intensity(
    percentage: float,
    count: int,
    frequency: float,
    amplitude: float,
    session_duration: float
) -> float:
    """
    Compute overall stereotypy intensity score.
    
    Components:
    - Session percentage (0-40 points)
    - Episode count (0-30 points)
    - Frequency/amplitude (0-30 points)
    """
    # Component 1: Session percentage
    pct_score = min(percentage * 2, 40)
    
    # Component 2: Episode count (normalized by session duration)
    episodes_per_min = (count / session_duration * 60) if session_duration > 0 else 0
    count_score = min(episodes_per_min * 10, 30)
    
    # Component 3: Movement characteristics
    char_score = min((frequency * 5 + amplitude * 50), 30)
    
    total = pct_score + count_score + char_score
    return float(np.clip(total, 0.0, 100.0))


def _generate_stereotypy_explanation(
    count: int,
    type_counts: Dict[str, int],
    percentage: float,
    frequency: float
) -> str:
    """Generate clinical interpretation."""
    
    if count == 0:
        return "No stereotyped movements detected during session."
    
    explanation = []
    
    # Overall prevalence
    if percentage < 5:
        explanation.append(f"Minimal stereotypies observed ({count} episodes, {percentage:.1f}% of session).")
    elif percentage < 15:
        explanation.append(f"Moderate stereotypies present ({count} episodes, {percentage:.1f}% of session).")
    else:
        explanation.append(f"Frequent stereotypies detected ({count} episodes, {percentage:.1f}% of session).")
    
    # Types
    if type_counts:
        types_str = ", ".join([f"{k}: {v}" for k, v in type_counts.items()])
        explanation.append(f"Types: {types_str}.")
    
    # Frequency
    explanation.append(f"Mean repetition rate: {frequency:.2f} Hz.")
    
    return " ".join(explanation)


def _create_empty_stereotypy_analysis() -> StereotypyAnalysis:
    """Create empty analysis for error cases."""
    return StereotypyAnalysis(
        total_duration=0.0,
        episode_count=0,
        stereotypy_types={},
        mean_episode_duration=0.0,
        mean_frequency=0.0,
        mean_amplitude=0.0,
        percentage_of_session=0.0,
        intensity_score=0.0,
        events=[],
        explanation="Insufficient data for stereotypy analysis"
    )
