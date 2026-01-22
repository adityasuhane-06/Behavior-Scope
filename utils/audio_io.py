"""
Audio I/O utilities for extracting and loading audio data.

Engineering decisions:
- Target 16kHz mono: Balance between feature resolution and computational cost
- Use librosa for robust audio processing across formats
- OpenCV for video handling (already used for video pipeline)
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import librosa
import soundfile as sf
import cv2

logger = logging.getLogger(__name__)


def extract_audio_from_video(
    video_path,
    output_path: Optional[Path] = None,
    sample_rate: int = 16000,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Extract audio from video file and optionally save to disk.
    
    Clinical rationale:
    - 16kHz is sufficient for speech prosody analysis (speech bandwidth ~8kHz)
    - Mono reduces computational cost without losing relevant information
      (spatial audio not needed for behavioral analysis)
    
    Args:
        video_path: Path to input video file (str or Path)
        output_path: Optional path to save extracted audio (WAV format)
        sample_rate: Target sample rate (default 16kHz for speech)
        mono: Convert to mono if True
        
    Returns:
        Tuple of (audio_data, sample_rate)
        - audio_data: numpy array of shape (n_samples,) if mono, (n_channels, n_samples) if stereo
        - sample_rate: sample rate in Hz
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If audio extraction fails
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    logger.info(f"Extracting audio from {video_path}")
    
    # Set output path
    if output_path is None:
        output_path = Path(f"{video_path.stem}_temp_audio.wav")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use FFmpeg directly via subprocess (most reliable method)
        import subprocess
        
        # Get FFmpeg path (try imageio_ffmpeg first, then system FFmpeg)
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            logger.info(f"Using FFmpeg from imageio_ffmpeg: {ffmpeg_path}")
        except ImportError:
            ffmpeg_path = 'ffmpeg'
            logger.info("Using system FFmpeg")
        
        # FFmpeg command to extract audio
        cmd = [
            ffmpeg_path,
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', str(sample_rate),  # Sample rate
            '-ac', '1' if mono else '2',  # Channels
            '-y',  # Overwrite output
            str(output_path)
        ]
        
        # Run FFmpeg
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        logger.info(f"FFmpeg extraction successful")
        
        # Load the extracted audio with librosa
        audio_data, sr = librosa.load(
            str(output_path),
            sr=sample_rate,
            mono=mono
        )
        
        logger.info(f"Extracted audio: {len(audio_data)/sr:.2f}s @ {sr}Hz")
        return audio_data, sr
        
    except FileNotFoundError:
        logger.error("FFmpeg not found. Please install FFmpeg and add to PATH")
        logger.error("Download from: https://ffmpeg.org/download.html")
        raise RuntimeError(
            "FFmpeg is required for video audio extraction. "
            "Please install FFmpeg and ensure it's in your system PATH."
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {e.stderr.decode()}")
        raise RuntimeError(f"Audio extraction failed: {e}")
    except Exception as e:
        logger.error(f"Failed to extract audio: {e}")
        raise RuntimeError(f"Audio extraction failed: {e}")


def load_audio(
    audio_path: Path,
    sample_rate: int = 16000,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio file from disk.
    
    Args:
        audio_path: Path to audio file (str or Path)
        sample_rate: Target sample rate (resamples if different)
        mono: Convert to mono if True
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    logger.info(f"Loading audio from {audio_path}")
    
    audio_data, sr = librosa.load(
        str(audio_path),
        sr=sample_rate,
        mono=mono
    )
    
    logger.info(f"Loaded audio: {len(audio_data)/sr:.2f}s @ {sr}Hz")
    
    return audio_data, sr


def get_audio_duration(audio_path: Path) -> float:
    """
    Get audio duration in seconds without loading full file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    duration = librosa.get_duration(path=str(audio_path))
    return duration


def normalize_audio(audio_data: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target dB level.
    
    Clinical rationale:
    - Standardizes volume across recordings with different gain settings
    - Ensures consistent feature extraction regardless of recording equipment
    
    Args:
        audio_data: Input audio array
        target_db: Target level in dB (default -20dB)
        
    Returns:
        Normalized audio array
    """
    # Calculate current RMS level in dB
    rms = np.sqrt(np.mean(audio_data ** 2))
    current_db = 20 * np.log10(rms + 1e-10)
    
    # Calculate gain needed
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20.0)
    
    # Apply gain
    normalized = audio_data * gain_linear
    
    # Prevent clipping
    max_val = np.abs(normalized).max()
    if max_val > 0.99:
        normalized = normalized / max_val * 0.99
    
    logger.debug(f"Normalized audio: {current_db:.2f}dB → {target_db:.2f}dB")
    
    return normalized
