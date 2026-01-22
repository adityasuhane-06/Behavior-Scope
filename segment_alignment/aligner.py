"""
Audio-to-video timestamp alignment.

Engineering challenge:
- Audio timestamps are in seconds (float precision)
- Video frames are discrete indices
- Must account for variable FPS, dropped frames
- Need efficient frame extraction

Clinical importance:
- Misaligned timestamps → analyzing wrong behavioral moments
- Precision matters: 1-second offset can miss critical regulatory shift
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """
    Audio segment with temporal boundaries.
    
    Attributes:
        start_time: Start time in seconds
        end_time: End time in seconds
        score: Associated score (e.g., instability score)
        metadata: Optional metadata dict
    """
    start_time: float
    end_time: float
    score: float = 0.0
    metadata: dict = None
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class VideoSegment:
    """
    Video segment defined by frame indices.
    
    Attributes:
        start_frame: Starting frame index (0-based)
        end_frame: Ending frame index (inclusive)
        start_time: Start time in seconds (for reference)
        end_time: End time in seconds (for reference)
        score: Associated score (inherited from audio)
        fps: Video frames per second
        metadata: Optional metadata dict
    """
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    score: float = 0.0
    fps: float = 30.0
    metadata: dict = None
    
    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def get_video_metadata(video_path: Path) -> dict:
    """
    Extract video metadata (FPS, frame count, duration).
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with metadata:
        - fps: Frames per second
        - frame_count: Total number of frames
        - duration: Duration in seconds
        - width: Frame width in pixels
        - height: Frame height in pixels
        
    Raises:
        FileNotFoundError: If video doesn't exist
        RuntimeError: If video cannot be opened
    """
    # Convert to Path if string
    if isinstance(video_path, str):
        video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    try:
        metadata = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        
        # Calculate duration
        if metadata['fps'] > 0:
            metadata['duration'] = metadata['frame_count'] / metadata['fps']
        else:
            metadata['duration'] = 0.0
        
        logger.info(
            f"Video metadata: {metadata['duration']:.1f}s, "
            f"{metadata['fps']:.2f} FPS, "
            f"{metadata['frame_count']} frames, "
            f"{metadata['width']}x{metadata['height']}"
        )
        
        return metadata
        
    finally:
        cap.release()


def align_audio_to_video(
    audio_segments: List[AudioSegment],
    video_path: Path,
    fps: Optional[float] = None
) -> List[VideoSegment]:
    """
    Convert audio time segments to video frame segments.
    
    Algorithm:
    1. Get video FPS
    2. For each audio segment:
       - start_frame = floor(start_time * fps)
       - end_frame = ceil(end_time * fps)
    3. Clamp to valid frame range [0, frame_count-1]
    
    Clinical consideration:
    - Use floor for start (don't miss early behavioral cues)
    - Use ceil for end (capture full episode)
    - Preserve score for prioritization
    
    Args:
        audio_segments: List of AudioSegment objects from instability detection
        video_path: Path to video file
        fps: Optional override for FPS (uses video metadata if None)
        
    Returns:
        List of VideoSegment objects with frame indices
        
    Engineering notes:
        - Handles edge cases (segments outside video duration)
        - Preserves metadata from audio segments
        - Logs any clipped segments
    """
    if not audio_segments:
        logger.warning("No audio segments to align")
        return []
    
    logger.info(f"Aligning {len(audio_segments)} audio segments to video frames")
    
    # Get video metadata
    metadata = get_video_metadata(video_path)
    
    if fps is None:
        fps = metadata['fps']
    
    frame_count = metadata['frame_count']
    duration = metadata['duration']
    
    video_segments = []
    clipped_count = 0
    
    for audio_seg in audio_segments:
        # Convert time to frame indices
        start_frame = int(np.floor(audio_seg.start_time * fps))
        end_frame = int(np.ceil(audio_seg.end_time * fps))
        
        # Clamp to valid range
        original_start = start_frame
        original_end = end_frame
        
        start_frame = max(0, min(start_frame, frame_count - 1))
        end_frame = max(0, min(end_frame, frame_count - 1))
        
        # Check if segment was clipped
        if (start_frame != original_start or end_frame != original_end):
            clipped_count += 1
            logger.debug(
                f"Clipped segment: [{original_start}, {original_end}] "
                f"→ [{start_frame}, {end_frame}]"
            )
        
        # Skip invalid segments
        if start_frame >= end_frame:
            logger.warning(
                f"Invalid segment after clipping: "
                f"[{audio_seg.start_time:.2f}s, {audio_seg.end_time:.2f}s]"
            )
            continue
        
        # Create video segment
        video_seg = VideoSegment(
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=audio_seg.start_time,
            end_time=audio_seg.end_time,
            score=audio_seg.score,
            fps=fps,
            metadata=audio_seg.metadata
        )
        
        video_segments.append(video_seg)
    
    if clipped_count > 0:
        logger.warning(
            f"Clipped {clipped_count} segments to valid frame range "
            f"[0, {frame_count-1}]"
        )
    
    logger.info(
        f"Aligned {len(video_segments)} video segments "
        f"(total frames: {sum(vs.num_frames for vs in video_segments)})"
    )
    
    return video_segments


def time_to_frame(time_sec: float, fps: float) -> int:
    """
    Convert time in seconds to frame index.
    
    Args:
        time_sec: Time in seconds
        fps: Frames per second
        
    Returns:
        Frame index (0-based)
    """
    return int(np.round(time_sec * fps))


def frame_to_time(frame_idx: int, fps: float) -> float:
    """
    Convert frame index to time in seconds.
    
    Args:
        frame_idx: Frame index (0-based)
        fps: Frames per second
        
    Returns:
        Time in seconds
    """
    return frame_idx / fps


def validate_segments(
    segments: List[VideoSegment],
    video_path: Path
) -> Tuple[List[VideoSegment], List[str]]:
    """
    Validate video segments against video constraints.
    
    Checks:
    - Segments within video duration
    - Non-negative frame indices
    - start_frame < end_frame
    - No duplicate segments
    
    Args:
        segments: List of VideoSegment objects
        video_path: Path to video file
        
    Returns:
        Tuple of (valid_segments, error_messages)
    """
    metadata = get_video_metadata(video_path)
    frame_count = metadata['frame_count']
    
    valid_segments = []
    errors = []
    
    for i, seg in enumerate(segments):
        # Check frame indices
        if seg.start_frame < 0:
            errors.append(f"Segment {i}: negative start_frame ({seg.start_frame})")
            continue
        
        if seg.end_frame >= frame_count:
            errors.append(
                f"Segment {i}: end_frame ({seg.end_frame}) "
                f"exceeds video length ({frame_count})"
            )
            continue
        
        if seg.start_frame >= seg.end_frame:
            errors.append(
                f"Segment {i}: start_frame ({seg.start_frame}) "
                f">= end_frame ({seg.end_frame})"
            )
            continue
        
        valid_segments.append(seg)
    
    if errors:
        logger.warning(f"Found {len(errors)} invalid segments")
        for error in errors[:5]:  # Log first 5 errors
            logger.warning(f"  {error}")
    
    return valid_segments, errors
