"""
Video I/O utilities for efficient frame extraction and processing.

Engineering decisions:
- OpenCV for video decoding (universal format support)
- Batch frame extraction (memory-efficient)
- Frame caching for repeated access
- Downsampling to target FPS (computational efficiency)
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Generator
import warnings

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoReader:
    """
    Efficient video reader with frame extraction and caching.
    
    Features:
    - Random frame access
    - Sequential streaming
    - FPS downsampling
    - Memory-efficient batch processing
    - Automatic resource cleanup
    
    Usage:
        with VideoReader('video.mp4', target_fps=5) as reader:
            for frame_idx, frame in reader.iter_frames():
                process(frame)
    """
    
    def __init__(
        self,
        video_path: Path,
        target_fps: Optional[float] = None,
        color_mode: str = 'RGB'
    ):
        """
        Initialize video reader.
        
        Args:
            video_path: Path to video file
            target_fps: Target FPS for downsampling (None = original FPS)
            color_mode: 'RGB' or 'BGR' (OpenCV default)
        """
        self.video_path = Path(video_path)
        self.target_fps = target_fps
        self.color_mode = color_mode
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        
        # Open video
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        
        # Get video properties
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.original_fps if self.original_fps > 0 else 0
        
        # Compute frame sampling rate
        if target_fps is not None and target_fps < self.original_fps:
            self.frame_skip = int(self.original_fps / target_fps)
        else:
            self.frame_skip = 1
            if target_fps is not None:
                logger.warning(
                    f"Target FPS ({target_fps}) >= original FPS ({self.original_fps:.2f}), "
                    f"no downsampling applied"
                )
        
        logger.info(
            f"Opened video: {self.duration:.1f}s, {self.original_fps:.2f} FPS, "
            f"{self.frame_count} frames, {self.width}x{self.height}"
        )
        if self.frame_skip > 1:
            logger.info(f"Downsampling: every {self.frame_skip} frames")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.release()
    
    def release(self):
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Read specific frame by index.
        
        Args:
            frame_idx: Frame index (0-based)
            
        Returns:
            Frame as numpy array (H, W, 3) or None if failed
        """
        if frame_idx < 0 or frame_idx >= self.frame_count:
            logger.warning(f"Frame index {frame_idx} out of range [0, {self.frame_count})")
            return None
        
        # Seek to frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read frame
        ret, frame = self.cap.read()
        
        if not ret:
            logger.warning(f"Failed to read frame {frame_idx}")
            return None
        
        # Convert color if needed
        if self.color_mode == 'RGB':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame
    
    def iter_frames(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Iterator over video frames with optional downsampling.
        
        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index (None = end of video)
            
        Yields:
            Tuple of (frame_index, frame_array)
        """
        if end_frame is None:
            end_frame = self.frame_count
        
        # Seek to start
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        current_frame = start_frame
        
        while current_frame < end_frame:
            ret, frame = self.cap.read()
            
            if not ret:
                logger.warning(f"Failed to read frame {current_frame}, stopping iteration")
                break
            
            # Apply downsampling
            if (current_frame - start_frame) % self.frame_skip == 0:
                # Convert color if needed
                if self.color_mode == 'RGB':
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                yield current_frame, frame
            
            current_frame += 1
    
    def extract_segment(
        self,
        start_frame: int,
        end_frame: int
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Extract all frames in a segment.
        
        Args:
            start_frame: Start frame index
            end_frame: End frame index (inclusive)
            
        Returns:
            List of (frame_idx, frame) tuples
        """
        frames = []
        for frame_idx, frame in self.iter_frames(start_frame, end_frame + 1):
            frames.append((frame_idx, frame))
        
        logger.debug(f"Extracted {len(frames)} frames from segment [{start_frame}, {end_frame}]")
        
        return frames


def extract_frames_from_segment(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    target_fps: Optional[float] = None
) -> List[np.ndarray]:
    """
    Convenience function to extract frames from a video segment.
    
    Args:
        video_path: Path to video file
        start_frame: Start frame index
        end_frame: End frame index (inclusive)
        target_fps: Optional target FPS for downsampling
        
    Returns:
        List of frame arrays
    """
    with VideoReader(video_path, target_fps=target_fps) as reader:
        frames = [frame for _, frame in reader.extract_segment(start_frame, end_frame)]
    
    return frames


def resize_frame(
    frame: np.ndarray,
    target_size: Tuple[int, int],
    keep_aspect_ratio: bool = True
) -> np.ndarray:
    """
    Resize frame with optional aspect ratio preservation.
    
    Args:
        frame: Input frame (H, W, C)
        target_size: Target (width, height)
        keep_aspect_ratio: Preserve aspect ratio by padding
        
    Returns:
        Resized frame
    """
    if keep_aspect_ratio:
        # Compute scaling factor
        h, w = frame.shape[:2]
        target_w, target_h = target_size
        
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        padded = cv2.copyMakeBorder(
            resized,
            top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        
        return padded
    else:
        # Direct resize
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)


def save_video_segment(
    frames: List[np.ndarray],
    output_path: Path,
    fps: float = 30.0,
    codec: str = 'mp4v'
):
    """
    Save frames as video file.
    
    Args:
        frames: List of frames (H, W, 3)
        output_path: Output video path
        fps: Output FPS
        codec: Video codec fourcc code
    """
    if not frames:
        logger.warning("No frames to save")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    h, w = frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    try:
        for frame in frames:
            # Convert RGB to BGR if needed
            if frame.shape[-1] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            out.write(frame_bgr)
        
        logger.info(f"Saved {len(frames)} frames to {output_path}")
    
    finally:
        out.release()
