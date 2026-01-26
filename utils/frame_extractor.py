"""
Frame extraction utility for generating evidence images from video frames.

This module provides functionality to extract and save specific frames from videos
to support the facial action units evidence feature.
"""

import cv2
import os
import logging
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


class FrameExtractor:
    """
    Utility for extracting frames from videos and saving them as images.
    """
    
    def __init__(self, output_dir: str = "data/frame_evidence"):
        """
        Initialize frame extractor.
        
        Args:
            output_dir: Directory to save extracted frame images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_frame(self, video_path: str, frame_number: int, 
                     session_id: str, au_number: Optional[int] = None) -> Optional[str]:
        """
        Extract a specific frame from video and save as image.
        
        Args:
            video_path: Path to video file
            frame_number: Frame number to extract
            session_id: Session identifier
            au_number: Action Unit number (for organizing files)
            
        Returns:
            Path to saved image file or None if extraction failed
        """
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return None
            
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Failed to read frame {frame_number} from {video_path}")
                cap.release()
                return None
            
            # Create output filename
            if au_number:
                filename = f"{session_id}_frame_{frame_number:06d}_AU{au_number:02d}.jpg"
            else:
                filename = f"{session_id}_frame_{frame_number:06d}.jpg"
            
            output_path = self.output_dir / filename
            
            # Save frame
            success = cv2.imwrite(str(output_path), frame)
            cap.release()
            
            if success:
                logger.info(f"✓ Frame extracted: {output_path}")
                return str(output_path)
            else:
                logger.error(f"Failed to save frame: {output_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting frame {frame_number}: {e}")
            return None
    
    def extract_au_evidence_frames(self, video_path: str, session_id: str, 
                                  au_evidence: List[dict]) -> List[str]:
        """
        Extract multiple frames for AU evidence.
        
        Args:
            video_path: Path to video file
            session_id: Session identifier
            au_evidence: List of AU evidence dictionaries with frame_number and au_number
            
        Returns:
            List of paths to extracted image files
        """
        extracted_paths = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return extracted_paths
            
            for evidence in au_evidence:
                frame_number = evidence.get('frame_number')
                au_number = evidence.get('au_number')
                
                if frame_number is None:
                    continue
                
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame {frame_number}")
                    continue
                
                # Create output filename
                filename = f"{session_id}_frame_{frame_number:06d}_AU{au_number:02d}.jpg"
                output_path = self.output_dir / filename
                
                # Save frame
                success = cv2.imwrite(str(output_path), frame)
                if success:
                    extracted_paths.append(str(output_path))
                    logger.debug(f"✓ AU evidence frame extracted: {output_path}")
            
            cap.release()
            logger.info(f"✓ Extracted {len(extracted_paths)} AU evidence frames")
            
        except Exception as e:
            logger.error(f"Error extracting AU evidence frames: {e}")
        
        return extracted_paths
    
    def get_frame_url(self, session_id: str, frame_number: int, 
                     au_number: Optional[int] = None) -> Optional[str]:
        """
        Get URL for a frame image if it exists.
        
        Args:
            session_id: Session identifier
            frame_number: Frame number
            au_number: Action Unit number (optional)
            
        Returns:
            Relative URL to frame image or None if not found
        """
        if au_number:
            filename = f"{session_id}_frame_{frame_number:06d}_AU{au_number:02d}.jpg"
        else:
            filename = f"{session_id}_frame_{frame_number:06d}.jpg"
        
        file_path = self.output_dir / filename
        
        if file_path.exists():
            # Return relative URL that can be served by the web server
            return f"/frame-evidence/{filename}"
        
        return None
    
    def cleanup_session_frames(self, session_id: str):
        """
        Clean up frame images for a specific session.
        
        Args:
            session_id: Session identifier
        """
        try:
            pattern = f"{session_id}_frame_*.jpg"
            deleted_count = 0
            
            for file_path in self.output_dir.glob(pattern):
                file_path.unlink()
                deleted_count += 1
            
            logger.info(f"✓ Cleaned up {deleted_count} frame images for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up frames for session {session_id}: {e}")


def extract_frames_for_session(video_path: str, session_id: str, 
                              frame_numbers: List[int]) -> List[str]:
    """
    Convenience function to extract multiple frames for a session.
    
    Args:
        video_path: Path to video file
        session_id: Session identifier
        frame_numbers: List of frame numbers to extract
        
    Returns:
        List of paths to extracted images
    """
    extractor = FrameExtractor()
    extracted_paths = []
    
    for frame_number in frame_numbers:
        path = extractor.extract_frame(video_path, frame_number, session_id)
        if path:
            extracted_paths.append(path)
    
    return extracted_paths


def get_frame_image_url(session_id: str, frame_number: int, 
                       au_number: Optional[int] = None) -> Optional[str]:
    """
    Get URL for a frame image.
    
    Args:
        session_id: Session identifier
        frame_number: Frame number
        au_number: Action Unit number (optional)
        
    Returns:
        URL string or None if image doesn't exist
    """
    extractor = FrameExtractor()
    return extractor.get_frame_url(session_id, frame_number, au_number)