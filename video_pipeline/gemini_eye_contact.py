"""
Gemini AI + MediaPipe Eye Contact Detection.

Combines:
- MediaPipe: Technical face/pose detection and landmarks
- Gemini Vision: Contextual understanding of eye contact behavior

This hybrid approach provides:
- More accurate eye contact detection
- Context-aware analysis (social situations)
- Better handling of edge cases
- Clinical-grade behavioral assessment
"""

import logging
import os
import base64
import io
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

import google.generativeai as genai
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class GeminiEyeContactFrame:
    """Eye contact analysis for a single frame."""
    frame_number: int
    timestamp: float
    eye_contact_detected: bool
    confidence: float
    gaze_direction: str  # "direct", "averted", "down", "up", "side"
    social_context: str  # "engaged", "avoidant", "distracted", "transitioning"
    explanation: str
    mediapipe_data: Dict  # Raw MediaPipe landmarks/pose


@dataclass
class GeminiEyeContactAnalysis:
    """Complete Gemini-enhanced eye contact analysis."""
    total_frames: int
    eye_contact_frames: int
    eye_contact_percentage: float
    average_confidence: float
    dominant_gaze_pattern: str
    social_engagement_level: str
    clinical_observations: List[str]
    frame_analyses: List[GeminiEyeContactFrame]


class GeminiEyeContactDetector:
    """
    Hybrid eye contact detector using Gemini Vision + MediaPipe.
    
    Workflow:
    1. MediaPipe extracts face landmarks and pose
    2. Gemini analyzes frames for contextual eye contact
    3. Combine both for robust detection
    """
    
    def __init__(self, api_key: str = None):
        """Initialize Gemini eye contact detector."""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        
        # Use Gemini 2.5 Flash for vision analysis
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        logger.info("Gemini eye contact detector initialized")
    
    def analyze_video_frames(
        self,
        frames: List[np.ndarray],
        mediapipe_data: List[Dict],
        timestamps: List[float],
        sample_rate: int = 5  # Analyze every 5th frame to manage API costs
    ) -> GeminiEyeContactAnalysis:
        """
        Analyze video frames for eye contact using Gemini + MediaPipe.
        
        Args:
            frames: List of video frames (numpy arrays)
            mediapipe_data: MediaPipe face/pose data for each frame
            timestamps: Frame timestamps
            sample_rate: Analyze every Nth frame (to manage API costs)
        
        Returns:
            GeminiEyeContactAnalysis with detailed results
        """
        logger.info(f"Analyzing {len(frames)} frames for eye contact (sampling every {sample_rate} frames)")
        
        frame_analyses = []
        
        # Sample frames to analyze (every Nth frame)
        sampled_indices = range(0, len(frames), sample_rate)
        
        for i, frame_idx in enumerate(sampled_indices):
            if frame_idx >= len(frames):
                break
                
            frame = frames[frame_idx]
            mp_data = mediapipe_data[frame_idx] if frame_idx < len(mediapipe_data) else {}
            timestamp = timestamps[frame_idx] if frame_idx < len(timestamps) else 0.0
            
            logger.info(f"Analyzing frame {frame_idx+1}/{len(frames)} (sample {i+1}/{len(sampled_indices)})")
            
            # Analyze this frame with Gemini
            frame_analysis = self._analyze_single_frame(
                frame, mp_data, frame_idx, timestamp
            )
            
            frame_analyses.append(frame_analysis)
        
        # Compute overall statistics
        return self._compute_overall_analysis(frame_analyses)
    
    def _analyze_single_frame(
        self,
        frame: np.ndarray,
        mediapipe_data: Dict,
        frame_number: int,
        timestamp: float
    ) -> GeminiEyeContactFrame:
        """Analyze a single frame for eye contact."""
        
        # Convert frame to base64 for Gemini
        frame_b64 = self._frame_to_base64(frame)
        
        # Create specialized prompt for eye contact detection
        prompt = self._create_eye_contact_prompt(mediapipe_data)
        
        try:
            # Send to Gemini for analysis
            response = self.model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": frame_b64}
            ])
            
            # Parse Gemini response
            analysis = self._parse_gemini_response(response.text)
            
            return GeminiEyeContactFrame(
                frame_number=frame_number,
                timestamp=timestamp,
                eye_contact_detected=analysis['eye_contact'],
                confidence=analysis['confidence'],
                gaze_direction=analysis['gaze_direction'],
                social_context=analysis['social_context'],
                explanation=analysis['explanation'],
                mediapipe_data=mediapipe_data
            )
            
        except Exception as e:
            logger.warning(f"Gemini analysis failed for frame {frame_number}: {e}")
            
            # Fallback to MediaPipe-only analysis
            return self._fallback_analysis(frame_number, timestamp, mediapipe_data)
    
    def _create_eye_contact_prompt(self, mediapipe_data: Dict) -> str:
        """Create specialized prompt for eye contact analysis."""
        
        # Include MediaPipe data context if available
        mp_context = ""
        if mediapipe_data.get('face_detected'):
            head_yaw = mediapipe_data.get('head_yaw', 'unknown')
            head_pitch = mediapipe_data.get('head_pitch', 'unknown')
            mp_context = f"\n\nMediaPipe detected: Head yaw={head_yaw}°, pitch={head_pitch}°"
        
        return f"""
You are an expert behavioral analyst specializing in autism assessment and eye contact detection.

Analyze this image for eye contact behavior. Focus on:

1. **Direct Eye Contact**: Is the person looking directly at the camera/viewer?
2. **Gaze Direction**: Where are the eyes directed? (direct, averted, down, up, side)
3. **Social Engagement**: Does the person appear socially engaged or avoidant?
4. **Confidence Level**: How certain are you about the eye contact assessment?

Consider these clinical factors:
- Eye contact in autism can be brief, indirect, or avoidant
- Some individuals may look "through" rather than "at" the camera
- Peripheral gaze (looking near but not at camera) should be noted
- Social context matters (engaged vs. withdrawn)

{mp_context}

Respond in this EXACT JSON format:
{{
  "eye_contact": true/false,
  "confidence": 0.0-1.0,
  "gaze_direction": "direct|averted|down|up|side",
  "social_context": "engaged|avoidant|distracted|transitioning",
  "explanation": "Brief clinical explanation of the assessment"
}}

CRITICAL: Return ONLY the JSON object, no additional text.
"""
    
    def _parse_gemini_response(self, response_text: str) -> Dict:
        """Parse Gemini's JSON response."""
        import json
        
        try:
            # Clean response (remove markdown if present)
            clean_text = response_text.strip()
            if clean_text.startswith('```'):
                lines = clean_text.split('\n')
                clean_text = '\n'.join(lines[1:-1])
            if clean_text.startswith('```json'):
                clean_text = clean_text[7:]
            if clean_text.endswith('```'):
                clean_text = clean_text[:-3]
            
            clean_text = clean_text.strip()
            
            # Parse JSON
            data = json.loads(clean_text)
            
            # Validate required fields
            required_fields = ['eye_contact', 'confidence', 'gaze_direction', 'social_context', 'explanation']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            return data
            
        except Exception as e:
            logger.warning(f"Failed to parse Gemini response: {e}")
            logger.warning(f"Response text: {response_text[:200]}...")
            
            # Return default analysis
            return {
                'eye_contact': False,
                'confidence': 0.5,
                'gaze_direction': 'unknown',
                'social_context': 'unknown',
                'explanation': 'Failed to parse Gemini response'
            }
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64 for Gemini."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Resize if too large (to save API costs)
        max_size = 512
        if max(pil_image.size) > max_size:
            pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        img_data = buffer.getvalue()
        
        return base64.b64encode(img_data).decode('utf-8')
    
    def _fallback_analysis(
        self,
        frame_number: int,
        timestamp: float,
        mediapipe_data: Dict
    ) -> GeminiEyeContactFrame:
        """Fallback to MediaPipe-only analysis if Gemini fails."""
        
        # Simple MediaPipe-based detection
        face_detected = mediapipe_data.get('face_detected', False)
        head_yaw = abs(mediapipe_data.get('head_yaw', 90))  # Default to no contact
        
        eye_contact = face_detected and head_yaw < 30.0
        
        return GeminiEyeContactFrame(
            frame_number=frame_number,
            timestamp=timestamp,
            eye_contact_detected=eye_contact,
            confidence=0.6 if face_detected else 0.3,
            gaze_direction="direct" if eye_contact else "averted",
            social_context="unknown",
            explanation="Fallback MediaPipe analysis (Gemini unavailable)",
            mediapipe_data=mediapipe_data
        )
    
    def _compute_overall_analysis(
        self,
        frame_analyses: List[GeminiEyeContactFrame]
    ) -> GeminiEyeContactAnalysis:
        """Compute overall eye contact analysis from frame results."""
        
        if not frame_analyses:
            return GeminiEyeContactAnalysis(
                total_frames=0,
                eye_contact_frames=0,
                eye_contact_percentage=0.0,
                average_confidence=0.0,
                dominant_gaze_pattern="unknown",
                social_engagement_level="unknown",
                clinical_observations=[],
                frame_analyses=[]
            )
        
        # Basic statistics
        total_frames = len(frame_analyses)
        eye_contact_frames = sum(1 for f in frame_analyses if f.eye_contact_detected)
        eye_contact_percentage = (eye_contact_frames / total_frames) * 100
        
        # Average confidence
        confidences = [f.confidence for f in frame_analyses]
        average_confidence = sum(confidences) / len(confidences)
        
        # Dominant patterns
        gaze_directions = [f.gaze_direction for f in frame_analyses]
        dominant_gaze = max(set(gaze_directions), key=gaze_directions.count)
        
        social_contexts = [f.social_context for f in frame_analyses]
        dominant_social = max(set(social_contexts), key=social_contexts.count)
        
        # Clinical observations
        observations = self._generate_clinical_observations(
            eye_contact_percentage,
            dominant_gaze,
            dominant_social,
            frame_analyses
        )
        
        return GeminiEyeContactAnalysis(
            total_frames=total_frames,
            eye_contact_frames=eye_contact_frames,
            eye_contact_percentage=eye_contact_percentage,
            average_confidence=average_confidence,
            dominant_gaze_pattern=dominant_gaze,
            social_engagement_level=dominant_social,
            clinical_observations=observations,
            frame_analyses=frame_analyses
        )
    
    def _generate_clinical_observations(
        self,
        eye_contact_pct: float,
        dominant_gaze: str,
        dominant_social: str,
        frame_analyses: List[GeminiEyeContactFrame]
    ) -> List[str]:
        """Generate clinical observations from analysis."""
        
        observations = []
        
        # Eye contact level
        if eye_contact_pct < 10:
            observations.append("Significant eye contact avoidance observed")
        elif eye_contact_pct < 30:
            observations.append("Below-typical eye contact frequency")
        elif eye_contact_pct < 60:
            observations.append("Moderate eye contact engagement")
        else:
            observations.append("Good eye contact maintenance")
        
        # Gaze pattern
        if dominant_gaze == "averted":
            observations.append("Predominantly averted gaze pattern")
        elif dominant_gaze == "side":
            observations.append("Frequent side-looking behavior")
        elif dominant_gaze == "down":
            observations.append("Downward gaze preference noted")
        
        # Social engagement
        if dominant_social == "avoidant":
            observations.append("Social avoidance behaviors present")
        elif dominant_social == "engaged":
            observations.append("Positive social engagement indicators")
        elif dominant_social == "distracted":
            observations.append("Attention regulation challenges observed")
        
        # Consistency analysis
        eye_contact_frames = [f for f in frame_analyses if f.eye_contact_detected]
        if len(eye_contact_frames) > 0:
            avg_confidence = sum(f.confidence for f in eye_contact_frames) / len(eye_contact_frames)
            if avg_confidence < 0.7:
                observations.append("Eye contact episodes show variable quality")
        
        return observations


def integrate_with_existing_pipeline(
    video_aggregated: List,
    video_path: str,
    config: Dict,
    clinical_transcript = None
) -> Dict:
    """
    Integration function for existing autism analysis pipeline.
    
    This replaces the current MediaPipe-only eye contact detection
    with the hybrid Gemini + MediaPipe approach using key frames
    identified by the clinical transcription.
    """
    
    # Check if Gemini eye contact is enabled
    gemini_config = config.get('autism_analysis', {}).get('gemini_eye_contact', {})
    if not gemini_config.get('enabled', False):
        logger.info("Gemini eye contact analysis disabled, using MediaPipe only")
        return None
    
    try:
        # Initialize Gemini detector
        detector = GeminiEyeContactDetector()
        
        # Extract frames and MediaPipe data from video_aggregated
        frames, mp_data, timestamps = _extract_frames_from_aggregated(
            video_aggregated, video_path
        )
        
        if not frames:
            logger.warning("No frames extracted for Gemini analysis")
            return None
        
        # Use key frames from clinical transcript if available
        if clinical_transcript and hasattr(clinical_transcript, 'key_frames_for_eye_contact'):
            logger.info("Using key frames from clinical transcript for optimal analysis...")
            
            key_frames = clinical_transcript.key_frames_for_eye_contact
            if key_frames:
                # Sort by priority (high first) and limit to max frames
                max_frames = gemini_config.get('max_frames', 10)
                priority_order = {'high': 3, 'medium': 2, 'low': 1}
                
                sorted_frames = sorted(
                    key_frames, 
                    key=lambda f: (priority_order.get(f.priority, 1), -f.timestamp),
                    reverse=True
                )[:max_frames]
                
                optimal_timestamps = [frame.timestamp for frame in sorted_frames]
                
                # Filter frames to only analyze optimal timestamps
                selected_frames = []
                selected_mp_data = []
                selected_timestamps = []
                
                for optimal_time in optimal_timestamps:
                    # Find closest frame to optimal timestamp
                    if timestamps:
                        closest_idx = min(range(len(timestamps)), 
                                        key=lambda i: abs(timestamps[i] - optimal_time))
                        
                        if closest_idx < len(frames):
                            selected_frames.append(frames[closest_idx])
                            selected_mp_data.append(mp_data[closest_idx] if closest_idx < len(mp_data) else {})
                            selected_timestamps.append(timestamps[closest_idx])
                
                frames = selected_frames
                mp_data = selected_mp_data
                timestamps = selected_timestamps
                
                logger.info(f"Clinical transcript key frames: analyzing {len(frames)} optimal frames")
                for i, frame in enumerate(sorted_frames[:len(frames)]):
                    logger.info(f"  {frame.timestamp:.1f}s - {frame.reason} (priority: {frame.priority})")
        
        # Fallback to uniform sampling if no key frames available
        if not frames:
            logger.info("No key frames available, using uniform sampling...")
            sample_rate = gemini_config.get('sample_rate', 10)
        else:
            sample_rate = 1  # Already pre-selected optimal frames
        
        # Run Gemini analysis
        analysis = detector.analyze_video_frames(
            frames, mp_data, timestamps, sample_rate
        )
        
        logger.info(f"Gemini eye contact analysis complete: {analysis.eye_contact_percentage:.1f}% eye contact")
        
        return {
            'gemini_analysis': analysis,
            'eye_contact_percentage': analysis.eye_contact_percentage,
            'social_engagement_level': analysis.social_engagement_level,
            'clinical_observations': analysis.clinical_observations
        }
        
    except Exception as e:
        logger.error(f"Gemini eye contact analysis failed: {e}")
        return None


def _extract_frames_from_aggregated(
    video_aggregated: List,
    video_path: str
) -> Tuple[List[np.ndarray], List[Dict], List[float]]:
    """Extract frames and data from video_aggregated for Gemini analysis."""
    
    import cv2
    
    frames = []
    mp_data = []
    timestamps = []
    
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {video_path}")
            return frames, mp_data, timestamps
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {total_frames} frames at {fps:.1f} FPS")
        
        # Extract frames corresponding to video_aggregated data
        for i, agg_data in enumerate(video_aggregated):
            # Calculate frame number from timestamp
            timestamp = getattr(agg_data, 'timestamp', i * 5.0)  # Assume 5s intervals
            frame_number = int(timestamp * fps)
            
            if frame_number < total_frames:
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret:
                    frames.append(frame)
                    timestamps.append(timestamp)
                    
                    # Extract MediaPipe data from aggregated results
                    mp_data_dict = {
                        'face_detected': True,  # Assume face detected if in aggregated data
                        'head_yaw': getattr(agg_data, 'head_yaw', 0.0),
                        'head_pitch': getattr(agg_data, 'head_pitch', 0.0),
                        'head_roll': getattr(agg_data, 'head_roll', 0.0),
                        'gaze_proxy': getattr(agg_data, 'gaze_proxy', 0.0)
                    }
                    mp_data.append(mp_data_dict)
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames for Gemini analysis")
        
    except Exception as e:
        logger.error(f"Frame extraction failed: {e}")
    
    return frames, mp_data, timestamps


# Example usage and testing
if __name__ == "__main__":
    # Test the Gemini eye contact detector
    detector = GeminiEyeContactDetector()
    
    # This would be called with actual video frames
    print("Gemini Eye Contact Detector initialized successfully")