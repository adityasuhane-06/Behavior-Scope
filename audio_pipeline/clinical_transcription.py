"""
Clinical Transcription Pipeline with Behavioral Analysis.

Uses Gemini's advanced capabilities for:
- Strict verbatim transcription
- Behavioral diarization
- Clinical pattern detection
- Sentiment analysis

This is separate from the basic audio pipeline and provides
specialized analysis for clinical/therapeutic sessions.
"""

import logging
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

import google.generativeai as genai

logger = logging.getLogger(__name__)


@dataclass
class BehavioralAnnotation:
    """Behavioral pattern detected in speech."""
    type: str  # echolalia, disfluency, latency, etc.
    start_time: float
    end_time: float
    description: str
    severity: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class ClinicalTranscriptSegment:
    """Clinical transcript segment with behavioral annotations."""
    start_time: float
    end_time: float
    speaker: str  # "therapist" or "child"
    text: str  # Verbatim text with all disfluencies
    sentiment: Optional[str] = None
    tone: Optional[str] = None
    behavioral_tags: List[str] = field(default_factory=list)
    annotations: List[BehavioralAnnotation] = field(default_factory=list)
    response_latency: Optional[float] = None


@dataclass
class KeyFrameForEyeContact:
    """Key frame timestamp for eye contact analysis."""
    timestamp: float
    reason: str
    priority: str  # "high", "medium", "low"
    context: str  # "therapist_question_child_reaction", "child_response_with_disfluency", etc.


@dataclass
class ClinicalTranscript:
    """Complete clinical transcript with behavioral analysis."""
    segments: List[ClinicalTranscriptSegment]
    summary: Dict
    behavioral_patterns: Dict
    clinical_insights: Dict
    key_frames_for_eye_contact: List[KeyFrameForEyeContact] = field(default_factory=list)


class ClinicalTranscriber:
    """
    Advanced clinical transcription using Gemini with behavioral analysis.
    
    This transcriber is specifically designed for clinical/therapeutic sessions
    and provides detailed behavioral annotations beyond basic transcription.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize clinical transcriber with Gemini."""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        
        # Use Gemini 2.5 Flash (latest stable multimodal model)
        # Flash is faster and cheaper while still providing excellent analysis
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        logger.info("Clinical transcriber initialized with Gemini 2.5 Flash")
    
    def transcribe_clinical_session(
        self,
        audio_path: str,
        session_type: str = "therapy"
    ) -> ClinicalTranscript:
        """
        Transcribe and analyze a clinical session.
        
        Args:
            audio_path: Path to audio file
            session_type: Type of session (therapy, assessment, etc.)
        
        Returns:
            ClinicalTranscript with behavioral analysis
        """
        logger.info(f"Starting clinical transcription: {audio_path}")
        
        # Upload audio to Gemini
        logger.info("Uploading audio to Gemini...")
        audio_file = genai.upload_file(audio_path, mime_type='audio/wav')
        
        # Create specialized clinical prompt
        prompt = self._create_clinical_prompt(session_type)
        
        # Send to Gemini for analysis
        logger.info("Analyzing with Gemini Pro (this may take 30-60 seconds)...")
        try:
            response = self.model.generate_content([
                prompt,
                audio_file
            ])
            
            response.resolve()
            
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            raise
        
        # Parse response
        logger.info("Parsing clinical analysis...")
        clinical_transcript = self._parse_clinical_response(response.text)
        
        logger.info(f"Clinical transcription complete: {len(clinical_transcript.segments)} segments")
        
        return clinical_transcript
    
    def _create_clinical_prompt(self, session_type: str) -> str:
        """Create specialized prompt for clinical analysis."""
        
        return f"""
Role: You are an expert Speech-Language Pathologist (SLP) and Behavioral Analyst specializing in neurodiversity and autism spectrum disorder (ASD).

Task: Analyze the provided audio of a {session_type} session. Perform "Strict Verbatim Transcription" and "Behavioral Diarization."

Requirements:

1. VERBATIM ACCURACY:
   - Capture EVERY "um," "ah," "uh," stutter, repetition, and false start
   - Use "—" for abrupt breaks (e.g., "I was going to— never mind")
   - Use "..." for silent blocks or pauses > 2 seconds
   - Preserve stuttering patterns exactly (e.g., "k-k-kids", "m-m-my")
   - Include all hesitations and fillers
   - Do NOT clean up or correct the speech

2. SPEAKER IDENTIFICATION:
   - Identify speakers as "therapist" or "child"
   - Note speaker changes with timestamps
   - Detect overlapping speech

3. BEHAVIORAL TAGGING:
   Identify and tag these patterns:
   
   a) Echolalia:
      - Immediate: Child repeats therapist's words right away
      - Delayed: Child repeats something said earlier
      - Tag with [ECHOLALIA: immediate/delayed]
   
   b) Disfluency:
      - Stuttering: Sound/syllable repetitions (k-k-kids)
      - Prolongations: Extended sounds (sssssnake)
      - Blocks: Silent pauses mid-word (...)
      - Tag with [DISFLUENCY: type]
   
   c) Response Latency:
      - Measure time between therapist's question and child's response
      - Note if latency is elevated (> 3 seconds)
      - Tag with [LATENCY: X.Xs]
   
   d) Sentiment & Tone:
      - Emotional state: anxious, calm, frustrated, excited, withdrawn
      - Engagement level: engaged, disengaged, partially engaged
      - Tag each segment with sentiment

4. CLINICAL PATTERNS:
   - Perseveration: Repetitive topics or phrases
   - Scripting: Memorized phrases or movie quotes
   - Topic maintenance: Staying on topic vs. tangential
   - Turn-taking: Appropriate vs. interrupting

5. KEY FRAME IDENTIFICATION FOR EYE CONTACT ANALYSIS:
   Identify the most clinically relevant timestamps for eye contact analysis:
   
   a) Question-Response Moments:
      - Right after therapist asks a question (child's reaction)
      - When child begins responding (engagement moment)
   
   b) Turn Transitions:
      - Speaker changes (social engagement shifts)
      - Pauses between speakers (processing time)
   
   c) Behavioral Moments:
      - During disfluencies (often show avoidance)
      - During elevated response latency (processing/anxiety)
      - During emotional moments (frustration, excitement)
   
   d) Social Engagement Peaks:
      - Moments of high engagement
      - Successful communication exchanges
      - Breakthrough moments

6. OUTPUT FORMAT:
   Return a JSON object with this EXACT structure:

{{
  "segments": [
    {{
      "start_time": 0.0,
      "end_time": 2.5,
      "speaker": "therapist",
      "text": "What is your name?",
      "sentiment": "neutral",
      "tone": "calm",
      "behavioral_tags": [],
      "annotations": []
    }},
    {{
      "start_time": 2.8,
      "end_time": 5.5,
      "speaker": "child",
      "text": "M-m-my name is... um... Ken",
      "sentiment": "anxious",
      "tone": "hesitant",
      "behavioral_tags": ["disfluency", "hesitation"],
      "response_latency": 0.3,
      "annotations": [
        {{
          "type": "disfluency",
          "start_time": 2.8,
          "end_time": 3.0,
          "description": "Sound repetition on 'm'",
          "severity": "mild"
        }},
        {{
          "type": "block",
          "start_time": 3.5,
          "end_time": 3.8,
          "description": "Silent block before 'um'",
          "severity": "mild"
        }}
      ]
    }}
  ],
  "summary": {{
    "total_duration": 60.0,
    "total_words": 150,
    "therapist_speaking_time": 25.0,
    "child_speaking_time": 35.0,
    "average_response_latency": 0.8,
    "turn_count": 12
  }},
  "behavioral_patterns": {{
    "echolalia_count": 2,
    "disfluency_count": 5,
    "elevated_latency_count": 3,
    "perseveration_detected": false,
    "scripting_detected": false
  }},
  "clinical_insights": {{
    "engagement_level": "moderate",
    "communication_effectiveness": "fair",
    "areas_of_concern": ["elevated response latency", "frequent disfluencies"],
    "strengths": ["maintained topic", "appropriate turn-taking"],
    "recommendations": ["Continue fluency exercises", "Monitor response latency"]
  }},
  "key_frames_for_eye_contact": [
    {{
      "timestamp": 2.3,
      "reason": "Question response moment - child's reaction to 'What is your name?'",
      "priority": "high",
      "context": "therapist_question_child_reaction"
    }},
    {{
      "timestamp": 2.8,
      "reason": "Child response initiation with disfluency",
      "priority": "high",
      "context": "child_response_with_disfluency"
    }},
    {{
      "timestamp": 4.0,
      "reason": "Turn transition - therapist to child",
      "priority": "medium",
      "context": "turn_transition"
    }}
  ]
}}

CRITICAL: Return ONLY the JSON object. No additional text before or after.
"""
    
    def _parse_clinical_response(self, response_text: str) -> ClinicalTranscript:
        """Parse Gemini's JSON response into ClinicalTranscript."""
        
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if json_text.startswith('```'):
                lines = json_text.split('\n')
                json_text = '\n'.join(lines[1:-1])  # Remove first and last lines
            
            if json_text.startswith('```json'):
                json_text = json_text[7:]  # Remove ```json
            if json_text.endswith('```'):
                json_text = json_text[:-3]  # Remove ```
            
            json_text = json_text.strip()
            
            # Parse JSON
            data = json.loads(json_text)
            
            # Convert to ClinicalTranscript objects
            segments = []
            for seg_data in data.get('segments', []):
                # Parse annotations
                annotations = []
                for ann_data in seg_data.get('annotations', []):
                    annotation = BehavioralAnnotation(
                        type=ann_data.get('type', ''),
                        start_time=ann_data.get('start_time', 0.0),
                        end_time=ann_data.get('end_time', 0.0),
                        description=ann_data.get('description', ''),
                        severity=ann_data.get('severity'),
                        metadata=ann_data.get('metadata', {})
                    )
                    annotations.append(annotation)
                
                # Create segment
                segment = ClinicalTranscriptSegment(
                    start_time=seg_data.get('start_time', 0.0),
                    end_time=seg_data.get('end_time', 0.0),
                    speaker=seg_data.get('speaker', 'unknown'),
                    text=seg_data.get('text', ''),
                    sentiment=seg_data.get('sentiment'),
                    tone=seg_data.get('tone'),
                    behavioral_tags=seg_data.get('behavioral_tags', []),
                    annotations=annotations,
                    response_latency=seg_data.get('response_latency')
                )
                segments.append(segment)
            
            # Parse key frames for eye contact
            key_frames = []
            for frame_data in data.get('key_frames_for_eye_contact', []):
                key_frame = KeyFrameForEyeContact(
                    timestamp=frame_data.get('timestamp', 0.0),
                    reason=frame_data.get('reason', ''),
                    priority=frame_data.get('priority', 'medium'),
                    context=frame_data.get('context', '')
                )
                key_frames.append(key_frame)
            
            # Create clinical transcript
            clinical_transcript = ClinicalTranscript(
                segments=segments,
                summary=data.get('summary', {}),
                behavioral_patterns=data.get('behavioral_patterns', {}),
                clinical_insights=data.get('clinical_insights', {}),
                key_frames_for_eye_contact=key_frames
            )
            
            return clinical_transcript
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response_text[:500]}...")
            raise
        except Exception as e:
            logger.error(f"Error parsing clinical response: {e}")
            raise
    
    def export_to_json(self, transcript: ClinicalTranscript, output_path: str):
        """Export clinical transcript to JSON file."""
        
        data = {
            'segments': [
                {
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'speaker': seg.speaker,
                    'text': seg.text,
                    'sentiment': seg.sentiment,
                    'tone': seg.tone,
                    'behavioral_tags': seg.behavioral_tags,
                    'response_latency': seg.response_latency,
                    'annotations': [
                        {
                            'type': ann.type,
                            'start_time': ann.start_time,
                            'end_time': ann.end_time,
                            'description': ann.description,
                            'severity': ann.severity,
                            'metadata': ann.metadata
                        }
                        for ann in seg.annotations
                    ]
                }
                for seg in transcript.segments
            ],
            'summary': transcript.summary,
            'behavioral_patterns': transcript.behavioral_patterns,
            'clinical_insights': transcript.clinical_insights,
            'key_frames_for_eye_contact': [
                {
                    'timestamp': frame.timestamp,
                    'reason': frame.reason,
                    'priority': frame.priority,
                    'context': frame.context
                }
                for frame in transcript.key_frames_for_eye_contact
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Clinical transcript exported: {output_path}")
    
    def export_to_text(self, transcript: ClinicalTranscript, output_path: str):
        """Export clinical transcript to readable text format."""
        
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("CLINICAL TRANSCRIPT WITH BEHAVIORAL ANALYSIS")
        lines.append("=" * 80)
        lines.append("")
        
        # Summary
        lines.append("SESSION SUMMARY:")
        for key, value in transcript.summary.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
        
        # Transcript
        lines.append("VERBATIM TRANSCRIPT:")
        lines.append("-" * 80)
        
        for seg in transcript.segments:
            # Timestamp and speaker
            time_str = f"[{seg.start_time:.2f}s - {seg.end_time:.2f}s]"
            speaker_str = seg.speaker.upper()
            
            # Sentiment/tone
            meta_str = ""
            if seg.sentiment or seg.tone:
                meta_parts = []
                if seg.sentiment:
                    meta_parts.append(f"sentiment: {seg.sentiment}")
                if seg.tone:
                    meta_parts.append(f"tone: {seg.tone}")
                meta_str = f" ({', '.join(meta_parts)})"
            
            lines.append(f"\n{time_str} {speaker_str}{meta_str}:")
            lines.append(f"  {seg.text}")
            
            # Behavioral tags
            if seg.behavioral_tags:
                lines.append(f"  Tags: {', '.join(seg.behavioral_tags)}")
            
            # Response latency
            if seg.response_latency is not None:
                lines.append(f"  Response Latency: {seg.response_latency:.2f}s")
            
            # Annotations
            if seg.annotations:
                lines.append("  Annotations:")
                for ann in seg.annotations:
                    lines.append(f"    - {ann.type}: {ann.description} (severity: {ann.severity})")
        
        lines.append("")
        lines.append("-" * 80)
        
        # Behavioral patterns
        lines.append("\nBEHAVIORAL PATTERNS:")
        for key, value in transcript.behavioral_patterns.items():
            lines.append(f"  {key}: {value}")
        
        # Clinical insights
        lines.append("\nCLINICAL INSIGHTS:")
        for key, value in transcript.clinical_insights.items():
            if isinstance(value, list):
                lines.append(f"  {key}:")
                for item in value:
                    lines.append(f"    - {item}")
            else:
                lines.append(f"  {key}: {value}")
        
        # Key frames for eye contact analysis
        if transcript.key_frames_for_eye_contact:
            lines.append("\nKEY FRAMES FOR EYE CONTACT ANALYSIS:")
            for frame in transcript.key_frames_for_eye_contact:
                lines.append(f"  {frame.timestamp:.1f}s - {frame.reason} (priority: {frame.priority})")
        
        lines.append("")
        lines.append("=" * 80)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Clinical transcript text exported: {output_path}")


def analyze_clinical_session(
    audio_path: str,
    output_dir: str,
    session_type: str = "therapy",
    api_key: str = None
) -> ClinicalTranscript:
    """
    Convenience function to analyze a clinical session.
    
    Args:
        audio_path: Path to audio file
        output_dir: Directory to save outputs
        session_type: Type of session
        api_key: Gemini API key
    
    Returns:
        ClinicalTranscript object
    """
    # Initialize transcriber
    transcriber = ClinicalTranscriber(api_key=api_key)
    
    # Analyze session
    transcript = transcriber.transcribe_clinical_session(
        audio_path,
        session_type=session_type
    )
    
    # Export results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(audio_path).stem
    
    # Export JSON
    json_path = output_dir / f"{base_name}_clinical.json"
    transcriber.export_to_json(transcript, str(json_path))
    
    # Export text
    text_path = output_dir / f"{base_name}_clinical.txt"
    transcriber.export_to_text(transcript, str(text_path))
    
    return transcript
