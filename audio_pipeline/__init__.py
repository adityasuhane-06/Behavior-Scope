"""
Audio processing pipeline for vocal dysregulation detection.

This package implements the audio-first approach:
1. Voice Activity Detection (VAD)
2. Speaker Diarization (patient vs clinician)
3. Speech Transcription (Google Gemini API with 99%+ accuracy)
4. Audio Embedding Extraction (HuBERT/Wav2Vec)
5. Prosodic Feature Analysis
6. Rule-based Instability Detection
"""

from .vad import run_voice_activity_detection, SpeechSegment
from .diarization import diarize_speakers, SpeakerSegment
from .embeddings import extract_audio_embeddings
from .prosody import compute_prosodic_features, ProsodicFeatures
from .instability import detect_vocal_instability, InstabilityWindow
from .transcription import (
    transcribe_audio_segments,
    align_transcription_with_speakers,
    TranscriptSegment,
    TranscriptWord,
    format_transcript_as_text,
    export_transcript_to_json,
    export_transcript_to_srt,
    get_transcript_statistics
)

__all__ = [
    'run_voice_activity_detection',
    'SpeechSegment',
    'diarize_speakers',
    'SpeakerSegment',
    'extract_audio_embeddings',
    'compute_prosodic_features',
    'ProsodicFeatures',
    'detect_vocal_instability',
    'InstabilityWindow',
    'transcribe_audio_segments',
    'align_transcription_with_speakers',
    'TranscriptSegment',
    'TranscriptWord',
    'format_transcript_as_text',
    'export_transcript_to_json',
    'export_transcript_to_srt',
    'get_transcript_statistics',
]
