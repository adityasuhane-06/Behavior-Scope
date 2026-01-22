"""
Audio processing pipeline for vocal dysregulation detection.

This package implements the audio-first approach:
1. Voice Activity Detection (VAD)
2. Speaker Diarization (patient vs clinician)
3. Audio Embedding Extraction (HuBERT/Wav2Vec)
4. Prosodic Feature Analysis
5. Rule-based Instability Detection
"""

from .vad import run_voice_activity_detection, SpeechSegment
from .diarization import diarize_speakers, SpeakerSegment
from .embeddings import extract_audio_embeddings
from .prosody import compute_prosodic_features, ProsodicFeatures
from .instability import detect_vocal_instability, InstabilityWindow

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
]
