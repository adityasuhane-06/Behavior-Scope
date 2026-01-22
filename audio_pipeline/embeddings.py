"""
Audio embedding extraction using self-supervised transformers.

Engineering decision: HuBERT vs Wav2Vec 2.0
- Both are self-supervised models trained on unlabeled speech
- HuBERT slightly better for prosodic representations
- Wav2Vec 2.0 faster inference
- For MVP: use HuBERT-Base (95M params, good accuracy-speed tradeoff)

Clinical rationale:
- Transformer embeddings capture rich vocal characteristics:
  * Prosodic patterns (pitch, rhythm, energy)
  * Voice quality (hoarseness, breathiness)
  * Speaking style variations
- Self-supervised = no labeled data needed
- Embeddings are NOT used for classification - only as features for rule-based analysis

Key insight:
- We extract embeddings to measure VARIABILITY and INSTABILITY, not to classify emotions
- High embedding variance over time → vocal instability → potential dysregulation
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, HubertModel

logger = logging.getLogger(__name__)


class AudioEmbeddingExtractor:
    """
    Extract contextualized audio embeddings using HuBERT.
    
    Architecture:
    - HuBERT-Base: 12 transformer layers, 768-dim embeddings
    - Input: 16kHz audio waveform
    - Output: Frame-level embeddings (every ~20ms)
    
    Usage:
        extractor = AudioEmbeddingExtractor()
        embeddings = extractor.extract(audio_data, sample_rate=16000)
    """
    
    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        device: Optional[str] = None
    ):
        """
        Initialize embedding extractor.
        
        Args:
            model_name: HuggingFace model identifier
                Options:
                - 'facebook/hubert-base-ls960' (95M params, recommended)
                - 'facebook/hubert-large-ls960-ft' (316M params, higher quality)
                - 'facebook/wav2vec2-base-960h' (95M params, alternative)
            device: Device for inference ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading {model_name} on {self.device}")
        
        # Load model and feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Inference mode
        
        logger.info(f"Model loaded: {self._count_parameters()} parameters")
    
    def extract(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        return_all_layers: bool = False,
        layer: int = -1
    ) -> np.ndarray:
        """
        Extract embeddings from audio.
        
        Args:
            audio_data: Audio waveform (mono)
            sample_rate: Sample rate (will resample to 16kHz if different)
            return_all_layers: If True, return embeddings from all layers
            layer: Which layer to extract (-1 = last layer, recommended)
            
        Returns:
            Embeddings array of shape:
            - (num_frames, 768) if return_all_layers=False
            - (num_layers, num_frames, 768) if return_all_layers=True
            
        Clinical interpretation:
            - Each frame represents ~20ms of speech
            - Embedding vectors capture vocal characteristics at that moment
            - Temporal variance in embeddings indicates vocal instability
        """
        # Resample if needed
        if sample_rate != 16000:
            import librosa
            audio_data = librosa.resample(
                audio_data,
                orig_sr=sample_rate,
                target_sr=16000
            )
            sample_rate = 16000
        
        # Prepare input
        inputs = self.feature_extractor(
            audio_data,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract embeddings
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=return_all_layers
            )
        
        if return_all_layers:
            # Stack all layer outputs
            # Shape: (num_layers, batch, num_frames, hidden_dim)
            hidden_states = outputs.hidden_states
            embeddings = torch.stack(hidden_states).cpu().numpy()
            # Remove batch dimension (assuming batch_size=1)
            embeddings = embeddings[:, 0, :, :]  # (num_layers, num_frames, 768)
        else:
            # Get specific layer
            if layer == -1:
                embeddings = outputs.last_hidden_state
            else:
                embeddings = outputs.hidden_states[layer]
            
            embeddings = embeddings.cpu().numpy()
            # Remove batch dimension
            embeddings = embeddings[0]  # (num_frames, 768)
        
        logger.debug(f"Extracted embeddings: shape={embeddings.shape}")
        
        return embeddings
    
    def extract_segment(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        start_time: float,
        end_time: float,
        max_chunk_duration: float = 120.0,
        **kwargs
    ) -> np.ndarray:
        """
        Extract embeddings for a specific time segment.
        Automatically chunks large segments to avoid memory issues.
        
        Args:
            audio_data: Full audio waveform
            sample_rate: Sample rate
            start_time: Segment start in seconds
            end_time: Segment end in seconds
            max_chunk_duration: Maximum chunk duration in seconds (default 120s = 2 minutes)
            **kwargs: Additional arguments passed to extract()
            
        Returns:
            Embeddings for the specified segment
        """
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        segment_audio = audio_data[start_sample:end_sample]
        segment_duration = end_time - start_time
        
        # If segment is small enough, process directly
        if segment_duration <= max_chunk_duration:
            return self.extract(segment_audio, sample_rate, **kwargs)
        
        # Otherwise, chunk it
        max_chunk_samples = int(max_chunk_duration * sample_rate)
        chunks = []
        
        for chunk_start in range(0, len(segment_audio), max_chunk_samples):
            chunk_end = min(chunk_start + max_chunk_samples, len(segment_audio))
            chunk = segment_audio[chunk_start:chunk_end]
            chunk_emb = self.extract(chunk, sample_rate, **kwargs)
            chunks.append(chunk_emb)
        
        # Concatenate chunks
        return np.concatenate(chunks, axis=0)
    
    def _count_parameters(self) -> int:
        """Count model parameters."""
        return sum(p.numel() for p in self.model.parameters())


def extract_audio_embeddings(
    audio_data: np.ndarray,
    sample_rate: int,
    segments: Optional[List] = None,
    model_name: str = "facebook/hubert-base-ls960"
) -> Dict:
    """
    Convenience function to extract embeddings.
    
    Args:
        audio_data: Audio waveform
        sample_rate: Sample rate
        segments: Optional list of (start_time, end_time) tuples to process
        model_name: HuggingFace model identifier
        
    Returns:
        Dictionary containing:
        - 'embeddings': Full audio embeddings or list of segment embeddings
        - 'timestamps': Frame timestamps in seconds
        - 'model': Model name used
    """
    extractor = AudioEmbeddingExtractor(model_name=model_name)
    
    if segments is None:
        # Process entire audio
        embeddings = extractor.extract(audio_data, sample_rate)
        
        # Compute frame timestamps
        # HuBERT outputs one frame per ~20ms (exact value depends on architecture)
        hop_length = len(audio_data) / embeddings.shape[0]
        timestamps = np.arange(embeddings.shape[0]) * hop_length / sample_rate
        
        return {
            'embeddings': embeddings,
            'timestamps': timestamps,
            'model': model_name
        }
    else:
        # Process specific segments
        segment_embeddings = []
        segment_timestamps = []
        
        for segment in segments:
            # Handle both tuple and object formats
            if isinstance(segment, tuple):
                start_time, end_time = segment
            else:
                start_time = segment.start_time
                end_time = segment.end_time
                
            emb = extractor.extract_segment(
                audio_data, sample_rate,
                start_time, end_time
            )
            segment_embeddings.append(emb)
            
            # Compute timestamps relative to segment start
            hop_length = (end_time - start_time) / emb.shape[0]
            timestamps = start_time + np.arange(emb.shape[0]) * hop_length
            segment_timestamps.append(timestamps)
        
        return {
            'embeddings': segment_embeddings,
            'timestamps': segment_timestamps,
            'model': model_name
        }


def compute_embedding_statistics(embeddings: np.ndarray, window_size: int = 50) -> Dict:
    """
    Compute statistical features from embeddings.
    
    Clinical rationale:
    - Mean embedding: average vocal characteristics
    - Std/variance: vocal variability (instability indicator)
    - Temporal gradient: rate of vocal change
    - Windowed variance: local instability detection
    
    Args:
        embeddings: Embedding array (num_frames, embedding_dim)
        window_size: Window size for local statistics (in frames)
        
    Returns:
        Dictionary of statistical features
    """
    stats = {}
    
    # Global statistics
    stats['mean'] = np.mean(embeddings, axis=0)  # (768,)
    stats['std'] = np.std(embeddings, axis=0)    # (768,)
    stats['global_variance'] = np.var(embeddings, axis=0)
    
    # Temporal dynamics
    stats['temporal_gradient'] = np.gradient(embeddings, axis=0)
    stats['gradient_magnitude'] = np.linalg.norm(stats['temporal_gradient'], axis=1)
    
    # Windowed statistics (sliding window)
    num_windows = len(embeddings) // window_size
    windowed_vars = []
    
    for i in range(num_windows):
        window = embeddings[i*window_size:(i+1)*window_size]
        window_var = np.var(window, axis=0).mean()  # Average variance across dims
        windowed_vars.append(window_var)
    
    stats['windowed_variance'] = np.array(windowed_vars)
    
    # Instability score (high variance + high gradient = unstable)
    stats['instability_score'] = (
        np.mean(stats['gradient_magnitude']) *
        np.mean(stats['global_variance'])
    )
    
    logger.debug(f"Embedding statistics: instability_score={stats['instability_score']:.4f}")
    
    return stats
