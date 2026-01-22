"""
Multimodal fusion module.

This package combines audio and video evidence using conservative fusion logic:
- Requires agreement between modalities for high-confidence detection
- Maintains explainability (which modality contributed)
- Produces unified timeline with confidence scores

Clinical rationale:
- Single-modality signals may be artifacts (coughing, camera movement)
- Multi-modal agreement increases precision
- Conservative approach minimizes false positives
"""

from .conservative_fusion import (
    FusionEngine,
    FusedEvidence,
    fuse_audio_video_evidence,
    compute_multimodal_confidence
)

__all__ = [
    'FusionEngine',
    'FusedEvidence',
    'fuse_audio_video_evidence',
    'compute_multimodal_confidence',
]
