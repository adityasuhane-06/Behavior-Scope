"""
Behavioral scoring module.

This package computes interpretable behavioral indices from multimodal features:
1. Vocal Regulation Index (0-100): Speech stability
2. Motor Agitation Index (0-100): Movement intensity
3. Attention Stability Score (0-100): Engagement/focus
4. Regulation Consistency Index (0-100): Temporal consistency
5. Facial Affect Index (0-100): Facial pattern diversity and appropriateness

All scores are:
- Interpretable (0-100 scale, higher = better for regulation/stability)
- Explainable (based on transparent calculations)
- Non-diagnostic (behavioral observation, not medical diagnosis)
"""

from .vocal_regulation import compute_vocal_regulation_index
from .motor_agitation import compute_motor_agitation_index
from .attention_stability import compute_attention_stability_score
from .consistency import compute_regulation_consistency_index
from .facial_affect_index import compute_facial_affect_index, FacialAffectIndex

__all__ = [
    'compute_vocal_regulation_index',
    'compute_motor_agitation_index',
    'compute_attention_stability_score',
    'compute_regulation_consistency_index',
    'compute_facial_affect_index',
    'FacialAffectIndex',
]
