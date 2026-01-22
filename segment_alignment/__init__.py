"""
Segment alignment module: Audio timestamp → Video frame mapping.

This module bridges audio analysis to video analysis by:
1. Converting audio timestamps (seconds) to video frame indices
2. Expanding temporal windows to capture behavioral context
3. Merging overlapping segments to avoid redundant processing
4. Prioritizing segments based on audio instability scores

Clinical rationale:
- Behavioral changes often begin before or extend after vocal markers
- Need temporal context (±3s) to capture full regulatory episode
- Video analysis is expensive - merge nearby segments for efficiency

Engineering approach:
- Precise timestamp alignment accounting for video FPS
- Conservative window expansion (configurable)
- Smart merging to balance context vs computation
"""

from .aligner import (
    align_audio_to_video,
    AudioSegment,
    VideoSegment,
    get_video_metadata
)
from .window_expander import (
    expand_temporal_windows,
    merge_overlapping_segments,
    prioritize_segments
)

__all__ = [
    'align_audio_to_video',
    'AudioSegment',
    'VideoSegment',
    'get_video_metadata',
    'expand_temporal_windows',
    'merge_overlapping_segments',
    'prioritize_segments',
]
