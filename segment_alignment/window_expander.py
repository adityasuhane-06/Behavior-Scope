"""
Temporal window expansion and segment merging.

Clinical rationale for window expansion:
- Behavioral dysregulation episodes have temporal dynamics
- Observable changes may precede or follow vocal markers
- Need context before/after instability to understand full episode

Engineering rationale for merging:
- Video analysis is computationally expensive (MediaPipe)
- Nearby segments likely part of same dysregulation episode
- Merging reduces redundant processing without losing information
- Balance: don't merge distant segments (lose temporal resolution)
"""

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def expand_temporal_windows(
    segments: List,
    pre_context_sec: float = 3.0,
    post_context_sec: float = 3.0,
    max_duration_sec: float = 30.0,
    video_duration_sec: float = None
) -> List:
    """
    Expand segment temporal windows to capture behavioral context.
    
    Clinical justification:
    - Pre-context (±3s before): Capture escalation/buildup to dysregulation
    - Post-context (±3s after): Capture recovery or sustained dysregulation
    - Example: If vocal instability at 45-50s, analyze video 42-53s
    
    Engineering considerations:
    - Respects video boundaries (doesn't expand beyond 0 or video_duration)
    - Caps maximum segment length (prevent excessive computation)
    - Maintains metadata and scores
    
    Args:
        segments: List of VideoSegment or AudioSegment objects
        pre_context_sec: Seconds to add before segment start
        post_context_sec: Seconds to add after segment end
        max_duration_sec: Maximum allowed segment duration
        video_duration_sec: Optional video duration for clamping
        
    Returns:
        List of expanded segments (same type as input)
        
    Example:
        Input: AudioSegment(45.0, 50.0, score=0.78)
        Output: AudioSegment(42.0, 53.0, score=0.78)  # ±3s expansion
    """
    if not segments:
        return []
    
    logger.info(
        f"Expanding {len(segments)} segments: "
        f"pre={pre_context_sec}s, post={post_context_sec}s"
    )
    
    expanded_segments = []
    
    for seg in segments:
        # Expand temporal boundaries
        new_start = seg.start_time - pre_context_sec
        new_end = seg.end_time + post_context_sec
        
        # Clamp to valid range
        if new_start < 0:
            new_start = 0.0
        
        if video_duration_sec is not None and new_end > video_duration_sec:
            new_end = video_duration_sec
        
        # Check maximum duration
        duration = new_end - new_start
        if duration > max_duration_sec:
            logger.debug(
                f"Segment exceeds max duration ({duration:.1f}s > {max_duration_sec}s), "
                f"capping to max"
            )
            # Keep centered on original segment
            original_center = (seg.start_time + seg.end_time) / 2
            new_start = max(0, original_center - max_duration_sec / 2)
            new_end = min(
                video_duration_sec if video_duration_sec else float('inf'),
                original_center + max_duration_sec / 2
            )
        
        # Create expanded segment (preserve type)
        if hasattr(seg, 'start_frame'):
            # VideoSegment - update both times and frames
            from .aligner import VideoSegment, time_to_frame
            
            expanded = VideoSegment(
                start_frame=time_to_frame(new_start, seg.fps),
                end_frame=time_to_frame(new_end, seg.fps),
                start_time=new_start,
                end_time=new_end,
                score=seg.score,
                fps=seg.fps,
                metadata=seg.metadata
            )
        else:
            # AudioSegment - update times only
            from .aligner import AudioSegment
            
            expanded = AudioSegment(
                start_time=new_start,
                end_time=new_end,
                score=seg.score,
                metadata=seg.metadata
            )
        
        expanded_segments.append(expanded)
    
    logger.info(
        f"Expanded segments: "
        f"total duration {sum(s.duration for s in segments):.1f}s → "
        f"{sum(s.duration for s in expanded_segments):.1f}s"
    )
    
    return expanded_segments


def merge_overlapping_segments(
    segments: List,
    merge_gap_threshold_sec: float = 2.0
) -> List:
    """
    Merge segments that overlap or are separated by small gaps.
    
    Clinical rationale:
    - Brief stable periods between unstable periods likely represent
      continuous dysregulation episode, not separate events
    - Reduces fragmentation for clearer clinical interpretation
    
    Engineering rationale:
    - Reduces number of video segments to process
    - Avoids redundant MediaPipe analysis of overlapping frames
    - Preserves highest instability score from merged segments
    
    Algorithm:
    1. Sort segments by start time
    2. Iterate, merging if gap < threshold or overlap exists
    3. Combined score = max(individual scores)
    
    Args:
        segments: List of VideoSegment or AudioSegment objects
        merge_gap_threshold_sec: Maximum gap to merge across (seconds)
        
    Returns:
        List of merged segments (fewer than input)
        
    Example:
        Input:
            Seg1: [40-50s, score=0.7]
            Seg2: [51-60s, score=0.8]  # 1s gap
        Output (with threshold=2s):
            Merged: [40-60s, score=0.8]
    """
    if not segments:
        return []
    
    if len(segments) == 1:
        return segments
    
    logger.info(
        f"Merging {len(segments)} segments "
        f"(gap threshold: {merge_gap_threshold_sec}s)"
    )
    
    # Sort by start time
    sorted_segments = sorted(segments, key=lambda s: s.start_time)
    
    merged = []
    current = sorted_segments[0]
    
    for next_seg in sorted_segments[1:]:
        # Calculate gap (negative if overlapping)
        gap = next_seg.start_time - current.end_time
        
        if gap <= merge_gap_threshold_sec:
            # Merge segments
            current = _merge_two_segments(current, next_seg)
        else:
            # Gap too large, save current and start new
            merged.append(current)
            current = next_seg
    
    # Add final segment
    merged.append(current)
    
    logger.info(
        f"Merged {len(segments)} → {len(merged)} segments "
        f"({len(segments) - len(merged)} merges, "
        f"{(1 - len(merged)/len(segments))*100:.1f}% reduction)"
    )
    
    return merged


def _merge_two_segments(seg1, seg2):
    """
    Merge two segments (internal helper).
    
    Takes earliest start time, latest end time, and highest score.
    """
    # Determine segment type
    if hasattr(seg1, 'start_frame'):
        # VideoSegment
        from .aligner import VideoSegment, time_to_frame
        
        new_start_time = min(seg1.start_time, seg2.start_time)
        new_end_time = max(seg1.end_time, seg2.end_time)
        
        merged = VideoSegment(
            start_frame=time_to_frame(new_start_time, seg1.fps),
            end_frame=time_to_frame(new_end_time, seg1.fps),
            start_time=new_start_time,
            end_time=new_end_time,
            score=max(seg1.score, seg2.score),
            fps=seg1.fps,
            metadata=seg1.metadata  # Keep first segment's metadata
        )
    else:
        # AudioSegment
        from .aligner import AudioSegment
        
        merged = AudioSegment(
            start_time=min(seg1.start_time, seg2.start_time),
            end_time=max(seg1.end_time, seg2.end_time),
            score=max(seg1.score, seg2.score),
            metadata=seg1.metadata
        )
    
    return merged


def prioritize_segments(
    segments: List,
    max_segments: int = None,
    min_score: float = 0.0
) -> List:
    """
    Prioritize segments for video analysis.
    
    Clinical rationale:
    - Limited computational budget → analyze highest-priority segments first
    - Highest instability scores → most likely dysregulation
    - Can process top N segments or all above threshold
    
    Args:
        segments: List of segments with scores
        max_segments: Maximum number of segments to return (None = all)
        min_score: Minimum score threshold (default 0.0)
        
    Returns:
        List of prioritized segments (sorted by score, descending)
        
    Example usage:
        # Get top 10 highest-scoring segments
        top_segments = prioritize_segments(all_segments, max_segments=10)
        
        # Get all segments with score > 0.6
        high_conf_segments = prioritize_segments(all_segments, min_score=0.6)
    """
    if not segments:
        return []
    
    # Filter by minimum score
    filtered = [seg for seg in segments if seg.score >= min_score]
    
    # Sort by score (descending)
    sorted_segments = sorted(filtered, key=lambda s: s.score, reverse=True)
    
    # Limit to max_segments if specified
    if max_segments is not None:
        sorted_segments = sorted_segments[:max_segments]
    
    logger.info(
        f"Prioritized segments: {len(segments)} → {len(sorted_segments)} "
        f"(min_score={min_score:.2f}"
        + (f", top {max_segments}" if max_segments else "")
        + ")"
    )
    
    return sorted_segments


def compute_coverage_statistics(segments: List) -> dict:
    """
    Compute coverage statistics for prioritized segments.
    
    Useful for understanding what proportion of the recording will be analyzed.
    
    Args:
        segments: List of segments
        
    Returns:
        Dictionary with statistics:
        - num_segments: Number of segments
        - total_duration: Total duration across all segments (seconds)
        - mean_duration: Mean segment duration
        - std_duration: Std dev of segment durations
        - min_score: Minimum score
        - max_score: Maximum score
        - mean_score: Mean score
    """
    if not segments:
        return {}
    
    durations = [seg.duration for seg in segments]
    scores = [seg.score for seg in segments]
    
    stats = {
        'num_segments': len(segments),
        'total_duration': sum(durations),
        'mean_duration': np.mean(durations),
        'std_duration': np.std(durations),
        'min_duration': min(durations),
        'max_duration': max(durations),
        'min_score': min(scores),
        'max_score': max(scores),
        'mean_score': np.mean(scores),
    }
    
    logger.info(
        f"Coverage stats: {stats['num_segments']} segments, "
        f"{stats['total_duration']:.1f}s total, "
        f"score range [{stats['min_score']:.2f}, {stats['max_score']:.2f}]"
    )
    
    return stats


def filter_short_segments(
    segments: List,
    min_duration_sec: float = 1.0
) -> List:
    """
    Remove segments shorter than minimum duration.
    
    Clinical rationale:
    - Very short segments (<1s) may not capture meaningful behavior
    - May be artifacts from audio processing
    
    Args:
        segments: List of segments
        min_duration_sec: Minimum duration threshold
        
    Returns:
        Filtered list
    """
    filtered = [seg for seg in segments if seg.duration >= min_duration_sec]
    
    if len(filtered) < len(segments):
        logger.info(
            f"Filtered {len(segments) - len(filtered)} segments "
            f"shorter than {min_duration_sec}s"
        )
    
    return filtered
