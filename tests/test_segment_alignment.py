"""
Unit tests for segment alignment module.

Tests cover:
- Audio-to-video timestamp conversion
- Temporal window expansion
- Segment merging logic
- Edge cases (boundary conditions, empty inputs)
"""

import pytest # pyright: ignore[reportMissingImports]
import numpy as np
from pathlib import Path

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from segment_alignment.aligner import (
    AudioSegment,
    VideoSegment,
    time_to_frame,
    frame_to_time,
    align_audio_to_video
)
from segment_alignment.window_expander import (
    expand_temporal_windows,
    merge_overlapping_segments,
    prioritize_segments,
    filter_short_segments
)


class TestTimeConversion:
    """Test time <-> frame conversion functions."""
    
    def test_time_to_frame_basic(self):
        """Test basic time to frame conversion."""
        assert time_to_frame(0.0, 30.0) == 0
        assert time_to_frame(1.0, 30.0) == 30
        assert time_to_frame(10.5, 30.0) == 315
    
    def test_frame_to_time_basic(self):
        """Test basic frame to time conversion."""
        assert frame_to_time(0, 30.0) == 0.0
        assert frame_to_time(30, 30.0) == 1.0
        assert frame_to_time(315, 30.0) == 10.5
    
    def test_roundtrip_conversion(self):
        """Test roundtrip time -> frame -> time."""
        fps = 30.0
        times = [0.0, 1.5, 10.25, 45.7]
        
        for t in times:
            frame = time_to_frame(t, fps)
            t_reconstructed = frame_to_time(frame, fps)
            assert abs(t - t_reconstructed) < 1.0 / fps  # Within 1 frame


class TestAudioSegment:
    """Test AudioSegment dataclass."""
    
    def test_creation(self):
        """Test segment creation."""
        seg = AudioSegment(10.0, 20.0, score=0.75)
        assert seg.start_time == 10.0
        assert seg.end_time == 20.0
        assert seg.score == 0.75
        assert seg.duration == 10.0
    
    def test_duration_property(self):
        """Test duration property calculation."""
        seg = AudioSegment(5.0, 15.5)
        assert seg.duration == 10.5


class TestWindowExpansion:
    """Test temporal window expansion."""
    
    def test_basic_expansion(self):
        """Test basic window expansion."""
        segments = [
            AudioSegment(10.0, 15.0, score=0.8)
        ]
        
        expanded = expand_temporal_windows(
            segments,
            pre_context_sec=2.0,
            post_context_sec=3.0
        )
        
        assert len(expanded) == 1
        assert expanded[0].start_time == 8.0  # 10 - 2
        assert expanded[0].end_time == 18.0   # 15 + 3
        assert expanded[0].score == 0.8
    
    def test_expansion_clamping_to_zero(self):
        """Test that expansion doesn't go negative."""
        segments = [
            AudioSegment(1.0, 3.0, score=0.7)
        ]
        
        expanded = expand_temporal_windows(
            segments,
            pre_context_sec=5.0  # Would go to -4.0
        )
        
        assert expanded[0].start_time == 0.0  # Clamped to 0
    
    def test_expansion_with_max_duration(self):
        """Test max duration capping."""
        segments = [
            AudioSegment(10.0, 20.0)  # 10s duration
        ]
        
        expanded = expand_temporal_windows(
            segments,
            pre_context_sec=10.0,
            post_context_sec=10.0,
            max_duration_sec=15.0  # Would be 30s without cap
        )
        
        assert expanded[0].duration <= 15.0


class TestSegmentMerging:
    """Test segment merging logic."""
    
    def test_merge_overlapping(self):
        """Test merging overlapping segments."""
        segments = [
            AudioSegment(10.0, 20.0, score=0.7),
            AudioSegment(15.0, 25.0, score=0.8)  # Overlaps with first
        ]
        
        merged = merge_overlapping_segments(segments, merge_gap_threshold_sec=0.0)
        
        assert len(merged) == 1
        assert merged[0].start_time == 10.0
        assert merged[0].end_time == 25.0
        assert merged[0].score == 0.8  # Takes max score
    
    def test_merge_nearby_gaps(self):
        """Test merging segments with small gaps."""
        segments = [
            AudioSegment(10.0, 15.0, score=0.7),
            AudioSegment(16.0, 20.0, score=0.6)  # 1s gap
        ]
        
        merged = merge_overlapping_segments(segments, merge_gap_threshold_sec=2.0)
        
        assert len(merged) == 1
        assert merged[0].start_time == 10.0
        assert merged[0].end_time == 20.0
    
    def test_no_merge_large_gaps(self):
        """Test that large gaps prevent merging."""
        segments = [
            AudioSegment(10.0, 15.0, score=0.7),
            AudioSegment(20.0, 25.0, score=0.6)  # 5s gap
        ]
        
        merged = merge_overlapping_segments(segments, merge_gap_threshold_sec=2.0)
        
        assert len(merged) == 2  # Should not merge
    
    def test_merge_multiple_segments(self):
        """Test merging chain of segments."""
        segments = [
            AudioSegment(10.0, 15.0, score=0.7),
            AudioSegment(16.0, 20.0, score=0.8),
            AudioSegment(21.0, 25.0, score=0.6)
        ]
        
        merged = merge_overlapping_segments(segments, merge_gap_threshold_sec=2.0)
        
        assert len(merged) == 1  # All merge into one
        assert merged[0].start_time == 10.0
        assert merged[0].end_time == 25.0
        assert merged[0].score == 0.8  # Max score


class TestSegmentPrioritization:
    """Test segment prioritization."""
    
    def test_sort_by_score(self):
        """Test sorting by score."""
        segments = [
            AudioSegment(10.0, 15.0, score=0.5),
            AudioSegment(20.0, 25.0, score=0.9),
            AudioSegment(30.0, 35.0, score=0.7)
        ]
        
        prioritized = prioritize_segments(segments)
        
        assert len(prioritized) == 3
        assert prioritized[0].score == 0.9
        assert prioritized[1].score == 0.7
        assert prioritized[2].score == 0.5
    
    def test_max_segments_limit(self):
        """Test limiting to top N segments."""
        segments = [
            AudioSegment(i*10, i*10+5, score=i*0.1)
            for i in range(10)
        ]
        
        prioritized = prioritize_segments(segments, max_segments=3)
        
        assert len(prioritized) == 3
        assert all(seg.score >= 0.7 for seg in prioritized)
    
    def test_min_score_filter(self):
        """Test filtering by minimum score."""
        segments = [
            AudioSegment(10.0, 15.0, score=0.3),
            AudioSegment(20.0, 25.0, score=0.7),
            AudioSegment(30.0, 35.0, score=0.9)
        ]
        
        prioritized = prioritize_segments(segments, min_score=0.6)
        
        assert len(prioritized) == 2
        assert all(seg.score >= 0.6 for seg in prioritized)


class TestFilterShortSegments:
    """Test filtering of short segments."""
    
    def test_filter_short(self):
        """Test filtering segments below threshold."""
        segments = [
            AudioSegment(10.0, 10.5, score=0.8),  # 0.5s - too short
            AudioSegment(20.0, 22.0, score=0.7),  # 2.0s - OK
            AudioSegment(30.0, 30.8, score=0.9)   # 0.8s - too short
        ]
        
        filtered = filter_short_segments(segments, min_duration_sec=1.0)
        
        assert len(filtered) == 1
        assert filtered[0].start_time == 20.0
    
    def test_no_filtering_if_all_long(self):
        """Test that long segments pass through."""
        segments = [
            AudioSegment(10.0, 15.0, score=0.8),
            AudioSegment(20.0, 25.0, score=0.7)
        ]
        
        filtered = filter_short_segments(segments, min_duration_sec=1.0)
        
        assert len(filtered) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
