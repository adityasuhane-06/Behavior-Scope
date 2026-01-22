"""
Unit tests for video pipeline modules.

Tests cover:
- Face analysis
- Pose analysis
- Temporal aggregation
- Video I/O utilities
- Edge cases and error handling
"""

import pytest # pyright: ignore[reportMissingImports]
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_pipeline.face_analyzer import FaceFeatures
from video_pipeline.pose_analyzer import PoseFeatures
from video_pipeline.temporal_agg import (
    TemporalAggregator,
    compute_sliding_window_stats,
    detect_motion_bursts
)


class TestFaceFeatures:
    """Test FaceFeatures dataclass."""
    
    def test_creation(self):
        """Test features creation."""
        features = FaceFeatures(
            frame_idx=10,
            timestamp=0.5,
            face_detected=True,
            head_pose=(10.5, -5.2, 2.1),
            facial_motion_energy=0.15,
            gaze_proxy=0.08,
            landmark_confidence=0.92,
            num_landmarks=468
        )
        
        assert features.frame_idx == 10
        assert features.face_detected is True
        assert features.head_pose[0] == 10.5


class TestPoseFeatures:
    """Test PoseFeatures dataclass."""
    
    def test_creation(self):
        """Test features creation."""
        features = PoseFeatures(
            frame_idx=10,
            timestamp=0.5,
            pose_detected=True,
            upper_body_motion=0.12,
            hand_velocity_left=15.3,
            hand_velocity_right=12.8,
            shoulder_stability=0.25,
            posture_angle=5.2,
            visibility_score=0.88,
            num_landmarks=33
        )
        
        assert features.pose_detected is True
        assert features.upper_body_motion == 0.12


class TestTemporalAggregator:
    """Test temporal aggregation."""
    
    def create_mock_face_features(self, num_frames: int, fps: float = 5.0) -> list:
        """Create mock face features for testing."""
        features = []
        for i in range(num_frames):
            features.append(FaceFeatures(
                frame_idx=i,
                timestamp=i / fps,
                face_detected=True,
                head_pose=(np.random.randn() * 10, np.random.randn() * 10, np.random.randn() * 5),
                facial_motion_energy=np.random.rand() * 0.2,
                gaze_proxy=np.random.rand() * 0.1,
                landmark_confidence=0.9 + np.random.rand() * 0.1,
                num_landmarks=468
            ))
        return features
    
    def create_mock_pose_features(self, num_frames: int, fps: float = 5.0) -> list:
        """Create mock pose features for testing."""
        features = []
        for i in range(num_frames):
            features.append(PoseFeatures(
                frame_idx=i,
                timestamp=i / fps,
                pose_detected=True,
                upper_body_motion=np.random.rand() * 0.2,
                hand_velocity_left=np.random.rand() * 20,
                hand_velocity_right=np.random.rand() * 20,
                shoulder_stability=0.2 + np.random.rand() * 0.1,
                posture_angle=np.random.randn() * 10,
                visibility_score=0.8 + np.random.rand() * 0.2,
                num_landmarks=33
            ))
        return features
    
    def test_aggregation_basic(self):
        """Test basic temporal aggregation."""
        # Create 50 frames (10 seconds at 5 FPS)
        face_features = self.create_mock_face_features(50, fps=5.0)
        pose_features = self.create_mock_pose_features(50, fps=5.0)
        
        aggregator = TemporalAggregator(
            window_duration=5.0,
            hop_duration=2.5
        )
        
        aggregated = aggregator.aggregate(face_features, pose_features, fps=5.0)
        
        # Should have windows
        assert len(aggregated) > 0
        
        # Check first window
        first_window = aggregated[0]
        assert first_window.window_start_time == 0.0
        assert first_window.window_end_time == 5.0
        assert first_window.num_frames > 0
        assert 'head_yaw_mean' in first_window.face_features
        assert 'upper_body_motion_mean' in first_window.pose_features
    
    def test_aggregation_functions(self):
        """Test different aggregation functions."""
        face_features = self.create_mock_face_features(25, fps=5.0)
        pose_features = self.create_mock_pose_features(25, fps=5.0)
        
        aggregator = TemporalAggregator(
            window_duration=5.0,
            hop_duration=5.0,
            aggregation_functions=['mean', 'std', 'max', 'min']
        )
        
        aggregated = aggregator.aggregate(face_features, pose_features, fps=5.0)
        
        if aggregated:
            first_window = aggregated[0]
            # Check all functions present
            assert 'head_yaw_mean' in first_window.face_features
            assert 'head_yaw_std' in first_window.face_features
            assert 'head_yaw_max' in first_window.face_features
            assert 'head_yaw_min' in first_window.face_features


class TestSlidingWindowStats:
    """Test sliding window statistics."""
    
    def test_sliding_window_mean(self):
        """Test sliding window mean computation."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        window_size = 3
        
        stats = compute_sliding_window_stats(values, window_size, np.mean)
        
        # Should have 8 windows (10 - 3 + 1)
        assert len(stats) == 8
        
        # Check first window mean
        assert stats[0] == 2.0  # mean of [1, 2, 3]
    
    def test_sliding_window_std(self):
        """Test sliding window std computation."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        window_size = 3
        
        stats = compute_sliding_window_stats(values, window_size, np.std)
        
        assert len(stats) == 3


class TestMotionBurstDetection:
    """Test motion burst detection."""
    
    def test_detect_single_burst(self):
        """Test detection of single motion burst."""
        # Create signal with burst
        motion = np.array([0.1, 0.1, 0.1, 0.5, 0.6, 0.7, 0.1, 0.1])
        
        bursts = detect_motion_bursts(
            motion,
            threshold_multiplier=2.0,
            min_burst_duration=2
        )
        
        # Should detect one burst
        assert len(bursts) >= 1
    
    def test_no_burst_detected(self):
        """Test no detection when motion is low."""
        motion = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        
        bursts = detect_motion_bursts(
            motion,
            threshold_multiplier=5.0,
            min_burst_duration=2
        )
        
        # Should detect no bursts
        assert len(bursts) == 0
    
    def test_min_duration_filter(self):
        """Test that short bursts are filtered out."""
        motion = np.array([0.1, 0.1, 0.5, 0.1, 0.1])  # 1-frame burst
        
        bursts = detect_motion_bursts(
            motion,
            threshold_multiplier=2.0,
            min_burst_duration=2  # Require 2 frames
        )
        
        # Should not detect 1-frame burst
        assert len(bursts) == 0


class TestVideoIOUtilities:
    """Test video I/O utilities."""
    
    def test_resize_frame(self):
        """Test frame resizing."""
        from utils.video_io import resize_frame
        
        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Resize without keeping aspect ratio
        resized = resize_frame(frame, (320, 240), keep_aspect_ratio=False)
        
        assert resized.shape == (240, 320, 3)
    
    def test_resize_with_aspect_ratio(self):
        """Test frame resizing with aspect ratio preservation."""
        from utils.video_io import resize_frame
        
        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Resize with aspect ratio preservation
        resized = resize_frame(frame, (320, 320), keep_aspect_ratio=True)
        
        # Should be padded to 320x320
        assert resized.shape == (320, 320, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
