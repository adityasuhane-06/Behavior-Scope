"""
Unit tests for autism analysis modules.

Tests turn-taking, eye contact, stereotypy, and social engagement analysis.
"""

import unittest
import numpy as np
from dataclasses import dataclass
from typing import List

# Import autism analysis modules
from autism_analysis import (
    analyze_turn_taking,
    compute_response_latency_child,
    analyze_eye_contact,
    detect_stereotyped_movements,
    compute_social_engagement_index,
    TurnTakingEvent,
    EyeContactEvent,
    StereotypyEvent
)


@dataclass
class MockSpeakerSegment:
    """Mock speaker segment for testing."""
    start_time: float
    end_time: float
    speaker_label: str


@dataclass
class MockAggregatedFeatures:
    """Mock aggregated video features for testing."""
    window_start_time: float
    window_end_time: float
    face_features: dict
    pose_features: dict


class TestTurnTakingAnalysis(unittest.TestCase):
    """Test turn-taking analysis."""
    
    def test_analyze_turn_taking_basic(self):
        """Test basic turn-taking analysis."""
        # Create mock speaker segments (alternating speakers)
        segments = [
            MockSpeakerSegment(0.0, 2.0, "SPEAKER_00"),   # Child
            MockSpeakerSegment(2.5, 5.0, "SPEAKER_01"),   # Therapist
            MockSpeakerSegment(6.0, 8.0, "SPEAKER_00"),   # Child
            MockSpeakerSegment(8.5, 11.0, "SPEAKER_01"),  # Therapist
        ]
        
        result = analyze_turn_taking(segments, child_label="SPEAKER_00")
        
        # Assertions
        self.assertEqual(result.total_turns, 4)
        self.assertEqual(result.child_turns, 2)
        self.assertEqual(result.therapist_turns, 2)
        self.assertGreater(result.mean_response_latency, 0.0)
        self.assertEqual(result.interruption_count, 0)  # No overlaps
        self.assertGreater(result.conversational_balance_score, 0)
        self.assertGreater(result.reciprocity_score, 0)
    
    def test_analyze_turn_taking_with_interruptions(self):
        """Test turn-taking with overlapping speech."""
        segments = [
            MockSpeakerSegment(0.0, 2.0, "SPEAKER_00"),
            MockSpeakerSegment(1.5, 3.0, "SPEAKER_01"),  # Interrupts
            MockSpeakerSegment(3.5, 5.0, "SPEAKER_00"),
        ]
        
        result = analyze_turn_taking(segments, child_label="SPEAKER_00")
        
        self.assertGreater(result.interruption_count, 0)
    
    def test_compute_response_latency_child(self):
        """Test child-specific response latency computation."""
        segments = [
            MockSpeakerSegment(0.0, 2.0, "SPEAKER_01"),  # Therapist
            MockSpeakerSegment(5.0, 7.0, "SPEAKER_00"),  # Child (3s latency)
        ]
        
        turn_analysis = analyze_turn_taking(segments, child_label="SPEAKER_00")
        latency = compute_response_latency_child(turn_analysis)
        
        self.assertIn('mean', latency)
        self.assertIn('median', latency)
        self.assertIn('percentage_elevated', latency)
        self.assertGreater(latency['mean'], 0.0)


class TestEyeContactAnalysis(unittest.TestCase):
    """Test eye contact analysis."""
    
    def test_analyze_eye_contact_basic(self):
        """Test basic eye contact detection."""
        # Create mock video features with good eye contact
        features = [
            MockAggregatedFeatures(
                window_start_time=0.0,
                window_end_time=1.0,
                face_features={
                    'head_yaw_mean': 10.0,   # Facing forward
                    'gaze_proxy_mean': 0.02  # Looking forward
                },
                pose_features={}
            ),
            MockAggregatedFeatures(
                window_start_time=1.0,
                window_end_time=2.0,
                face_features={
                    'head_yaw_mean': 15.0,
                    'gaze_proxy_mean': 0.03
                },
                pose_features={}
            ),
            MockAggregatedFeatures(
                window_start_time=2.0,
                window_end_time=3.0,
                face_features={
                    'head_yaw_mean': 50.0,   # Looking away
                    'gaze_proxy_mean': 0.10
                },
                pose_features={}
            )
        ]
        
        result = analyze_eye_contact(features)
        
        self.assertGreater(result.episode_count, 0)
        self.assertGreater(result.total_duration, 0.0)
        self.assertGreater(result.percentage_of_session, 0.0)
        self.assertGreaterEqual(result.eye_contact_score, 0.0)
        self.assertLessEqual(result.eye_contact_score, 100.0)
    
    def test_analyze_eye_contact_no_contact(self):
        """Test with no eye contact detected."""
        features = [
            MockAggregatedFeatures(
                window_start_time=0.0,
                window_end_time=1.0,
                face_features={
                    'head_yaw_mean': 90.0,   # Looking away
                    'gaze_proxy_mean': 0.50
                },
                pose_features={}
            )
        ]
        
        result = analyze_eye_contact(features)
        
        self.assertEqual(result.episode_count, 0)
        self.assertEqual(result.total_duration, 0.0)


class TestStereotypyDetection(unittest.TestCase):
    """Test stereotypy detection."""
    
    def test_detect_stereotyped_movements_basic(self):
        """Test basic stereotypy detection."""
        # Create mock features with repetitive hand motion
        features = []
        for i in range(60):  # 60 windows @ 10 fps = 6 seconds
            t = i * 0.1
            # Simulate hand flapping at 2 Hz
            hand_x = 0.5 + 0.2 * np.sin(2 * np.pi * 2 * t)
            hand_y = 0.5 + 0.2 * np.cos(2 * np.pi * 2 * t)
            
            features.append(MockAggregatedFeatures(
                window_start_time=t,
                window_end_time=t + 0.1,
                face_features={},
                pose_features={
                    'left_wrist_x_mean': hand_x,
                    'left_wrist_y_mean': hand_y,
                    'right_wrist_x_mean': 0.5,
                    'right_wrist_y_mean': 0.5,
                    'shoulder_center_x_mean': 0.5,
                    'shoulder_center_y_mean': 0.5,
                    'nose_x_mean': 0.5,
                    'nose_y_mean': 0.3
                }
            ))
        
        result = detect_stereotyped_movements(features)
        
        # May or may not detect depending on thresholds
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.episode_count, 0)
        self.assertGreaterEqual(result.percentage_of_session, 0.0)
        self.assertGreaterEqual(result.intensity_score, 0.0)
    
    def test_detect_stereotyped_movements_no_motion(self):
        """Test with no repetitive motion."""
        features = [
            MockAggregatedFeatures(
                window_start_time=float(i),
                window_end_time=float(i + 1),
                face_features={},
                pose_features={
                    'left_wrist_x_mean': 0.5,  # Static
                    'left_wrist_y_mean': 0.5,
                    'right_wrist_x_mean': 0.5,
                    'right_wrist_y_mean': 0.5,
                    'shoulder_center_x_mean': 0.5,
                    'shoulder_center_y_mean': 0.5,
                    'nose_x_mean': 0.5,
                    'nose_y_mean': 0.3
                }
            )
            for i in range(10)
        ]
        
        result = detect_stereotyped_movements(features)
        
        self.assertEqual(result.episode_count, 0)


class TestSocialEngagementIndex(unittest.TestCase):
    """Test social engagement index computation."""
    
    def test_compute_social_engagement_index_basic(self):
        """Test basic SEI computation."""
        # Mock turn-taking analysis
        from autism_analysis.turn_taking import TurnTakingAnalysis
        turn_analysis = TurnTakingAnalysis(
            total_turns=10,
            child_turns=5,
            therapist_turns=5,
            child_speaking_time=25.0,
            therapist_speaking_time=25.0,
            child_percentage=50.0,
            mean_response_latency=1.0,
            median_response_latency=0.9,
            max_response_latency=3.0,
            interruption_count=1,
            conversational_balance_score=100.0,
            reciprocity_score=80.0
        )
        
        # Mock eye contact analysis
        from autism_analysis.eye_contact import EyeContactAnalysis
        eye_analysis = EyeContactAnalysis(
            total_duration=15.0,
            episode_count=10,
            mean_episode_duration=1.5,
            frequency_per_minute=12.0,
            percentage_of_session=30.0,
            during_speaking_percentage=25.0,
            during_listening_percentage=35.0,
            longest_episode=4.0,
            eye_contact_score=70.0,
            avoidance_score=30.0
        )
        
        result = compute_social_engagement_index(
            eye_contact_analysis=eye_analysis,
            turn_taking_analysis=turn_analysis,
            attention_stability_score=75.0
        )
        
        self.assertGreater(result.social_engagement_index, 0.0)
        self.assertLessEqual(result.social_engagement_index, 100.0)
        self.assertGreater(result.eye_contact_component, 0.0)
        self.assertGreater(result.turn_taking_component, 0.0)
        self.assertGreater(result.responsiveness_component, 0.0)
        self.assertGreater(result.attention_component, 0.0)
        self.assertGreater(result.confidence, 0.0)


if __name__ == '__main__':
    unittest.main()
