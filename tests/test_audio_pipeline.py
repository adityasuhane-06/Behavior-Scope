
import pytest # pyright: ignore[reportMissingImports]
import numpy as np
from pathlib import Path

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_pipeline.vad import SpeechSegment
from audio_pipeline.diarization import SpeakerSegment
from audio_pipeline.prosody import ProsodicFeatures
from audio_pipeline.instability import InstabilityWindow, _compute_zscore


class TestSpeechSegment:
    """Test SpeechSegment dataclass."""
    
    def test_creation(self):
        """Test segment creation."""
        seg = SpeechSegment(1.5, 3.2, 0.95)
        assert seg.start_time == 1.5
        assert seg.end_time == 3.2
        assert seg.confidence == 0.95
    
    def test_duration_property(self):
        """Test duration calculation."""
        seg = SpeechSegment(10.0, 15.5, 0.9)
        assert seg.duration == 5.5


class TestSpeakerSegment:
    """Test SpeakerSegment dataclass."""
    
    def test_creation(self):
        """Test segment creation."""
        seg = SpeakerSegment(5.0, 10.0, "SPEAKER_00", is_patient=True)
        assert seg.speaker_id == "SPEAKER_00"
        assert seg.is_patient is True
        assert seg.duration == 5.0


class TestProsodicFeatures:
    """Test ProsodicFeatures dataclass."""
    
    def test_creation(self):
        """Test feature creation with all fields."""
        features = ProsodicFeatures(
            start_time=0.0,
            end_time=5.0,
            speech_rate=4.5,
            pause_count=3,
            pause_duration_mean=0.5,
            pause_duration_std=0.2,
            pitch_mean=180.0,
            pitch_std=35.0,
            pitch_range=120.0,
            energy_mean=-20.0,
            energy_std=5.0,
            speaking_ratio=0.75
        )
        
        assert features.speech_rate == 4.5
        assert features.pitch_mean == 180.0


class TestInstabilityWindow:
    """Test InstabilityWindow dataclass."""
    
    def test_creation(self):
        """Test window creation."""
        window = InstabilityWindow(
            start_time=45.0,
            end_time=52.0,
            instability_score=0.78,
            contributing_features={'speech_rate': 2.3},
            explanation="Speech rate elevated"
        )
        
        assert window.duration == 7.0
        assert window.instability_score == 0.78


class TestZScoreComputation:
    """Test z-score computation."""
    
    def test_basic_zscore(self):
        """Test basic z-score calculation."""
        z = _compute_zscore(value=15.0, mean=10.0, std=2.0)
        assert z == 2.5
    
    def test_negative_zscore(self):
        """Test negative z-score."""
        z = _compute_zscore(value=5.0, mean=10.0, std=2.0)
        assert z == -2.5
    
    def test_zero_std_protection(self):
        """Test that zero std doesn't cause division by zero."""
        z = _compute_zscore(value=10.0, mean=10.0, std=0.0)
        assert z == 0.0
    
    def test_at_mean(self):
        """Test z-score when value equals mean."""
        z = _compute_zscore(value=10.0, mean=10.0, std=2.0)
        assert z == 0.0


class TestAudioProcessing:
    """Test audio processing utilities."""
    
    def test_synthetic_audio_generation(self):
        """Test generating synthetic audio for testing."""
        # Generate 1 second of silence
        sample_rate = 16000
        duration = 1.0
        audio = np.zeros(int(sample_rate * duration))
        
        assert len(audio) == sample_rate
        assert audio.dtype == np.float64
    
    def test_audio_normalization(self):
        """Test audio normalization."""
        from utils.audio_io import normalize_audio
        
        # Generate audio with known amplitude
        audio = np.random.randn(16000) * 0.1
        
        normalized = normalize_audio(audio, target_db=-20.0)
        
        # Check that normalization doesn't introduce NaN or inf
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))
        
        # Check that max value is within range
        assert np.max(np.abs(normalized)) <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
