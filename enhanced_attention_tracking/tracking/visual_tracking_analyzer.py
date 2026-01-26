"""
Visual Tracking Analyzer implementation for Enhanced Eye Contact & Attention Tracking System.

This module provides eye movement pattern analysis, fixation and saccade detection,
and repetitive behavior identification.
"""

import logging
from typing import List, Dict, Optional, Any
import numpy as np
from scipy import signal
from collections import deque

from ..core.interfaces import VisualTrackingAnalyzer
from ..core.data_models import GazeVector, VisualTrackingData
from ..core.enums import ScanningPattern

logger = logging.getLogger(__name__)


class VisualTrackingAnalyzerImpl(VisualTrackingAnalyzer):
    """
    Implementation of visual tracking pattern analysis.
    
    Analyzes eye movement patterns including saccades, fixations, scanning strategies,
    and repetitive behaviors.
    """
    
    def __init__(self):
        """Initialize the visual tracking analyzer."""
        self.movement_history: deque = deque(maxlen=1000)  # Store recent movements
        
        # Analysis parameters
        self.fixation_velocity_threshold = 30.0  # degrees/second
        self.saccade_velocity_threshold = 100.0  # degrees/second
        self.minimum_fixation_duration = 0.1  # seconds
        self.repetitive_pattern_threshold = 0.7  # similarity threshold for repetitive patterns
        
        # Scanning pattern analysis
        self.central_region_radius = 0.3  # Normalized radius for central region
        self.peripheral_threshold = 0.7  # Threshold for peripheral bias
        
        logger.info("Visual Tracking Analyzer initialized")
    
    def analyze_eye_movements(self, gaze_sequence: List[GazeVector], 
                             timestamps: List[float]) -> VisualTrackingData:
        """
        Analyze eye movement patterns from gaze sequence.
        
        Args:
            gaze_sequence: Sequence of gaze vectors
            timestamps: Corresponding timestamps
            
        Returns:
            VisualTrackingData with movement analysis results
        """
        try:
            if len(gaze_sequence) < 2 or len(timestamps) < 2:
                return self._create_empty_tracking_data(timestamps[0] if timestamps else 0.0)
            
            # Calculate velocities and accelerations
            velocities = self._calculate_eye_movement_velocities(gaze_sequence, timestamps)
            
            # Detect fixations and saccades
            fixation_metrics = self.calculate_fixation_metrics(gaze_sequence)
            saccade_metrics = self.calculate_saccade_metrics(gaze_sequence)
            
            # Analyze scanning patterns
            scanning_pattern = self.detect_scanning_patterns(gaze_sequence, [])
            
            # Detect repetitive behaviors
            repetitive_score = self.identify_repetitive_behaviors(gaze_sequence)
            
            # Calculate attention stability
            attention_stability = self.assess_attention_stability(gaze_sequence, timestamps[-1] - timestamps[0])
            
            # Create tracking data
            tracking_data = VisualTrackingData(
                timestamp=timestamps[0],
                window_duration=timestamps[-1] - timestamps[0],
                eye_movement_velocity=np.mean(velocities) if velocities else 0.0,
                saccade_count=saccade_metrics.get('saccade_count', 0),
                fixation_duration=fixation_metrics.get('average_duration', 0.0),
                scanning_pattern=scanning_pattern,
                repetitive_behavior_score=repetitive_score,
                attention_stability=attention_stability
            )
            
            # Store in movement history
            self.movement_history.append({
                'timestamp': timestamps[0],
                'tracking_data': tracking_data,
                'gaze_sequence': gaze_sequence
            })
            
            return tracking_data
            
        except Exception as e:
            logger.error(f"Error analyzing eye movements: {e}")
            return self._create_empty_tracking_data(timestamps[0] if timestamps else 0.0)
    
    def _calculate_eye_movement_velocities(self, gaze_sequence: List[GazeVector], 
                                         timestamps: List[float]) -> List[float]:
        """Calculate eye movement velocities between consecutive gaze points."""
        try:
            velocities = []
            
            for i in range(1, len(gaze_sequence)):
                prev_gaze = gaze_sequence[i-1]
                curr_gaze = gaze_sequence[i]
                
                # Calculate angular distance
                angular_distance = prev_gaze.angle_to(curr_gaze)
                
                # Calculate time difference
                time_diff = timestamps[i] - timestamps[i-1]
                
                if time_diff > 0:
                    # Convert to degrees per second
                    velocity = np.degrees(angular_distance) / time_diff
                    velocities.append(velocity)
            
            return velocities
            
        except Exception as e:
            logger.debug(f"Error calculating velocities: {e}")
            return []
    
    def detect_scanning_patterns(self, gaze_path: List[GazeVector], 
                                attention_zones: List[Any]) -> ScanningPattern:
        """
        Identify the dominant scanning pattern.
        
        Args:
            gaze_path: Sequence of gaze vectors
            attention_zones: Defined attention zones (optional)
            
        Returns:
            ScanningPattern classification
        """
        try:
            if len(gaze_path) < 5:
                return ScanningPattern.RANDOM
            
            # Extract 2D coordinates for analysis
            x_coords = [gv.x for gv in gaze_path]
            y_coords = [gv.y for gv in gaze_path]
            
            # Analyze spatial distribution
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)
            
            # Calculate center bias
            center_distances = [np.sqrt(gv.x**2 + gv.y**2) for gv in gaze_path]
            avg_center_distance = np.mean(center_distances)
            
            # Analyze movement patterns
            movement_regularity = self._calculate_movement_regularity(gaze_path)
            repetitive_score = self._calculate_pattern_repetitiveness(gaze_path)
            
            # Classification logic
            if repetitive_score > self.repetitive_pattern_threshold:
                return ScanningPattern.REPETITIVE
            
            elif avg_center_distance < self.central_region_radius:
                return ScanningPattern.CENTRAL_BIAS
            
            elif avg_center_distance > self.peripheral_threshold:
                return ScanningPattern.PERIPHERAL_BIAS
            
            elif movement_regularity > 0.7:
                return ScanningPattern.SYSTEMATIC
            
            elif x_range < 0.2 and y_range < 0.2:
                return ScanningPattern.CENTRAL_BIAS
            
            else:
                return ScanningPattern.RANDOM
                
        except Exception as e:
            logger.error(f"Error detecting scanning patterns: {e}")
            return ScanningPattern.RANDOM
    
    def _calculate_movement_regularity(self, gaze_path: List[GazeVector]) -> float:
        """Calculate how regular/systematic the movement pattern is."""
        try:
            if len(gaze_path) < 3:
                return 0.0
            
            # Calculate direction changes
            direction_changes = []
            for i in range(2, len(gaze_path)):
                prev_vec = np.array([gaze_path[i-1].x - gaze_path[i-2].x,
                                   gaze_path[i-1].y - gaze_path[i-2].y])
                curr_vec = np.array([gaze_path[i].x - gaze_path[i-1].x,
                                   gaze_path[i].y - gaze_path[i-1].y])
                
                # Calculate angle between movement vectors
                if np.linalg.norm(prev_vec) > 0 and np.linalg.norm(curr_vec) > 0:
                    cos_angle = np.dot(prev_vec, curr_vec) / (np.linalg.norm(prev_vec) * np.linalg.norm(curr_vec))
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    direction_changes.append(angle)
            
            if not direction_changes:
                return 0.0
            
            # Regular movement has consistent direction changes
            regularity = 1.0 - (np.std(direction_changes) / np.pi)
            return max(0.0, regularity)
            
        except Exception as e:
            logger.debug(f"Error calculating movement regularity: {e}")
            return 0.0
    
    def _calculate_pattern_repetitiveness(self, gaze_path: List[GazeVector]) -> float:
        """Calculate how repetitive the gaze pattern is."""
        try:
            if len(gaze_path) < 6:
                return 0.0
            
            # Look for repeating subsequences
            path_length = len(gaze_path)
            max_pattern_length = min(10, path_length // 3)
            
            best_repetition_score = 0.0
            
            for pattern_length in range(2, max_pattern_length + 1):
                for start_idx in range(path_length - pattern_length * 2):
                    pattern = gaze_path[start_idx:start_idx + pattern_length]
                    
                    # Look for repetitions of this pattern
                    repetitions = 0
                    for check_idx in range(start_idx + pattern_length, path_length - pattern_length + 1):
                        candidate = gaze_path[check_idx:check_idx + pattern_length]
                        
                        # Calculate similarity between pattern and candidate
                        similarity = self._calculate_pattern_similarity(pattern, candidate)
                        
                        if similarity > 0.8:  # High similarity threshold
                            repetitions += 1
                    
                    # Calculate repetition score
                    if repetitions > 0:
                        repetition_score = (repetitions * pattern_length) / path_length
                        best_repetition_score = max(best_repetition_score, repetition_score)
            
            return min(1.0, best_repetition_score)
            
        except Exception as e:
            logger.debug(f"Error calculating pattern repetitiveness: {e}")
            return 0.0
    
    def _calculate_pattern_similarity(self, pattern1: List[GazeVector], 
                                    pattern2: List[GazeVector]) -> float:
        """Calculate similarity between two gaze patterns."""
        try:
            if len(pattern1) != len(pattern2):
                return 0.0
            
            similarities = []
            for gv1, gv2 in zip(pattern1, pattern2):
                # Calculate angular similarity
                angle = gv1.angle_to(gv2)
                similarity = 1.0 - (angle / np.pi)  # Convert to 0-1 scale
                similarities.append(max(0.0, similarity))
            
            return np.mean(similarities)
            
        except Exception as e:
            logger.debug(f"Error calculating pattern similarity: {e}")
            return 0.0
    
    def calculate_fixation_metrics(self, gaze_sequence: List[GazeVector]) -> Dict[str, float]:
        """
        Calculate fixation duration and frequency metrics.
        
        Args:
            gaze_sequence: Sequence of gaze vectors
            
        Returns:
            Dictionary with fixation metrics
        """
        try:
            if len(gaze_sequence) < 2:
                return {
                    'fixation_count': 0,
                    'average_duration': 0.0,
                    'total_fixation_time': 0.0,
                    'fixation_rate': 0.0
                }
            
            # Detect fixations based on low movement velocity
            fixations = []
            current_fixation_start = None
            current_fixation_positions = []
            
            for i in range(1, len(gaze_sequence)):
                prev_gaze = gaze_sequence[i-1]
                curr_gaze = gaze_sequence[i]
                
                # Calculate movement velocity (simplified)
                angular_distance = prev_gaze.angle_to(curr_gaze)
                time_diff = curr_gaze.timestamp - prev_gaze.timestamp
                
                if time_diff > 0:
                    velocity = np.degrees(angular_distance) / time_diff
                    
                    if velocity < self.fixation_velocity_threshold:
                        # Low velocity - potential fixation
                        if current_fixation_start is None:
                            current_fixation_start = prev_gaze.timestamp
                            current_fixation_positions = [prev_gaze, curr_gaze]
                        else:
                            current_fixation_positions.append(curr_gaze)
                    else:
                        # High velocity - end of fixation
                        if current_fixation_start is not None:
                            fixation_duration = curr_gaze.timestamp - current_fixation_start
                            
                            if fixation_duration >= self.minimum_fixation_duration:
                                # Calculate fixation center
                                center_x = np.mean([gv.x for gv in current_fixation_positions])
                                center_y = np.mean([gv.y for gv in current_fixation_positions])
                                center_z = np.mean([gv.z for gv in current_fixation_positions])
                                
                                fixations.append({
                                    'start_time': current_fixation_start,
                                    'end_time': curr_gaze.timestamp,
                                    'duration': fixation_duration,
                                    'center': (center_x, center_y, center_z),
                                    'stability': self._calculate_fixation_stability(current_fixation_positions)
                                })
                            
                            current_fixation_start = None
                            current_fixation_positions = []
            
            # Handle ongoing fixation at end
            if current_fixation_start is not None and current_fixation_positions:
                fixation_duration = gaze_sequence[-1].timestamp - current_fixation_start
                if fixation_duration >= self.minimum_fixation_duration:
                    center_x = np.mean([gv.x for gv in current_fixation_positions])
                    center_y = np.mean([gv.y for gv in current_fixation_positions])
                    center_z = np.mean([gv.z for gv in current_fixation_positions])
                    
                    fixations.append({
                        'start_time': current_fixation_start,
                        'end_time': gaze_sequence[-1].timestamp,
                        'duration': fixation_duration,
                        'center': (center_x, center_y, center_z),
                        'stability': self._calculate_fixation_stability(current_fixation_positions)
                    })
            
            # Calculate metrics
            fixation_count = len(fixations)
            total_fixation_time = sum(f['duration'] for f in fixations)
            average_duration = total_fixation_time / fixation_count if fixation_count > 0 else 0.0
            
            total_time = gaze_sequence[-1].timestamp - gaze_sequence[0].timestamp
            fixation_rate = fixation_count / total_time if total_time > 0 else 0.0
            
            return {
                'fixation_count': fixation_count,
                'average_duration': float(average_duration),
                'total_fixation_time': float(total_fixation_time),
                'fixation_rate': float(fixation_rate),
                'fixations': fixations
            }
            
        except Exception as e:
            logger.error(f"Error calculating fixation metrics: {e}")
            return {
                'fixation_count': 0,
                'average_duration': 0.0,
                'total_fixation_time': 0.0,
                'fixation_rate': 0.0
            }
    
    def _calculate_fixation_stability(self, fixation_positions: List[GazeVector]) -> float:
        """Calculate how stable a fixation is based on position variance."""
        try:
            if len(fixation_positions) < 2:
                return 1.0
            
            x_coords = [gv.x for gv in fixation_positions]
            y_coords = [gv.y for gv in fixation_positions]
            
            x_var = np.var(x_coords)
            y_var = np.var(y_coords)
            
            # Convert variance to stability score (lower variance = higher stability)
            stability = 1.0 / (1.0 + x_var + y_var)
            return min(1.0, stability)
            
        except Exception as e:
            logger.debug(f"Error calculating fixation stability: {e}")
            return 0.5
    
    def calculate_saccade_metrics(self, gaze_sequence: List[GazeVector]) -> Dict[str, float]:
        """
        Calculate saccadic eye movement metrics.
        
        Args:
            gaze_sequence: Sequence of gaze vectors
            
        Returns:
            Dictionary with saccade metrics
        """
        try:
            if len(gaze_sequence) < 2:
                return {
                    'saccade_count': 0,
                    'average_amplitude': 0.0,
                    'average_velocity': 0.0,
                    'saccade_rate': 0.0
                }
            
            saccades = []
            
            for i in range(1, len(gaze_sequence)):
                prev_gaze = gaze_sequence[i-1]
                curr_gaze = gaze_sequence[i]
                
                # Calculate movement metrics
                angular_distance = prev_gaze.angle_to(curr_gaze)
                time_diff = curr_gaze.timestamp - prev_gaze.timestamp
                
                if time_diff > 0:
                    velocity = np.degrees(angular_distance) / time_diff
                    
                    # Detect saccades based on high velocity
                    if velocity > self.saccade_velocity_threshold:
                        saccades.append({
                            'start_time': prev_gaze.timestamp,
                            'end_time': curr_gaze.timestamp,
                            'amplitude': np.degrees(angular_distance),
                            'velocity': velocity,
                            'duration': time_diff
                        })
            
            # Calculate metrics
            saccade_count = len(saccades)
            average_amplitude = np.mean([s['amplitude'] for s in saccades]) if saccades else 0.0
            average_velocity = np.mean([s['velocity'] for s in saccades]) if saccades else 0.0
            
            total_time = gaze_sequence[-1].timestamp - gaze_sequence[0].timestamp
            saccade_rate = saccade_count / total_time if total_time > 0 else 0.0
            
            return {
                'saccade_count': saccade_count,
                'average_amplitude': float(average_amplitude),
                'average_velocity': float(average_velocity),
                'saccade_rate': float(saccade_rate),
                'saccades': saccades
            }
            
        except Exception as e:
            logger.error(f"Error calculating saccade metrics: {e}")
            return {
                'saccade_count': 0,
                'average_amplitude': 0.0,
                'average_velocity': 0.0,
                'saccade_rate': 0.0
            }
    
    def identify_repetitive_behaviors(self, visual_patterns: List[Any]) -> float:
        """
        Identify and score repetitive visual behaviors (0-1).
        
        Args:
            visual_patterns: List of visual patterns or gaze vectors
            
        Returns:
            Repetitive behavior score (0-1, higher = more repetitive)
        """
        try:
            if isinstance(visual_patterns[0], GazeVector):
                return self._calculate_pattern_repetitiveness(visual_patterns)
            else:
                # Handle other pattern types
                return 0.0
                
        except Exception as e:
            logger.error(f"Error identifying repetitive behaviors: {e}")
            return 0.0
    
    def assess_attention_stability(self, gaze_data: List[GazeVector], window_duration: float) -> float:
        """
        Assess how stable attention is over time (0-1).
        
        Args:
            gaze_data: List of gaze vectors
            window_duration: Duration of the analysis window
            
        Returns:
            Attention stability score (0-1, higher = more stable)
        """
        try:
            if len(gaze_data) < 3:
                return 1.0
            
            # Calculate position variance over time
            x_coords = [gv.x for gv in gaze_data]
            y_coords = [gv.y for gv in gaze_data]
            z_coords = [gv.z for gv in gaze_data]
            
            x_var = np.var(x_coords)
            y_var = np.var(y_coords)
            z_var = np.var(z_coords)
            
            # Calculate movement consistency
            movements = []
            for i in range(1, len(gaze_data)):
                movement = gaze_data[i-1].angle_to(gaze_data[i])
                movements.append(movement)
            
            movement_consistency = 1.0 - (np.std(movements) / np.pi) if movements else 1.0
            
            # Combine variance and consistency measures
            position_stability = 1.0 / (1.0 + x_var + y_var + z_var)
            
            # Overall stability score
            stability = (position_stability * 0.6 + movement_consistency * 0.4)
            return min(1.0, max(0.0, stability))
            
        except Exception as e:
            logger.error(f"Error assessing attention stability: {e}")
            return 0.0
    
    def _create_empty_tracking_data(self, timestamp: float) -> VisualTrackingData:
        """Create empty tracking data for error cases."""
        return VisualTrackingData(
            timestamp=timestamp,
            window_duration=0.0,
            eye_movement_velocity=0.0,
            saccade_count=0,
            fixation_duration=0.0,
            scanning_pattern=ScanningPattern.RANDOM,
            repetitive_behavior_score=0.0,
            attention_stability=1.0
        )
    
    def get_movement_summary(self, time_window: float = 30.0) -> Dict[str, Any]:
        """Get summary of recent eye movement patterns."""
        try:
            if not self.movement_history:
                return {}
            
            # Filter recent movements
            current_time = self.movement_history[-1]['timestamp']
            recent_movements = [
                entry for entry in self.movement_history
                if current_time - entry['timestamp'] <= time_window
            ]
            
            if not recent_movements:
                return {}
            
            # Calculate summary statistics
            tracking_data_list = [entry['tracking_data'] for entry in recent_movements]
            
            avg_velocity = np.mean([td.eye_movement_velocity for td in tracking_data_list])
            avg_fixation_duration = np.mean([td.fixation_duration for td in tracking_data_list])
            avg_saccade_count = np.mean([td.saccade_count for td in tracking_data_list])
            avg_stability = np.mean([td.attention_stability for td in tracking_data_list])
            avg_repetitive_score = np.mean([td.repetitive_behavior_score for td in tracking_data_list])
            
            # Most common scanning pattern
            patterns = [td.scanning_pattern for td in tracking_data_list]
            pattern_counts = {}
            for pattern in patterns:
                pattern_counts[pattern.value] = pattern_counts.get(pattern.value, 0) + 1
            
            dominant_pattern = max(pattern_counts.items(), key=lambda x: x[1])[0] if pattern_counts else "unknown"
            
            return {
                'average_velocity': float(avg_velocity),
                'average_fixation_duration': float(avg_fixation_duration),
                'average_saccade_count': float(avg_saccade_count),
                'attention_stability': float(avg_stability),
                'repetitive_behavior_score': float(avg_repetitive_score),
                'dominant_scanning_pattern': dominant_pattern,
                'analysis_count': len(recent_movements)
            }
            
        except Exception as e:
            logger.error(f"Error generating movement summary: {e}")
            return {}
    
    def reset(self) -> None:
        """Reset the analyzer state."""
        self.movement_history.clear()
        logger.info("Visual Tracking Analyzer reset")