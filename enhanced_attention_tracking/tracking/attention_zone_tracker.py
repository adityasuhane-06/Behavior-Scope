"""
Attention Zone Tracker implementation for Enhanced Eye Contact & Attention Tracking System.

This module provides attention zone configuration, tracking, and analysis capabilities.
"""

import logging
from typing import List, Dict, Optional, Any
import numpy as np
from collections import defaultdict, deque

from ..core.interfaces import AttentionZoneTracker
from ..core.data_models import GazeVector, AttentionZoneEvent, TimeWindow, Coordinate, ZoneCoordinates
from ..core.enums import ZoneType

logger = logging.getLogger(__name__)


class AttentionZoneTrackerImpl(AttentionZoneTracker):
    """
    Implementation of attention zone tracking.
    
    Tracks attention distribution across predefined zones, calculates dwell times,
    and analyzes transition patterns.
    """
    
    def __init__(self):
        """Initialize the attention zone tracker."""
        self.zones: Dict[str, Dict[str, Any]] = {}
        self.zone_events: List[AttentionZoneEvent] = []
        self.current_zone_occupancy: Dict[str, Optional[float]] = {}  # zone_id -> entry_time
        
        # Tracking parameters
        self.entry_threshold = 0.6  # Confidence threshold for zone entry
        self.exit_threshold = 0.4   # Confidence threshold for zone exit (hysteresis)
        self.minimum_dwell_time = 0.1  # Minimum time to register as zone attention
        self.transition_history: deque = deque(maxlen=100)
        
        logger.info("Attention Zone Tracker initialized")
    
    def configure_zones(self, zone_definitions: List[Dict[str, Any]]) -> None:
        """
        Configure attention zones for tracking.
        
        Args:
            zone_definitions: List of zone definition dictionaries
        """
        try:
            self.zones.clear()
            self.current_zone_occupancy.clear()
            
            for zone_def in zone_definitions:
                zone_id = zone_def.get('zone_id')
                if not zone_id:
                    logger.warning("Zone definition missing zone_id, skipping")
                    continue
                
                # Validate and store zone definition
                zone_config = {
                    'zone_id': zone_id,
                    'zone_name': zone_def.get('zone_name', zone_id),
                    'zone_type': ZoneType(zone_def.get('zone_type', 'custom')),
                    'coordinates': self._parse_zone_coordinates(zone_def.get('coordinates', {})),
                    'is_dynamic': zone_def.get('is_dynamic', False),
                    'tracking_sensitivity': zone_def.get('tracking_sensitivity', 1.0),
                    'minimum_dwell_time': zone_def.get('minimum_dwell_time', self.minimum_dwell_time)
                }
                
                self.zones[zone_id] = zone_config
                self.current_zone_occupancy[zone_id] = None
                
                logger.info(f"Configured zone: {zone_id} ({zone_config['zone_name']})")
            
            logger.info(f"Configured {len(self.zones)} attention zones")
            
        except Exception as e:
            logger.error(f"Error configuring zones: {e}")
    
    def _parse_zone_coordinates(self, coord_data: Dict[str, Any]) -> ZoneCoordinates:
        """Parse zone coordinate data into ZoneCoordinates object."""
        try:
            zone_type = coord_data.get('type', 'bounding_box')
            coord_list = coord_data.get('coordinates', [])
            
            coordinates = []
            for coord in coord_list:
                if isinstance(coord, dict):
                    coordinates.append(Coordinate(
                        x=coord.get('x', 0.0),
                        y=coord.get('y', 0.0),
                        confidence=coord.get('confidence', 1.0)
                    ))
                elif isinstance(coord, (list, tuple)) and len(coord) >= 2:
                    coordinates.append(Coordinate(
                        x=float(coord[0]),
                        y=float(coord[1]),
                        confidence=coord[2] if len(coord) > 2 else 1.0
                    ))
            
            return ZoneCoordinates(zone_type=zone_type, coordinates=coordinates)
            
        except Exception as e:
            logger.debug(f"Error parsing zone coordinates: {e}")
            return ZoneCoordinates(zone_type='bounding_box', coordinates=[])
    
    def track_zone_attention(self, gaze_vector: GazeVector, timestamp: float) -> Optional[AttentionZoneEvent]:
        """
        Track attention within configured zones.
        
        Args:
            gaze_vector: Current gaze vector
            timestamp: Current timestamp
            
        Returns:
            AttentionZoneEvent if zone transition occurred, None otherwise
        """
        try:
            if not self.zones:
                return None
            
            # Determine which zones the gaze is currently in
            current_zones = self._determine_gaze_zones(gaze_vector)
            
            # Process zone entries and exits
            zone_event = None
            
            for zone_id, zone_config in self.zones.items():
                is_in_zone = zone_id in current_zones
                was_in_zone = self.current_zone_occupancy[zone_id] is not None
                
                if is_in_zone and not was_in_zone:
                    # Zone entry
                    self.current_zone_occupancy[zone_id] = timestamp
                    logger.debug(f"Entered zone: {zone_id} at {timestamp:.3f}s")
                
                elif not is_in_zone and was_in_zone:
                    # Zone exit
                    entry_time = self.current_zone_occupancy[zone_id]
                    duration = timestamp - entry_time
                    
                    if duration >= zone_config['minimum_dwell_time']:
                        # Create zone attention event
                        zone_event = AttentionZoneEvent(
                            zone_id=zone_id,
                            zone_name=zone_config['zone_name'],
                            entry_time=entry_time,
                            exit_time=timestamp,
                            duration=duration,
                            attention_intensity=gaze_vector.confidence,
                            peak_intensity=gaze_vector.confidence  # Simplified
                        )
                        
                        self.zone_events.append(zone_event)
                        
                        # Record transition
                        self.transition_history.append({
                            'timestamp': timestamp,
                            'from_zone': zone_id,
                            'to_zone': None,  # Will be updated when entering new zone
                            'duration': duration
                        })
                        
                        logger.debug(f"Exited zone: {zone_id}, duration: {duration:.3f}s")
                    
                    self.current_zone_occupancy[zone_id] = None
            
            return zone_event
            
        except Exception as e:
            logger.error(f"Error tracking zone attention: {e}")
            return None
    
    def _determine_gaze_zones(self, gaze_vector: GazeVector) -> List[str]:
        """Determine which zones the gaze vector intersects."""
        try:
            intersecting_zones = []
            
            for zone_id, zone_config in self.zones.items():
                if self._is_gaze_in_zone(gaze_vector, zone_config):
                    intersecting_zones.append(zone_id)
            
            return intersecting_zones
            
        except Exception as e:
            logger.debug(f"Error determining gaze zones: {e}")
            return []
    
    def _is_gaze_in_zone(self, gaze_vector: GazeVector, zone_config: Dict[str, Any]) -> bool:
        """Check if gaze vector intersects with a specific zone."""
        try:
            zone_type = zone_config['zone_type']
            coordinates = zone_config['coordinates']
            sensitivity = zone_config['tracking_sensitivity']
            
            # Apply sensitivity to confidence threshold
            effective_threshold = self.entry_threshold / sensitivity
            
            if gaze_vector.confidence < effective_threshold:
                return False
            
            # Zone-specific intersection logic
            if zone_type == ZoneType.FACE:
                return self._is_gaze_in_face_zone(gaze_vector)
            
            elif zone_type == ZoneType.OBJECT:
                return self._is_gaze_in_object_zone(gaze_vector, coordinates)
            
            elif zone_type == ZoneType.BACKGROUND:
                return self._is_gaze_in_background_zone(gaze_vector)
            
            elif zone_type == ZoneType.SOCIAL_PARTNER:
                return self._is_gaze_in_social_partner_zone(gaze_vector)
            
            elif zone_type == ZoneType.CUSTOM:
                return self._is_gaze_in_custom_zone(gaze_vector, coordinates)
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking gaze in zone: {e}")
            return False
    
    def _is_gaze_in_face_zone(self, gaze_vector: GazeVector) -> bool:
        """Check if gaze is in face/social partner zone."""
        # Face zone: forward gaze with moderate deviation
        return (gaze_vector.z > 0.5 and 
                abs(gaze_vector.x) < 0.4 and 
                abs(gaze_vector.y) < 0.4)
    
    def _is_gaze_in_object_zone(self, gaze_vector: GazeVector, coordinates: ZoneCoordinates) -> bool:
        """Check if gaze is in object zone."""
        # Simplified object zone detection
        if coordinates.zone_type == 'bounding_box' and len(coordinates.coordinates) >= 2:
            # Use bounding box coordinates
            x_coords = [coord.x for coord in coordinates.coordinates]
            y_coords = [coord.y for coord in coordinates.coordinates]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            return (min_x <= gaze_vector.x <= max_x and 
                    min_y <= gaze_vector.y <= max_y and
                    gaze_vector.z > 0.3)
        
        return False
    
    def _is_gaze_in_background_zone(self, gaze_vector: GazeVector) -> bool:
        """Check if gaze is in background zone."""
        # Background: forward gaze but not in face region
        return (gaze_vector.z > 0.2 and 
                (abs(gaze_vector.x) > 0.5 or abs(gaze_vector.y) > 0.5))
    
    def _is_gaze_in_social_partner_zone(self, gaze_vector: GazeVector) -> bool:
        """Check if gaze is in social partner zone."""
        # Similar to face zone but slightly broader
        return (gaze_vector.z > 0.4 and 
                abs(gaze_vector.x) < 0.5 and 
                abs(gaze_vector.y) < 0.5)
    
    def _is_gaze_in_custom_zone(self, gaze_vector: GazeVector, coordinates: ZoneCoordinates) -> bool:
        """Check if gaze is in custom-defined zone."""
        # Use the coordinates' contains_point method
        return coordinates.contains_point(gaze_vector.x, gaze_vector.y)
    
    def calculate_zone_metrics(self, time_window: TimeWindow) -> Dict[str, Any]:
        """
        Calculate attention metrics for each zone.
        
        Args:
            time_window: Time window for analysis
            
        Returns:
            Dictionary with zone metrics
        """
        try:
            # Filter events within time window
            relevant_events = [
                event for event in self.zone_events
                if (time_window.start_time <= event.entry_time <= time_window.start_time + time_window.duration)
            ]
            
            zone_metrics = {}
            
            for zone_id, zone_config in self.zones.items():
                zone_events = [e for e in relevant_events if e.zone_id == zone_id]
                
                if zone_events:
                    total_duration = sum(e.duration for e in zone_events)
                    avg_duration = total_duration / len(zone_events)
                    max_duration = max(e.duration for e in zone_events)
                    avg_intensity = np.mean([e.attention_intensity for e in zone_events])
                    peak_intensity = max(e.peak_intensity for e in zone_events)
                    
                    # Calculate percentage of time window
                    percentage = (total_duration / time_window.duration) * 100
                    
                else:
                    total_duration = 0.0
                    avg_duration = 0.0
                    max_duration = 0.0
                    avg_intensity = 0.0
                    peak_intensity = 0.0
                    percentage = 0.0
                
                zone_metrics[zone_id] = {
                    'zone_name': zone_config['zone_name'],
                    'zone_type': zone_config['zone_type'].value,
                    'event_count': len(zone_events),
                    'total_duration': float(total_duration),
                    'average_duration': float(avg_duration),
                    'max_duration': float(max_duration),
                    'percentage_of_window': float(percentage),
                    'average_intensity': float(avg_intensity),
                    'peak_intensity': float(peak_intensity),
                    'events': zone_events
                }
            
            return zone_metrics
            
        except Exception as e:
            logger.error(f"Error calculating zone metrics: {e}")
            return {}
    
    def generate_attention_heatmap(self, attention_data: List[AttentionZoneEvent]) -> Any:
        """
        Generate attention heatmap visualization data.
        
        Args:
            attention_data: List of attention zone events
            
        Returns:
            Heatmap data structure
        """
        try:
            if not attention_data:
                return None
            
            # Create heatmap grid (simplified 2D representation)
            grid_size = 20
            heatmap = np.zeros((grid_size, grid_size))
            
            # Map zone events to grid positions
            for event in attention_data:
                zone_config = self.zones.get(event.zone_id)
                if zone_config and zone_config['coordinates'].coordinates:
                    # Get zone center
                    coords = zone_config['coordinates'].coordinates
                    center_x = np.mean([coord.x for coord in coords])
                    center_y = np.mean([coord.y for coord in coords])
                    
                    # Map to grid coordinates (assuming normalized -1 to 1 range)
                    grid_x = int((center_x + 1) * grid_size / 2)
                    grid_y = int((center_y + 1) * grid_size / 2)
                    
                    # Clamp to grid bounds
                    grid_x = max(0, min(grid_size - 1, grid_x))
                    grid_y = max(0, min(grid_size - 1, grid_y))
                    
                    # Add attention intensity weighted by duration
                    intensity = event.attention_intensity * event.duration
                    heatmap[grid_y, grid_x] += intensity
            
            # Normalize heatmap
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
            
            return {
                'heatmap_data': heatmap.tolist(),
                'grid_size': grid_size,
                'total_events': len(attention_data),
                'max_intensity': float(np.max(heatmap))
            }
            
        except Exception as e:
            logger.error(f"Error generating attention heatmap: {e}")
            return None
    
    def detect_zone_transitions(self, attention_sequence: List[AttentionZoneEvent]) -> List[Dict[str, Any]]:
        """
        Detect and analyze transitions between attention zones.
        
        Args:
            attention_sequence: Sequence of attention zone events
            
        Returns:
            List of transition analysis results
        """
        try:
            if len(attention_sequence) < 2:
                return []
            
            transitions = []
            
            for i in range(1, len(attention_sequence)):
                prev_event = attention_sequence[i-1]
                curr_event = attention_sequence[i]
                
                # Calculate transition metrics
                transition_latency = curr_event.entry_time - prev_event.exit_time
                
                transition = {
                    'from_zone': prev_event.zone_id,
                    'to_zone': curr_event.zone_id,
                    'from_zone_name': prev_event.zone_name,
                    'to_zone_name': curr_event.zone_name,
                    'transition_time': curr_event.entry_time,
                    'transition_latency': float(transition_latency),
                    'from_duration': prev_event.duration,
                    'to_duration': curr_event.duration,
                    'intensity_change': curr_event.attention_intensity - prev_event.attention_intensity
                }
                
                transitions.append(transition)
            
            # Analyze transition patterns
            transition_analysis = self._analyze_transition_patterns(transitions)
            
            return {
                'transitions': transitions,
                'analysis': transition_analysis
            }
            
        except Exception as e:
            logger.error(f"Error detecting zone transitions: {e}")
            return []
    
    def _analyze_transition_patterns(self, transitions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in zone transitions."""
        try:
            if not transitions:
                return {}
            
            # Calculate transition frequencies
            transition_counts = defaultdict(int)
            for trans in transitions:
                key = f"{trans['from_zone']} -> {trans['to_zone']}"
                transition_counts[key] += 1
            
            # Most common transitions
            most_common = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Average transition latency
            latencies = [trans['transition_latency'] for trans in transitions if trans['transition_latency'] >= 0]
            avg_latency = np.mean(latencies) if latencies else 0.0
            
            # Transition rate
            if transitions:
                total_time = transitions[-1]['transition_time'] - transitions[0]['transition_time']
                transition_rate = len(transitions) / total_time if total_time > 0 else 0.0
            else:
                transition_rate = 0.0
            
            # Identify circular patterns (A -> B -> A)
            circular_patterns = self._identify_circular_patterns(transitions)
            
            return {
                'total_transitions': len(transitions),
                'unique_transition_types': len(transition_counts),
                'most_common_transitions': most_common,
                'average_transition_latency': float(avg_latency),
                'transition_rate': float(transition_rate),
                'circular_patterns': circular_patterns
            }
            
        except Exception as e:
            logger.debug(f"Error analyzing transition patterns: {e}")
            return {}
    
    def _identify_circular_patterns(self, transitions: List[Dict[str, Any]]) -> List[str]:
        """Identify circular transition patterns."""
        try:
            patterns = []
            
            for i in range(len(transitions) - 1):
                trans1 = transitions[i]
                trans2 = transitions[i + 1]
                
                # Check for A -> B -> A pattern
                if (trans1['from_zone'] == trans2['to_zone'] and 
                    trans1['to_zone'] == trans2['from_zone']):
                    pattern = f"{trans1['from_zone']} <-> {trans1['to_zone']}"
                    if pattern not in patterns:
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.debug(f"Error identifying circular patterns: {e}")
            return []
    
    def calculate_zone_preferences(self, zone_events: List[AttentionZoneEvent]) -> Dict[str, float]:
        """
        Calculate preference scores for each attention zone.
        
        Args:
            zone_events: List of attention zone events
            
        Returns:
            Dictionary mapping zone_id to preference score (0-1)
        """
        try:
            if not zone_events:
                return {}
            
            # Calculate total attention time per zone
            zone_durations = defaultdict(float)
            zone_intensities = defaultdict(list)
            
            for event in zone_events:
                zone_durations[event.zone_id] += event.duration
                zone_intensities[event.zone_id].append(event.attention_intensity)
            
            # Calculate total time across all zones
            total_time = sum(zone_durations.values())
            
            if total_time == 0:
                return {}
            
            # Calculate preference scores
            preferences = {}
            for zone_id in self.zones.keys():
                duration = zone_durations.get(zone_id, 0.0)
                intensities = zone_intensities.get(zone_id, [0.0])
                
                # Preference based on time proportion and average intensity
                time_proportion = duration / total_time
                avg_intensity = np.mean(intensities)
                
                # Combined preference score
                preference_score = (time_proportion * 0.7 + avg_intensity * 0.3)
                preferences[zone_id] = float(preference_score)
            
            return preferences
            
        except Exception as e:
            logger.error(f"Error calculating zone preferences: {e}")
            return {}
    
    def get_zone_summary(self, time_window: float = 60.0) -> Dict[str, Any]:
        """Get summary of zone attention patterns."""
        try:
            if not self.zone_events:
                return {
                    'total_zones': len(self.zones),
                    'active_zones': 0,
                    'total_events': 0,
                    'zone_metrics': {}
                }
            
            # Filter recent events
            current_time = self.zone_events[-1].exit_time
            recent_events = [
                event for event in self.zone_events
                if current_time - event.entry_time <= time_window
            ]
            
            # Calculate metrics
            active_zones = len(set(event.zone_id for event in recent_events))
            
            # Zone preferences
            preferences = self.calculate_zone_preferences(recent_events)
            
            # Most preferred zone
            most_preferred = max(preferences.items(), key=lambda x: x[1]) if preferences else None
            
            return {
                'total_zones': len(self.zones),
                'active_zones': active_zones,
                'total_events': len(recent_events),
                'zone_preferences': preferences,
                'most_preferred_zone': most_preferred[0] if most_preferred else None,
                'analysis_window': time_window
            }
            
        except Exception as e:
            logger.error(f"Error generating zone summary: {e}")
            return {}
    
    def reset(self) -> None:
        """Reset the tracker state."""
        self.zone_events.clear()
        self.transition_history.clear()
        for zone_id in self.current_zone_occupancy:
            self.current_zone_occupancy[zone_id] = None
        logger.info("Attention Zone Tracker reset")