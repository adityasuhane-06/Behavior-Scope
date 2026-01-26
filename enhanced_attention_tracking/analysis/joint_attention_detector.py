"""
Joint Attention Detector implementation for Enhanced Eye Contact & Attention Tracking System.

This module provides joint attention detection between subjects and social partners,
attention type classification, and response latency measurement.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from datetime import datetime, timedelta

from ..core.interfaces import JointAttentionDetector
from ..core.data_models import GazeVector, JointAttentionEvent, AlignmentScore
from ..core.enums import AttentionType

logger = logging.getLogger(__name__)


class JointAttentionDetectorImpl(JointAttentionDetector):
    """
    Implementation of joint attention detection.
    
    Detects moments when subject and social partner focus on the same object or area,
    classifies attention types, and measures response latencies.
    """
    
    def __init__(self):
        """Initialize the joint attention detector."""
        self.attention_history: List[JointAttentionEvent] = []
        self.gaze_alignment_threshold = 0.7  # Threshold for considering gazes aligned
        self.temporal_alignment_window = 2.0  # Seconds for temporal alignment
        self.minimum_episode_duration = 0.5  # Minimum duration for valid joint attention
        self.response_latency_threshold = 3.0  # Maximum latency for responding attention
        
        # Track attention cues and responses
        self.attention_cues: List[Dict[str, Any]] = []
        self.pending_responses: List[Dict[str, Any]] = []
        
        logger.info("Joint Attention Detector initialized")
    
    def detect_joint_attention(self, subject_gaze: GazeVector, partner_gaze: Optional[GazeVector], 
                              objects: List[Any]) -> Optional[JointAttentionEvent]:
        """
        Detect joint attention between subject and partner.
        
        Args:
            subject_gaze: Subject's gaze vector
            partner_gaze: Partner's gaze vector (if available)
            objects: List of objects in the scene
            
        Returns:
            JointAttentionEvent if joint attention detected, None otherwise
        """
        try:
            if subject_gaze is None:
                return None
            
            current_time = subject_gaze.timestamp
            
            # Method 1: Partner-based joint attention (if partner gaze available)
            if partner_gaze is not None:
                alignment = self.calculate_attention_alignment([subject_gaze, partner_gaze])
                
                if alignment.overall_score >= self.gaze_alignment_threshold:
                    # Determine attention type based on temporal context
                    attention_type = self._determine_attention_type_from_alignment(
                        subject_gaze, partner_gaze, current_time
                    )
                    
                    # Find target object if available
                    target_object = self._identify_target_object(subject_gaze, objects)
                    
                    # Create joint attention event
                    event = JointAttentionEvent(
                        start_time=current_time,
                        end_time=current_time + 0.1,  # Initial duration, will be updated
                        duration=0.1,
                        attention_type=attention_type,
                        target_object=target_object,
                        alignment_score=alignment.overall_score,
                        confidence=min(subject_gaze.confidence, partner_gaze.confidence)
                    )
                    
                    return event
            
            # Method 2: Object-based joint attention (infer from subject's attention to objects)
            elif objects:
                object_attention_event = self._detect_object_based_joint_attention(
                    subject_gaze, objects, current_time
                )
                if object_attention_event:
                    return object_attention_event
            
            # Method 3: Sequential attention pattern analysis
            sequential_event = self._detect_sequential_attention_patterns(
                subject_gaze, current_time
            )
            if sequential_event:
                return sequential_event
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting joint attention: {e}")
            return None
    
    def _determine_attention_type_from_alignment(self, subject_gaze: GazeVector, 
                                               partner_gaze: GazeVector, 
                                               current_time: float) -> AttentionType:
        """Determine if attention is initiated, responding, or mutual."""
        try:
            # Check recent attention history to determine who initiated
            recent_window = 3.0  # Look back 3 seconds
            
            # Find recent attention cues
            recent_cues = [
                cue for cue in self.attention_cues
                if current_time - cue['timestamp'] <= recent_window
            ]
            
            if recent_cues:
                # Check if subject initiated attention shift
                subject_initiated = any(
                    cue['initiator'] == 'subject' for cue in recent_cues
                )
                
                # Check if partner initiated attention shift
                partner_initiated = any(
                    cue['initiator'] == 'partner' for cue in recent_cues
                )
                
                if subject_initiated and not partner_initiated:
                    return AttentionType.INITIATED
                elif partner_initiated and not subject_initiated:
                    return AttentionType.RESPONDING
                else:
                    return AttentionType.MUTUAL
            
            # If no recent cues, assume mutual attention
            return AttentionType.MUTUAL
            
        except Exception as e:
            logger.debug(f"Error determining attention type: {e}")
            return AttentionType.MUTUAL
    
    def _detect_object_based_joint_attention(self, subject_gaze: GazeVector, 
                                           objects: List[Any], 
                                           current_time: float) -> Optional[JointAttentionEvent]:
        """Detect joint attention based on subject's attention to objects."""
        try:
            # This would integrate with object detection/tracking
            # For now, create a simplified implementation
            
            target_object = self._identify_target_object(subject_gaze, objects)
            
            if target_object:
                # Check if this represents a shift in attention (potential joint attention)
                if self._is_attention_shift_to_object(subject_gaze, target_object, current_time):
                    return JointAttentionEvent(
                        start_time=current_time,
                        end_time=current_time + 0.1,
                        duration=0.1,
                        attention_type=AttentionType.INITIATED,  # Assume subject-initiated
                        target_object=target_object,
                        alignment_score=0.8,  # High score for clear object attention
                        confidence=subject_gaze.confidence
                    )
            
            return None
            
        except Exception as e:
            logger.debug(f"Error in object-based joint attention detection: {e}")
            return None
    
    def _detect_sequential_attention_patterns(self, subject_gaze: GazeVector, 
                                            current_time: float) -> Optional[JointAttentionEvent]:
        """Detect joint attention from sequential attention patterns."""
        try:
            # Look for patterns like: face -> object -> face (triadic attention)
            # This is a simplified implementation
            
            # Check if subject is looking at face region after looking elsewhere
            from ..core.enums import GazeTarget
            from ..analysis.gaze_direction_analyzer import GazeDirectionAnalyzerImpl
            
            # Create temporary analyzer for classification
            analyzer = GazeDirectionAnalyzerImpl()
            current_target = analyzer.classify_gaze_target(subject_gaze, [])
            
            if current_target in [GazeTarget.FACE_REGION, GazeTarget.CAMERA_DIRECT]:
                # Check if this follows attention to an object
                if self._was_recent_object_attention(current_time):
                    return JointAttentionEvent(
                        start_time=current_time,
                        end_time=current_time + 0.1,
                        duration=0.1,
                        attention_type=AttentionType.RESPONDING,
                        target_object="social_partner",
                        alignment_score=0.6,
                        confidence=subject_gaze.confidence * 0.8  # Lower confidence for inferred attention
                    )
            
            return None
            
        except Exception as e:
            logger.debug(f"Error in sequential attention pattern detection: {e}")
            return None
    
    def _identify_target_object(self, gaze_vector: GazeVector, objects: List[Any]) -> Optional[str]:
        """Identify which object the gaze is directed toward."""
        try:
            if not objects:
                return None
            
            # This would integrate with object detection/tracking system
            # For now, create a simplified implementation based on gaze direction
            
            # Map gaze direction to potential objects
            if gaze_vector.z > 0.8:  # Looking forward
                if abs(gaze_vector.x) < 0.3:
                    return "central_object"
                elif gaze_vector.x > 0.3:
                    return "right_object"
                elif gaze_vector.x < -0.3:
                    return "left_object"
            
            return None
            
        except Exception as e:
            logger.debug(f"Error identifying target object: {e}")
            return None
    
    def _is_attention_shift_to_object(self, gaze_vector: GazeVector, 
                                    target_object: str, current_time: float) -> bool:
        """Check if this represents a new attention shift to an object."""
        try:
            # Check recent attention history
            recent_window = 2.0
            recent_events = [
                event for event in self.attention_history
                if current_time - event.start_time <= recent_window
            ]
            
            # If no recent attention to this object, it's a new shift
            recent_to_same_object = any(
                event.target_object == target_object for event in recent_events
            )
            
            return not recent_to_same_object
            
        except Exception as e:
            logger.debug(f"Error checking attention shift: {e}")
            return False
    
    def _was_recent_object_attention(self, current_time: float) -> bool:
        """Check if there was recent attention to an object (not face/social)."""
        try:
            recent_window = 3.0
            recent_events = [
                event for event in self.attention_history
                if current_time - event.start_time <= recent_window
            ]
            
            return any(
                event.target_object and event.target_object not in ["social_partner", "face"]
                for event in recent_events
            )
            
        except Exception as e:
            logger.debug(f"Error checking recent object attention: {e}")
            return False
    
    def classify_attention_type(self, event: JointAttentionEvent, 
                               temporal_context: Dict[str, Any]) -> AttentionType:
        """
        Classify the type of joint attention (initiated, responding, mutual).
        
        Args:
            event: Joint attention event to classify
            temporal_context: Context information about recent events
            
        Returns:
            AttentionType classification
        """
        try:
            # Extract context information
            recent_cues = temporal_context.get('recent_attention_cues', [])
            partner_actions = temporal_context.get('partner_actions', [])
            subject_actions = temporal_context.get('subject_actions', [])
            
            event_time = event.start_time
            
            # Check for recent partner cues (pointing, looking, vocalizing)
            partner_cue_times = [
                action['timestamp'] for action in partner_actions
                if action.get('type') in ['point', 'look', 'vocalize'] and
                abs(event_time - action['timestamp']) <= self.response_latency_threshold
            ]
            
            # Check for recent subject initiations
            subject_initiation_times = [
                action['timestamp'] for action in subject_actions
                if action.get('type') in ['look_shift', 'point'] and
                abs(event_time - action['timestamp']) <= 1.0  # Shorter window for initiations
            ]
            
            # Classification logic
            if partner_cue_times and not subject_initiation_times:
                # Partner cued, subject responded
                response_latency = min(abs(event_time - t) for t in partner_cue_times)
                event.response_latency = response_latency
                return AttentionType.RESPONDING
            
            elif subject_initiation_times and not partner_cue_times:
                # Subject initiated
                return AttentionType.INITIATED
            
            elif partner_cue_times and subject_initiation_times:
                # Both parties active - determine who was first
                earliest_partner = min(partner_cue_times)
                earliest_subject = min(subject_initiation_times)
                
                if earliest_subject < earliest_partner:
                    return AttentionType.INITIATED
                else:
                    response_latency = event_time - earliest_partner
                    event.response_latency = response_latency
                    return AttentionType.RESPONDING
            
            else:
                # No clear cues - assume mutual attention
                return AttentionType.MUTUAL
                
        except Exception as e:
            logger.error(f"Error classifying attention type: {e}")
            return AttentionType.MUTUAL
    
    def calculate_attention_alignment(self, gaze_vectors: List[GazeVector]) -> AlignmentScore:
        """
        Calculate how well aligned multiple gaze vectors are.
        
        Args:
            gaze_vectors: List of gaze vectors to compare
            
        Returns:
            AlignmentScore with spatial and temporal alignment metrics
        """
        try:
            if len(gaze_vectors) < 2:
                return AlignmentScore(
                    spatial_alignment=0.0,
                    temporal_alignment=0.0,
                    overall_score=0.0
                )
            
            # Calculate spatial alignment (angular similarity)
            spatial_alignments = []
            for i in range(len(gaze_vectors)):
                for j in range(i + 1, len(gaze_vectors)):
                    angle = gaze_vectors[i].angle_to(gaze_vectors[j])
                    # Convert angle to alignment score (0 = perfect alignment, π = opposite)
                    alignment = 1.0 - (angle / np.pi)
                    spatial_alignments.append(max(0.0, alignment))
            
            spatial_score = np.mean(spatial_alignments) if spatial_alignments else 0.0
            
            # Calculate temporal alignment (timestamp proximity)
            timestamps = [gv.timestamp for gv in gaze_vectors]
            time_span = max(timestamps) - min(timestamps)
            
            # Temporal alignment is higher when timestamps are closer
            temporal_score = max(0.0, 1.0 - (time_span / self.temporal_alignment_window))
            
            # Overall score (weighted combination)
            overall_score = spatial_score * 0.7 + temporal_score * 0.3
            
            return AlignmentScore(
                spatial_alignment=float(spatial_score),
                temporal_alignment=float(temporal_score),
                overall_score=float(overall_score)
            )
            
        except Exception as e:
            logger.error(f"Error calculating attention alignment: {e}")
            return AlignmentScore(
                spatial_alignment=0.0,
                temporal_alignment=0.0,
                overall_score=0.0
            )
    
    def track_attention_shifts(self, attention_history: List[Any]) -> Dict[str, Any]:
        """
        Track patterns in attention shifting behavior.
        
        Args:
            attention_history: List of attention events or gaze data
            
        Returns:
            Dictionary with attention shift patterns and metrics
        """
        try:
            if len(attention_history) < 2:
                return {
                    'shift_frequency': 0.0,
                    'average_shift_latency': 0.0,
                    'shift_patterns': [],
                    'attention_stability': 1.0
                }
            
            # Analyze attention shifts
            shifts = []
            for i in range(1, len(attention_history)):
                prev_event = attention_history[i-1]
                curr_event = attention_history[i]
                
                # Calculate shift metrics
                if hasattr(prev_event, 'target_object') and hasattr(curr_event, 'target_object'):
                    if prev_event.target_object != curr_event.target_object:
                        shift_latency = curr_event.start_time - prev_event.end_time
                        shifts.append({
                            'from_target': prev_event.target_object,
                            'to_target': curr_event.target_object,
                            'latency': shift_latency,
                            'timestamp': curr_event.start_time
                        })
            
            # Calculate metrics
            total_time = attention_history[-1].start_time - attention_history[0].start_time
            shift_frequency = len(shifts) / total_time if total_time > 0 else 0.0
            
            average_latency = np.mean([s['latency'] for s in shifts]) if shifts else 0.0
            
            # Attention stability (inverse of shift frequency)
            stability = max(0.0, 1.0 - min(1.0, shift_frequency / 2.0))  # Normalize to 0-1
            
            # Identify common patterns
            patterns = self._identify_shift_patterns(shifts)
            
            return {
                'shift_frequency': float(shift_frequency),
                'average_shift_latency': float(average_latency),
                'shift_patterns': patterns,
                'attention_stability': float(stability),
                'total_shifts': len(shifts)
            }
            
        except Exception as e:
            logger.error(f"Error tracking attention shifts: {e}")
            return {
                'shift_frequency': 0.0,
                'average_shift_latency': 0.0,
                'shift_patterns': [],
                'attention_stability': 0.0
            }
    
    def _identify_shift_patterns(self, shifts: List[Dict[str, Any]]) -> List[str]:
        """Identify common attention shift patterns."""
        try:
            patterns = []
            
            if len(shifts) < 3:
                return patterns
            
            # Look for triadic attention patterns (face -> object -> face)
            for i in range(len(shifts) - 2):
                shift1 = shifts[i]
                shift2 = shifts[i + 1]
                shift3 = shifts[i + 2]
                
                # Check for face -> object -> face pattern
                if (shift1['to_target'] in ['face', 'social_partner'] and
                    shift2['to_target'] not in ['face', 'social_partner'] and
                    shift3['to_target'] in ['face', 'social_partner']):
                    patterns.append('triadic_attention')
                
                # Check for object -> object -> object pattern (object exploration)
                if (shift1['to_target'] not in ['face', 'social_partner'] and
                    shift2['to_target'] not in ['face', 'social_partner'] and
                    shift3['to_target'] not in ['face', 'social_partner']):
                    patterns.append('object_exploration')
            
            # Look for rapid shifting (multiple shifts in short time)
            rapid_shifts = 0
            for i in range(len(shifts) - 1):
                if shifts[i + 1]['timestamp'] - shifts[i]['timestamp'] < 1.0:
                    rapid_shifts += 1
            
            if rapid_shifts > len(shifts) * 0.5:
                patterns.append('rapid_shifting')
            
            return list(set(patterns))  # Remove duplicates
            
        except Exception as e:
            logger.debug(f"Error identifying shift patterns: {e}")
            return []
    
    def measure_response_latency(self, cue_timestamp: float, response_timestamp: float) -> float:
        """
        Measure latency between attention cue and response.
        
        Args:
            cue_timestamp: Timestamp of attention cue
            response_timestamp: Timestamp of attention response
            
        Returns:
            Response latency in seconds
        """
        try:
            latency = response_timestamp - cue_timestamp
            
            # Validate latency is reasonable
            if latency < 0:
                logger.warning(f"Negative response latency: {latency}")
                return 0.0
            
            if latency > self.response_latency_threshold:
                logger.debug(f"Long response latency: {latency:.2f}s")
            
            return float(latency)
            
        except Exception as e:
            logger.error(f"Error measuring response latency: {e}")
            return 0.0
    
    def add_attention_cue(self, timestamp: float, cue_type: str, initiator: str, 
                         target: Optional[str] = None) -> None:
        """Add an attention cue to the tracking system."""
        try:
            cue = {
                'timestamp': timestamp,
                'type': cue_type,
                'initiator': initiator,
                'target': target
            }
            
            self.attention_cues.append(cue)
            
            # Keep only recent cues
            cutoff_time = timestamp - 10.0  # Keep last 10 seconds
            self.attention_cues = [
                cue for cue in self.attention_cues 
                if cue['timestamp'] > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error adding attention cue: {e}")
    
    def get_joint_attention_summary(self, time_window: float = 60.0) -> Dict[str, Any]:
        """Get summary of joint attention events in recent time window."""
        try:
            if not self.attention_history:
                return {
                    'total_episodes': 0,
                    'total_duration': 0.0,
                    'initiated_episodes': 0,
                    'responding_episodes': 0,
                    'mutual_episodes': 0,
                    'average_duration': 0.0,
                    'average_response_latency': 0.0
                }
            
            # Filter recent events
            current_time = self.attention_history[-1].start_time
            recent_events = [
                event for event in self.attention_history
                if current_time - event.start_time <= time_window
            ]
            
            if not recent_events:
                return {
                    'total_episodes': 0,
                    'total_duration': 0.0,
                    'initiated_episodes': 0,
                    'responding_episodes': 0,
                    'mutual_episodes': 0,
                    'average_duration': 0.0,
                    'average_response_latency': 0.0
                }
            
            # Calculate summary statistics
            total_episodes = len(recent_events)
            total_duration = sum(event.duration for event in recent_events)
            
            initiated_episodes = len([e for e in recent_events if e.attention_type == AttentionType.INITIATED])
            responding_episodes = len([e for e in recent_events if e.attention_type == AttentionType.RESPONDING])
            mutual_episodes = len([e for e in recent_events if e.attention_type == AttentionType.MUTUAL])
            
            average_duration = total_duration / total_episodes if total_episodes > 0 else 0.0
            
            # Calculate average response latency for responding episodes
            responding_latencies = [
                event.response_latency for event in recent_events
                if event.attention_type == AttentionType.RESPONDING and event.response_latency is not None
            ]
            average_response_latency = np.mean(responding_latencies) if responding_latencies else 0.0
            
            return {
                'total_episodes': total_episodes,
                'total_duration': float(total_duration),
                'initiated_episodes': initiated_episodes,
                'responding_episodes': responding_episodes,
                'mutual_episodes': mutual_episodes,
                'average_duration': float(average_duration),
                'average_response_latency': float(average_response_latency)
            }
            
        except Exception as e:
            logger.error(f"Error generating joint attention summary: {e}")
            return {}
    
    def reset(self) -> None:
        """Reset the detector state."""
        self.attention_history.clear()
        self.attention_cues.clear()
        self.pending_responses.clear()
        logger.info("Joint Attention Detector reset")