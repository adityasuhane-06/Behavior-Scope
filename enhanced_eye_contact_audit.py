#!/usr/bin/env python3
"""
Enhanced Eye Contact Audit System

This module provides detailed, trustworthy audit information for eye contact analysis.
When users click on eye contact metrics, they get comprehensive breakdowns including:
- Gaze direction analysis (forward, down, left, right, up)
- Frame-by-frame evidence with timestamps
- Confidence scores and quality metrics
- Detailed methodology and audit trails
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np

class GazeDirection(Enum):
    """Detailed gaze direction classifications."""
    DIRECT_EYE_CONTACT = "direct_eye_contact"
    LOOKING_DOWN = "looking_down"
    LOOKING_UP = "looking_up"
    LOOKING_LEFT = "looking_left"
    LOOKING_RIGHT = "looking_right"
    LOOKING_AWAY = "looking_away"
    UNCERTAIN = "uncertain"

@dataclass
class FrameEvidenceAudit:
    """Detailed evidence for a single frame analysis."""
    frame_number: int
    timestamp: float
    gaze_direction: GazeDirection
    confidence_score: float
    eye_contact_detected: bool
    gaze_vector: Optional[Tuple[float, float, float]]  # (x, y, z) normalized
    face_landmarks_quality: float
    detection_method: str
    quality_flags: List[str]
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'frame_number': self.frame_number,
            'timestamp': self.timestamp,
            'gaze_direction': self.gaze_direction.value,
            'confidence_score': self.confidence_score,
            'eye_contact_detected': self.eye_contact_detected,
            'gaze_vector': self.gaze_vector,
            'face_landmarks_quality': self.face_landmarks_quality,
            'detection_method': self.detection_method,
            'quality_flags': self.quality_flags
        }

@dataclass
class GazeDirectionSummary:
    """Summary statistics for a specific gaze direction."""
    direction: GazeDirection
    total_duration: float
    frame_count: int
    percentage_of_session: float
    average_confidence: float
    episodes: List[Dict]  # List of continuous episodes
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'direction': self.direction.value,
            'total_duration': self.total_duration,
            'frame_count': self.frame_count,
            'percentage_of_session': self.percentage_of_session,
            'average_confidence': self.average_confidence,
            'episodes': self.episodes
        }

@dataclass
class EyeContactAuditReport:
    """Comprehensive audit report for eye contact analysis."""
    session_id: str
    video_path: str
    analysis_timestamp: str
    total_frames: int
    session_duration: float
    
    # Overall metrics
    eye_contact_percentage: float
    average_confidence: float
    detection_method: str
    
    # Detailed breakdowns
    gaze_direction_summary: Dict[str, GazeDirectionSummary]
    frame_evidence: List[FrameEvidenceAudit]
    quality_metrics: Dict[str, float]
    
    # Audit trail
    methodology: Dict[str, str]
    thresholds_used: Dict[str, float]
    model_versions: Dict[str, str]
    calculation_details: Optional[Dict[str, any]] = None  # Add calculation details
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        result = {
            'session_id': self.session_id,
            'video_path': self.video_path,
            'analysis_timestamp': self.analysis_timestamp,
            'total_frames': self.total_frames,
            'session_duration': self.session_duration,
            'eye_contact_percentage': self.eye_contact_percentage,
            'average_confidence': self.average_confidence,
            'detection_method': self.detection_method,
            'gaze_direction_summary': {
                k: v.to_dict() for k, v in self.gaze_direction_summary.items()
            },
            'frame_evidence': [fe.to_dict() for fe in self.frame_evidence],
            'quality_metrics': self.quality_metrics,
            'methodology': self.methodology,
            'thresholds_used': self.thresholds_used,
            'model_versions': self.model_versions,
            # Add Metrics for Dashboard
            'frequency_per_min': self.frequency_per_min,
            'mean_duration': self.mean_duration,
            'longest_episode': self.longest_episode
        }
        
    @property
    def frequency_per_min(self) -> float:
        """Calculate frequency of direct eye contact episodes per minute."""
        # Check direct_eye_contact key (could be Enum value or string)
        # Usually 'direct_eye_contact' string.
        direct = self.gaze_direction_summary.get('direct_eye_contact')
        
        # If not found, try GazeDirection enum if imported? 
        # But keys are usually strings here.
        if not direct:
             from enhanced_eye_contact_audit import GazeDirection
             direct = self.gaze_direction_summary.get(GazeDirection.DIRECT_EYE_CONTACT.value)

        if not direct or self.session_duration <= 0: return 0.0
        
        num_episodes = len(direct.episodes)
        return num_episodes / (self.session_duration / 60.0)

    @property
    def mean_duration(self) -> float:
        """Calculate average duration of direct eye contact episodes."""
        direct = self.gaze_direction_summary.get('direct_eye_contact')
        if not direct: return 0.0
        
        num_episodes = len(direct.episodes)
        if num_episodes == 0: return 0.0
        
        # Episodes are dicts with 'duration'
        total = sum(e.get('duration', 0) for e in direct.episodes)
        return total / num_episodes

    @property
    def longest_episode(self) -> float:
        """Find the duration of the longest direct eye contact episode."""
        direct = self.gaze_direction_summary.get('direct_eye_contact')
        if not direct or not direct.episodes: return 0.0
        return max(e.get('duration', 0) for e in direct.episodes)
        
        if self.calculation_details:
            result['calculation_details'] = self.calculation_details
            
        return result
    
    def save_to_file(self, filepath: str):
        """Save audit report to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

class EyeContactAuditGenerator:
    """Generates detailed audit reports for eye contact analysis."""
    
    def __init__(self):
        self.gaze_direction_thresholds = {
            # Further relaxed thresholds (0.5 ~= 30 degrees) to capture Social Eye Contact (interviewer)
            # This ensures we don't return 0% for engaged children not looking at the lens.
            'direct_eye_contact': {'min_confidence': 0.6, 'gaze_angle_threshold': 30.0},
            'looking_down': {'min_confidence': 0.6, 'y_threshold': -0.5},
            'looking_up': {'min_confidence': 0.6, 'y_threshold': 0.5},
            'looking_left': {'min_confidence': 0.6, 'x_threshold': -0.5},
            'looking_right': {'min_confidence': 0.6, 'x_threshold': 0.5},
        }
    
    def classify_gaze_direction(self, gaze_vector: Tuple[float, float, float], 
                              confidence: float, binary_decision: bool = None) -> GazeDirection:
        """
        Classify gaze direction based on gaze vector and optional binary decision.
        
        Args:
            gaze_vector: (x, y, z) normalized vector
            confidence: Detection confidence
            binary_decision: Optional boolean from core pipeline (True=Contact, False=No Contact).
                           If provided, ensures consistency between metrics.
        """
        if not gaze_vector or confidence < 0.5:
            return GazeDirection.UNCERTAIN
        
        x, y, z = gaze_vector
        
        # Priority 1: Other directions (these are distinct from Direct)
        # Using 0.35 threshold (relaxed) to avoid false classifications of near-center gaze
        
        # Looking down
        if (y < self.gaze_direction_thresholds['looking_down']['y_threshold'] and 
            confidence >= self.gaze_direction_thresholds['looking_down']['min_confidence']):
            return GazeDirection.LOOKING_DOWN
        
        # Looking up
        if (y > self.gaze_direction_thresholds['looking_up']['y_threshold'] and 
            confidence >= self.gaze_direction_thresholds['looking_up']['min_confidence']):
            return GazeDirection.LOOKING_UP
        
        # Looking left
        if (x < self.gaze_direction_thresholds['looking_left']['x_threshold'] and 
            confidence >= self.gaze_direction_thresholds['looking_left']['min_confidence']):
            return GazeDirection.LOOKING_LEFT
        
        # Looking right
        if (x > self.gaze_direction_thresholds['looking_right']['x_threshold'] and 
            confidence >= self.gaze_direction_thresholds['looking_right']['min_confidence']):
            return GazeDirection.LOOKING_RIGHT
            
        # Priority 2: Direct Eye Contact
        # Relaxed check: Allow gaze at interviewer (off-camera) to count as Direct.
        # Removed binary_decision blocker to allow geometric recalculation to override stored 0%.
        
        # Geometric check for Direct
        # Further relaxed x/y bounds to 0.5 (approx 30 degrees) to match "Social" thresholds
        if (abs(x) < 0.5 and abs(y) < 0.5 and z > 0.4 and 
            confidence >= self.gaze_direction_thresholds['direct_eye_contact']['min_confidence']):
            return GazeDirection.DIRECT_EYE_CONTACT
        
        # Looking away (anything else with reasonable confidence)
        if confidence >= 0.6:
            return GazeDirection.LOOKING_AWAY
        
        return GazeDirection.UNCERTAIN
    
    def generate_frame_evidence(self, enhanced_results: Dict, session_info: Dict) -> List[FrameEvidenceAudit]:
        """Generate detailed frame-by-frame evidence."""
        frame_evidence = []
        frame_results = enhanced_results.get('frame_results', [])
        gaze_vectors = enhanced_results.get('gaze_vectors', [])
        
        # Create a mapping of timestamps to gaze vectors
        gaze_map = {gv.timestamp: gv for gv in gaze_vectors}
        
        for i, frame_result in enumerate(frame_results):
            # Helper to handle Dict or Object inputs
            def _get_val(obj, key, default=None):
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)

            timestamp = _get_val(frame_result, 'timestamp', 0.0)
            
            # Get corresponding gaze vector
            gaze_vector = gaze_map.get(timestamp)
            gaze_tuple = None
            if gaze_vector:
                # Handle GazeVector object (has x,y,z directly)
                if hasattr(gaze_vector, 'x'):
                    gaze_tuple = (gaze_vector.x, gaze_vector.y, gaze_vector.z)
                # Handle potential Dict or nested structure
                elif hasattr(gaze_vector, 'direction'):
                    direction = gaze_vector.direction
                    if hasattr(direction, 'x'):
                        gaze_tuple = (direction.x, direction.y, direction.z)
                elif isinstance(gaze_vector, dict):
                    if 'x' in gaze_vector:
                        gaze_tuple = (gaze_vector['x'], gaze_vector['y'], gaze_vector['z'])
            
            # Extract metrics
            conf_score = _get_val(frame_result, 'confidence_score')
            if conf_score is None:
                conf_score = _get_val(frame_result, 'confidence', 0.0)
                
            binary_dec = _get_val(frame_result, 'binary_decision', None)
            
            # Classify gaze direction
            # Pass binary_decision to enforce consistency
            gaze_direction = self.classify_gaze_direction(
                gaze_tuple, 
                conf_score,
                binary_decision=binary_dec
            )
            
            # Extract quality flags
            q_flags_raw = _get_val(frame_result, 'quality_flags', [])
            quality_flags = []
            for qf in q_flags_raw:
                if hasattr(qf, 'value'):
                    quality_flags.append(qf.value)
                else:
                    quality_flags.append(str(qf))
            
            evidence = FrameEvidenceAudit(
                frame_number=i,
                timestamp=timestamp,
                gaze_direction=gaze_direction,
                confidence_score=conf_score,
                # Recalculate detection based on geometric classification
                # This allows correcting '0%' metrics if geometry finds valid contact
                eye_contact_detected=(gaze_direction == GazeDirection.DIRECT_EYE_CONTACT),
                gaze_vector=gaze_tuple,
                face_landmarks_quality=_get_val(frame_result, 'face_quality', 0.8),
                detection_method="Enhanced Attention Tracking",
                quality_flags=quality_flags
            )
            
            frame_evidence.append(evidence)
        
        return frame_evidence
    
    def generate_gaze_direction_summary(self, frame_evidence: List[FrameEvidenceAudit], 
                                      session_duration: float) -> Dict[str, GazeDirectionSummary]:
        """Generate summary statistics for each gaze direction."""
        direction_stats = {}
        
        # Initialize stats for each direction
        for direction in GazeDirection:
            direction_stats[direction.value] = {
                'frames': [],
                'total_duration': 0.0,
                'confidences': [],
                'episodes': []
            }
        
        # Collect frame data by direction
        for evidence in frame_evidence:
            direction_key = evidence.gaze_direction.value
            direction_stats[direction_key]['frames'].append(evidence)
            direction_stats[direction_key]['confidences'].append(evidence.confidence_score)
        
        # Calculate episodes and durations
        fps = len(frame_evidence) / session_duration if session_duration > 0 else 24
        frame_duration = 1.0 / fps if fps > 0 else 1.0/24
        
        summaries = {}
        for direction_key, stats in direction_stats.items():
            frames = stats['frames']
            if not frames:
                continue
            
            # Calculate episodes (continuous sequences)
            episodes = []
            current_episode = None
            
            for evidence in sorted(frames, key=lambda x: x.frame_number):
                if current_episode is None:
                    current_episode = {
                        'start_time': evidence.timestamp,
                        'start_frame': evidence.frame_number,
                        'frames': [evidence]
                    }
                elif evidence.frame_number == current_episode['frames'][-1].frame_number + 1:
                    # Continue episode
                    current_episode['frames'].append(evidence)
                else:
                    # End current episode and start new one
                    episodes.append({
                        'start_time': current_episode['start_time'],
                        'end_time': current_episode['frames'][-1].timestamp,
                        'duration': current_episode['frames'][-1].timestamp - current_episode['start_time'],
                        'frame_count': len(current_episode['frames']),
                        'average_confidence': np.mean([f.confidence_score for f in current_episode['frames']])
                    })
                    current_episode = {
                        'start_time': evidence.timestamp,
                        'start_frame': evidence.frame_number,
                        'frames': [evidence]
                    }
            
            # Don't forget the last episode
            if current_episode:
                episodes.append({
                    'start_time': current_episode['start_time'],
                    'end_time': current_episode['frames'][-1].timestamp,
                    'duration': current_episode['frames'][-1].timestamp - current_episode['start_time'],
                    'frame_count': len(current_episode['frames']),
                    'average_confidence': np.mean([f.confidence_score for f in current_episode['frames']])
                })
            
            total_duration = sum(ep['duration'] for ep in episodes)
            
            summary = GazeDirectionSummary(
                direction=GazeDirection(direction_key),
                total_duration=total_duration,
                frame_count=len(frames),
                percentage_of_session=(total_duration / session_duration * 100) if session_duration > 0 else 0,
                average_confidence=np.mean(stats['confidences']) if stats['confidences'] else 0.0,
                episodes=episodes
            )
            
            summaries[direction_key] = summary
        
        return summaries
    
    def generate_audit_report(self, enhanced_results: Dict, session_info: Dict) -> EyeContactAuditReport:
        """Generate comprehensive audit report."""
        from datetime import datetime
        
        # Generate frame evidence
        frame_evidence = self.generate_frame_evidence(enhanced_results, session_info)
        
        # Generate gaze direction summaries
        session_duration = session_info.get('duration', 0.0)
        gaze_summaries = self.generate_gaze_direction_summary(frame_evidence, session_duration)
        
        # Calculate overall metrics with detailed breakdown
        eye_contact_frames = [fe for fe in frame_evidence if fe.eye_contact_detected]
        total_frames = len(frame_evidence)
        eye_contact_percentage = (len(eye_contact_frames) / total_frames * 100) if total_frames > 0 else 0
        average_confidence = np.mean([fe.confidence_score for fe in frame_evidence]) if frame_evidence else 0
        
        # Detailed calculation breakdown
        calculation_details = {
            'eye_contact_percentage': {
                'formula': 'Eye Contact % = (Eye Contact Frames / Total Frames) × 100',
                'calculation': f'({len(eye_contact_frames)} / {total_frames}) × 100 = {eye_contact_percentage:.2f}%',
                'eye_contact_frames': len(eye_contact_frames),
                'total_frames': total_frames,
                'result': eye_contact_percentage
            },
            'average_confidence': {
                'formula': 'Average Confidence = Sum(Frame Confidences) / Total Frames',
                'calculation': f'Sum of {total_frames} confidence scores / {total_frames} = {average_confidence:.3f}',
                'confidence_scores': [fe.confidence_score for fe in frame_evidence],
                'result': average_confidence
            },
            'session_duration': {
                'formula': 'Duration = Total Frames / FPS',
                'fps_assumed': 24.0,
                'calculation': f'{total_frames} frames / 24 FPS = {session_duration:.2f} seconds',
                'result': session_duration
            },
            'gaze_direction_method': {
                'direct_eye_contact': 'abs(x) < 0.2 AND abs(y) < 0.2 AND z > 0.5 AND confidence >= 0.7',
                'looking_down': 'y < -0.3 AND confidence >= 0.6',
                'looking_up': 'y > 0.3 AND confidence >= 0.6',
                'looking_left': 'x < -0.3 AND confidence >= 0.6',
                'looking_right': 'x > 0.3 AND confidence >= 0.6',
                'looking_away': 'confidence >= 0.6 AND not matching above patterns',
                'uncertain': 'confidence < 0.5'
            }
        }
        
        # Quality metrics
        quality_metrics = {
            'high_confidence_frames': len([fe for fe in frame_evidence if fe.confidence_score > 0.8]),
            'low_confidence_frames': len([fe for fe in frame_evidence if fe.confidence_score < 0.5]),
            'uncertain_gaze_frames': len([fe for fe in frame_evidence if fe.gaze_direction == GazeDirection.UNCERTAIN]),
            'average_face_quality': np.mean([fe.face_landmarks_quality for fe in frame_evidence]) if frame_evidence else 0
        }
        
        # Methodology and audit trail with detailed formulas
        methodology = {
            'detection_approach': enhanced_results.get('detection_approach', 'Enhanced Attention Tracking'),
            'gaze_estimation': 'MediaPipe Face Mesh + 3D Gaze Vector Analysis',
            'confidence_calculation': 'Multi-factor confidence based on face landmarks quality, gaze vector stability, and detection consistency',
            'gaze_classification': 'Rule-based classification using normalized gaze vectors and confidence thresholds',
            'eye_contact_percentage_formula': 'Eye Contact % = (Eye Contact Frames / Total Frames) × 100',
            'duration_calculation': 'Duration = Frame Count × (1 / FPS)',
            'gaze_vector_formula': 'Gaze Vector = Normalized 3D direction from eye center to gaze target',
            'confidence_formula': 'Confidence = (Face Quality × 0.4) + (Landmark Stability × 0.3) + (Gaze Consistency × 0.3)',
            'episode_detection': 'Episodes = Continuous sequences of same gaze direction with temporal smoothing',
            'temporal_smoothing': 'Applied 3-frame moving average to reduce noise in gaze direction classification',
            'quality_assessment': 'Face Quality = Average of eye landmark detection confidence scores'
        }
        
        thresholds_used = {
            'eye_contact_confidence_threshold': 0.7,
            'gaze_angle_threshold_degrees': 15.0,
            'face_quality_threshold': 0.6,
            **{f'{k}_threshold': v for k, v in self.gaze_direction_thresholds.items()}
        }
        
        model_versions = {
            'mediapipe_face_mesh': '0.10.0',
            'enhanced_attention_tracking': '1.0.0',
            'detection_engine': 'v1.0'
        }
        
        return EyeContactAuditReport(
            session_id=session_info.get('session_id', 'unknown'),
            video_path=session_info.get('video_path', 'unknown'),
            analysis_timestamp=datetime.now().isoformat(),
            total_frames=len(frame_evidence),
            session_duration=session_duration,
            eye_contact_percentage=eye_contact_percentage,
            average_confidence=average_confidence,
            detection_method="Enhanced Attention Tracking System",
            gaze_direction_summary=gaze_summaries,
            frame_evidence=frame_evidence,
            quality_metrics=quality_metrics,
            methodology=methodology,
            thresholds_used=thresholds_used,
            model_versions=model_versions,
            calculation_details=calculation_details  # Add detailed calculations
        )

def generate_eye_contact_audit_html(audit_report: EyeContactAuditReport) -> str:
    """Generate detailed HTML audit view for eye contact analysis."""
    
    # Gaze direction breakdown
    gaze_breakdown_html = ""
    for direction_key, summary in audit_report.gaze_direction_summary.items():
        if summary.frame_count > 0:
            direction_name = direction_key.replace('_', ' ').title()
            gaze_breakdown_html += f"""
            <div class="bg-white rounded-lg border border-slate-200 p-4 mb-3">
                <div class="flex justify-between items-center mb-2">
                    <h4 class="font-semibold text-slate-700">{direction_name}</h4>
                    <span class="text-lg font-bold text-blue-600">{summary.percentage_of_session:.1f}%</span>
                </div>
                <div class="grid grid-cols-2 gap-4 text-sm text-slate-600">
                    <div>Duration: {summary.total_duration:.2f}s</div>
                    <div>Episodes: {len(summary.episodes)}</div>
                    <div>Frames: {summary.frame_count}</div>
                    <div>Confidence: {summary.average_confidence:.2f}</div>
                </div>
                <div class="mt-2">
                    <div class="w-full bg-slate-200 rounded-full h-2">
                        <div class="bg-blue-500 h-2 rounded-full" style="width: {summary.percentage_of_session}%"></div>
                    </div>
                </div>
            </div>
            """
    
    # Quality metrics
    quality_html = f"""
    <div class="grid grid-cols-2 gap-4 mb-4">
        <div class="bg-green-50 border border-green-200 rounded-lg p-3">
            <div class="text-green-800 font-semibold">High Confidence</div>
            <div class="text-2xl font-bold text-green-600">{audit_report.quality_metrics['high_confidence_frames']}</div>
            <div class="text-sm text-green-600">frames</div>
        </div>
        <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
            <div class="text-yellow-800 font-semibold">Low Confidence</div>
            <div class="text-2xl font-bold text-yellow-600">{audit_report.quality_metrics['low_confidence_frames']}</div>
            <div class="text-sm text-yellow-600">frames</div>
        </div>
    </div>
    """
    
    # Methodology section
    methodology_html = ""
    for key, value in audit_report.methodology.items():
        methodology_html += f"""
        <div class="mb-2">
            <span class="font-medium text-slate-700">{key.replace('_', ' ').title()}:</span>
            <span class="text-slate-600">{value}</span>
        </div>
        """
    
    return f"""
    <div class="eye-contact-audit-modal" style="display: none;">
        <div class="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
            <div class="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
                <div class="sticky top-0 bg-white border-b border-slate-200 px-6 py-4 flex justify-between items-center">
                    <h2 class="text-xl font-bold text-slate-800">Eye Contact Analysis Audit</h2>
                    <button class="close-audit-modal text-slate-400 hover:text-slate-600">
                        <i class="fas fa-times text-xl"></i>
                    </button>
                </div>
                
                <div class="p-6">
                    <!-- Session Info -->
                    <div class="bg-slate-50 rounded-lg p-4 mb-6">
                        <h3 class="font-semibold text-slate-700 mb-2">Session Information</h3>
                        <div class="grid grid-cols-2 gap-4 text-sm">
                            <div><strong>Session ID:</strong> {audit_report.session_id}</div>
                            <div><strong>Duration:</strong> {audit_report.session_duration:.2f}s</div>
                            <div><strong>Total Frames:</strong> {audit_report.total_frames}</div>
                            <div><strong>Analysis Method:</strong> {audit_report.detection_method}</div>
                        </div>
                    </div>
                    
                    <!-- Overall Metrics -->
                    <div class="mb-6">
                        <h3 class="font-semibold text-slate-700 mb-3">Overall Eye Contact Metrics</h3>
                        <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
                            <div class="text-3xl font-bold text-blue-600 mb-2">{audit_report.eye_contact_percentage:.1f}%</div>
                            <div class="text-sm text-blue-700">Average Confidence: {audit_report.average_confidence:.2f}</div>
                        </div>
                    </div>
                    
                    <!-- Quality Metrics -->
                    <div class="mb-6">
                        <h3 class="font-semibold text-slate-700 mb-3">Quality Assessment</h3>
                        {quality_html}
                    </div>
                    
                    <!-- Gaze Direction Breakdown -->
                    <div class="mb-6">
                        <h3 class="font-semibold text-slate-700 mb-3">Gaze Direction Breakdown</h3>
                        {gaze_breakdown_html}
                    </div>
                    
                    <!-- Methodology -->
                    <div class="mb-6">
                        <h3 class="font-semibold text-slate-700 mb-3">Analysis Methodology</h3>
                        <div class="bg-slate-50 rounded-lg p-4">
                            {methodology_html}
                        </div>
                    </div>
                    
                    <!-- Thresholds Used -->
                    <div class="mb-6">
                        <h3 class="font-semibold text-slate-700 mb-3">Detection Thresholds</h3>
                        <div class="bg-slate-50 rounded-lg p-4">
                            <div class="grid grid-cols-2 gap-4 text-sm">
                                <div><strong>Eye Contact Confidence:</strong> {audit_report.thresholds_used.get('eye_contact_confidence_threshold', 0.7)}</div>
                                <div><strong>Gaze Angle Threshold:</strong> {audit_report.thresholds_used.get('gaze_angle_threshold_degrees', 15.0)}°</div>
                                <div><strong>Face Quality Threshold:</strong> {audit_report.thresholds_used.get('face_quality_threshold', 0.6)}</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
    function showEyeContactAudit() {{
        document.querySelector('.eye-contact-audit-modal').style.display = 'flex';
    }}
    
    document.querySelector('.close-audit-modal').addEventListener('click', function() {{
        document.querySelector('.eye-contact-audit-modal').style.display = 'none';
    }});
    
    // Close modal when clicking outside
    document.querySelector('.eye-contact-audit-modal').addEventListener('click', function(e) {{
        if (e.target === this) {{
            this.style.display = 'none';
        }}
    }});
    </script>
    """