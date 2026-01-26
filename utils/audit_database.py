"""
Audit Database Module for Behavior Scope.

Provides comprehensive audit trail storage for all analysis results,
enabling verification, querying, and tracking of system outputs.

Key features:
- Centralized SQLite database for all analysis results
- Structured storage of scores, metrics, and metadata
- Query interface for historical analysis lookup
- Audit trail with timestamps, versions, and confidence levels
- Human-readable data export
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class SessionAudit:
    """Complete audit record for a single analysis session."""
    session_id: str
    video_path: str
    timestamp: str
    config_hash: str  # Hash of configuration used

    # Behavioral scores
    vocal_regulation_index: float
    motor_agitation_index: float
    attention_stability_score: float
    regulation_consistency_index: float
    facial_affect_index: Optional[float] = None

    # Autism analysis (if enabled)
    turn_taking_score: Optional[float] = None
    eye_contact_score: Optional[float] = None
    social_engagement_index: Optional[float] = None
    stereotypy_percentage: Optional[float] = None

    # Clinical analysis (if enabled)
    stuttering_severity_index: Optional[float] = None
    responsiveness_index: Optional[float] = None

    # Metadata
    audio_segments_detected: int = 0
    instability_windows_detected: int = 0
    fused_evidence_count: int = 0
    high_confidence_segments: int = 0
    medium_confidence_segments: int = 0
    low_confidence_segments: int = 0

    # Model/system versions
    system_version: str = "1.0.0"
    config_version: str = "unknown"

    # Processing info
    processing_duration_sec: Optional[float] = None
    participant_age: Optional[int] = None


class AuditDatabase:
    """
    Audit database manager for Behavior Scope analysis results.

    Stores all analysis results in a structured SQLite database for:
    - Auditability: Track what the system said and when
    - Verification: Compare results across sessions
    - Querying: Retrieve historical analyses
    - Reporting: Generate aggregate statistics
    """

    def __init__(self, db_path: str = "data/audit/behavior_scope_audit.db"):
        """
        Initialize audit database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database schema
        self._init_database()

        logger.info(f"Audit database initialized: {self.db_path}")

    def _init_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Main sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    video_path TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    config_hash TEXT,

                    -- Behavioral Scores
                    vocal_regulation_index REAL,
                    motor_agitation_index REAL,
                    attention_stability_score REAL,
                    regulation_consistency_index REAL,
                    facial_affect_index REAL,

                    -- Autism Analysis
                    turn_taking_score REAL,
                    eye_contact_score REAL,
                    social_engagement_index REAL,
                    stereotypy_percentage REAL,

                    -- Enhanced Attention Tracking
                    enhanced_eye_contact_percentage REAL,
                    enhanced_gaze_stability REAL,
                    enhanced_detection_approach TEXT,
                    enhanced_total_frames INTEGER,
                    enhanced_gaze_vectors INTEGER,
                    enhanced_joint_attention_episodes INTEGER,
                    enhanced_zone_events INTEGER,
                    enhanced_attention_stability REAL,
                    enhanced_repetitive_behavior_score REAL,
                    enhanced_scanning_pattern TEXT,
                    enhanced_data_quality_score REAL,

                    -- Clinical Analysis
                    stuttering_severity_index REAL,
                    responsiveness_index REAL,

                    -- Transcript Data
                    transcript_text TEXT,
                    transcript_json TEXT,
                    transcript_word_count INTEGER,
                    transcript_speaker_count INTEGER,

                    -- Metadata
                    audio_segments_detected INTEGER,
                    instability_windows_detected INTEGER,
                    fused_evidence_count INTEGER,
                    high_confidence_segments INTEGER,
                    medium_confidence_segments INTEGER,
                    low_confidence_segments INTEGER,

                    -- System Info
                    system_version TEXT,
                    config_version TEXT,
                    processing_duration_sec REAL,
                    participant_age INTEGER,

                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Detailed scores table (stores full score dictionaries)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detailed_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    score_type TEXT NOT NULL,
                    score_data JSON NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # Autism analysis details
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS autism_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    analysis_data JSON NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # Clinical analysis details
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS clinical_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    analysis_data JSON NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # Segments table (individual dysregulation segments)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    start_frame INTEGER,
                    end_frame INTEGER,
                    confidence_level TEXT,
                    audio_score REAL,
                    video_score REAL,
                    combined_score REAL,
                    indicators JSON,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # Enhanced Attention Tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS enhanced_attention_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    detection_approach TEXT NOT NULL,
                    confidence_threshold REAL NOT NULL,
                    total_frames_processed INTEGER NOT NULL,
                    eye_contact_frames INTEGER NOT NULL,
                    eye_contact_percentage REAL NOT NULL,
                    average_confidence REAL NOT NULL,
                    gaze_stability_score REAL NOT NULL,
                    gaze_vectors_generated INTEGER NOT NULL,
                    joint_attention_episodes INTEGER NOT NULL,
                    zone_events INTEGER NOT NULL,
                    visual_tracking_data JSON,
                    joint_attention_data JSON,
                    zone_attention_data JSON,
                    quality_flags JSON,
                    frame_results JSON,
                    clinical_interpretation JSON,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # Configuration snapshots
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS configurations (
                    config_hash TEXT PRIMARY KEY,
                    config_data JSON NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Frame-level analysis (detailed frame-by-frame tracking)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS frame_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    frame_number INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    analysis_type TEXT NOT NULL,
                    face_detected BOOLEAN,
                    pose_detected BOOLEAN,
                    eye_contact_detected BOOLEAN,
                    movement_detected BOOLEAN,
                    action_units JSON,
                    confidence_score REAL,
                    details JSON,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # Analysis log (for tracking all system operations)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # Create indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_timestamp
                ON sessions(timestamp)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_video_path
                ON sessions(video_path)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_segments_session
                ON segments(session_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_frame_analysis_session
                ON frame_analysis(session_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_frame_analysis_frame
                ON frame_analysis(frame_number)
            """)

            conn.commit()

    def save_session(self, audit_record: SessionAudit):
        """
        Save complete session audit record.

        Args:
            audit_record: SessionAudit object with all analysis results
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            record_dict = asdict(audit_record)

            cursor.execute("""
                INSERT INTO sessions (
                    session_id, video_path, timestamp, config_hash,
                    vocal_regulation_index, motor_agitation_index,
                    attention_stability_score, regulation_consistency_index,
                    facial_affect_index,
                    turn_taking_score, eye_contact_score,
                    social_engagement_index, stereotypy_percentage,
                    stuttering_severity_index, responsiveness_index,
                    audio_segments_detected, instability_windows_detected,
                    fused_evidence_count, high_confidence_segments,
                    medium_confidence_segments, low_confidence_segments,
                    system_version, config_version,
                    processing_duration_sec, participant_age
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record_dict['session_id'],
                record_dict['video_path'],
                record_dict['timestamp'],
                record_dict['config_hash'],
                record_dict['vocal_regulation_index'],
                record_dict['motor_agitation_index'],
                record_dict['attention_stability_score'],
                record_dict['regulation_consistency_index'],
                record_dict['facial_affect_index'],
                record_dict['turn_taking_score'],
                record_dict['eye_contact_score'],
                record_dict['social_engagement_index'],
                record_dict['stereotypy_percentage'],
                record_dict['stuttering_severity_index'],
                record_dict['responsiveness_index'],
                record_dict['audio_segments_detected'],
                record_dict['instability_windows_detected'],
                record_dict['fused_evidence_count'],
                record_dict['high_confidence_segments'],
                record_dict['medium_confidence_segments'],
                record_dict['low_confidence_segments'],
                record_dict['system_version'],
                record_dict['config_version'],
                record_dict['processing_duration_sec'],
                record_dict['participant_age']
            ))

            conn.commit()
            logger.info(f"✓ Session audit saved: {audit_record.session_id}")

    def save_detailed_scores(self, session_id: str, scores: Dict[str, Any]):
        """
        Save detailed score breakdowns.

        Args:
            session_id: Session identifier
            scores: Dictionary of scores with detailed breakdowns
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for score_type, score_data in scores.items():
                cursor.execute("""
                    INSERT INTO detailed_scores (session_id, score_type, score_data)
                    VALUES (?, ?, ?)
                """, (session_id, score_type, json.dumps(score_data)))

            conn.commit()
            logger.info(f"✓ Detailed scores saved for session: {session_id}")

    def save_enhanced_attention_tracking(self, session_id: str, enhanced_data: Dict[str, Any], enhanced_results: Dict[str, Any]):
        """
        Save enhanced attention tracking analysis results.

        Args:
            session_id: Session identifier
            enhanced_data: Enhanced attention tracking summary data
            enhanced_results: Raw enhanced attention tracking results
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Extract key metrics
            eye_contact_percentage = enhanced_data.get('eye_contact_percentage', 0.0)
            gaze_stability = enhanced_data.get('gaze_stability', 0.0)
            detection_approach = enhanced_data.get('detection_approach', 'hybrid')
            total_frames = enhanced_data.get('total_frames', 0)
            gaze_vectors = enhanced_data.get('gaze_vectors_generated', 0)
            
            # Count events
            joint_attention_episodes = len(enhanced_results.get('joint_attention_events', []))
            zone_events = len(enhanced_results.get('zone_events', []))
            eye_contact_frames = sum(1 for fr in enhanced_results.get('frame_results', []) if fr.binary_decision)
            
            # Calculate average confidence
            frame_results = enhanced_results.get('frame_results', [])
            avg_confidence = sum(fr.confidence_score for fr in frame_results) / len(frame_results) if frame_results else 0.0
            
            # Extract visual tracking data
            visual_tracking = enhanced_data.get('visual_tracking', {})
            joint_attention_data = enhanced_data.get('joint_attention', {})
            zone_attention_data = enhanced_data.get('zone_attention', {})
            quality_flags = enhanced_data.get('quality_flags', {})
            
            # Generate clinical interpretation
            clinical_interpretation = {
                'eye_contact_assessment': 'Strong' if eye_contact_percentage >= 70 else 'Moderate' if eye_contact_percentage >= 40 else 'Reduced',
                'gaze_stability_assessment': 'Excellent' if gaze_stability >= 0.8 else 'Good' if gaze_stability >= 0.6 else 'Moderate' if gaze_stability >= 0.4 else 'Low',
                'attention_patterns': [],
                'recommendations': []
            }
            
            if gaze_stability < 0.4:
                clinical_interpretation['attention_patterns'].append('Attention instability noted')
                clinical_interpretation['recommendations'].append('Consider attention stability assessment')
            
            if visual_tracking.get('repetitive_behavior_score', 0) > 0.7:
                clinical_interpretation['attention_patterns'].append('High repetitive visual behavior patterns')
                clinical_interpretation['recommendations'].append('Monitor repetitive visual behaviors')
            
            if joint_attention_episodes == 0:
                clinical_interpretation['attention_patterns'].append('Limited joint attention episodes')
                clinical_interpretation['recommendations'].append('Consider joint attention intervention')

            cursor.execute("""
                INSERT INTO enhanced_attention_tracking (
                    session_id, detection_approach, confidence_threshold,
                    total_frames_processed, eye_contact_frames, eye_contact_percentage,
                    average_confidence, gaze_stability_score, gaze_vectors_generated,
                    joint_attention_episodes, zone_events,
                    visual_tracking_data, joint_attention_data, zone_attention_data,
                    quality_flags, frame_results, clinical_interpretation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                detection_approach,
                enhanced_data.get('confidence_threshold', 0.6),
                total_frames,
                eye_contact_frames,
                eye_contact_percentage,
                avg_confidence,
                gaze_stability,
                gaze_vectors,
                joint_attention_episodes,
                zone_events,
                json.dumps(visual_tracking),
                json.dumps(joint_attention_data),
                json.dumps(zone_attention_data),
                json.dumps(quality_flags),
                json.dumps([{
                    'timestamp': fr.timestamp,
                    'confidence': fr.confidence_score,
                    'binary_decision': fr.binary_decision,
                    'gaze_target': fr.gaze_target.value if fr.gaze_target else None
                } for fr in frame_results[:100]]),  # Store first 100 frames to avoid size issues
                json.dumps(clinical_interpretation)
            ))

            # Also update the main sessions table with enhanced data
            cursor.execute("""
                UPDATE sessions SET
                    enhanced_eye_contact_percentage = ?,
                    enhanced_gaze_stability = ?,
                    enhanced_detection_approach = ?,
                    enhanced_total_frames = ?,
                    enhanced_gaze_vectors = ?,
                    enhanced_joint_attention_episodes = ?,
                    enhanced_zone_events = ?,
                    enhanced_attention_stability = ?,
                    enhanced_repetitive_behavior_score = ?,
                    enhanced_scanning_pattern = ?,
                    enhanced_data_quality_score = ?
                WHERE session_id = ?
            """, (
                eye_contact_percentage,
                gaze_stability,
                detection_approach,
                total_frames,
                gaze_vectors,
                joint_attention_episodes,
                zone_events,
                visual_tracking.get('attention_stability', 0.0),
                visual_tracking.get('repetitive_behavior_score', 0.0),
                visual_tracking.get('scanning_pattern', 'unknown'),
                (gaze_vectors / total_frames) * 100 if total_frames > 0 else 0.0,  # Data quality score
                session_id
            ))

            conn.commit()
            logger.info(f"✓ Enhanced attention tracking data saved for session: {session_id}")

    def save_autism_analysis(self, session_id: str, autism_results: Dict[str, Any]):
        """
        Save autism-specific analysis results.

        Args:
            session_id: Session identifier
            autism_results: Dictionary of autism analysis results
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for analysis_type, analysis_data in autism_results.items():
                # Convert dataclass objects to dicts
                if hasattr(analysis_data, '__dict__'):
                    data_dict = asdict(analysis_data) if hasattr(analysis_data, '__dataclass_fields__') else vars(analysis_data)
                else:
                    data_dict = analysis_data

                cursor.execute("""
                    INSERT INTO autism_analysis (session_id, analysis_type, analysis_data)
                    VALUES (?, ?, ?)
                """, (session_id, analysis_type, json.dumps(data_dict, default=str)))

            conn.commit()
            logger.info(f"✓ Autism analysis saved for session: {session_id}")

    def save_clinical_analysis(self, session_id: str, clinical_results: Dict[str, Any]):
        """
        Save clinical analysis results.

        Args:
            session_id: Session identifier
            clinical_results: Dictionary of clinical analysis results
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for analysis_type, analysis_data in clinical_results.items():
                # Convert dataclass objects to dicts
                if hasattr(analysis_data, '__dict__'):
                    data_dict = asdict(analysis_data) if hasattr(analysis_data, '__dataclass_fields__') else vars(analysis_data)
                else:
                    data_dict = analysis_data

                cursor.execute("""
                    INSERT INTO clinical_analysis (session_id, analysis_type, analysis_data)
                    VALUES (?, ?, ?)
                """, (session_id, analysis_type, json.dumps(data_dict, default=str)))

            conn.commit()
            logger.info(f"✓ Clinical analysis saved for session: {session_id}")
    
    def save_clinical_transcript(self, session_id: str, clinical_transcript):
        """
        Save clinical transcript with behavioral annotations.

        Args:
            session_id: Session identifier
            clinical_transcript: ClinicalTranscript object
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Convert clinical transcript to dict
            clinical_data = {
                'segments': [
                    {
                        'start_time': seg.start_time,
                        'end_time': seg.end_time,
                        'speaker': seg.speaker,
                        'text': seg.text,
                        'sentiment': seg.sentiment,
                        'tone': seg.tone,
                        'behavioral_tags': seg.behavioral_tags,
                        'response_latency': seg.response_latency,
                        'annotations': [
                            {
                                'type': ann.type,
                                'start_time': ann.start_time,
                                'end_time': ann.end_time,
                                'description': ann.description,
                                'severity': ann.severity,
                                'metadata': ann.metadata
                            }
                            for ann in seg.annotations
                        ]
                    }
                    for seg in clinical_transcript.segments
                ],
                'summary': clinical_transcript.summary,
                'behavioral_patterns': clinical_transcript.behavioral_patterns,
                'clinical_insights': clinical_transcript.clinical_insights
            }

            # Save to clinical_analysis table
            cursor.execute("""
                INSERT INTO clinical_analysis (session_id, analysis_type, analysis_data)
                VALUES (?, ?, ?)
            """, (session_id, 'clinical_transcript', json.dumps(clinical_data, default=str)))

            conn.commit()
            logger.info(f"✓ Clinical transcript saved for session: {session_id}")
    
    def get_clinical_transcript(self, session_id: str) -> Dict:
        """
        Get clinical transcript for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with clinical transcript data
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT analysis_data
                FROM clinical_analysis
                WHERE session_id = ? AND analysis_type = 'clinical_transcript'
                ORDER BY id DESC
                LIMIT 1
            """, (session_id,))

            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            else:
                return {}

    def save_transcript_data(self, session_id: str, transcript_segments: List, transcript_text: str = None):
        """
        Save transcript data to the database.

        Args:
            session_id: Session identifier
            transcript_segments: List of TranscriptSegment objects
            transcript_text: Formatted transcript text (optional)
        """
        import json
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Convert transcript segments to JSON
            transcript_json = json.dumps([
                {
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'speaker_id': seg.speaker_id,
                    'text': seg.text,
                    'confidence': seg.confidence,
                    'language': seg.language,
                    'words': [
                        {
                            'word': w.word,
                            'start_time': w.start_time,
                            'end_time': w.end_time,
                            'confidence': w.confidence
                        }
                        for w in seg.words
                    ]
                }
                for seg in transcript_segments
            ], default=str)

            # Calculate statistics
            word_count = sum(len(seg.text.split()) for seg in transcript_segments if seg.text)
            speaker_count = len(set(seg.speaker_id for seg in transcript_segments if seg.speaker_id))

            # Generate text if not provided
            if transcript_text is None:
                from audio_pipeline.transcription import format_transcript_as_text
                transcript_text = format_transcript_as_text(transcript_segments)

            # Update session with transcript data
            cursor.execute("""
                UPDATE sessions 
                SET transcript_text = ?, transcript_json = ?, 
                    transcript_word_count = ?, transcript_speaker_count = ?
                WHERE session_id = ?
            """, (
                transcript_text,
                transcript_json,
                word_count,
                speaker_count,
                session_id
            ))

            conn.commit()
            logger.info(f"✓ Transcript data saved for session: {session_id} ({word_count} words, {speaker_count} speakers)")

    def update_transcript_text(self, session_id: str, updated_text: str):
        """
        Update the transcript text (for editing functionality).

        Args:
            session_id: Session identifier
            updated_text: Updated transcript text
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE sessions 
                SET transcript_text = ?
                WHERE session_id = ?
            """, (updated_text, session_id))

            conn.commit()
            logger.info(f"✓ Transcript text updated for session: {session_id}")

    def get_transcript_data(self, session_id: str) -> Dict:
        """
        Get transcript data for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with transcript data
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT transcript_text, transcript_json, transcript_word_count, transcript_speaker_count
                FROM sessions 
                WHERE session_id = ?
            """, (session_id,))

            row = cursor.fetchone()
            if row:
                return {
                    'transcript_text': row[0],
                    'transcript_json': row[1],
                    'word_count': row[2],
                    'speaker_count': row[3]
                }
            else:
                return {}

    def save_segments(self, session_id: str, fused_evidence: List, fps: float = 30.0):
        """
        Save individual dysregulation segments with frame information.

        Args:
            session_id: Session identifier
            fused_evidence: List of FusedEvidence objects
            fps: Frames per second (for time-to-frame conversion)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for evidence in fused_evidence:
                # Extract indicators
                indicators = {
                    'audio_indicators': evidence.audio_indicators if hasattr(evidence, 'audio_indicators') else [],
                    'video_indicators': evidence.video_indicators if hasattr(evidence, 'video_indicators') else []
                }

                # Calculate frame numbers from timestamps
                start_frame = int(evidence.start_time * fps) if hasattr(evidence, 'start_time') else None
                end_frame = int(evidence.end_time * fps) if hasattr(evidence, 'end_time') else None

                cursor.execute("""
                    INSERT INTO segments (
                        session_id, start_time, end_time, start_frame, end_frame,
                        confidence_level, audio_score, video_score, combined_score, indicators
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    evidence.start_time,
                    evidence.end_time,
                    start_frame,
                    end_frame,
                    evidence.confidence_level,
                    evidence.audio_confidence,
                    evidence.video_confidence,
                    evidence.combined_score,
                    json.dumps(indicators)
                ))

            conn.commit()
            logger.info(f"✓ {len(fused_evidence)} segments saved for session: {session_id}")

    def save_frame_analysis(self, session_id: str, frame_data: List[Dict]):
        """
        Save frame-level analysis data for detailed tracking.

        Args:
            session_id: Session identifier
            frame_data: List of dictionaries containing frame analysis info
                       Each dict should have: frame_number, timestamp, analysis_type, etc.
        """
        if not frame_data:
            return

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for frame in frame_data:
                # Safely serialize action units and details
                try:
                    action_units_json = json.dumps(frame.get('action_units', []), default=str)
                except (TypeError, ValueError):
                    action_units_json = '[]'

                try:
                    details_json = json.dumps(frame.get('details', {}), default=str)
                except (TypeError, ValueError):
                    details_json = '{}'

                cursor.execute("""
                    INSERT INTO frame_analysis (
                        session_id, frame_number, timestamp, analysis_type,
                        face_detected, pose_detected, eye_contact_detected,
                        movement_detected, action_units, confidence_score, details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    frame.get('frame_number'),
                    frame.get('timestamp'),
                    frame.get('analysis_type', 'general'),
                    frame.get('face_detected', False),
                    frame.get('pose_detected', False),
                    frame.get('eye_contact_detected', False),
                    frame.get('movement_detected', False),
                    action_units_json,
                    frame.get('confidence_score'),
                    details_json
                ))

            conn.commit()
            logger.info(f"✓ {len(frame_data)} frame analyses saved for session: {session_id}")

    def save_configuration(self, config_hash: str, config: Dict):
        """
        Save configuration snapshot.

        Args:
            config_hash: Hash of configuration
            config: Configuration dictionary
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            try:
                cursor.execute("""
                    INSERT INTO configurations (config_hash, config_data)
                    VALUES (?, ?)
                """, (config_hash, json.dumps(config)))
                conn.commit()
                logger.info(f"✓ Configuration saved: {config_hash}")
            except sqlite3.IntegrityError:
                # Configuration already exists
                pass

    def log_operation(self, session_id: str, stage: str, operation: str,
                     status: str, details: str = ""):
        """
        Log analysis operation for audit trail.

        Args:
            session_id: Session identifier
            stage: Pipeline stage (e.g., "audio_analysis", "video_analysis")
            operation: Specific operation (e.g., "voice_activity_detection")
            status: Operation status ("success", "warning", "error")
            details: Additional details or error messages
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO analysis_log (session_id, stage, operation, status, details)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, stage, operation, status, details))

            conn.commit()

    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve complete session data.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with all session data or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get main session data
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()

            if not row:
                return None

            session_data = dict(row)

            # Get detailed scores
            cursor.execute(
                "SELECT score_type, score_data FROM detailed_scores WHERE session_id = ?",
                (session_id,)
            )
            session_data['detailed_scores'] = {
                row['score_type']: json.loads(row['score_data'])
                for row in cursor.fetchall()
            }

            # Get segments
            cursor.execute(
                "SELECT * FROM segments WHERE session_id = ? ORDER BY start_time",
                (session_id,)
            )
            session_data['segments'] = [dict(row) for row in cursor.fetchall()]

            # Get autism analysis
            cursor.execute(
                "SELECT analysis_type, analysis_data FROM autism_analysis WHERE session_id = ?",
                (session_id,)
            )
            session_data['autism_analysis'] = {
                row['analysis_type']: json.loads(row['analysis_data'])
                for row in cursor.fetchall()
            }

            # Get clinical analysis
            cursor.execute(
                "SELECT analysis_type, analysis_data FROM clinical_analysis WHERE session_id = ?",
                (session_id,)
            )
            session_data['clinical_analysis'] = {
                row['analysis_type']: json.loads(row['analysis_data'])
                for row in cursor.fetchall()
            }

            # Get operation log
            cursor.execute(
                "SELECT * FROM analysis_log WHERE session_id = ? ORDER BY timestamp",
                (session_id,)
            )
            session_data['operation_log'] = [dict(row) for row in cursor.fetchall()]

            # Get frame-level analysis (limit to reasonable number for display)
            cursor.execute(
                "SELECT COUNT(*) FROM frame_analysis WHERE session_id = ?",
                (session_id,)
            )
            frame_count = cursor.fetchone()[0]
            session_data['frame_analysis_count'] = frame_count

            if frame_count > 0:
                # Get sample of frame data (first 100 frames)
                cursor.execute(
                    """SELECT * FROM frame_analysis
                       WHERE session_id = ?
                       ORDER BY frame_number
                       LIMIT 100""",
                    (session_id,)
                )
                session_data['frame_analysis_sample'] = [dict(row) for row in cursor.fetchall()]

            # Get enhanced attention tracking data
            cursor.execute(
                "SELECT * FROM enhanced_attention_tracking WHERE session_id = ?",
                (session_id,)
            )
            enhanced_row = cursor.fetchone()
            if enhanced_row:
                enhanced_data = dict(enhanced_row)
                
                # Parse JSON fields
                for json_field in ['visual_tracking_data', 'joint_attention_data', 'zone_attention_data', 
                                 'quality_flags', 'frame_results', 'clinical_interpretation']:
                    if enhanced_data.get(json_field):
                        try:
                            enhanced_data[json_field] = json.loads(enhanced_data[json_field])
                        except:
                            enhanced_data[json_field] = {}
                
                session_data['enhanced_attention_tracking'] = enhanced_data

            return session_data

    def list_sessions(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        List recent sessions.

        Args:
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip

        Returns:
            List of session summary dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    session_id, video_path, timestamp,
                    vocal_regulation_index, motor_agitation_index,
                    attention_stability_score, regulation_consistency_index,
                    fused_evidence_count, high_confidence_segments
                FROM sessions
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))

            return [dict(row) for row in cursor.fetchall()]

    def search_sessions(self, video_path: str = None, min_date: str = None,
                       max_date: str = None) -> List[Dict]:
        """
        Search sessions by criteria.

        Args:
            video_path: Filter by video path (partial match)
            min_date: Minimum timestamp (ISO format)
            max_date: Maximum timestamp (ISO format)

        Returns:
            List of matching session dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM sessions WHERE 1=1"
            params = []

            if video_path:
                query += " AND video_path LIKE ?"
                params.append(f"%{video_path}%")

            if min_date:
                query += " AND timestamp >= ?"
                params.append(min_date)

            if max_date:
                query += " AND timestamp <= ?"
                params.append(max_date)

            query += " ORDER BY timestamp DESC"

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def export_session_report(self, session_id: str, output_path: str):
        """
        Export complete session data as human-readable JSON.

        Args:
            session_id: Session identifier
            output_path: Path to save JSON report
        """
        session_data = self.get_session(session_id)

        if not session_data:
            logger.error(f"Session not found: {session_id}")
            return

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, default=str)

        logger.info(f"✓ Session report exported: {output_path}")

    def get_frame_range(self, session_id: str, start_frame: int = 0,
                       end_frame: int = None, limit: int = 1000) -> List[Dict]:
        """
        Get frame-level analysis data for a specific frame range.

        Args:
            session_id: Session identifier
            start_frame: Starting frame number
            end_frame: Ending frame number (None = all after start)
            limit: Maximum number of frames to return

        Returns:
            List of frame analysis dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if end_frame is not None:
                cursor.execute("""
                    SELECT * FROM frame_analysis
                    WHERE session_id = ? AND frame_number >= ? AND frame_number <= ?
                    ORDER BY frame_number
                    LIMIT ?
                """, (session_id, start_frame, end_frame, limit))
            else:
                cursor.execute("""
                    SELECT * FROM frame_analysis
                    WHERE session_id = ? AND frame_number >= ?
                    ORDER BY frame_number
                    LIMIT ?
                """, (session_id, start_frame, limit))

            return [dict(row) for row in cursor.fetchall()]

    def get_frames_with_detections(self, session_id: str,
                                   detection_type: str = None) -> List[Dict]:
        """
        Get frames where specific detections occurred.

        Args:
            session_id: Session identifier
            detection_type: Type of detection ('face', 'pose', 'eye_contact', 'movement')

        Returns:
            List of frames with detections
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if detection_type == 'face':
                cursor.execute("""
                    SELECT * FROM frame_analysis
                    WHERE session_id = ? AND face_detected = 1
                    ORDER BY frame_number
                """, (session_id,))
            elif detection_type == 'pose':
                cursor.execute("""
                    SELECT * FROM frame_analysis
                    WHERE session_id = ? AND pose_detected = 1
                    ORDER BY frame_number
                """, (session_id,))
            elif detection_type == 'eye_contact':
                cursor.execute("""
                    SELECT * FROM frame_analysis
                    WHERE session_id = ? AND eye_contact_detected = 1
                    ORDER BY frame_number
                """, (session_id,))
            elif detection_type == 'movement':
                cursor.execute("""
                    SELECT * FROM frame_analysis
                    WHERE session_id = ? AND movement_detected = 1
                    ORDER BY frame_number
                """, (session_id,))
            else:
                cursor.execute("""
                    SELECT * FROM frame_analysis
                    WHERE session_id = ?
                    ORDER BY frame_number
                """, (session_id,))

            return [dict(row) for row in cursor.fetchall()]

    def get_statistics(self) -> Dict:
        """
        Get aggregate statistics across all sessions.

        Returns:
            Dictionary with summary statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM sessions")
            total_sessions = cursor.fetchone()[0]

            cursor.execute("""
                SELECT
                    AVG(vocal_regulation_index) as avg_vri,
                    AVG(motor_agitation_index) as avg_mai,
                    AVG(attention_stability_score) as avg_ass,
                    AVG(regulation_consistency_index) as avg_rci,
                    SUM(fused_evidence_count) as total_segments
                FROM sessions
            """)

            stats = cursor.fetchone()

            cursor.execute("SELECT COUNT(*) FROM frame_analysis")
            total_frames = cursor.fetchone()[0]

            return {
                'total_sessions': total_sessions,
                'avg_vocal_regulation_index': stats[0],
                'avg_motor_agitation_index': stats[1],
                'avg_attention_stability_score': stats[2],
                'avg_regulation_consistency_index': stats[3],
                'total_segments_analyzed': stats[4],
                'total_frames_analyzed': total_frames
            }

    def get_session_frames(self, session_id: str) -> List[Dict]:
        """
        Get all frame analysis data for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of all frames with analysis data
        """
        return self.get_frame_range(session_id, start_frame=0, end_frame=None, limit=10000)

    def get_operation_logs(self, session_id: str) -> List[Dict]:
        """
        Get operation logs for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of operation log entries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM analysis_log
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,))

            return [dict(row) for row in cursor.fetchall()]

    def get_frames_with_action_units(self, session_id: str) -> List[Dict]:
        """
        Get frames that have action units data.

        Args:
            session_id: Session identifier

        Returns:
            List of frames with action units data
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM frame_analysis
                WHERE session_id = ? AND action_units IS NOT NULL AND action_units != '[]' AND action_units != '{}'
                ORDER BY frame_number
            """, (session_id,))

            return [dict(row) for row in cursor.fetchall()]

    def get_au_evidence_frames(self, session_id: str, au_number: int) -> List[Dict]:
        """
        Get frames where a specific Action Unit was activated.

        Args:
            session_id: Session identifier
            au_number: Action Unit number

        Returns:
            List of frames with AU activation evidence
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get frames with action units data
            cursor.execute("""
                SELECT * FROM frame_analysis
                WHERE session_id = ? AND action_units IS NOT NULL AND action_units != '[]' AND action_units != '{}'
                ORDER BY frame_number
            """, (session_id,))

            frames = cursor.fetchall()
            evidence_frames = []

            # Filter frames that contain the specific AU
            for frame in frames:
                try:
                    action_units = json.loads(frame['action_units']) if frame['action_units'] else {}
                    au_data = action_units.get(str(au_number), {})
                    
                    if au_data and au_data.get('present', False):
                        evidence_frames.append(dict(frame))
                except (json.JSONDecodeError, TypeError):
                    continue

            return evidence_frames
