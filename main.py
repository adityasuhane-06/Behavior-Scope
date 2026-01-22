#!/usr/bin/env python3
"""
Main orchestration script for Behavior Scope.

This script coordinates the complete multimodal behavioral analysis pipeline:
1. Audio extraction and analysis (VAD, diarization, prosody, instability)
2. Segment alignment (audio-to-video timestamp mapping)
3. Video analysis (face detection, pose tracking, temporal aggregation)
4. Multimodal fusion (conservative audio-video evidence combination)
5. Behavioral scoring (VRI, MAI, ASS, RCI indices)
6. Visualization and reporting (timelines, segments, HTML report)

Usage:
    python main.py --video path/to/video.mp4 --config configs/thresholds.yaml --output results/

Clinical rationale:
- Audio-first approach (audio signals often precede visual changes)
- Conservative fusion (requires multi-modal agreement)
- Explainable outputs (rule-based, transparent)
- Non-diagnostic (behavioral observation only)

Engineering approach:
- Modular pipeline (each stage can be run independently)
- Robust error handling (graceful degradation)
- Configurable thresholds (clinician-adjustable)
- Comprehensive logging
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Dict
from datetime import datetime
import pickle

# Import all pipeline modules
from audio_pipeline import (
    run_voice_activity_detection,
    diarize_speakers,
    extract_audio_embeddings,
    compute_prosodic_features,
    detect_vocal_instability
)
from segment_alignment import align_audio_to_video, expand_temporal_windows
from video_pipeline import FaceAnalyzer, PoseAnalyzer, TemporalAggregator
from video_pipeline.face_analyzer import analyze_face_segment
from video_pipeline.pose_analyzer import analyze_pose_segment
from fusion import fuse_audio_video_evidence
from scoring import (
    compute_vocal_regulation_index,
    compute_motor_agitation_index,
    compute_attention_stability_score,
    compute_regulation_consistency_index,
    compute_facial_affect_index
)
from autism_analysis import (
    analyze_turn_taking,
    compute_response_latency_child,
    analyze_eye_contact,
    detect_stereotyped_movements,
    compute_social_engagement_index
)
from clinical_analysis import (
    analyze_stuttering,
    analyze_question_response_ability,
    analyze_facial_action_units
)
from visualization import (
    plot_multimodal_timeline,
    plot_behavioral_scores,
    export_dysregulation_segments,
    create_segment_annotations,
    generate_html_report,
    BehavioralReport
)
from utils.config_loader import load_config
from utils.audio_io import extract_audio_from_video, load_audio
from utils.video_io import VideoReader, extract_frames_from_segment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('behavior_scope.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def run_pipeline(video_path: str, config: Dict, output_dir: str):
    """
    Execute the complete behavioral analysis pipeline.
    
    Args:
        video_path: Path to input video file
        config: Configuration dictionary
        output_dir: Directory for output files
    """
    logger.info("="*80)
    logger.info("BEHAVIOR SCOPE - Multimodal Behavioral Analysis Pipeline")
    logger.info("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = output_path / f"{session_id}_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(name, data):
        """Save checkpoint to disk."""
        path = checkpoint_dir / f"{name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"✓ Checkpoint saved: {name}")
    
    def load_checkpoint(name):
        """Load checkpoint from disk."""
        path = checkpoint_dir / f"{name}.pkl"
        if path.exists():
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None
    
    # =========================================================================
    # STAGE 1: Audio Extraction and Analysis
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("STAGE 1: Audio Analysis")
    logger.info("="*80)
    
    # Check for cached results
    cached_audio = load_checkpoint('stage1_audio')
    if cached_audio:
        logger.info("♻ Loading cached audio analysis...")
        audio_signal, sr, speech_segments, speaker_segments = cached_audio['data']
        embedding_results = cached_audio['embeddings']
        prosodic_features = cached_audio['prosodic']
        logger.info(f"♻ Restored: {len(speech_segments)} speech segments, {len(prosodic_features)} prosodic windows")
    else:
        # Extract audio from video
        audio_path = output_path / f"{session_id}_audio.wav"
        logger.info(f"Extracting audio to: {audio_path}")
        extract_audio_from_video(video_path, str(audio_path))
        
        # Load audio
        audio_signal, sr = load_audio(str(audio_path))
        logger.info(f"Audio loaded: {len(audio_signal)/sr:.1f}s @ {sr}Hz")
        
        # Voice activity detection
        logger.info("Running voice activity detection...")
        speech_segments = run_voice_activity_detection(audio_signal, sr, config)
        logger.info(f"Detected {len(speech_segments)} speech segments")
        
        # Speaker diarization
        logger.info("Performing speaker diarization...")
        speaker_segments = diarize_speakers(str(audio_path), config)
        logger.info(f"Identified {len(set(s.speaker_id for s in speaker_segments))} speakers")
        
        # Extract audio embeddings
        logger.info("Extracting audio embeddings...")
        embedding_results = extract_audio_embeddings(
            audio_signal, sr, speaker_segments
        )
        logger.info(f"Extracted embeddings for speaker segments")
        
        # Compute prosodic features
        logger.info("Computing prosodic features...")
        prosodic_features = compute_prosodic_features(
            audio_signal, 
            sr, 
            speech_segments,
            window_duration=config['audio']['instability']['window_duration_sec'],
            hop_duration=config['audio']['instability']['hop_duration_sec']
        )
        logger.info(f"Computed prosodic features for {len(prosodic_features)} windows")
        
        # Save checkpoint
        save_checkpoint('stage1_audio', {
            'data': (audio_signal, sr, speech_segments, speaker_segments),
            'embeddings': embedding_results,
            'prosodic': prosodic_features
        })
    
    # Detect vocal instability
    logger.info("Detecting vocal instability...")
    audio_baseline = _compute_audio_baseline(prosodic_features)
    instability_windows = detect_vocal_instability(
        prosodic_features, config, baseline=audio_baseline
    )
    logger.info(f"Detected {len(instability_windows)} instability windows")
    
    # =========================================================================
    # STAGE 2: Segment Alignment
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("STAGE 2: Segment Alignment")
    logger.info("="*80)
    
    # Align audio windows to video frames
    # Use speech segments if no instability detected (for normal sessions)
    segments_to_align = instability_windows if len(instability_windows) > 0 else []
    
    # If no instability but we have speech, analyze during speech for eye contact
    if len(segments_to_align) == 0 and len(speech_segments) > 0:
        from segment_alignment.aligner import AudioSegment
        # Create audio segments from speech for video analysis
        segments_to_align = [
            AudioSegment(start_time=seg.start_time, end_time=seg.end_time, score=1.0)
            for seg in speech_segments[:10]  # Limit to first 10 speech segments
        ]
        logger.info(f"No instability detected, using {len(segments_to_align)} speech segments for video analysis")
    
    logger.info("Aligning audio windows to video...")
    video_segments = align_audio_to_video(segments_to_align, video_path)
    logger.info(f"Aligned {len(video_segments)} segments")
    
    # Expand temporal windows
    logger.info("Expanding temporal windows...")
    expanded_segments = expand_temporal_windows(
        video_segments,
        pre_context_sec=config['alignment']['pre_context_sec'],
        post_context_sec=config['alignment']['post_context_sec'],
        max_duration_sec=config['alignment']['max_segment_duration_sec']
    )
    logger.info(f"Expanded to {len(expanded_segments)} segments")
    
    # =========================================================================
    # STAGE 3: Video Analysis
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("STAGE 3: Video Analysis")
    logger.info("="*80)
    
    # Initialize analyzers
    face_analyzer = FaceAnalyzer()
    pose_analyzer = PoseAnalyzer()
    temporal_aggregator = TemporalAggregator(
        window_duration=config['video']['temporal']['window_duration_sec'],
        hop_duration=config['video']['temporal']['window_duration_sec'] / 2,  # 50% overlap
        aggregation_functions=config['video']['temporal']['aggregation_functions']
    )
    
    # Analyze each video segment
    logger.info("Analyzing video segments...")
    video_aggregated = []
    facial_au_sequence = []  # Store facial AUs for all segments
    
    for segment in expanded_segments:
        logger.info(f"Processing segment [{segment.start_time:.1f}s - {segment.end_time:.1f}s]")
        
        # Extract frames
        frames = extract_frames_from_segment(
            video_path,
            segment.start_frame,
            segment.end_frame,
            target_fps=config['video']['target_fps']
        )
        
        if not frames:
            logger.warning(f"No frames extracted for segment")
            continue
        
        # Analyze face features
        face_features_list = analyze_face_segment(frames, segment.start_frame, fps=config['video']['target_fps'])
        
        # Analyze pose features
        pose_features_list = analyze_pose_segment(frames, segment.start_frame, fps=config['video']['target_fps'])
        
        # Analyze Facial Action Units (if enabled)
        if config.get('clinical_analysis', {}).get('facial_action_units', {}).get('enabled', False):
            from clinical_analysis.facial_action_units import FacialActionUnitAnalyzer
            import mediapipe as mp
            
            # Initialize MediaPipe and AU analyzer
            mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,  # Process each frame independently
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.3  # Lower threshold for better detection
            )
            
            au_analyzer = FacialActionUnitAnalyzer(
                intensity_threshold=config['clinical_analysis']['facial_action_units'].get('intensity_threshold', 0.3)
            )
            
            # Process each frame for facial landmarks
            import cv2
            import numpy as np
            
            logger.info(f"  Processing {len(frames)} frames for AU extraction...")
            if len(frames) > 0:
                logger.info(f"  Frame shape: {frames[0].shape}, dtype: {frames[0].dtype}")
            faces_detected_for_au = 0
            
            for i, frame in enumerate(frames):
                frame_idx = segment.start_frame + i
                timestamp = frame_idx / config['video']['target_fps']
                
                # MediaPipe Face Mesh expects BGR format (OpenCV native format)
                # Do NOT convert to RGB despite what documentation says
                results = mp_face_mesh.process(frame)
                
                if results.multi_face_landmarks:
                    faces_detected_for_au += 1
                    h, w = frame.shape[:2]
                    landmarks = np.array([
                        [lm.x * w, lm.y * h, lm.z]
                        for lm in results.multi_face_landmarks[0].landmark
                    ])
                    
                    # Analyze Action Units
                    aus = au_analyzer.analyze_landmarks(landmarks, frame_idx, timestamp)
                    facial_au_sequence.append(aus)
            
            logger.info(f"  Detected faces in {faces_detected_for_au}/{len(frames)} frames for AU analysis")
            
            mp_face_mesh.close()
            
            segment_aus_count = len([au for au in facial_au_sequence if au.timestamp >= segment.start_time and au.timestamp <= segment.end_time])
            logger.info(f"  Extracted {segment_aus_count} facial AU frames from segment")
        
        # Temporal aggregation
        aggregated = temporal_aggregator.aggregate(
            face_features_list,
            pose_features_list,
            fps=config['video']['target_fps']
        )
        video_aggregated.extend(aggregated)  # extend instead of append since aggregate returns a list
    
    logger.info(f"Analyzed {len(video_aggregated)} video segments")
    logger.info(f"Total facial AU frames collected: {len(facial_au_sequence)}")
    
    # Calculate duration from audio length
    duration = len(audio_signal) / sr if 'audio_signal' in locals() and 'sr' in locals() else 8.0
    
    # =========================================================================
    # STAGE 4: Multimodal Fusion
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("STAGE 4: Multimodal Fusion")
    logger.info("="*80)
    
    # Fuse audio and video evidence
    logger.info("Fusing multimodal evidence...")
    fused_evidence = fuse_audio_video_evidence(
        instability_windows, video_aggregated, config
    )
    logger.info(f"Fused {len(fused_evidence)} evidence windows")
    
    # Log confidence distribution
    if fused_evidence:
        strong = len([fe for fe in fused_evidence if fe.confidence_level == 'strong'])
        moderate = len([fe for fe in fused_evidence if fe.confidence_level == 'moderate'])
        weak = len([fe for fe in fused_evidence if fe.confidence_level == 'weak'])
        logger.info(f"Confidence distribution: {strong} strong, {moderate} moderate, {weak} weak")
    
    # Compute Facial Affect Index (if AUs were analyzed)
    facial_affect_index = None
    if len(facial_au_sequence) > 0 and config.get('clinical_analysis', {}).get('facial_action_units', {}).get('enabled', False):
        logger.info("\nComputing Facial Affect Index...")
        facial_affect_index = compute_facial_affect_index(
            au_sequence=facial_au_sequence,
            duration=duration
        )
        logger.info(f"Facial Affect Index: {facial_affect_index.facial_affect_index:.1f}/100")
        logger.info(f"  Affect Range: {facial_affect_index.affect_range_score:.1f}/100")
        logger.info(f"  Facial Mobility: {facial_affect_index.facial_mobility_index:.1f}/100")
        logger.info(f"  Flat Affect: {facial_affect_index.flat_affect_indicator:.1f}/100")
        logger.info(f"  Symmetry: {facial_affect_index.symmetry_index:.1f}/100")
        logger.info(f"  Dominant AUs: {facial_affect_index.dominant_aus}")
    
    # =========================================================================
    # STAGE 5: Behavioral Scoring
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("STAGE 5: Behavioral Scoring")
    logger.info("="*80)
    
    scores = {}
    
    # Vocal Regulation Index
    logger.info("Computing Vocal Regulation Index...")
    scores['vocal_regulation'] = compute_vocal_regulation_index(
        prosodic_features, audio_baseline, config
    )
    logger.info(f"VRI: {scores['vocal_regulation']['score']:.1f}/100")
    
    # Motor Agitation Index
    logger.info("Computing Motor Agitation Index...")
    scores['motor_agitation'] = compute_motor_agitation_index(
        video_aggregated, config
    )
    logger.info(f"MAI: {scores['motor_agitation']['score']:.1f}/100")
    
    # Attention Stability Score
    logger.info("Computing Attention Stability Score...")
    scores['attention_stability'] = compute_attention_stability_score(
        video_aggregated, config
    )
    logger.info(f"ASS: {scores['attention_stability']['score']:.1f}/100")
    
    # Regulation Consistency Index
    logger.info("Computing Regulation Consistency Index...")
    scores['regulation_consistency'] = compute_regulation_consistency_index(
        fused_evidence, config
    )
    logger.info(f"RCI: {scores['regulation_consistency']['score']:.1f}/100")
    
    # Add Facial Affect Index to scores (if available)
    if facial_affect_index is not None:
        scores['facial_affect'] = {
            'score': facial_affect_index.facial_affect_index,
            'affect_range': facial_affect_index.affect_range_score,
            'facial_mobility': facial_affect_index.facial_mobility_index,
            'flat_affect': facial_affect_index.flat_affect_indicator,
            'facial_symmetry': facial_affect_index.symmetry_index,
            'dominant_aus': facial_affect_index.dominant_aus
        }
    
    # =========================================================================
    # STAGE 5B: Autism-Specific Analysis (if enabled)
    # =========================================================================
    autism_results = None
    
    if config.get('autism_analysis', {}).get('enabled', False):
        logger.info("\n" + "="*80)
        logger.info("STAGE 5B: Autism-Specific Analysis")
        logger.info("="*80)
        
        autism_results = {}
        
        # Turn-taking analysis
        logger.info("Analyzing turn-taking dynamics...")
        turn_analysis = analyze_turn_taking(speaker_segments, config=config)
        autism_results['turn_taking'] = turn_analysis
        logger.info(f"Turn-taking: {turn_analysis.child_turns} child turns, "
                   f"{turn_analysis.therapist_turns} therapist turns")
        logger.info(f"Response latency: {turn_analysis.mean_response_latency:.2f}s")
        logger.info(f"Reciprocity score: {turn_analysis.reciprocity_score:.1f}/100")
        
        # Response latency details
        child_latency = compute_response_latency_child(turn_analysis)
        autism_results['child_latency'] = child_latency
        logger.info(f"Child latency: mean={child_latency['mean']:.2f}s, "
                   f"elevated={child_latency['percentage_elevated']:.1f}%")
        
        # Eye contact analysis
        logger.info("Analyzing eye contact patterns...")
        eye_contact = analyze_eye_contact(
            video_aggregated,
            turn_analysis=turn_analysis,
            config=config
        )
        autism_results['eye_contact'] = eye_contact
        logger.info(f"Eye contact: {eye_contact.episode_count} episodes, "
                   f"{eye_contact.percentage_of_session:.1f}% of session")
        logger.info(f"Eye contact score: {eye_contact.eye_contact_score:.1f}/100")
        
        # Stereotypy detection
        logger.info("Detecting stereotyped movements...")
        stereotypy = detect_stereotyped_movements(video_aggregated, config=config)
        autism_results['stereotypy'] = stereotypy
        logger.info(f"Stereotypies: {stereotypy.episode_count} episodes, "
                   f"{stereotypy.percentage_of_session:.1f}% of session")
        logger.info(f"Types: {stereotypy.stereotypy_types}")
        
        # Social engagement index
        logger.info("Computing Social Engagement Index...")
        social_engagement = compute_social_engagement_index(
            eye_contact_analysis=eye_contact,
            turn_taking_analysis=turn_analysis,
            attention_stability_score=scores['attention_stability']['score'],
            config=config
        )
        autism_results['social_engagement'] = social_engagement
        logger.info(f"Social Engagement Index: {social_engagement.social_engagement_index:.1f}/100")
        logger.info(f"Components: EC={social_engagement.eye_contact_component:.1f}, "
                   f"TT={social_engagement.turn_taking_component:.1f}, "
                   f"Resp={social_engagement.responsiveness_component:.1f}, "
                   f"Attn={social_engagement.attention_component:.1f}")
    
    # =========================================================================
    # STAGE 5C: Clinical Analysis (if enabled)
    # =========================================================================
    clinical_results = None
    
    if config.get('clinical_analysis', {}).get('enabled', False):
        logger.info("\n" + "="*80)
        logger.info("STAGE 5C: Clinical Analysis")
        logger.info("="*80)
        
        clinical_results = {}
        
        # Stuttering/disfluency analysis
        logger.info("Analyzing stuttering/disfluency patterns...")
        stuttering = analyze_stuttering(
            prosodic_features,
            speaker_segments,
            config=config
        )
        clinical_results['stuttering'] = stuttering
        logger.info(f"Disfluencies: {stuttering.total_disfluencies} events, "
                   f"rate={stuttering.disfluency_rate:.1f}%")
        logger.info(f"Stuttering Severity Index: {stuttering.stuttering_severity_index:.1f}/100")
        logger.info(f"Types: {stuttering.disfluency_types}")
        
        # Question-response ability
        logger.info("Analyzing question-response ability...")
        qr_ability = analyze_question_response_ability(
            speaker_segments,
            prosodic_features,
            config=config
        )
        clinical_results['question_response'] = qr_ability
        logger.info(f"Questions detected: {qr_ability.total_questions}")
        logger.info(f"Response rate: {qr_ability.response_rate:.1f}%")
        logger.info(f"Mean response latency: {qr_ability.mean_response_latency:.2f}s")
        logger.info(f"Responsiveness Index: {qr_ability.responsiveness_index:.1f}/100")
    
    # =========================================================================
    # STAGE 6: Visualization and Reporting
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("STAGE 6: Visualization and Reporting")
    logger.info("="*80)
    
    # Generate timeline plot
    logger.info("Generating multimodal timeline...")
    timeline_path = output_path / f"{session_id}_timeline.png"
    plot_multimodal_timeline(
        instability_windows,
        video_aggregated,
        fused_evidence,
        str(timeline_path),
        title=f"Behavioral Analysis Timeline - {session_id}"
    )
    
    # Generate scores plot
    logger.info("Generating behavioral scores plot...")
    scores_path = output_path / f"{session_id}_scores.png"
    plot_behavioral_scores(
        scores,
        str(scores_path),
        title=f"Behavioral Indices - {session_id}"
    )
    
    # Export top dysregulation segments
    logger.info("Exporting dysregulation video segments...")
    segments_dir = output_path / f"{session_id}_segments"
    exported_segments = export_dysregulation_segments(
        video_path,
        fused_evidence,
        str(segments_dir),
        n_segments=5,
        config=config
    )
    logger.info(f"Exported {len(exported_segments)} video segments")
    
    # Create annotations
    logger.info("Creating segment annotations...")
    annotations_path = output_path / f"{session_id}_annotations.json"
    create_segment_annotations(
        fused_evidence,
        str(annotations_path),
        format='json'
    )
    
    # Generate HTML report
    logger.info("Generating HTML report...")
    report_data = BehavioralReport(
        session_id=session_id,
        video_path=video_path,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        scores=scores,
        fused_evidence=fused_evidence,
        audio_windows=instability_windows,
        video_aggregated=video_aggregated,
        autism_results=autism_results,
        clinical_results=clinical_results,  # NEW: Include clinical analysis
        metadata={
            'audio_segments': len(speech_segments),
            'instability_windows': len(instability_windows),
            'fused_segments': len(fused_evidence),
            'video_segments_analyzed': len(video_aggregated),
            'autism_analysis_enabled': autism_results is not None,
            'clinical_analysis_enabled': clinical_results is not None
        }
    )
    
    report_path = output_path / f"{session_id}_report.html"
    generate_html_report(
        report_data,
        str(report_path),
        timeline_plot_path=str(timeline_path),
        scores_plot_path=str(scores_path)
    )
    
    # =========================================================================
    # Pipeline Complete
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"\nGenerated files:")
    logger.info(f"  - HTML Report: {report_path.name}")
    logger.info(f"  - Timeline Plot: {timeline_path.name}")
    logger.info(f"  - Scores Plot: {scores_path.name}")
    logger.info(f"  - Annotations: {annotations_path.name}")
    logger.info(f"  - Video Segments: {segments_dir.name}/")
    
    return {
        'status': 'success',
        'session_id': session_id,
        'output_dir': str(output_dir),
        'scores': scores,
        'report_path': str(report_path)
    }


def _compute_audio_baseline(prosodic_features: list) -> Dict:
    """
    Compute baseline statistics from prosodic features.
    
    Args:
        prosodic_features: List of ProsodicFeatures
        
    Returns:
        Dictionary with mean and std for each feature
    """
    import numpy as np
    
    if not prosodic_features:
        return _get_default_baseline()
    
    # Extract feature arrays
    speech_rates = [f.speech_rate for f in prosodic_features if f.speech_rate > 0]
    pause_durations = [f.pause_duration_mean for f in prosodic_features]
    pitch_stds = [f.pitch_std for f in prosodic_features if f.pitch_std > 0]
    energy_stds = [f.energy_std for f in prosodic_features if f.energy_std > 0]
    
    baseline = {}
    
    if speech_rates:
        baseline['speech_rate_mean'] = float(np.mean(speech_rates))
        baseline['speech_rate_std'] = float(np.std(speech_rates)) if len(speech_rates) > 1 else 1.0
    else:
        baseline['speech_rate_mean'] = 4.5
        baseline['speech_rate_std'] = 1.0
    
    if pause_durations:
        baseline['pause_duration_mean'] = float(np.mean(pause_durations))
        baseline['pause_duration_std'] = float(np.std(pause_durations)) if len(pause_durations) > 1 else 0.5
    else:
        baseline['pause_duration_mean'] = 0.5
        baseline['pause_duration_std'] = 0.5
    
    if pitch_stds:
        baseline['pitch_variability_mean'] = float(np.mean(pitch_stds))
        baseline['pitch_variability_std'] = float(np.std(pitch_stds)) if len(pitch_stds) > 1 else 15.0
    else:
        baseline['pitch_variability_mean'] = 30.0
        baseline['pitch_variability_std'] = 15.0
    
    if energy_stds:
        baseline['energy_variability_mean'] = float(np.mean(energy_stds))
        baseline['energy_variability_std'] = float(np.std(energy_stds)) if len(energy_stds) > 1 else 5.0
    else:
        baseline['energy_variability_mean'] = 8.0
        baseline['energy_variability_std'] = 5.0
    
    return baseline


def _get_default_baseline() -> Dict:
    """Return default baseline values for short/sparse audio."""
    return {
        'speech_rate_mean': 4.5,
        'speech_rate_std': 1.0,
        'pause_duration_mean': 0.5,
        'pause_duration_std': 0.5,
        'pitch_variability_mean': 30.0,
        'pitch_variability_std': 15.0,
        'energy_variability_mean': 8.0,
        'energy_variability_std': 5.0
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Behavior Scope - Multimodal Behavioral Regulation Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py --video session.mp4 --output results/

  # With custom config
  python main.py --video session.mp4 --config custom.yaml --output results/

  # Specify number of segments to export
  python main.py --video session.mp4 --output results/ --segments 10
        """
    )
    
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path to input video file'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/thresholds.yaml',
        help='Path to configuration YAML file (default: configs/thresholds.yaml)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/outputs',
        help='Output directory for results (default: data/outputs)'
    )
    
    parser.add_argument(
        '--segments',
        type=int,
        default=5,
        help='Number of top dysregulation segments to export (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Validate video path
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)
    
    # Validate config path
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Load configuration
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(str(config_path))
    
    # Update config with command-line args
    if 'visualization' not in config:
        config['visualization'] = {}
    config['visualization']['n_segments'] = args.segments
    
    try:
        # Run pipeline
        result = run_pipeline(
            video_path=str(video_path),
            config=config,
            output_dir=args.output
        )
        
        logger.info("\n" + "="*80)
        logger.info("✓ SUCCESS: Analysis completed successfully!")
        logger.info(f"  Report: {result['report_path']}")
        logger.info("="*80)
        
        sys.exit(0)
    
    except KeyboardInterrupt:
        logger.warning("\nAnalysis interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"\n✗ ERROR: Pipeline failed with exception:")
        logger.error(f"  {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == '__main__':
    main()
