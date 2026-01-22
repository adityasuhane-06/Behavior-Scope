"""
Dysregulation segment export for clinical review.

Exports top-N highest-confidence dysregulation video segments:
- Video clips with temporal context (±2s buffer)
- Segment metadata (timestamps, scores, features)
- Annotation files for review software

Clinical rationale:
- Concrete examples for clinical review
- Context preservation (before/during/after)
- Enables detailed behavioral coding
- Supports inter-rater reliability studies

Engineering approach:
- FFmpeg-based video extraction
- JSON metadata for each segment
- Optional annotation format export
- Organized output directory structure
"""

import logging
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import asdict

logger = logging.getLogger(__name__)


def export_dysregulation_segments(
    video_path: str,
    fused_evidence: List,
    output_dir: str,
    n_segments: int = 5,
    buffer_sec: float = 2.0,
    config: Dict = None
) -> List[Dict]:
    """
    Export top-N dysregulation video segments.
    
    Process:
    1. Sort fused evidence by confidence (descending)
    2. Select top N segments
    3. Add temporal buffer
    4. Extract video clips using FFmpeg
    5. Generate metadata JSON for each segment
    
    Args:
        video_path: Path to source video file
        fused_evidence: List of FusedEvidence objects
        output_dir: Directory to save exported segments
        n_segments: Number of top segments to export
        buffer_sec: Temporal buffer to add (seconds)
        config: Optional configuration dict
        
    Returns:
        List of exported segment metadata dicts
    """
    if not fused_evidence:
        logger.warning("No fused evidence to export")
        return []
    
    logger.info(f"Exporting top {n_segments} dysregulation segments from {video_path}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sort by confidence (descending)
    sorted_evidence = sorted(
        fused_evidence,
        key=lambda fe: fe.fused_confidence,
        reverse=True
    )
    
    # Select top N
    top_segments = sorted_evidence[:n_segments]
    
    # Export each segment
    exported_metadata = []
    
    for idx, segment in enumerate(top_segments, 1):
        # Add buffer
        start_time = max(0, segment.start_time - buffer_sec)
        end_time = segment.end_time + buffer_sec
        duration = end_time - start_time
        
        # Generate output filename
        output_filename = f"segment_{idx:02d}_conf_{segment.fused_confidence:.2f}.mp4"
        output_filepath = output_path / output_filename
        
        # Extract video segment
        success = _extract_video_clip(
            video_path,
            str(output_filepath),
            start_time,
            duration
        )
        
        if success:
            # Create metadata
            metadata = {
                'segment_id': idx,
                'filename': output_filename,
                'original_video': str(Path(video_path).name),
                'start_time': float(segment.start_time),
                'end_time': float(segment.end_time),
                'duration': float(segment.duration),
                'buffer_start_time': float(start_time),
                'buffer_end_time': float(end_time),
                'buffer_duration': float(duration),
                'fused_confidence': float(segment.fused_confidence),
                'confidence_level': segment.confidence_level,
                'audio_score': float(segment.audio_score),
                'video_score': float(segment.video_score),
                'explanation': segment.explanation,
                'contributing_features': segment.contributing_features
            }
            
            # Save metadata JSON
            metadata_filename = f"segment_{idx:02d}_metadata.json"
            metadata_filepath = output_path / metadata_filename
            
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            exported_metadata.append(metadata)
            logger.info(f"Exported segment {idx}: {output_filename}")
        else:
            logger.error(f"Failed to export segment {idx}")
    
    # Create summary file
    summary_filepath = output_path / "segments_summary.json"
    with open(summary_filepath, 'w') as f:
        json.dump({
            'total_segments': len(exported_metadata),
            'source_video': str(Path(video_path).name),
            'buffer_seconds': buffer_sec,
            'segments': exported_metadata
        }, f, indent=2)
    
    logger.info(f"Exported {len(exported_metadata)} segments to {output_dir}")
    
    return exported_metadata


def _extract_video_clip(
    input_video: str,
    output_video: str,
    start_time: float,
    duration: float
) -> bool:
    """
    Extract video clip using FFmpeg.
    
    Args:
        input_video: Source video path
        output_video: Output video path
        start_time: Start time in seconds
        duration: Duration in seconds
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # FFmpeg command
        # -ss: start time
        # -t: duration
        # -c copy: copy codec (fast, no re-encoding)
        # -avoid_negative_ts 1: handle timestamps correctly
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-ss', str(start_time),
            '-i', input_video,
            '-t', str(duration),
            '-c', 'copy',
            '-avoid_negative_ts', '1',
            output_video
        ]
        
        # Run FFmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            return True
        else:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False
    
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg timeout")
        return False
    except FileNotFoundError:
        logger.error("FFmpeg not found - please install FFmpeg")
        return False
    except Exception as e:
        logger.error(f"Video extraction error: {e}")
        return False


def create_segment_annotations(
    fused_evidence: List,
    output_path: str,
    format: str = 'json'
) -> str:
    """
    Create annotation file for all segments.
    
    Supports multiple formats:
    - 'json': Simple JSON format
    - 'elan': ELAN annotation format (XML)
    - 'csv': Comma-separated values
    
    Args:
        fused_evidence: List of FusedEvidence objects
        output_path: Path to save annotation file
        format: Output format ('json', 'elan', or 'csv')
        
    Returns:
        Path to saved annotation file
    """
    logger.info(f"Creating annotation file: {output_path} (format: {format})")
    
    if format == 'json':
        return _create_json_annotations(fused_evidence, output_path)
    elif format == 'csv':
        return _create_csv_annotations(fused_evidence, output_path)
    elif format == 'elan':
        return _create_elan_annotations(fused_evidence, output_path)
    else:
        logger.error(f"Unsupported annotation format: {format}")
        return ""


def _create_json_annotations(fused_evidence: List, output_path: str) -> str:
    """Create JSON annotation file."""
    
    annotations = []
    
    for idx, segment in enumerate(sorted(fused_evidence, key=lambda fe: fe.start_time), 1):
        annotations.append({
            'id': idx,
            'start_time': float(segment.start_time),
            'end_time': float(segment.end_time),
            'duration': float(segment.duration),
            'confidence': float(segment.fused_confidence),
            'level': segment.confidence_level,
            'audio_score': float(segment.audio_score),
            'video_score': float(segment.video_score),
            'explanation': segment.explanation
        })
    
    with open(output_path, 'w') as f:
        json.dump({
            'annotation_type': 'behavioral_dysregulation',
            'total_segments': len(annotations),
            'annotations': annotations
        }, f, indent=2)
    
    logger.info(f"JSON annotations saved: {output_path}")
    return output_path


def _create_csv_annotations(fused_evidence: List, output_path: str) -> str:
    """Create CSV annotation file."""
    
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'ID',
            'Start_Time_Sec',
            'End_Time_Sec',
            'Duration_Sec',
            'Fused_Confidence',
            'Confidence_Level',
            'Audio_Score',
            'Video_Score',
            'Explanation'
        ])
        
        # Rows
        for idx, segment in enumerate(sorted(fused_evidence, key=lambda fe: fe.start_time), 1):
            writer.writerow([
                idx,
                f"{segment.start_time:.2f}",
                f"{segment.end_time:.2f}",
                f"{segment.duration:.2f}",
                f"{segment.fused_confidence:.3f}",
                segment.confidence_level,
                f"{segment.audio_score:.3f}",
                f"{segment.video_score:.3f}",
                segment.explanation
            ])
    
    logger.info(f"CSV annotations saved: {output_path}")
    return output_path


def _create_elan_annotations(fused_evidence: List, output_path: str) -> str:
    """
    Create ELAN-compatible XML annotation file.
    
    ELAN (EUDICO Linguistic Annotator) is widely used for behavioral coding.
    """
    
    # Simplified ELAN XML structure
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<ANNOTATION_DOCUMENT xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv3.0.xsd">',
        '    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">',
        '    </HEADER>',
        '    <TIME_ORDER>'
    ]
    
    # Time slots
    time_id = 1
    for segment in sorted(fused_evidence, key=lambda fe: fe.start_time):
        start_ms = int(segment.start_time * 1000)
        end_ms = int(segment.end_time * 1000)
        
        xml_lines.append(f'        <TIME_SLOT TIME_SLOT_ID="ts{time_id}" TIME_VALUE="{start_ms}"/>')
        xml_lines.append(f'        <TIME_SLOT TIME_SLOT_ID="ts{time_id+1}" TIME_VALUE="{end_ms}"/>')
        time_id += 2
    
    xml_lines.append('    </TIME_ORDER>')
    
    # Tier for dysregulation annotations
    xml_lines.append('    <TIER LINGUISTIC_TYPE_REF="default-lt" TIER_ID="Dysregulation">')
    
    annotation_id = 1
    time_id = 1
    
    for segment in sorted(fused_evidence, key=lambda fe: fe.start_time):
        annotation_value = f"{segment.confidence_level.upper()} (conf={segment.fused_confidence:.2f}): {segment.explanation}"
        
        xml_lines.append(f'        <ANNOTATION>')
        xml_lines.append(f'            <ALIGNABLE_ANNOTATION ANNOTATION_ID="a{annotation_id}" '
                        f'TIME_SLOT_REF1="ts{time_id}" TIME_SLOT_REF2="ts{time_id+1}">')
        xml_lines.append(f'                <ANNOTATION_VALUE>{annotation_value}</ANNOTATION_VALUE>')
        xml_lines.append(f'            </ALIGNABLE_ANNOTATION>')
        xml_lines.append(f'        </ANNOTATION>')
        
        annotation_id += 1
        time_id += 2
    
    xml_lines.append('    </TIER>')
    xml_lines.append('    <LINGUISTIC_TYPE LINGUISTIC_TYPE_ID="default-lt" TIME_ALIGNABLE="true" '
                    'GRAPHIC_REFERENCES="false"/>')
    xml_lines.append('</ANNOTATION_DOCUMENT>')
    
    # Write file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(xml_lines))
    
    logger.info(f"ELAN annotations saved: {output_path}")
    return output_path


def get_segment_summary_stats(exported_metadata: List[Dict]) -> Dict:
    """
    Compute summary statistics for exported segments.
    
    Args:
        exported_metadata: List of segment metadata dicts
        
    Returns:
        Dictionary with summary statistics
    """
    if not exported_metadata:
        return {}
    
    confidences = [seg['fused_confidence'] for seg in exported_metadata]
    durations = [seg['duration'] for seg in exported_metadata]
    audio_scores = [seg['audio_score'] for seg in exported_metadata]
    video_scores = [seg['video_score'] for seg in exported_metadata]
    
    return {
        'total_segments': len(exported_metadata),
        'mean_confidence': float(np.mean(confidences)),
        'max_confidence': float(np.max(confidences)),
        'mean_duration': float(np.mean(durations)),
        'total_duration': float(np.sum(durations)),
        'mean_audio_score': float(np.mean(audio_scores)),
        'mean_video_score': float(np.mean(video_scores)),
        'confidence_levels': {
            'strong': len([s for s in exported_metadata if s['confidence_level'] == 'strong']),
            'moderate': len([s for s in exported_metadata if s['confidence_level'] == 'moderate']),
            'weak': len([s for s in exported_metadata if s['confidence_level'] == 'weak'])
        }
    }


# Import numpy for stats (delayed import to avoid hard dependency)
try:
    import numpy as np
except ImportError:
    logger.warning("NumPy not available - segment statistics will be limited")
    np = None
