#!/usr/bin/env python3
"""
Audit Query CLI Tool for Behavior Scope.

Command-line interface for querying and verifying analysis results
from the audit database.

Usage:
    # List recent sessions
    python audit_query.py list

    # View specific session details
    python audit_query.py view SESSION_ID

    # Search sessions
    python audit_query.py search --video "session_video.mp4"

    # Export session report
    python audit_query.py export SESSION_ID --output report.json

    # Show database statistics
    python audit_query.py stats
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from utils.audit_database import AuditDatabase


def format_score(score: float, threshold_good: float = 70, threshold_fair: float = 50) -> str:
    """Format score with color indicator."""
    if score is None:
        return "N/A"

    score_str = f"{score:.1f}/100"

    if score >= threshold_good:
        return f"✓ {score_str} (Good)"
    elif score >= threshold_fair:
        return f"⚠ {score_str} (Fair)"
    else:
        return f"✗ {score_str} (Concern)"


def print_session_summary(session: Dict):
    """Print formatted session summary."""
    print("\n" + "="*80)
    print(f"SESSION: {session['session_id']}")
    print("="*80)

    print(f"\nVideo:     {session['video_path']}")
    print(f"Timestamp: {session['timestamp']}")
    print(f"Age:       {session.get('participant_age', 'N/A')}")
    print(f"Duration:  {session.get('processing_duration_sec', 0):.1f}s")

    print("\n" + "-"*80)
    print("BEHAVIORAL SCORES")
    print("-"*80)

    print(f"Vocal Regulation Index (VRI):        {format_score(session['vocal_regulation_index'])}")
    print(f"Motor Agitation Index (MAI):         {format_score(session['motor_agitation_index'])}")
    print(f"Attention Stability Score (ASS):     {format_score(session['attention_stability_score'])}")
    print(f"Regulation Consistency Index (RCI):  {format_score(session['regulation_consistency_index'])}")

    if session['facial_affect_index']:
        print(f"Facial Affect Index (FAI):           {format_score(session['facial_affect_index'])}")

    if session.get('turn_taking_score') or session.get('eye_contact_score') or session.get('social_engagement_index'):
        print("\n" + "-"*80)
        print("AUTISM-SPECIFIC ANALYSIS")
        print("-"*80)

        if session.get('turn_taking_score'):
            print(f"Turn-Taking Score:                   {format_score(session['turn_taking_score'])}")

        if session.get('eye_contact_score'):
            print(f"Eye Contact Score:                   {format_score(session['eye_contact_score'])}")

        if session.get('social_engagement_index'):
            print(f"Social Engagement Index (SEI):       {format_score(session['social_engagement_index'])}")

        if session.get('stereotypy_percentage') is not None:
            print(f"Stereotypy Percentage:               {session['stereotypy_percentage']:.1f}%")

    if session.get('stuttering_severity_index') or session.get('responsiveness_index'):
        print("\n" + "-"*80)
        print("CLINICAL ANALYSIS")
        print("-"*80)

        if session.get('stuttering_severity_index'):
            print(f"Stuttering Severity Index (SSI):     {format_score(session['stuttering_severity_index'])}")

        if session.get('responsiveness_index'):
            print(f"Responsiveness Index (RI):           {format_score(session['responsiveness_index'])}")

    print("\n" + "-"*80)
    print("ANALYSIS SUMMARY")
    print("-"*80)

    print(f"Audio segments detected:             {session['audio_segments_detected']}")
    print(f"Instability windows detected:        {session['instability_windows_detected']}")
    print(f"Fused evidence segments:             {session['fused_evidence_count']}")

    if session['high_confidence_segments'] or session['medium_confidence_segments'] or session['low_confidence_segments']:
        print(f"\nConfidence distribution:")
        print(f"  High confidence:    {session['high_confidence_segments']}")
        print(f"  Medium confidence:  {session['medium_confidence_segments']}")
        print(f"  Low confidence:     {session['low_confidence_segments']}")

    print("\n" + "="*80 + "\n")


def print_detailed_session(session: Dict):
    """Print detailed session information including segments and logs."""
    print_session_summary(session)

    # Show segments
    if session.get('segments'):
        print("\n" + "-"*80)
        print(f"DYSREGULATION SEGMENTS ({len(session['segments'])} total)")
        print("-"*80)

        for i, seg in enumerate(session['segments'][:10], 1):  # Show first 10
            time_info = f"[{seg['start_time']:.1f}s - {seg['end_time']:.1f}s]"
            frame_info = ""
            if seg.get('start_frame') is not None and seg.get('end_frame') is not None:
                frame_info = f" [Frames {seg['start_frame']}-{seg['end_frame']}]"

            print(f"\n#{i} {time_info}{frame_info} ({seg['confidence_level'].upper()})")
            print(f"  Combined Score: {seg['combined_score']:.2f}")
            print(f"  Audio Score:    {seg['audio_score']:.2f}")
            print(f"  Video Score:    {seg['video_score']:.2f}")

            if seg.get('indicators'):
                indicators = json.loads(seg['indicators']) if isinstance(seg['indicators'], str) else seg['indicators']
                if indicators.get('audio_indicators'):
                    print(f"  Audio Indicators: {', '.join(indicators['audio_indicators'])}")
                if indicators.get('video_indicators'):
                    print(f"  Video Indicators: {', '.join(indicators['video_indicators'])}")

        if len(session['segments']) > 10:
            print(f"\n... and {len(session['segments']) - 10} more segments")

    # Show autism analysis details
    if session.get('autism_analysis'):
        print("\n" + "-"*80)
        print("AUTISM ANALYSIS DETAILS")
        print("-"*80)

        for analysis_type, data in session['autism_analysis'].items():
            print(f"\n{analysis_type.upper()}:")
            # Print relevant fields from data
            if isinstance(data, dict):
                for key, value in list(data.items())[:5]:  # Show first 5 fields
                    if not key.startswith('_') and key not in ['turn_events', 'episodes']:
                        print(f"  {key}: {value}")

    # Show clinical analysis details
    if session.get('clinical_analysis'):
        print("\n" + "-"*80)
        print("CLINICAL ANALYSIS DETAILS")
        print("-"*80)

        for analysis_type, data in session['clinical_analysis'].items():
            print(f"\n{analysis_type.upper()}:")
            if isinstance(data, dict):
                for key, value in list(data.items())[:5]:  # Show first 5 fields
                    if not key.startswith('_') and key not in ['events', 'episodes']:
                        print(f"  {key}: {value}")

    # Show frame analysis summary
    if session.get('frame_analysis_count', 0) > 0:
        print("\n" + "-"*80)
        print(f"FRAME-LEVEL ANALYSIS ({session['frame_analysis_count']} frames)")
        print("-"*80)

        if session.get('frame_analysis_sample'):
            print("\nSample of first frames:")
            for frame in session['frame_analysis_sample'][:5]:
                detections = []
                if frame.get('face_detected'):
                    detections.append("Face")
                if frame.get('pose_detected'):
                    detections.append("Pose")
                if frame.get('eye_contact_detected'):
                    detections.append("Eye Contact")
                if frame.get('movement_detected'):
                    detections.append("Movement")

                detection_str = ", ".join(detections) if detections else "None"
                print(f"  Frame #{frame['frame_number']} @ {frame['timestamp']:.2f}s - {detection_str}")

            print(f"\nTo view all frames: python audit_query.py frames {session['session_id']}")

    # Show operation log
    if session.get('operation_log'):
        print("\n" + "-"*80)
        print(f"OPERATION LOG ({len(session['operation_log'])} entries)")
        print("-"*80)

        for log in session['operation_log'][:20]:  # Show first 20
            status_icon = "✓" if log['status'] == 'success' else "✗" if log['status'] == 'error' else "⚠"
            print(f"{status_icon} [{log['timestamp']}] {log['stage']} > {log['operation']}")
            if log.get('details'):
                print(f"  {log['details']}")

        if len(session['operation_log']) > 20:
            print(f"\n... and {len(session['operation_log']) - 20} more log entries")


def list_sessions_cmd(args):
    """List recent sessions."""
    db = AuditDatabase(args.db_path)
    sessions = db.list_sessions(limit=args.limit, offset=args.offset)

    if not sessions:
        print("No sessions found in audit database.")
        return

    print("\n" + "="*80)
    print(f"RECENT SESSIONS ({len(sessions)} shown)")
    print("="*80 + "\n")

    for session in sessions:
        print(f"Session ID: {session['session_id']}")
        print(f"  Video:     {session['video_path']}")
        print(f"  Timestamp: {session['timestamp']}")
        print(f"  VRI: {session['vocal_regulation_index']:.1f}, "
              f"MAI: {session['motor_agitation_index']:.1f}, "
              f"ASS: {session['attention_stability_score']:.1f}, "
              f"RCI: {session['regulation_consistency_index']:.1f}")
        print(f"  Segments:  {session['fused_evidence_count']} "
              f"({session['high_confidence_segments']} high confidence)")
        print()


def view_session_cmd(args):
    """View detailed session information."""
    db = AuditDatabase(args.db_path)
    session = db.get_session(args.session_id)

    if not session:
        print(f"Session not found: {args.session_id}")
        sys.exit(1)

    if args.detailed:
        print_detailed_session(session)
    else:
        print_session_summary(session)


def search_sessions_cmd(args):
    """Search sessions by criteria."""
    db = AuditDatabase(args.db_path)
    sessions = db.search_sessions(
        video_path=args.video,
        min_date=args.from_date,
        max_date=args.to_date
    )

    if not sessions:
        print("No matching sessions found.")
        return

    print(f"\nFound {len(sessions)} matching session(s):\n")

    for session in sessions:
        print(f"Session ID: {session['session_id']}")
        print(f"  Video:     {session['video_path']}")
        print(f"  Timestamp: {session['timestamp']}")
        print()


def export_session_cmd(args):
    """Export session to JSON file."""
    db = AuditDatabase(args.db_path)
    db.export_session_report(args.session_id, args.output)
    print(f"✓ Session exported to: {args.output}")


def stats_cmd(args):
    """Show database statistics."""
    db = AuditDatabase(args.db_path)
    stats = db.get_statistics()

    print("\n" + "="*80)
    print("AUDIT DATABASE STATISTICS")
    print("="*80 + "\n")

    print(f"Total sessions analyzed:              {stats['total_sessions']}")

    if stats['total_sessions'] > 0:
        print(f"\nAverage Scores Across All Sessions:")
        print(f"  Vocal Regulation Index:             {stats['avg_vocal_regulation_index']:.1f}/100")
        print(f"  Motor Agitation Index:              {stats['avg_motor_agitation_index']:.1f}/100")
        print(f"  Attention Stability Score:          {stats['avg_attention_stability_score']:.1f}/100")
        print(f"  Regulation Consistency Index:       {stats['avg_regulation_consistency_index']:.1f}/100")

        print(f"\nTotal segments analyzed:              {stats['total_segments_analyzed']}")
        print(f"Total frames analyzed:                {stats.get('total_frames_analyzed', 0)}")

    print("\n" + "="*80 + "\n")


def frames_cmd(args):
    """Show frame-level analysis data."""
    db = AuditDatabase(args.db_path)

    if args.detection_type:
        frames = db.get_frames_with_detections(args.session_id, args.detection_type)
        detection_label = args.detection_type.replace('_', ' ').title()
        print(f"\n{detection_label} Detections for {args.session_id}:")
    else:
        frames = db.get_frame_range(
            args.session_id,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            limit=args.limit
        )
        print(f"\nFrame Analysis for {args.session_id}:")

    if not frames:
        print("No frame data found.")
        return

    print(f"Showing {len(frames)} frames\n")
    print("-" * 80)

    for frame in frames[:args.limit]:
        print(f"\nFrame #{frame['frame_number']} @ {frame['timestamp']:.2f}s")
        print(f"  Analysis Type: {frame['analysis_type']}")

        detections = []
        if frame['face_detected']:
            detections.append("Face")
        if frame['pose_detected']:
            detections.append("Pose")
        if frame['eye_contact_detected']:
            detections.append("Eye Contact")
        if frame['movement_detected']:
            detections.append("Movement")

        if detections:
            print(f"  Detections: {', '.join(detections)}")

        if frame['confidence_score'] is not None:
            print(f"  Confidence: {frame['confidence_score']:.2f}")

        if frame['action_units']:
            try:
                aus = json.loads(frame['action_units']) if isinstance(frame['action_units'], str) else frame['action_units']
                if aus:
                    print(f"  Action Units: {len(aus)} detected")
            except:
                pass

    if len(frames) > args.limit:
        print(f"\n... and {len(frames) - args.limit} more frames")

    print("\n" + "="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Behavior Scope Audit Query Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List recent sessions
  python audit_query.py list

  # View specific session
  python audit_query.py view session_20260123_135106

  # View with full details
  python audit_query.py view session_20260123_135106 --detailed

  # Search by video name
  python audit_query.py search --video "therapy_session.mp4"

  # Export session data
  python audit_query.py export session_20260123_135106 --output report.json

  # Show statistics
  python audit_query.py stats

  # View frame-level analysis
  python audit_query.py frames session_20260123_135106

  # View specific frame range
  python audit_query.py frames session_20260123_135106 --start-frame 100 --end-frame 200

  # Filter frames by detection type
  python audit_query.py frames session_20260123_135106 --detection-type eye_contact
        """
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default='data/audit/behavior_scope_audit.db',
        help='Path to audit database (default: data/audit/behavior_scope_audit.db)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # List command
    list_parser = subparsers.add_parser('list', help='List recent sessions')
    list_parser.add_argument('--limit', type=int, default=10, help='Number of sessions to show')
    list_parser.add_argument('--offset', type=int, default=0, help='Number of sessions to skip')

    # View command
    view_parser = subparsers.add_parser('view', help='View session details')
    view_parser.add_argument('session_id', help='Session ID to view')
    view_parser.add_argument('--detailed', action='store_true', help='Show detailed information')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search sessions')
    search_parser.add_argument('--video', help='Filter by video path (partial match)')
    search_parser.add_argument('--from-date', help='Minimum timestamp (YYYY-MM-DD)')
    search_parser.add_argument('--to-date', help='Maximum timestamp (YYYY-MM-DD)')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export session to JSON')
    export_parser.add_argument('session_id', help='Session ID to export')
    export_parser.add_argument('--output', required=True, help='Output JSON file path')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')

    # Frames command
    frames_parser = subparsers.add_parser('frames', help='Show frame-level analysis')
    frames_parser.add_argument('session_id', help='Session ID to view frames for')
    frames_parser.add_argument('--start-frame', type=int, default=0, help='Starting frame number')
    frames_parser.add_argument('--end-frame', type=int, help='Ending frame number')
    frames_parser.add_argument('--limit', type=int, default=50, help='Maximum frames to display')
    frames_parser.add_argument('--detection-type', choices=['face', 'pose', 'eye_contact', 'movement'],
                              help='Filter by detection type')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == 'list':
        list_sessions_cmd(args)
    elif args.command == 'view':
        view_session_cmd(args)
    elif args.command == 'search':
        search_sessions_cmd(args)
    elif args.command == 'export':
        export_session_cmd(args)
    elif args.command == 'stats':
        stats_cmd(args)
    elif args.command == 'frames':
        frames_cmd(args)


if __name__ == '__main__':
    main()
