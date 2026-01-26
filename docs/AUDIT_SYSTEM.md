# Audit System Documentation

## Overview

The Behavior Scope audit system provides **complete traceability and verification** of all analysis results. Every time you run an analysis, all results are automatically stored in a structured SQLite database, allowing you to:

- ✓ **Know exactly what the system said** about each session
- ✓ **Verify results** are correct and traceable to inputs
- ✓ **Query historical analyses** by session, video, or date
- ✓ **Track confidence levels** for all detections
- ✓ **Compare sessions** over time
- ✓ **Export detailed reports** in human-readable formats

## Key Features

### 1. Automatic Storage
Every analysis is automatically saved to the audit database with:
- All behavioral scores (VRI, MAI, ASS, RCI, FAI)
- Autism-specific analysis (turn-taking, eye contact, social engagement)
- Clinical analysis (stuttering, responsiveness)
- Individual dysregulation segments with confidence levels and **frame numbers**
- **Frame-level tracking** (timestamp, detections, action units for each frame)
- Configuration used (for reproducibility)
- Processing timestamps and duration
- Complete operation log

### 2. Queryable Database
Easily retrieve and verify past analyses:
```bash
# List recent sessions
python audit_query.py list

# View specific session details
python audit_query.py view session_20260123_135106

# Search by video name
python audit_query.py search --video "therapy_session.mp4"
```

### 3. Human-Readable Exports
Export any session to JSON format for review:
```bash
python audit_query.py export session_20260123_135106 --output report.json
```

### 4. Aggregate Statistics
View statistics across all analyzed sessions:
```bash
python audit_query.py stats
```

## Database Schema

The audit database stores data in the following tables:

### `sessions` - Main session records
- Session ID, video path, timestamp
- All behavioral scores (VRI, MAI, ASS, RCI, FAI)
- Autism analysis scores (turn-taking, eye contact, SEI)
- Clinical analysis scores (stuttering, responsiveness)
- Metadata (segment counts, confidence distribution)
- System/config versions
- Processing duration and participant age

### `detailed_scores` - Full score breakdowns
- Detailed score components for each behavioral index
- Stored as JSON for flexibility

### `segments` - Individual dysregulation segments
- Start/end times and **frame numbers**
- Confidence levels (strong, moderate, weak)
- Audio, video, and combined scores
- Specific indicators detected

### `frame_analysis` - Frame-by-frame tracking (NEW)
- Frame number and timestamp for each analyzed frame
- Detection flags (face, pose, eye contact, movement)
- Action units detected in each frame
- Confidence scores per frame
- Additional analysis details (head pose, bounding boxes, etc.)

### `autism_analysis` - Autism-specific results
- Turn-taking analysis details
- Eye contact patterns
- Stereotypy detection results
- Social engagement components

### `clinical_analysis` - Clinical assessment results
- Stuttering/disfluency patterns
- Question-response analysis
- Facial action units

### `configurations` - Config snapshots
- Configuration hashes
- Full configuration data
- Enables reproducibility

### `analysis_log` - Operation audit trail
- Every major pipeline operation
- Success/warning/error status
- Timestamps and details
- Complete processing history

## Usage Examples

### Example 1: View What the System Said About a Session

```bash
# Run analysis
python main.py --video therapy_session.mp4 --output results/

# After completion, view results
python audit_query.py view session_20260123_135106
```

Output:
```
================================================================================
SESSION: session_20260123_135106
================================================================================

Video:     therapy_session.mp4
Timestamp: 2026-01-23 13:51:06
Age:       8
Duration:  245.3s

--------------------------------------------------------------------------------
BEHAVIORAL SCORES
--------------------------------------------------------------------------------
Vocal Regulation Index (VRI):        ✓ 75.2/100 (Good)
Motor Agitation Index (MAI):         ⚠ 62.1/100 (Fair)
Attention Stability Score (ASS):     ✓ 78.5/100 (Good)
Regulation Consistency Index (RCI):  ✓ 71.3/100 (Good)

--------------------------------------------------------------------------------
AUTISM-SPECIFIC ANALYSIS
--------------------------------------------------------------------------------
Turn-Taking Score:                   ⚠ 65.4/100 (Fair)
Eye Contact Score:                   ✗ 42.1/100 (Concern)
Social Engagement Index (SEI):       ⚠ 58.3/100 (Fair)
Stereotypy Percentage:               2.3%

--------------------------------------------------------------------------------
ANALYSIS SUMMARY
--------------------------------------------------------------------------------
Audio segments detected:             45
Instability windows detected:        8
Fused evidence segments:             6

Confidence distribution:
  High confidence:    4
  Medium confidence:  2
  Low confidence:     0
```

### Example 2: Compare Sessions Over Time

```bash
# List all sessions for a specific video
python audit_query.py search --video "child_therapy.mp4"
```

Output:
```
Found 3 matching session(s):

Session ID: session_20260123_135106
  Video:     child_therapy.mp4
  Timestamp: 2026-01-23 13:51:06
  VRI: 75.2, MAI: 62.1, ASS: 78.5, RCI: 71.3
  Segments:  6 (4 high confidence)

Session ID: session_20260120_103045
  Video:     child_therapy.mp4
  Timestamp: 2026-01-20 10:30:45
  VRI: 68.5, MAI: 70.2, ASS: 72.1, RCI: 65.8
  Segments:  9 (5 high confidence)

Session ID: session_20260118_145512
  Video:     child_therapy.mp4
  Timestamp: 2026-01-18 14:55:12
  VRI: 72.1, MAI: 65.4, ASS: 75.3, RCI: 69.2
  Segments:  7 (6 high confidence)
```

### Example 3: Export Detailed Report for Review

```bash
# Export complete session data
python audit_query.py export session_20260123_135106 --output detailed_report.json
```

The JSON file includes:
- All scores with full breakdowns
- Every dysregulation segment with timestamps
- Autism and clinical analysis details
- Complete operation log
- Configuration used

### Example 4: View Detailed Session Information

```bash
# View with full details including segments and logs
python audit_query.py view session_20260123_135106 --detailed
```

This shows:
- All behavioral scores
- Individual dysregulation segments with indicators
- Autism analysis breakdown
- Clinical analysis breakdown
- Complete operation log

### Example 5: Database Statistics

```bash
python audit_query.py stats
```

Output:
```
================================================================================
AUDIT DATABASE STATISTICS
================================================================================

Total sessions analyzed:              25

Average Scores Across All Sessions:
  Vocal Regulation Index:             72.4/100
  Motor Agitation Index:              65.8/100
  Attention Stability Score:          74.2/100
  Regulation Consistency Index:       68.9/100

Total segments analyzed:              185
Total frames analyzed:                3,542
```

### Example 6: View Frame-Level Analysis

```bash
# View all frames for a session
python audit_query.py frames session_20260123_135106

# View specific frame range
python audit_query.py frames session_20260123_135106 --start-frame 100 --end-frame 200

# Filter by detection type
python audit_query.py frames session_20260123_135106 --detection-type eye_contact
```

Output:
```
Frame Analysis for session_20260123_135106:
Showing 50 frames

--------------------------------------------------------------------------------

Frame #125 @ 4.17s
  Analysis Type: video_multimodal
  Detections: Face, Pose, Eye Contact
  Confidence: 0.87
  Action Units: 12 detected

Frame #126 @ 4.20s
  Analysis Type: video_multimodal
  Detections: Face, Pose
  Confidence: 0.82
  Action Units: 10 detected

Frame #127 @ 4.23s
  Analysis Type: video_multimodal
  Detections: Face, Pose, Eye Contact
  Confidence: 0.91
  Action Units: 14 detected

...
```

## Database Location

By default, the audit database is stored at:
```
data/audit/behavior_scope_audit.db
```

You can specify a different location:
```bash
python audit_query.py --db-path /path/to/custom.db list
```

## Benefits

### 1. **Auditability**
Every analysis result is permanently stored with full context. You can always answer:
- What did the system say about this session?
- What scores were computed?
- What segments were detected?
- What confidence levels were assigned?
- **Which specific frames had detections?**
- **What was detected in each frame (face, pose, eye contact, action units)?**

### 2. **Verification**
Compare results across sessions to verify:
- Consistency of detections
- Progress over time
- Configuration impact
- System reliability

### 3. **Traceability**
Track every analysis back to:
- Original video file
- Configuration used
- Date and time processed
- System version
- All intermediate results

### 4. **Reproducibility**
Configuration snapshots enable:
- Rerunning with same settings
- Understanding why results differ
- Version control for analysis parameters

### 5. **Quality Assurance**
Operation logs provide:
- Complete processing history
- Success/failure tracking
- Error diagnostics
- Performance monitoring

## Integration with Workflow

The audit system is fully integrated into the main pipeline:

```python
# Run analysis (audit happens automatically)
python main.py --video session.mp4 --output results/

# After analysis, query results
python audit_query.py view <session_id>

# Or export for external review
python audit_query.py export <session_id> --output report.json
```

## Advanced Queries

### Search by Date Range
```bash
python audit_query.py search --from-date 2026-01-20 --to-date 2026-01-23
```

### List Recent Sessions with Custom Limit
```bash
python audit_query.py list --limit 50
```

### Pagination
```bash
python audit_query.py list --limit 10 --offset 20
```

### Frame-Level Queries

View frame-by-frame analysis for detailed inspection:

```bash
# View all frames for a session (shows first 50 by default)
python audit_query.py frames session_20260123_135106

# View more frames
python audit_query.py frames session_20260123_135106 --limit 100

# View specific frame range
python audit_query.py frames session_20260123_135106 --start-frame 100 --end-frame 200

# Filter by detection type
python audit_query.py frames session_20260123_135106 --detection-type face
python audit_query.py frames session_20260123_135106 --detection-type eye_contact
python audit_query.py frames session_20260123_135106 --detection-type movement
```

This allows you to:
- **Verify detections** - See exactly which frames had face/pose/eye contact detections
- **Inspect action units** - View which facial expressions were detected in each frame
- **Trace timestamps** - Map frame numbers to exact timestamps
- **Debug issues** - Identify frames where detections failed or succeeded

## Data Privacy

The audit database stores:
- ✓ Analysis results and scores
- ✓ Timestamps and metadata
- ✓ Configuration parameters
- ✗ **NO video content** (only paths)
- ✗ **NO audio data** (only derived features)
- ✗ **NO personal identifying information** (only age if provided)

## Backup and Export

### Backup Database
```bash
# Copy the database file
cp data/audit/behavior_scope_audit.db data/audit/backup_$(date +%Y%m%d).db
```

### Export All Sessions
```bash
# Export each session to JSON
for session_id in $(python audit_query.py list | grep "Session ID:" | awk '{print $3}'); do
    python audit_query.py export $session_id --output exports/${session_id}.json
done
```

## Troubleshooting

### Database Not Found
If you get "Database not found" errors:
1. Run an analysis first to create the database
2. Check the path: `data/audit/behavior_scope_audit.db`
3. Verify write permissions

### Session Not Found
If a session is not in the database:
1. Check if the pipeline completed successfully
2. Look for errors in `behavior_scope.log`
3. Verify the session ID is correct

### Corrupted Database
If the database becomes corrupted:
1. Restore from backup
2. Delete and rerun analyses
3. Contact support with error logs

## Summary

The audit system ensures that:
- ✓ You **always know** what the system said
- ✓ Results are **verifiable** and **traceable**
- ✓ Analyses are **reproducible**
- ✓ Historical data is **queryable**
- ✓ Reports are **exportable** for review

No more wondering "What did the system detect?" - it's all stored, queryable, and verifiable in the audit database!
