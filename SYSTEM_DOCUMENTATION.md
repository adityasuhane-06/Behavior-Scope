# Behavior Scope - System Architecture & Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Workflow](#workflow)
4. [File Structure](#file-structure)
5. [Measurement Methods](#measurement-methods)
6. [Report Parameters & Ranges](#report-parameters--ranges)

---

## System Overview

**Behavior Scope** is a multimodal behavioral analysis system that analyzes videos to detect dysregulation patterns through synchronized audio and video analysis. It uses AI models to extract behavioral features and generate clinical-grade reports.

### Core Capabilities
- **Audio Analysis**: Voice activity, prosody, speech patterns
- **Video Analysis**: Facial expressions, body movements, head pose
- **Multimodal Fusion**: Combines audio + video evidence
- **Clinical Metrics**: Autism-specific & clinical analysis
- **Behavioral Scoring**: 4 interpretable indices (VRI, MAI, ASS, RCI)

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT VIDEO                             │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌──────────────┐      ┌──────────────┐
│ AUDIO        │      │ VIDEO        │
│ EXTRACTION   │      │ FRAMES       │
└──────┬───────┘      └──────┬───────┘
       │                     │
       ▼                     ▼
┌──────────────┐      ┌──────────────┐
│ Audio        │      │ Video        │
│ Pipeline     │      │ Pipeline     │
│              │      │              │
│ • VAD        │      │ • Face       │
│ • Diarization│      │ • Pose       │
│ • Prosody    │      │ • Motion     │
│ • Embeddings │      │ • AU (FACS)  │
└──────┬───────┘      └──────┬───────┘
       │                     │
       └──────────┬──────────┘
                  │
                  ▼
         ┌────────────────┐
         │ SEGMENT        │
         │ ALIGNMENT      │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │ MULTIMODAL     │
         │ FUSION         │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │ BEHAVIORAL     │
         │ SCORING        │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │ CLINICAL       │
         │ ANALYSIS       │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │ REPORT         │
         │ GENERATION     │
         └────────────────┘
```

---

## Workflow

### Stage 1: Audio Analysis

**Input**: Video file → Extract audio track

**Steps**:
1. **Voice Activity Detection (VAD)**
   - Model: Silero VAD (neural network)
   - Detects speech vs. silence segments
   - Fallback: Energy-based VAD if Silero fails

2. **Speaker Diarization**
   - Model: pyannote.audio
   - Identifies different speakers (e.g., therapist vs. child)
   - Fallback: Turn-based detection using silence gaps

3. **Prosodic Features**
   - Window: 5.0s sliding window, 2.5s hop
   - Extracts:
     - **Pitch** (F0): Fundamental frequency
     - **Energy**: Voice intensity/loudness
     - **Speech Rate**: Words per second
     - **Pause Patterns**: Silence duration & frequency

4. **Audio Embeddings**
   - Model: HuBERT (facebook/hubert-base-ls960)
   - Generates 768-dimensional vectors
   - Captures deep acoustic features

5. **Instability Detection**
   - Analyzes prosodic features for anomalies
   - Thresholds defined in `configs/thresholds.yaml`
   - Marks segments with vocal dysregulation

**Output**: Audio instability segments with confidence scores

---

### Stage 2: Segment Alignment

**Purpose**: Synchronize audio and video timelines

**Steps**:
1. Map audio segments to video frame indices
2. Compute frame-accurate timestamps
3. Expand temporal windows (±3s context)
4. Handle edge cases (start/end of video)

**Output**: Time-aligned multimodal segments

---

### Stage 3: Video Analysis

**Input**: Video frames for each segment

**Steps**:
1. **Face Analysis** (MediaPipe Face Mesh)
   - Detects 468 facial landmarks
   - Extracts:
     - Head pose (pitch, yaw, roll)
     - Gaze direction
     - Face presence/absence
   - Computes facial motion velocity

2. **Pose Analysis** (MediaPipe Pose)
   - Detects 33 body landmarks
   - Extracts:
     - Body motion velocity
     - Hand motion velocity
     - Posture changes

3. **Facial Action Units (FACS)**
   - Model: MediaPipe Face Mesh + AU mapping
   - Detects 18 Action Units:
     - AU1: Inner brow raiser
     - AU2: Outer brow raiser
     - AU4: Brow lowerer
     - AU5: Upper lid raiser
     - AU6: Cheek raiser
     - AU7: Lid tightener
     - AU9: Nose wrinkler
     - AU10: Upper lip raiser
     - AU12: Lip corner puller
     - AU14: Dimpler
     - AU15: Lip corner depressor
     - AU17: Chin raiser
     - AU18: Lip puckerer
     - AU20: Lip stretcher
     - AU23: Lip tightener
     - AU24: Lip pressor
     - AU25: Lips part
     - AU26: Jaw drop

4. **Temporal Aggregation**
   - Window: 5.0s
   - Computes: mean, std, max, 95th percentile

**Output**: Video feature vectors with motion metrics

---

### Stage 4: Multimodal Fusion

**Engine**: Conservative Fusion

**Logic**:
- Requires **overlapping** audio AND video anomalies
- Strong threshold: Both audio_score ≥ 0.7 AND video_score ≥ 0.7
- Moderate threshold: min_confidence ≥ 0.5

**Fusion Formula**:
```python
fused_confidence = (audio_score + video_score) / 2
```

**Confidence Levels**:
- **Strong**: fused_confidence ≥ 0.7
- **Moderate**: 0.5 ≤ fused_confidence < 0.7
- **Weak**: fused_confidence < 0.5

**Output**: Fused evidence segments with explanations

---

### Stage 5: Behavioral Scoring

Computes 4 interpretable behavioral indices:

#### 1. Vocal Regulation Index (VRI)
**Range**: 0-100 (Higher is better)

**Formula**:
```python
VRI = (speech_rate_score + pause_score + prosody_score) / 3
```

**Components**:
- **Speech Rate Score**: Deviation from normal rate (120-180 WPM)
- **Pause Score**: Frequency and duration of pauses
- **Prosody Score**: Pitch variation, intonation patterns

**Interpretation**:
- **70-100**: Well-regulated speech
- **40-70**: Moderate dysregulation
- **0-40**: Significant vocal dysregulation

---

#### 2. Motor Agitation Index (MAI)
**Range**: 0-100 (Lower is better - this is inverted)

**Formula**:
```python
MAI = (head_motion_score + body_motion_score + hand_motion_score) / 3
```

**Components**:
- **Head Motion**: Angular velocity (degrees/second)
- **Body Motion**: Center-of-mass displacement
- **Hand Motion**: Hand landmark velocity

**Interpretation**:
- **0-30**: Low agitation (good)
- **30-70**: Moderate agitation
- **70-100**: High agitation (concern)

---

#### 3. Attention Stability Score (ASS)
**Range**: 0-100 (Higher is better)

**Formula**:
```python
ASS = (head_pose_stability + gaze_stability + presence_score) / 3
```

**Components**:
- **Head Pose Stability**: Consistency of head orientation
- **Gaze Stability**: Eye tracking consistency
- **Presence Score**: Face detection reliability

**Interpretation**:
- **70-100**: Stable attention
- **40-70**: Moderate instability
- **0-40**: Attention difficulties

---

#### 4. Regulation Consistency Index (RCI)
**Range**: 0-100 (Higher is better)

**Formula**:
```python
RCI = 100 - (variability * 100)
```

**Components**:
- **Autocorrelation**: Temporal pattern consistency
- **Variability**: Standard deviation of confidence scores
- **Trend**: Improving/stable/declining pattern

**Interpretation**:
- **70-100**: Consistent regulation
- **40-70**: Inconsistent patterns
- **0-40**: High variability (concern)

---

#### 5. Facial Affect Index (FAI)
**Range**: 0-100 (Higher is better)

**Formula**:
```python
FAI = (affect_range + facial_mobility - flat_affect_penalty) / 3
```

**Components**:
- **Affect Range**: Diversity of AU activations
- **Facial Mobility**: Frequency of expression changes
- **Flat Affect Penalty**: Deduction for lack of expression
- **Symmetry**: Left-right facial symmetry

**Interpretation**:
- **70-100**: Rich, expressive affect
- **40-70**: Moderate expressiveness
- **0-40**: Flat/restricted affect (concern)

**Dominant AUs**: Top 5 most frequently activated Action Units

---

### Stage 6: Clinical Analysis

#### A. Autism-Specific Analysis

**1. Social Engagement Index (SEI)**
**Range**: 0-100 (Higher is better)

**Formula**:
```python
SEI = (eye_contact * 0.3 + turn_taking * 0.3 + 
       responsiveness * 0.2 + attention * 0.2)
```

**Components**:
- Eye contact percentage
- Turn-taking reciprocity
- Response to questions
- Attention stability

**Interpretation**:
- **70-100**: Good social engagement
- **50-70**: Emerging skills
- **0-50**: Significant difficulties

---

**2. Turn-Taking Dynamics**

**Metrics**:
- **Total Turns**: Count of speaker alternations
- **Child Percentage**: % of turns taken by child
- **Response Latency**: Mean time between therapist speech end → child speech start
- **Reciprocity Score**: 0-100 (balance + timing)

**Optimal Values**:
- Child percentage: 40-60%
- Response latency: < 1.0s
- Reciprocity score: > 70

---

**3. Eye Contact Patterns**

**Metrics**:
- **Session Coverage**: % of time with detected eye contact
- **Episode Count**: Number of eye contact periods
- **During Speaking**: % eye contact while child speaks
- **During Listening**: % eye contact while child listens
- **Eye Contact Score**: 0-100

**Optimal Values**:
- Session coverage: > 50%
- Speaking: > 30%
- Listening: > 50%

---

**4. Stereotyped Movements**

**Detection**:
- Repetitive head movements
- Hand flapping/twisting
- Body rocking
- Pacing patterns

**Metrics**:
- **Episode Count**: Number of stereotypy events
- **Session Coverage**: % of time with stereotypies
- **Intensity Score**: 0-100 (Higher is worse)
- **Types**: Dictionary of detected patterns

**Interpretation**:
- **0-20**: Minimal stereotypy
- **20-40**: Occasional stereotypy
- **40+**: Frequent stereotypy (concern)

---

#### B. Clinical Analysis

**1. Stuttering/Disfluency Analysis**

**Metrics**:
- **Total Disfluencies**: Count of stuttering events
- **Disfluency Rate**: Per 100 syllables
- **Stuttering Severity Index (SSI)**: 0-100

**SSI Ranges**:
- **0-20**: Mild
- **20-40**: Moderate
- **40-60**: Moderately Severe
- **60-100**: Severe

**Disfluency Types**:
- Repetitions (sound, syllable, word)
- Prolongations
- Blocks

---

**2. Question-Response Ability**

**Metrics**:
- **Questions Detected**: Count of therapist questions
- **Questions Answered**: Count of child responses
- **Response Rate**: % of questions answered
- **Response Latency**: Mean time to respond
- **Appropriate Responses**: Context-relevant answers
- **Responsiveness Index**: 0-100

**Interpretation**:
- **60-100**: Functional communication
- **40-60**: Below typical
- **0-40**: Below expected (concern)

---

## File Structure

```
Behavior Scope/
├── main.py                          # Main pipeline orchestrator
├── configs/
│   └── thresholds.yaml              # Detection thresholds
├── audio_pipeline/
│   ├── vad.py                       # Voice Activity Detection
│   ├── diarization.py               # Speaker identification
│   ├── embeddings.py                # Audio feature extraction
│   ├── prosody.py                   # Prosodic feature extraction
│   └── instability.py               # Vocal instability detection
├── video_pipeline/
│   ├── face_analyzer.py             # MediaPipe Face Mesh
│   ├── pose_analyzer.py             # MediaPipe Pose
│   └── temporal_agg.py              # Temporal aggregation
├── segment_alignment/
│   ├── aligner.py                   # Audio-video sync
│   └── window_expander.py           # Temporal context expansion
├── fusion/
│   └── conservative_fusion.py       # Multimodal fusion engine
├── scoring/
│   ├── vocal_regulation.py          # VRI computation
│   ├── motor_agitation.py           # MAI computation
│   ├── attention_stability.py       # ASS computation
│   ├── consistency.py               # RCI computation
│   └── facial_affect_index.py       # FAI computation
├── clinical_analysis/
│   ├── facial_action_units.py       # FACS AU detection
│   ├── stuttering.py                # Disfluency analysis
│   └── question_response.py         # Q&A ability
├── autism_analysis/
│   ├── turn_taking.py               # Turn dynamics
│   ├── eye_contact.py               # Eye contact detection
│   ├── stereotypy.py                # Repetitive behaviors
│   └── social_engagement.py         # SEI computation
├── visualization/
│   ├── report_generator.py          # HTML report builder
│   ├── timeline_plots.py            # Matplotlib visualizations
│   └── segment_marker.py            # Annotation export
├── utils/
│   ├── audio_io.py                  # Audio file handling
│   ├── video_io.py                  # Video file handling
│   └── config_loader.py             # YAML config loader
└── data/
    ├── raw/                         # Input videos
    │   ├── test_short.mp4
    │   └── ted_talk.mp4
    └── outputs/                     # Generated reports
        ├── session_*_report.html
        ├── session_*_timeline.png
        ├── session_*_scores.png
        └── session_*_annotations.json
```

---

## Measurement Methods

### Audio Measurements

**1. Pitch (F0)**
- **Method**: Autocorrelation-based pitch detection
- **Range**: 50-500 Hz
- **Typical**: 100-250 Hz (adults)

**2. Speech Rate**
- **Method**: Syllable counting / duration
- **Range**: 0-300 WPM
- **Typical**: 120-180 WPM

**3. Energy**
- **Method**: RMS amplitude
- **Range**: 0-1 (normalized)

**4. Pause Duration**
- **Method**: Silence detection (< -40 dB)
- **Range**: 0.1-5.0 seconds

---

### Video Measurements

**1. Head Pose**
- **Method**: 3D facial landmark triangulation
- **Angles**: Pitch (up/down), Yaw (left/right), Roll (tilt)
- **Range**: -90° to +90°

**2. Motion Velocity**
- **Method**: Frame-to-frame displacement
- **Unit**: pixels/frame
- **Range**: 0-100+

**3. Facial Action Units**
- **Method**: Landmark distance ratios
- **Activation**: 0.0-1.0 (threshold: 0.3)

---

## Report Parameters Summary

| Parameter | Range | Optimal | Measurement |
|-----------|-------|---------|-------------|
| **VRI** | 0-100 | 70-100 | Audio prosody |
| **MAI** | 0-100 | 0-30 | Motion velocity |
| **ASS** | 0-100 | 70-100 | Head/gaze stability |
| **RCI** | 0-100 | 70-100 | Temporal consistency |
| **FAI** | 0-100 | 70-100 | Facial expressiveness |
| **SEI** | 0-100 | 70-100 | Social engagement |
| **Eye Contact** | 0-100% | 50-80% | Face orientation |
| **Turn Reciprocity** | 0-100 | 70-100 | Speaker alternation |
| **Stereotypy** | 0-100 | 0-20 | Repetitive motion |
| **SSI** | 0-100 | 0-20 | Disfluency rate |
| **Responsiveness** | 0-100 | 60-100 | Q&A ability |

---

## Key Concepts

### Conservative Fusion
- Only creates fused evidence when BOTH audio AND video show anomalies
- Reduces false positives
- High precision, lower recall
- Suitable for clinical applications

### Temporal Windows
- **Window Size**: 5.0 seconds
- **Hop Size**: 2.5 seconds (50% overlap)
- **Context Expansion**: ±3.0 seconds around detected anomalies

### Confidence Levels
- **Strong**: ≥ 0.7 (70%)
- **Moderate**: 0.5-0.7 (50-70%)
- **Weak**: < 0.5 (< 50%)

### Thresholds (configurable in `configs/thresholds.yaml`)
```yaml
audio:
  pitch_std_threshold: 30.0      # Hz
  energy_std_threshold: 0.15     # normalized
  speech_rate_min: 60            # WPM
  speech_rate_max: 250           # WPM

video:
  head_motion_threshold: 5.0     # pixels/frame
  body_motion_threshold: 10.0    # pixels/frame
  hand_motion_threshold: 15.0    # pixels/frame

fusion:
  strong_threshold: 0.7
  min_confidence: 0.5
```

---

## Pipeline Execution

**Command**:
```bash
python main.py --video "data/raw/video.mp4" --output "data/outputs"
```

**Duration**: ~2-3 minutes per minute of video

**Output Files**:
1. `session_*_report.html` - Interactive dashboard
2. `session_*_timeline.png` - Multimodal timeline visualization
3. `session_*_scores.png` - Behavioral scores plot
4. `session_*_annotations.json` - Machine-readable segments
5. `session_*_audio.wav` - Extracted audio track

---

## Dependencies

**Core Models**:
- Silero VAD (torch hub)
- pyannote.audio (Hugging Face)
- HuBERT (facebook/hubert-base-ls960)
- MediaPipe Face Mesh
- MediaPipe Pose

**Libraries**:
- Python 3.8+
- PyTorch 2.8.0
- OpenCV 4.x
- MediaPipe 0.10.x
- NumPy, SciPy, Pandas
- Matplotlib (visualization)
- PyYAML (config)

---

## Clinical Disclaimer

This system provides **observational data** and **computational metrics** for behavioral analysis. It is **NOT a diagnostic tool**. All findings must be interpreted by qualified clinicians within the context of:

- Comprehensive clinical evaluation
- Developmental history
- Validated assessment tools (ADOS-2, ADI-R, etc.)
- Multi-informant observations
- Standardized testing

**Metrics are supplementary information only.**

---

## Version Information

- **System Version**: 1.0
- **Last Updated**: January 22, 2026
- **Platform**: Windows/Linux/macOS
- **License**: Research Use

---

## Contact & Support

For technical questions, threshold adjustments, or custom analysis needs, refer to the system administrator or development team.
