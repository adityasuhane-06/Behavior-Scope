# Clinical Analysis Metrics - Complete Technical Explanation

## How Each Metric is Calculated

---

## A. AUTISM-SPECIFIC METRICS

---

### 1. Social Engagement Index (SEI)

**Purpose:** Overall measure of social interaction quality  
**Range:** 0-100 (higher = better engagement)  
**Formula:** Weighted combination of 4 components

#### Calculation Formula:
```
SEI = (eye_contact × 0.35) + (turn_taking × 0.30) + 
      (responsiveness × 0.20) + (attention × 0.15)
```

#### Component Breakdown:

**Eye Contact Component (35% weight):**
- Extracted from eye contact analysis (see section below)
- Based on % of session with eye contact
- Formula: `eye_contact_score` (0-100)

**Turn-Taking Component (30% weight):**
- Extracted from turn-taking analysis
- Based on reciprocity score
- Formula: `reciprocity_score` (0-100)

**Responsiveness Component (20% weight):**
- Based on how quickly the child responds
- Formula: Convert response latency to score
```python
def latency_to_score(latency_sec):
    if latency < 0.5:
        return 70  # Too fast (impulsivity)
    elif latency <= 1.0:
        return 100  # Optimal (typical)
    elif latency <= 2.0:
        return 80   # Slightly delayed
    elif latency <= 3.0:
        return 60   # Moderately delayed
    else:
        return 40   # Significantly delayed
```

**Attention Component (15% weight):**
- Uses Attention Stability Score (ASS) from behavioral scoring
- Formula: `attention_stability_score` (0-100)

#### Interpretation:
| SEI Score | Interpretation |
|-----------|----------------|
| 70-100 | Good social engagement |
| 50-70 | Emerging skills |
| 30-50 | Moderate difficulties |
| 0-30 | Significant challenges |

#### Code Location:
```
autism_analysis/social_engagement.py
Function: compute_social_engagement_index()
```

---

### 2. Turn-Taking Analysis

**Purpose:** Measure conversational reciprocity  
**Key Metrics:** Balance, response latency, interruptions

#### What is Measured:

**A. Speaking Time Balance**
```python
# Calculate who talks more
child_time = sum(duration for all child turns)
therapist_time = sum(duration for all therapist turns)
total_time = child_time + therapist_time

child_percentage = (child_time / total_time) × 100

# Balance score (50 = perfect balance)
balance_score = 100 - |50 - child_percentage|
```

**Optimal Range:** 40-60% (child should speak 40-60% of the time)

**B. Response Latency**
```python
# Time between therapist speech end and child speech start
for each turn_pair:
    if previous_speaker == therapist AND current_speaker == child:
        latency = current_turn.start_time - previous_turn.end_time
        response_latencies.append(latency)

mean_latency = average(response_latencies)
median_latency = median(response_latencies)
```

**Interpretation:**
| Latency | Meaning |
|---------|---------|
| < 0.5s | Very fast (normal or impulsive) |
| 0.5-1.5s | Typical response time |
| 1.5-3.0s | Slightly delayed (may indicate processing difficulty) |
| > 3.0s | Significantly delayed (concern) |

**C. Turn Count**
```python
# Count speaker changes
turn_changes = 0
for i in range(1, len(segments)):
    if segments[i].speaker != segments[i-1].speaker:
        turn_changes += 1

total_turns = turn_changes + 1
```

**Typical:** 10-30 turns in a 5-minute session

**D. Interruptions**
```python
# Negative gap = overlap/interruption
for each turn:
    gap = turn.start_time - previous_turn.end_time
    if gap < -0.5:  # 0.5s overlap threshold
        interruption_count += 1
```

**E. Reciprocity Score**
```python
def compute_reciprocity_score(turns, mean_latency, interruptions):
    # Start at 100
    score = 100.0
    
    # Penalty for delayed responses
    if mean_latency > 1.0:
        score -= (mean_latency - 1.0) × 10  # -10 per second delay
    
    # Penalty for interruptions
    score -= interruptions × 5  # -5 per interruption
    
    # Penalty for imbalanced conversation
    imbalance_penalty = abs(50 - child_percentage) × 0.5
    score -= imbalance_penalty
    
    return max(0, min(100, score))  # Clip to 0-100
```

#### Interpretation:
| Reciprocity Score | Meaning |
|-------------------|---------|
| 70-100 | Good turn-taking |
| 50-70 | Adequate with some difficulties |
| 30-50 | Significant challenges |
| 0-30 | Poor reciprocity |

#### Code Location:
```
autism_analysis/turn_taking.py
Function: analyze_turn_taking()
```

---

### 3. Eye Contact Detection

**Purpose:** Quantify eye contact patterns  
**Key Metrics:** Duration, frequency, context (speaking vs. listening)

#### Detection Algorithm:

**Step 1: Determine if looking at camera**
```python
# Eye contact criteria:
# 1. Head facing forward (yaw angle close to 0°)
# 2. Gaze directed forward (eyes looking straight)

for each video_window:
    head_yaw = head_pose.yaw_angle  # -90° to +90°
    gaze_proxy = gaze_forward_score  # 0.0 to 1.0
    
    # Thresholds (configurable)
    head_threshold = 30°  # Head within ±30° of center
    gaze_threshold = 0.05  # Low gaze value = forward
    
    is_eye_contact = (abs(head_yaw) < head_threshold) AND 
                     (gaze_proxy < gaze_threshold)
```

**Step 2: Build eye contact episodes**
```python
episodes = []
in_contact = False
contact_start = 0.0

for window in video_windows:
    if is_eye_contact(window) and not in_contact:
        # Start new episode
        in_contact = True
        contact_start = window.start_time
    
    elif not is_eye_contact(window) and in_contact:
        # End episode
        duration = window.start_time - contact_start
        
        if duration >= min_duration (0.5s):
            episodes.append(EyeContactEvent(
                start=contact_start,
                end=window.start_time,
                duration=duration
            ))
        
        in_contact = False
```

**Step 3: Compute statistics**
```python
# Overall metrics
total_duration = sum(episode.duration for all episodes)
episode_count = len(episodes)
mean_duration = total_duration / episode_count

session_duration = video_end - video_start
percentage = (total_duration / session_duration) × 100

# Frequency
frequency_per_minute = (episode_count / session_duration) × 60
```

**Step 4: Context-specific metrics**
```python
# During speaking (child talking)
speaking_episodes = [e for e in episodes if child_speaking_at(e.time)]
speaking_duration = sum(e.duration for e in speaking_episodes)

child_speaking_time = sum(turn.duration for turns where speaker=child)
speaking_percentage = (speaking_duration / child_speaking_time) × 100

# During listening (therapist talking)
listening_episodes = [e for e in episodes if therapist_speaking_at(e.time)]
listening_duration = sum(e.duration for e in listening_episodes)

therapist_speaking_time = sum(turn.duration for turns where speaker=therapist)
listening_percentage = (listening_duration / therapist_speaking_time) × 100
```

**Step 5: Eye contact score**
```python
def compute_eye_contact_score(percentage, mean_duration, frequency):
    # Weighted combination
    score = (
        percentage × 0.5 +           # 50% weight on coverage
        (mean_duration × 10) × 0.3 + # 30% weight on episode length
        (frequency × 2) × 0.2         # 20% weight on frequency
    )
    
    return min(100, score)  # Cap at 100
```

#### Interpretation:
| Metric | Typical Range | ASD Range |
|--------|---------------|-----------|
| **Session Coverage** | 50-80% | 10-30% |
| **Episode Count** | 20-40 per 5 min | 5-15 per 5 min |
| **Mean Duration** | 2-5 seconds | 0.5-2 seconds |
| **During Speaking** | 30-50% | 10-30% |
| **During Listening** | 50-70% | 20-40% |
| **Eye Contact Score** | 70-100 | 20-50 |

**Clinical Note:** Eye contact during speaking is typically more challenging than during listening, especially for individuals with autism.

#### Code Location:
```
autism_analysis/eye_contact.py
Function: analyze_eye_contact()
```

---

### 4. Stereotyped Movements (Stereotypies)

**Purpose:** Detect repetitive motor behaviors  
**Types:** Hand flapping, body rocking, head movements

#### Detection Algorithm:

**Step 1: Extract body part motion**
```python
# Get pose landmark positions over time
time_series = {
    'timestamps': [t0, t1, t2, ...],
    'left_hand_x': [x0, x1, x2, ...],
    'left_hand_y': [y0, y1, y2, ...],
    'right_hand_x': [...],
    'right_hand_y': [...],
    'torso_x': [...],
    'torso_y': [...],
    'head_x': [...],
    'head_y': [...]
}
```

**Step 2: Detect cyclic/repetitive motion using FFT**
```python
from scipy.fft import fft, fftfreq

def detect_cyclic_motion(timestamps, x_coords, y_coords):
    # Compute 2D motion magnitude
    motion = sqrt(x_coords² + y_coords²)
    
    # Remove DC component (mean)
    motion_centered = motion - mean(motion)
    
    # Fast Fourier Transform to find frequencies
    fft_values = fft(motion_centered)
    frequencies = fftfreq(len(motion), d=timestamps[1]-timestamps[0])
    
    # Find dominant frequency (peak in FFT)
    power_spectrum = abs(fft_values)²
    dominant_freq = frequencies[argmax(power_spectrum)]
    
    # Amplitude (how much movement)
    amplitude = max(motion) - min(motion)
    
    return dominant_freq, amplitude
```

**Step 3: Classification criteria**
```python
# Stereotypy criteria:
# 1. Frequency in repetitive range (0.5-5 Hz)
# 2. Amplitude above threshold (significant movement)
# 3. Minimum cycles (at least 3 repetitions)

if (0.5 <= frequency <= 5.0) AND 
   (amplitude > threshold) AND 
   (duration × frequency >= 3):  # At least 3 cycles
    
    # Classify type based on body part and characteristics
    if body_part == 'hand' and frequency > 2.0:
        type = 'hand_flapping'
    elif body_part == 'hand' and frequency <= 2.0:
        type = 'hand_wringing'
    elif body_part == 'body' and 0.5 <= frequency <= 2.0:
        type = 'body_rocking'
    elif body_part == 'head':
        type = 'head_nodding' or 'head_shaking'
```

**Step 4: Compute metrics**
```python
# Episode statistics
total_duration = sum(episode.duration for all episodes)
episode_count = len(episodes)
mean_duration = total_duration / episode_count

session_duration = video_end - video_start
percentage = (total_duration / session_duration) × 100

# Count by type
type_counts = {
    'flapping': count,
    'rocking': count,
    'head_movement': count,
    'other': count
}

# Frequency and amplitude averages
mean_frequency = average([e.frequency for e in episodes])
mean_amplitude = average([e.amplitude for e in episodes])
```

**Step 5: Intensity score**
```python
def compute_stereotypy_intensity(percentage, count, freq, amplitude, duration):
    # Start at 0 (no stereotypy)
    intensity = 0
    
    # Add points for session coverage
    intensity += percentage × 0.5  # Max +50 points
    
    # Add points for frequency
    intensity += (count / duration × 60) × 2  # Episodes per minute
    
    # Add points for high repetition rate
    if freq > 2.0:
        intensity += 20
    
    # Add points for high amplitude
    if amplitude > 0.25:
        intensity += 10
    
    return min(100, intensity)  # Cap at 100
```

#### Interpretation:
| Intensity Score | Meaning |
|----------------|---------|
| 0-20 | Minimal/absent |
| 20-40 | Occasional stereotypies |
| 40-60 | Frequent stereotypies |
| 60-100 | Pervasive stereotypies (concern) |

**Common Patterns:**
- **Hand Flapping:** 2-4 Hz, high amplitude (>0.2), bilateral
- **Body Rocking:** 0.5-2 Hz, moderate amplitude, torso movement
- **Head Nodding:** 1-3 Hz, vertical head motion
- **Hand Wringing:** < 2 Hz, complex hand movements

#### Code Location:
```
autism_analysis/stereotypy.py
Function: detect_stereotyped_movements()
```

---

## B. CLINICAL METRICS

---

### 5. Stuttering Analysis (SSI)

**Purpose:** Quantify speech disfluencies  
**Output:** Stuttering Severity Index (SSI, 0-100)

#### Types of Disfluencies Detected:

**1. Sound/Syllable Repetitions**
```python
# Example: "b-b-b-ball" or "ba-ba-ball"
# Detection: Analyze audio for rapid repetitive patterns

def detect_repetitions(audio_segment):
    # Compute autocorrelation to find periodicity
    autocorr = correlate(audio, audio, mode='same')
    
    # Find peaks in autocorrelation (repetition cycles)
    peaks = find_peaks(autocorr, distance=min_cycle_length)
    
    # If 3+ peaks within short time → repetition
    if len(peaks) >= 3:
        cycle_period = mean(diff(peaks))
        frequency = 1 / cycle_period
        
        if 3 < frequency < 10:  # 3-10 Hz typical for repetitions
            return DisfluencyEvent(
                type='repetition',
                frequency=frequency,
                duration=len(peaks) × cycle_period
            )
```

**2. Prolongations**
```python
# Example: "sssssnake" (extended 's' sound)
# Detection: Sustained energy in specific frequency band

def detect_prolongations(audio_segment):
    # Compute energy envelope
    energy = librosa.feature.rms(audio)
    
    # Find sustained regions (low variance)
    is_sustained = (energy > threshold) AND (std(energy) < low_variance)
    
    # Prolongation if sustained > 0.5 seconds
    sustained_duration = count_true(is_sustained) × frame_time
    
    if sustained_duration > 0.5:
        return DisfluencyEvent(
            type='prolongation',
            duration=sustained_duration
        )
```

**3. Blocks**
```python
# Example: Pause mid-word (silent struggle)
# Detection: Unusual pause in speech context

def detect_blocks(prosodic_features, speech_segments):
    for segment in speech_segments:
        pauses = find_silence_periods(segment)
        
        for pause in pauses:
            # Block if:
            # 1. Pause is mid-word (not at natural boundary)
            # 2. Pause duration > threshold (0.3s)
            # 3. Preceded/followed by speech attempt
            
            if (pause.duration > 0.3) AND not_at_word_boundary(pause):
                return DisfluencyEvent(
                    type='block',
                    duration=pause.duration
                )
```

#### Metrics Computed:

**A. Disfluency Rate**
```python
# Standard metric: disfluencies per 100 syllables

total_disfluencies = len(detected_events)
estimated_syllables = speaking_time × 3  # ~3 syllables/second average

disfluency_rate = (total_disfluencies / estimated_syllables) × 100
```

**Interpretation:**
| Rate | Severity |
|------|----------|
| < 3% | Normal/mild |
| 3-5% | Mild stuttering |
| 5-10% | Moderate stuttering |
| 10-20% | Moderate-severe |
| > 20% | Severe stuttering |

**B. Stuttering Severity Index (SSI-4 inspired)**
```python
def compute_stuttering_severity_index(rate, events, speaking_time):
    score = 0
    
    # Component 1: Frequency (disfluency rate)
    if rate < 2:
        score += 0
    elif rate < 5:
        score += 20
    elif rate < 10:
        score += 40
    elif rate < 15:
        score += 60
    else:
        score += 80
    
    # Component 2: Duration (longest block/prolongation)
    longest = max([e.duration for e in events if e.type in ['block', 'prolongation']])
    
    if longest < 0.5:
        score += 0
    elif longest < 1.0:
        score += 10
    elif longest < 2.0:
        score += 15
    else:
        score += 20
    
    # Component 3: Physical concomitants (not available, skip)
    
    return min(100, score)
```

**SSI Interpretation:**
| SSI Score | Severity |
|-----------|----------|
| 0-20 | Very mild |
| 20-40 | Mild |
| 40-60 | Moderate |
| 60-80 | Severe |
| 80-100 | Very severe |

#### Code Location:
```
clinical_analysis/stuttering.py
Function: analyze_stuttering()
```

---

### 6. Question-Response Ability

**Purpose:** Assess comprehension and communication  
**Key Metrics:** Response rate, latency, appropriateness

#### Detection Algorithm:

**Step 1: Detect questions from therapist**
```python
# Method 1: Rising intonation (pitch rise at end)
def is_question_prosodic(prosodic_features):
    # Check if pitch rises at end of utterance
    pitch_trend = prosodic_features.pitch_slope
    
    if pitch_trend > 30:  # Rising intonation (30+ Hz increase)
        return True
    
    # Also check final pitch vs. mean
    final_pitch = prosodic_features.pitch[-1]
    mean_pitch = prosodic_features.pitch_mean
    
    if final_pitch > mean_pitch + 50:  # 50 Hz above mean
        return True
    
    return False

# Method 2: Turn-taking pattern
# Questions are typically short therapist turns followed by longer patient turns
```

**Step 2: Pair questions with responses**
```python
question_response_pairs = []

for i, segment in enumerate(therapist_segments):
    if is_question(segment):
        # Look for patient response within next 10 seconds
        question_end = segment.end_time
        
        for patient_seg in patient_segments:
            gap = patient_seg.start_time - question_end
            
            if 0 <= gap <= 10.0:  # Response within 10s
                pair = {
                    'question': segment,
                    'response': patient_seg,
                    'latency': gap,
                    'response_duration': patient_seg.duration
                }
                question_response_pairs.append(pair)
                break  # Found response, move to next question
        else:
            # No response found
            question_response_pairs.append({
                'question': segment,
                'response': None,
                'latency': None,
                'response_duration': None
            })
```

**Step 3: Compute metrics**
```python
total_questions = len(questions_detected)
answered_questions = len([p for p in pairs if p['response'] is not None])

# Response rate
response_rate = (answered_questions / total_questions) × 100

# Latencies (only for answered questions)
latencies = [p['latency'] for p in pairs if p['response'] is not None]
mean_latency = mean(latencies)
median_latency = median(latencies)

# Response durations
durations = [p['response_duration'] for p in pairs if p['response'] is not None]
mean_duration = mean(durations)
```

**Step 4: Appropriateness assessment**
```python
def assess_response_appropriateness(pair):
    response = pair['response']
    
    # Heuristic criteria:
    score = 100
    
    # Too fast (< 0.3s) may be interruption/not listening
    if pair['latency'] < 0.3:
        score -= 20
    
    # Too slow (> 5s) may indicate difficulty
    if pair['latency'] > 5.0:
        score -= 10
    
    # Too brief (< 1s) may be incomplete
    if response.duration < 1.0:
        score -= 30
    
    # Very long (> 30s) may be off-topic
    if response.duration > 30.0:
        score -= 20
    
    return max(0, score)

# Average across all responses
appropriateness_scores = [assess(p) for p in pairs if p['response']]
appropriateness_rate = mean(appropriateness_scores)
```

**Step 5: Responsiveness Index**
```python
def compute_responsiveness_index(response_rate, mean_latency, appropriateness):
    # Weighted combination
    index = (
        response_rate × 0.50 +        # 50% weight on answering
        latency_to_score(mean_latency) × 0.30 +  # 30% on speed
        appropriateness × 0.20         # 20% on quality
    )
    
    return index

def latency_to_score(latency):
    if latency <= 1.0:
        return 100
    elif latency <= 2.0:
        return 85
    elif latency <= 3.0:
        return 70
    elif latency <= 5.0:
        return 50
    else:
        return 30
```

#### Interpretation:
| Responsiveness Index | Meaning |
|---------------------|---------|
| 80-100 | Excellent responsiveness |
| 60-80 | Good responsiveness |
| 40-60 | Adequate with some difficulty |
| 20-40 | Significant challenges |
| 0-20 | Poor responsiveness |

**Clinical Significance:**
- **High response rate + Fast latency:** Good comprehension and engagement
- **Low response rate:** May indicate comprehension difficulty or withdrawal
- **Slow latency + Good responses:** Processing delay but intact understanding
- **Fast latency + Brief responses:** May not be processing questions fully

#### Code Location:
```
clinical_analysis/question_response.py
Function: analyze_question_response_ability()
```

---

## Summary Table: All Clinical Metrics

| Metric | Range | Optimal | Calculation Method |
|--------|-------|---------|-------------------|
| **SEI** | 0-100 | 70-100 | Weighted: eye contact (35%) + turn-taking (30%) + responsiveness (20%) + attention (15%) |
| **Turn Balance** | 0-100% | 40-60% | (child_time / total_time) × 100 |
| **Response Latency** | 0-10s | 0.5-1.5s | Gap between therapist speech end and child speech start |
| **Reciprocity** | 0-100 | 70-100 | 100 - penalties(latency + interruptions + imbalance) |
| **Eye Contact %** | 0-100% | 50-80% | (contact_time / session_time) × 100 |
| **Eye Contact Score** | 0-100 | 70-100 | Weighted: coverage (50%) + duration (30%) + frequency (20%) |
| **Stereotypy Intensity** | 0-100 | 0-20 | Coverage + frequency + amplitude factors |
| **Disfluency Rate** | 0-100% | < 3% | (disfluencies / syllables) × 100 |
| **SSI** | 0-100 | 0-20 | Frequency + duration + concomitants |
| **Response Rate** | 0-100% | 60-100% | (answered / total_questions) × 100 |
| **Responsiveness Index** | 0-100 | 60-100 | Weighted: response_rate (50%) + latency (30%) + quality (20%) |

---

## Configuration Thresholds

All detection thresholds are configurable in `configs/thresholds.yaml`:

```yaml
autism_analysis:
  eye_contact:
    head_facing_threshold_deg: 30.0      # Head within ±30° = looking
    gaze_forward_threshold: 0.05         # Gaze proxy threshold
    min_episode_duration_sec: 0.5        # Minimum eye contact duration
  
  turn_taking:
    interruption_threshold_sec: 0.5      # Overlap = interruption
    typical_latency_sec: 1.0             # Normal response time
    elevated_latency_sec: 3.0            # Delayed threshold
  
  stereotypy:
    frequency_min_hz: 0.5                # Min repetition rate
    frequency_max_hz: 5.0                # Max repetition rate
    amplitude_threshold: 0.15            # Movement size threshold
    min_cycles: 3                        # Minimum repetitions

clinical_analysis:
  stuttering:
    repetition_cycle_threshold: 0.15     # Repetition detection
    prolongation_duration_sec: 0.5       # Sound prolongation
    block_silence_sec: 0.3               # Block pause duration
  
  question_response:
    max_response_latency_sec: 10.0       # Max time to respond
    min_response_duration_sec: 0.5       # Minimum response length
    pitch_rise_threshold_hz: 30.0        # Rising intonation
```

---

## Clinical Interpretation Guidelines

### When to Be Concerned:

**Autism-Specific:**
- SEI < 50 (significant social difficulties)
- Eye contact < 30% of session
- Response latency consistently > 3 seconds
- Stereotypy intensity > 40 (frequent repetitive behaviors)
- Reciprocity score < 50 (poor turn-taking)

**Communication:**
- Disfluency rate > 10% (moderate-severe stuttering)
- SSI > 60 (severe stuttering)
- Response rate < 40% (comprehension or engagement issues)
- Responsiveness index < 40 (significant challenges)

### Contextual Factors:

These metrics should ALWAYS be interpreted within context:
- **Age/developmental level** (younger children have different norms)
- **Cultural background** (eye contact norms vary)
- **Anxiety/stress** (performance pressure affects all metrics)
- **Medication effects** (can influence behavior)
- **Time of day** (fatigue affects performance)
- **Rapport with clinician** (comfort level matters)

### Clinical Use:

✅ **Appropriate:**
- Supplementary objective data for clinical assessment
- Progress monitoring over time (comparing sessions)
- Identifying specific areas for intervention
- Documenting behavioral patterns

❌ **Inappropriate:**
- Sole basis for diagnosis
- Comparing across individuals (metrics are relative)
- Replacing comprehensive clinical evaluation
- Making treatment decisions without clinical context

---

## Frequently Asked Questions

**Q: Why are some metrics 0?**  
A: Short videos or stable behavior may not trigger detection thresholds. This is normal and indicates well-regulated behavior.

**Q: How accurate is eye contact detection?**  
A: Depends on video quality and face visibility. Accuracy is ~80-90% with good frontal video. Low confidence if face is occluded or turned away.

**Q: Can these metrics diagnose autism?**  
A: **NO.** These are observational measures only. Diagnosis requires comprehensive clinical evaluation (ADOS-2, ADI-R, etc.) by qualified professionals.

**Q: What if speaker diarization fails?**  
A: Turn-taking metrics may be inaccurate. Check if diarization correctly identified speakers in the report. Manual review recommended.

**Q: How do I improve detection accuracy?**  
A: Use high-quality video (1080p), ensure good lighting, frontal face visible, minimal background noise, clear audio.

---

## References

1. **Autism Assessment:**
   - Lord et al. (2012). Autism Diagnostic Observation Schedule, Second Edition (ADOS-2)
   - Mundy et al. (2007). Joint attention and social-communication in autism
   
2. **Stuttering:**
   - Riley, G. D. (2009). Stuttering Severity Instrument (SSI-4)
   - Yairi & Ambrose (2005). Early childhood stuttering
   
3. **Eye Contact:**
   - Jones & Klin (2013). Attention to eyes in autism
   - Tanaka & Sung (2016). The "eye avoidance" hypothesis
   
4. **Turn-Taking:**
   - Tager-Flusberg & Joseph (2003). Identifying neurocognitive phenotypes in autism

---

**Document Version:** 2.0  
**Last Updated:** January 22, 2026  
**Contact:** System Administrator
