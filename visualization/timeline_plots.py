"""
Timeline visualization for multimodal behavioral analysis.

Generates interactive and static plots showing:
- Audio features (speech rate, pauses, prosody) over time
- Video features (head motion, body motion, hand velocity) over time
- Fused dysregulation confidence scores
- Behavioral score trends

Clinical rationale:
- Timeline visualization reveals temporal patterns
- Multi-modal overlay shows feature relationships
- Dysregulation markers guide clinician attention
- Trend analysis reveals session dynamics

Engineering approach:
- Matplotlib for static publication-quality plots
- Plotly for interactive HTML exports
- Color-coded confidence levels
- Annotation support for key moments
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available - interactive plots disabled")

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10


def plot_multimodal_timeline(
    audio_windows: List,
    video_aggregated: List,
    fused_evidence: List,
    output_path: str,
    title: str = "Multimodal Behavioral Analysis Timeline",
    interactive: bool = True
) -> str:
    """
    Create comprehensive multimodal timeline plot.
    
    Layout:
    - Panel 1: Audio features (speech rate, pauses, pitch variance)
    - Panel 2: Video features (head motion, body motion, hand velocity)
    - Panel 3: Fused dysregulation confidence with color-coded levels
    
    Args:
        audio_windows: List of InstabilityWindow objects
        video_aggregated: List of AggregatedFeatures
        fused_evidence: List of FusedEvidence
        output_path: Path to save plot (PNG or HTML)
        title: Plot title
        interactive: If True and plotly available, create interactive HTML
        
    Returns:
        Path to saved plot file
    """
    logger.info(f"Generating multimodal timeline plot: {output_path}")
    
    output_path = Path(output_path)
    
    # Choose plot type
    if interactive and PLOTLY_AVAILABLE and output_path.suffix == '.html':
        return _create_interactive_timeline(
            audio_windows,
            video_aggregated,
            fused_evidence,
            output_path,
            title
        )
    else:
        return _create_static_timeline(
            audio_windows,
            video_aggregated,
            fused_evidence,
            output_path,
            title
        )


def _create_static_timeline(
    audio_windows: List,
    video_aggregated: List,
    fused_evidence: List,
    output_path: Path,
    title: str
) -> str:
    """Create static matplotlib timeline."""
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 1, height_ratios=[1, 1, 1, 0.8], hspace=0.3)
    
    # Panel 1: Audio features
    ax1 = fig.add_subplot(gs[0])
    _plot_audio_features(ax1, audio_windows)
    ax1.set_title("Audio Features", fontweight='bold', fontsize=12)
    ax1.set_ylabel("Normalized Value")
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Video features
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    _plot_video_features(ax2, video_aggregated)
    ax2.set_title("Video Features", fontweight='bold', fontsize=12)
    ax2.set_ylabel("Normalized Value")
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Fused confidence
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    _plot_fused_confidence(ax3, fused_evidence)
    ax3.set_title("Multimodal Dysregulation Confidence", fontweight='bold', fontsize=12)
    ax3.set_ylabel("Confidence")
    ax3.set_xlabel("Time (seconds)")
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Confidence level markers
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    _plot_confidence_markers(ax4, fused_evidence)
    ax4.set_ylabel("Level")
    ax4.set_xlabel("Time (seconds)")
    ax4.set_yticks([])
    
    # Overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Static timeline saved: {output_path}")
    return str(output_path)


def _plot_audio_features(ax, audio_windows: List):
    """Plot audio features on axis."""
    if not audio_windows:
        ax.text(0.5, 0.5, 'No audio data', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Extract time series
    times = [(w.start_time + w.end_time) / 2 for w in audio_windows]
    
    # Normalize features for plotting
    speech_rates = [w.contributing_features.get('speech_rate', 0) for w in audio_windows]
    pause_irregularity = [w.contributing_features.get('pause_irregularity', 0) for w in audio_windows]
    pitch_variance = [w.contributing_features.get('pitch_variance_zscore', 0) for w in audio_windows]
    
    # Plot
    ax.plot(times, _normalize_series(speech_rates), 'o-', label='Speech Rate', alpha=0.7, linewidth=2)
    ax.plot(times, _normalize_series(pause_irregularity), 's-', label='Pause Irregularity', alpha=0.7, linewidth=2)
    ax.plot(times, _normalize_series(pitch_variance), '^-', label='Pitch Variance', alpha=0.7, linewidth=2)
    
    ax.set_ylim(-0.1, 1.1)


def _plot_video_features(ax, video_aggregated: List):
    """Plot video features on axis."""
    if not video_aggregated:
        ax.text(0.5, 0.5, 'No video data', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Extract time series
    times = [(w.window_start_time + w.window_end_time) / 2 for w in video_aggregated]
    
    # Extract features
    head_motion = []
    body_motion = []
    hand_velocity = []
    
    for window in video_aggregated:
        # Head motion (average of yaw/pitch/roll std)
        face_feat = window.face_features
        head = np.mean([
            face_feat.get('head_yaw_std', 0),
            face_feat.get('head_pitch_std', 0),
            face_feat.get('head_roll_std', 0)
        ])
        head_motion.append(head)
        
        # Body motion
        pose_feat = window.pose_features
        body_motion.append(pose_feat.get('upper_body_motion_mean', 0))
        
        # Hand velocity
        hand_velocity.append(pose_feat.get('hand_velocity_max_mean', 0))
    
    # Plot
    ax.plot(times, _normalize_series(head_motion), 'o-', label='Head Motion', alpha=0.7, linewidth=2)
    ax.plot(times, _normalize_series(body_motion), 's-', label='Body Motion', alpha=0.7, linewidth=2)
    ax.plot(times, _normalize_series(hand_velocity), '^-', label='Hand Velocity', alpha=0.7, linewidth=2)
    
    ax.set_ylim(-0.1, 1.1)


def _plot_fused_confidence(ax, fused_evidence: List):
    """Plot fused confidence scores."""
    if not fused_evidence:
        ax.text(0.5, 0.5, 'No fused evidence', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Sort by time
    sorted_evidence = sorted(fused_evidence, key=lambda fe: fe.start_time)
    
    # Extract series
    times = [(fe.start_time + fe.end_time) / 2 for fe in sorted_evidence]
    confidences = [fe.fused_confidence for fe in sorted_evidence]
    audio_scores = [fe.audio_score for fe in sorted_evidence]
    video_scores = [fe.video_score for fe in sorted_evidence]
    
    # Plot
    ax.plot(times, confidences, 'o-', label='Fused Confidence', color='red', alpha=0.8, linewidth=3)
    ax.plot(times, audio_scores, 's--', label='Audio Score', alpha=0.5, linewidth=1.5)
    ax.plot(times, video_scores, '^--', label='Video Score', alpha=0.5, linewidth=1.5)
    
    # Threshold lines
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.3, label='Strong Threshold')
    ax.axhline(y=0.5, color='yellow', linestyle='--', alpha=0.3, label='Moderate Threshold')
    
    ax.set_ylim(0, 1.05)


def _plot_confidence_markers(ax, fused_evidence: List):
    """Plot confidence level markers as colored bars."""
    if not fused_evidence:
        return
    
    # Color mapping
    color_map = {
        'strong': '#d62728',  # Red
        'moderate': '#ff7f0e',  # Orange
        'weak': '#ffbb78',  # Light orange
        'none': '#e0e0e0'  # Gray
    }
    
    # Plot each segment
    for fe in fused_evidence:
        color = color_map.get(fe.confidence_level, '#808080')
        ax.axvspan(fe.start_time, fe.end_time, alpha=0.6, color=color)
    
    # Create legend
    patches = [
        mpatches.Patch(color=color_map['strong'], label='Strong', alpha=0.6),
        mpatches.Patch(color=color_map['moderate'], label='Moderate', alpha=0.6),
        mpatches.Patch(color=color_map['weak'], label='Weak', alpha=0.6),
    ]
    ax.legend(handles=patches, loc='upper right', fontsize=8)
    
    ax.set_ylim(0, 1)


def _create_interactive_timeline(
    audio_windows: List,
    video_aggregated: List,
    fused_evidence: List,
    output_path: Path,
    title: str
) -> str:
    """Create interactive plotly timeline."""
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Audio Features', 'Video Features', 'Multimodal Confidence'),
        vertical_spacing=0.08,
        row_heights=[0.33, 0.33, 0.34]
    )
    
    # Audio features
    if audio_windows:
        times = [(w.start_time + w.end_time) / 2 for w in audio_windows]
        speech_rates = _normalize_series([w.contributing_features.get('speech_rate', 0) for w in audio_windows])
        
        fig.add_trace(
            go.Scatter(x=times, y=speech_rates, mode='lines+markers', name='Speech Rate', line=dict(width=2)),
            row=1, col=1
        )
    
    # Video features
    if video_aggregated:
        times = [(w.window_start_time + w.window_end_time) / 2 for w in video_aggregated]
        head_motion = []
        for window in video_aggregated:
            face_feat = window.face_features
            head = np.mean([
                face_feat.get('head_yaw_std', 0),
                face_feat.get('head_pitch_std', 0),
                face_feat.get('head_roll_std', 0)
            ])
            head_motion.append(head)
        
        fig.add_trace(
            go.Scatter(x=times, y=_normalize_series(head_motion), mode='lines+markers', name='Head Motion', line=dict(width=2)),
            row=2, col=1
        )
    
    # Fused confidence
    if fused_evidence:
        sorted_evidence = sorted(fused_evidence, key=lambda fe: fe.start_time)
        times = [(fe.start_time + fe.end_time) / 2 for fe in sorted_evidence]
        confidences = [fe.fused_confidence for fe in sorted_evidence]
        
        fig.add_trace(
            go.Scatter(
                x=times, y=confidences,
                mode='lines+markers',
                name='Fused Confidence',
                line=dict(width=3, color='red'),
                hovertemplate='Time: %{x:.1f}s<br>Confidence: %{y:.2f}<extra></extra>'
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=900,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
    fig.update_yaxes(title_text="Normalized Value", row=1, col=1)
    fig.update_yaxes(title_text="Normalized Value", row=2, col=1)
    fig.update_yaxes(title_text="Confidence", row=3, col=1)
    
    # Save
    fig.write_html(str(output_path))
    
    logger.info(f"Interactive timeline saved: {output_path}")
    return str(output_path)


def plot_behavioral_scores(
    scores: Dict,
    output_path: str,
    title: str = "Behavioral Indices Summary"
) -> str:
    """
    Create bar chart of behavioral scores.
    
    Args:
        scores: Dictionary with score results (VRI, MAI, ASS, RCI)
        output_path: Path to save plot
        title: Plot title
        
    Returns:
        Path to saved plot
    """
    logger.info(f"Generating behavioral scores plot: {output_path}")
    
    # Extract scores
    score_names = []
    score_values = []
    
    if 'vocal_regulation' in scores:
        score_names.append('Vocal\nRegulation')
        score_values.append(scores['vocal_regulation']['score'])
    
    if 'motor_agitation' in scores:
        score_names.append('Motor\nAgitation')
        score_values.append(scores['motor_agitation']['score'])
    
    if 'attention_stability' in scores:
        score_names.append('Attention\nStability')
        score_values.append(scores['attention_stability']['score'])
    
    if 'regulation_consistency' in scores:
        score_names.append('Regulation\nConsistency')
        score_values.append(scores['regulation_consistency']['score'])
    
    if not score_names:
        logger.warning("No scores to plot")
        return ""
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color by score value
    colors = [_get_score_color(val) for val in score_values]
    
    bars = ax.bar(score_names, score_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, score_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{value:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Styling
    ax.set_ylim(0, 110)
    ax.set_ylabel('Score (0-100)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Midpoint')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Behavioral scores plot saved: {output_path}")
    return output_path


def save_timeline_plot(
    fig,
    output_path: str,
    dpi: int = 150
):
    """
    Save matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure object
        output_path: Path to save
        dpi: Resolution
    """
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Plot saved: {output_path}")


def _normalize_series(values: List[float]) -> np.ndarray:
    """Normalize values to 0-1 range for plotting."""
    arr = np.array(values)
    if len(arr) == 0:
        return arr
    
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    if max_val - min_val < 1e-6:
        return np.ones_like(arr) * 0.5
    
    return (arr - min_val) / (max_val - min_val)


def _get_score_color(value: float) -> str:
    """Get color based on score value."""
    if value >= 75:
        return '#2ca02c'  # Green
    elif value >= 50:
        return '#ff7f0e'  # Orange
    else:
        return '#d62728'  # Red
