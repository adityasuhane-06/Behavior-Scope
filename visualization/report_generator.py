

import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass
import json
import base64

logger = logging.getLogger(__name__)


@dataclass
class BehavioralReport:
    """
    Container for complete behavioral analysis report.
    
    Attributes:
        session_id: Unique session identifier
        video_path: Path to analyzed video
        timestamp: Analysis timestamp
        scores: Dict of behavioral scores (VRI, MAI, ASS, RCI)
        fused_evidence: List of FusedEvidence objects
        audio_windows: List of audio InstabilityWindow objects
        video_aggregated: List of video AggregatedFeatures
        autism_results: Optional dict of autism-specific analysis results
        metadata: Additional session metadata
    """
    session_id: str
    video_path: str
    timestamp: str
    scores: Dict
    fused_evidence: List
    audio_windows: List
    video_aggregated: List
    metadata: Dict
    autism_results: Optional[Dict] = None
    clinical_results: Optional[Dict] = None


def generate_html_report(
    report_data: BehavioralReport,
    output_path: str,
    timeline_plot_path: Optional[str] = None,
    scores_plot_path: Optional[str] = None
) -> str:
    """
    Generate comprehensive HTML report.
    
    Args:
        report_data: BehavioralReport object with analysis results
        output_path: Path to save HTML report
        timeline_plot_path: Optional path to timeline plot image
        scores_plot_path: Optional path to scores plot image
        
    Returns:
        Path to generated HTML report
    """
    logger.info(f"Generating HTML report: {output_path}")
    
    # Build HTML content
    html = _build_html_structure(
        report_data,
        timeline_plot_path,
        scores_plot_path
    )
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    logger.info(f"HTML report saved: {output_path}")
    return output_path


def _build_html_structure(
    report: BehavioralReport,
    timeline_plot_path: Optional[str],
    scores_plot_path: Optional[str]
) -> str:
    """Build complete HTML document with Tailwind CSS."""
    
    # Embed images as base64 if provided
    timeline_img = _embed_image(timeline_plot_path) if timeline_plot_path else ""
    scores_img = _embed_image(scores_plot_path) if scores_plot_path else ""
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Behavior Scope Analysis - {report.session_id}</title>
    
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <!-- Tailwind Config -->
    <script>
        tailwind.config = {{
            theme: {{
                extend: {{
                    colors: {{
                        primary: '#3b82f6',
                        secondary: '#64748b',
                        success: '#22c55e',
                        warning: '#f59e0b',
                        danger: '#ef4444',
                        dark: '#0f172a',
                    }},
                    fontFamily: {{
                        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif'],
                    }}
                }}
            }}
        }}
    </script>
    
    <style>
        {_get_css_styles()} 
    </style>
</head>
<body class="bg-slate-50 text-slate-800 antialiased font-sans">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {_build_header(report)}
        {_build_executive_summary(report)}
        {_build_scores_section(report, scores_img)}
        {_build_facial_action_units_section(report)}
        {_build_autism_section(report) if report.autism_results else ""}
        {_build_clinical_section(report) if report.clinical_results else ""}
        {_build_timeline_section(report, timeline_img)}
        {_build_segments_section(report)}
        {_build_methodology_section()}
        {_build_footer(report)}
    </div>
</body>
</html>
"""
    return html


def _build_header(report: BehavioralReport) -> str:
    """Build report header with Tailwind."""
    return f"""
    <header class="bg-white rounded-xl shadow-sm p-6 border-l-4 border-primary">
        <div class="flex flex-col md:flex-row md:justify-between md:items-start">
            <div>
                <h1 class="text-3xl font-bold text-dark mb-2">
                    <i class="fas fa-brain text-primary mr-2"></i>Behavioral Scope Analysis
                </h1>
                <div class="flex flex-wrap gap-4 text-sm text-secondary mt-3">
                    <div class="flex items-center">
                        <i class="fas fa-fingerprint mr-1.5 w-5 text-center"></i>
                        <span class="font-medium">ID:</span>&nbsp;{report.session_id}
                    </div>
                    <div class="flex items-center">
                        <i class="far fa-file-video mr-1.5 w-5 text-center"></i>
                        <span class="font-medium">Source:</span>&nbsp;{Path(report.video_path).name}
                    </div>
                    <div class="flex items-center">
                        <i class="far fa-calendar-alt mr-1.5 w-5 text-center"></i>
                        <span class="font-medium">Date:</span>&nbsp;{report.timestamp}
                    </div>
                </div>
            </div>
            <div class="mt-4 md:mt-0">
                <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                    Behavioral Analysis
                </span>
            </div>
        </div>
        
        <div class="mt-6 bg-amber-50 rounded-lg p-4 border border-amber-100 flex items-start">
            <i class="fas fa-exclamation-triangle text-amber-500 mt-1 mr-3 flex-shrink-0"></i>
            <p class="text-sm text-amber-800">
                <strong>Clinical Use Only:</strong> This report is generated by an assistance tool and analyzes behavioral 
                patterns from audio/video data. It is <strong>not</strong> a medical diagnosis. Results should be interpreted by qualified 
                professionals in conjunction with other clinical assessments.
            </p>
        </div>
    </header>
    """


def _build_executive_summary(report: BehavioralReport) -> str:
    """Build executive summary section with Tailwind."""
    
    # Extract key metrics
    n_audio_windows = len(report.audio_windows)
    # Count only video windows with dysregulation (video_score >= 0.7)
    n_video_windows = len([v for v in report.video_aggregated if hasattr(v, 'video_score') and v.video_score >= 0.7])
    n_fused_segments = len(report.fused_evidence)
    
    # High-confidence segments
    high_conf_segments = [fe for fe in report.fused_evidence if fe.fused_confidence >= 0.7]
    
    summary = f"""
    <section class="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div class="px-6 py-4 border-b border-slate-100 bg-slate-50">
            <h2 class="text-xl font-bold text-slate-800 flex items-center">
                <i class="fas fa-clipboard-check text-secondary mr-2"></i>Executive Summary
            </h2>
        </div>
        
        <div class="p-6 grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="bg-blue-50 rounded-lg p-5 border border-blue-100">
                <h3 class="font-bold text-blue-800 mb-3 flex items-center">
                    <i class="fas fa-search-plus mr-2"></i>Dysregulation Detection Coverage
                </h3>
                <ul class="space-y-2 text-sm text-blue-900">
                    <li class="flex items-center"><i class="fas fa-microphone text-blue-500 mr-2"></i><span class="font-medium">Audio dysregulation events:</span>&nbsp;{n_audio_windows} detected</li>
                    <li class="flex items-center"><i class="fas fa-video text-blue-500 mr-2"></i><span class="font-medium">Video dysregulation events:</span>&nbsp;{n_video_windows} detected</li>
                    <li class="flex items-center"><i class="fas fa-layer-group text-blue-500 mr-2"></i><span class="font-medium">Multimodal fusion events:</span>&nbsp;{n_fused_segments} detected</li>
                    <li class="flex items-center"><i class="fas fa-exclamation-triangle text-blue-500 mr-2"></i><span class="font-medium">High-confidence crises:</span>&nbsp;{len(high_conf_segments)} detected</li>
                </ul>
                <p class="text-xs text-blue-700 mt-3 italic border-t border-blue-200 pt-3">
                    <i class="fas fa-info-circle mr-1"></i>Note: Clinical metrics (stuttering, turn-taking, eye contact) are analyzed separately and always processed regardless of dysregulation events.
                </p>
            </div>
            
            <div class="bg-slate-50 rounded-lg p-5 border border-slate-200">
                <h3 class="font-bold text-slate-800 mb-3 flex items-center">
                    <i class="fas fa-star mr-2 text-warning"></i>Key Findings
                </h3>
                <div class="prose prose-sm text-slate-600">
                    {_generate_key_findings(report)}
                </div>
            </div>
        </div>
    </section>
    """
    return summary


def _build_scores_section(report: BehavioralReport, scores_img: str) -> str:
    """Build behavioral scores section with Tailwind."""
    
    scores = report.scores
    
    score_html = """
    <section class="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <div class="mb-6">
            <h2 class="text-xl font-bold text-slate-800 flex items-center mb-2">
                <i class="fas fa-chart-line text-secondary mr-2"></i>Behavioral Indices
            </h2>
            <p class="text-slate-500 text-sm">
                Four interpretable indices derived from multimodal analysis. 
                Scores range from 0-100 with clinical interpretations provided.
            </p>
        </div>
    """
    
    # Embedded scores plot
    if scores_img:
        score_html += f'<div class="mb-8 rounded-lg overflow-hidden border border-slate-200 bg-slate-50 p-4 flex justify-center"><img src="{scores_img}" alt="Behavioral Scores" class="max-h-96" /></div>'
    
    # Detailed score breakdown
    score_html += '<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-6">'
    
    # Vocal Regulation Index
    if 'vocal_regulation' in scores:
        vri = scores['vocal_regulation']
        score_html += f"""
        <div class="bg-white rounded-lg border border-slate-200 p-5 hover:shadow-md transition-shadow">
            <div class="flex items-start justify-between mb-4">
                <div class="flex items-center">
                    <div class="p-2 rounded-lg bg-blue-50 text-blue-600 mr-3">
                        <i class="fas fa-microphone-alt text-lg"></i>
                    </div>
                    <h3 class="font-semibold text-slate-800">Vocal Regulation</h3>
                </div>
                <div class="px-2.5 py-0.5 rounded-full text-sm font-bold bg-slate-100 text-slate-700">
                    {vri['score']:.1f}
                </div>
            </div>
            
            <p class="text-sm text-slate-600 mb-4">{vri['explanation']}</p>
            
            <div class="grid grid-cols-3 gap-2 pt-3 border-t border-slate-100">
                <div class="text-center">
                    <div class="text-xs text-slate-400 uppercase tracking-wider">Rate</div>
                    <div class="font-semibold text-slate-700">{vri['speech_rate_score']:.1f}</div>
                </div>
                <div class="text-center border-l border-slate-100">
                    <div class="text-xs text-slate-400 uppercase tracking-wider">Pauses</div>
                    <div class="font-semibold text-slate-700">{vri['pause_score']:.1f}</div>
                </div>
                <div class="text-center border-l border-slate-100">
                    <div class="text-xs text-slate-400 uppercase tracking-wider">Prosody</div>
                    <div class="font-semibold text-slate-700">{vri['prosody_score']:.1f}</div>
                </div>
            </div>
        </div>
        """
    
    # Motor Agitation Index
    if 'motor_agitation' in scores:
        mai = scores['motor_agitation']
        score_html += f"""
        <div class="bg-white rounded-lg border border-slate-200 p-5 hover:shadow-md transition-shadow">
            <div class="flex items-start justify-between mb-4">
                <div class="flex items-center">
                    <div class="p-2 rounded-lg bg-orange-50 text-orange-600 mr-3">
                        <i class="fas fa-running text-lg"></i>
                    </div>
                    <h3 class="font-semibold text-slate-800">Motor Agitation</h3>
                </div>
                <div class="px-2.5 py-0.5 rounded-full text-sm font-bold bg-slate-100 text-slate-700">
                    {mai['score']:.1f}
                </div>
            </div>
            
            <p class="text-sm text-slate-600 mb-4">{mai['explanation']}</p>
            
            <div class="grid grid-cols-3 gap-2 pt-3 border-t border-slate-100">
                <div class="text-center">
                    <div class="text-xs text-slate-400 uppercase tracking-wider">Head</div>
                    <div class="font-semibold text-slate-700">{mai['head_motion_score']:.1f}</div>
                </div>
                <div class="text-center border-l border-slate-100">
                    <div class="text-xs text-slate-400 uppercase tracking-wider">Body</div>
                    <div class="font-semibold text-slate-700">{mai['body_motion_score']:.1f}</div>
                </div>
                <div class="text-center border-l border-slate-100">
                    <div class="text-xs text-slate-400 uppercase tracking-wider">Hands</div>
                    <div class="font-semibold text-slate-700">{mai['hand_motion_score']:.1f}</div>
                </div>
            </div>
        </div>
        """
    
    # Attention Stability Score
    if 'attention_stability' in scores:
        ass = scores['attention_stability']
        score_html += f"""
        <div class="bg-white rounded-lg border border-slate-200 p-5 hover:shadow-md transition-shadow">
            <div class="flex items-start justify-between mb-4">
                <div class="flex items-center">
                    <div class="p-2 rounded-lg bg-purple-50 text-purple-600 mr-3">
                        <i class="fas fa-eye text-lg"></i>
                    </div>
                    <h3 class="font-semibold text-slate-800">Attention Stability</h3>
                </div>
                <div class="px-2.5 py-0.5 rounded-full text-sm font-bold bg-slate-100 text-slate-700">
                    {ass['score']:.1f}
                </div>
            </div>
            
            <p class="text-sm text-slate-600 mb-4">{ass['explanation']}</p>
            
            <div class="grid grid-cols-3 gap-2 pt-3 border-t border-slate-100">
                <div class="text-center">
                    <div class="text-xs text-slate-400 uppercase tracking-wider">Head</div>
                    <div class="font-semibold text-slate-700">{ass['head_pose_stability']:.1f}</div>
                </div>
                <div class="text-center border-l border-slate-100">
                    <div class="text-xs text-slate-400 uppercase tracking-wider">Gaze</div>
                    <div class="font-semibold text-slate-700">{ass['gaze_stability']:.1f}</div>
                </div>
                <div class="text-center border-l border-slate-100">
                    <div class="text-xs text-slate-400 uppercase tracking-wider">Focus</div>
                    <div class="font-semibold text-slate-700">{ass['presence_score']:.1f}</div>
                </div>
            </div>
        </div>
        """
    
    # Regulation Consistency Index
    if 'regulation_consistency' in scores:
        rci = scores['regulation_consistency']
        score_html += f"""
        <div class="bg-white rounded-lg border border-slate-200 p-5 hover:shadow-md transition-shadow">
            <div class="flex items-start justify-between mb-4">
                <div class="flex items-center">
                    <div class="p-2 rounded-lg bg-teal-50 text-teal-600 mr-3">
                        <i class="fas fa-wave-square text-lg"></i>
                    </div>
                    <h3 class="font-semibold text-slate-800">Consistency</h3>
                </div>
                <div class="px-2.5 py-0.5 rounded-full text-sm font-bold bg-slate-100 text-slate-700">
                    {rci['score']:.1f}
                </div>
            </div>
            
            <p class="text-sm text-slate-600 mb-4">{rci['explanation']}</p>
            
            <div class="grid grid-cols-3 gap-2 pt-3 border-t border-slate-100">
                <div class="text-center">
                    <div class="text-xs text-slate-400 uppercase tracking-wider">Temporal</div>
                    <div class="font-semibold text-slate-700">{rci['autocorrelation']:.2f}</div>
                </div>
                <div class="text-center border-l border-slate-100">
                    <div class="text-xs text-slate-400 uppercase tracking-wider">Var</div>
                    <div class="font-semibold text-slate-700">{rci['variability']:.2f}</div>
                </div>
                <div class="text-center border-l border-slate-100">
                    <div class="text-xs text-slate-400 uppercase tracking-wider">Trend</div>
                    <div class="font-semibold text-slate-700">{rci['trend'].capitalize()}</div>
                </div>
            </div>
        </div>
        """
    
    # Facial Affect Index (if available)
    if 'facial_affect' in scores:
        fai = scores['facial_affect']
        dominant_aus_text = ', '.join([str(au) for au in fai.get('dominant_aus', [])])
        score_html += f"""
        <div class="bg-white rounded-lg border border-slate-200 p-5 hover:shadow-md transition-shadow">
            <div class="flex items-start justify-between mb-4">
                <div class="flex items-center">
                    <div class="p-2 rounded-lg bg-pink-50 text-pink-600 mr-3">
                        <i class="fas fa-smile text-lg"></i>
                    </div>
                    <h3 class="font-semibold text-slate-800">Facial Affect</h3>
                </div>
                <div class="px-2.5 py-0.5 rounded-full text-sm font-bold bg-slate-100 text-slate-700">
                    {fai['score']:.1f}
                </div>
            </div>
            
            <p class="text-sm text-slate-600 mb-4">Facial expressiveness and affect range based on Action Unit activation.</p>
            
            <div class="grid grid-cols-3 gap-2 pt-3 border-t border-slate-100">
                <div class="text-center">
                    <div class="text-xs text-slate-400 uppercase tracking-wider">Range</div>
                    <div class="font-semibold text-slate-700">{fai['affect_range']:.1f}</div>
                </div>
                <div class="text-center border-l border-slate-100">
                    <div class="text-xs text-slate-400 uppercase tracking-wider">Mobility</div>
                    <div class="font-semibold text-slate-700">{fai['facial_mobility']:.1f}</div>
                </div>
                <div class="text-center border-l border-slate-100 col-span-1">
                    <div class="text-xs text-slate-400 uppercase tracking-wider">Sym</div>
                    <div class="font-semibold text-slate-700">{fai['facial_symmetry']:.1f}</div>
                </div>
            </div>
        </div>
        """
    
    score_html += '</div></section>'
    
    return score_html


def _build_timeline_section(report: BehavioralReport, timeline_img: str) -> str:
    """Build timeline visualization section with Tailwind."""

    # If no significant fused evidence, hide the timeline graph to avoid blank plots
    if not report.fused_evidence:
        return """
        <section class="bg-slate-50 rounded-xl border border-slate-200 p-6 mt-8">
            <h2 class="text-xl font-bold text-slate-800 flex items-center mb-2">
                <i class="fas fa-chart-area text-secondary mr-2"></i>Multimodal Timeline
            </h2>
            <div class="bg-white p-6 rounded-lg border border-slate-100 text-center">
                <div class="inline-flex items-center justify-center w-12 h-12 rounded-full bg-green-100 text-green-600 mb-3">
                    <i class="fas fa-check"></i>
                </div>
                <h3 class="text-lg font-medium text-slate-800">No Dysregulation Events Detected</h3>
                <p class="text-slate-500 mt-1">Both audio and video signals remained within baseline thresholds throughout the session.</p>
            </div>
        </section>
        """
    
    timeline_html = """
    <section class="bg-white rounded-xl shadow-sm border border-slate-200 p-6 mb-8">
        <h2 class="text-xl font-bold text-slate-800 flex items-center mb-4">
            <i class="fas fa-chart-area text-secondary mr-2"></i>Multimodal Timeline
        </h2>
        <p class="text-slate-500 text-sm mb-6 max-w-3xl">
            Temporal visualization of audio features, video features, and fused dysregulation confidence.
            Color-coded markers indicate confidence levels: <span class="bg-red-500 text-white px-2 py-0.5 rounded text-xs mx-1">Strong</span>, 
            <span class="bg-orange-500 text-white px-2 py-0.5 rounded text-xs mx-1">Moderate</span>, <span class="bg-yellow-400 text-slate-800 px-2 py-0.5 rounded text-xs mx-1">Weak</span>.
        </p>
    """
    
    if timeline_img:
        timeline_html += f'<div class="rounded-lg overflow-hidden border border-slate-200 bg-slate-50 p-2"><img src="{timeline_img}" alt="Multimodal Timeline" class="w-full h-auto" /></div>'
    else:
        timeline_html += '<div class="p-6 bg-slate-50 text-slate-400 text-center italic rounded-lg">Timeline visualization not available.</div>'
    
    timeline_html += '</section>'
    
    return timeline_html


def _build_segments_section(report: BehavioralReport) -> str:
    """Build top segments section with Tailwind."""
    
    # Get top 5 segments
    top_segments = sorted(
        report.fused_evidence,
        key=lambda fe: fe.fused_confidence,
        reverse=True
    )[:5]
    
    if not top_segments:
        return "" 
    
    segments_html = """
    <section class="bg-white rounded-xl shadow-sm border border-slate-200 p-6 mt-8">
        <h2 class="text-xl font-bold text-slate-800 flex items-center mb-4">
            <i class="fas fa-list-ul text-secondary mr-2"></i>Top Dysregulation Segments
        </h2>
        <p class="text-slate-500 text-sm mb-6">
            Highest-confidence multimodal dysregulation segments with feature explanations.
        </p>
        
        <div class="overflow-x-auto">
            <table class="min-w-full divide-y divide-slate-200">
                <thead class="bg-slate-50">
                    <tr>
                        <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Rank</th>
                        <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Time Window</th>
                        <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Confidence</th>
                        <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Level</th>
                        <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Audio</th>
                        <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Video</th>
                        <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Explanation</th>
                    </tr>
                </thead>
                <tbody class="bg-white divide-y divide-slate-200">
    """
    
    for idx, segment in enumerate(top_segments, 1):
        if segment.confidence_level == 'strong':
            badge_class = "bg-red-100 text-red-800"
        elif segment.confidence_level == 'moderate':
            badge_class = "bg-orange-100 text-orange-800"
        else:
            badge_class = "bg-yellow-100 text-yellow-800"
        
        segments_html += f"""
                <tr class="hover:bg-slate-50 transition-colors">
                    <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-slate-900">{idx}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-slate-700 font-mono">{_format_time_range(segment.start_time, segment.end_time)}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm font-bold text-slate-900">{segment.fused_confidence:.2f}</td>
                    <td class="px-6 py-4 whitespace-nowrap">
                        <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full {badge_class}">
                            {segment.confidence_level.upper()}
                        </span>
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-slate-600">{segment.audio_score:.2f}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-slate-600">{segment.video_score:.2f}</td>
                    <td class="px-6 py-4 text-sm text-slate-500 max-w-xs truncate" title="{segment.explanation}">{segment.explanation}</td>
                </tr>
        """
    
    segments_html += """
            </tbody>
        </table>
        </div>
    </section>
    """
    
    return segments_html


def _build_methodology_section() -> str:
    """Build methodology explanation section with Tailwind."""
    
    return """
    <section class="bg-slate-50 rounded-xl border border-slate-200 p-6 mt-8">
        <h2 class="text-xl font-bold text-slate-800 mb-6 flex items-center">
            <i class="fas fa-cogs text-secondary mr-2"></i>Methodology
        </h2>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div class="bg-white p-4 rounded-lg shadow-sm border border-slate-100">
                <h3 class="font-bold text-slate-700 mb-2 border-b border-slate-100 pb-2">Audio Analysis</h3>
                <ul class="text-xs text-slate-600 space-y-1.5">
                    <li class="flex items-start"><i class="fas fa-check text-green-500 mt-0.5 mr-1.5 text-[10px]"></i><span><strong>VAD:</strong> Silero neural detection</span></li>
                    <li class="flex items-start"><i class="fas fa-check text-green-500 mt-0.5 mr-1.5 text-[10px]"></i><span><strong>Diarization:</strong> pyannote.audio</span></li>
                    <li class="flex items-start"><i class="fas fa-check text-green-500 mt-0.5 mr-1.5 text-[10px]"></i><span><strong>Prosody:</strong> Pitch, energy, rhythm</span></li>
                    <li class="flex items-start"><i class="fas fa-check text-green-500 mt-0.5 mr-1.5 text-[10px]"></i><span><strong>Embeddings:</strong> HuBERT models</span></li>
                </ul>
            </div>
            <div class="bg-white p-4 rounded-lg shadow-sm border border-slate-100">
                <h3 class="font-bold text-slate-700 mb-2 border-b border-slate-100 pb-2">Video Analysis</h3>
                <ul class="text-xs text-slate-600 space-y-1.5">
                    <li class="flex items-start"><i class="fas fa-check text-green-500 mt-0.5 mr-1.5 text-[10px]"></i><span><strong>Face:</strong> MediaPipe Mesh (468 pts)</span></li>
                    <li class="flex items-start"><i class="fas fa-check text-green-500 mt-0.5 mr-1.5 text-[10px]"></i><span><strong>Pose:</strong> Head orientation (PnP)</span></li>
                    <li class="flex items-start"><i class="fas fa-check text-green-500 mt-0.5 mr-1.5 text-[10px]"></i><span><strong>Body:</strong> Upper-body kinematics</span></li>
                    <li class="flex items-start"><i class="fas fa-check text-green-500 mt-0.5 mr-1.5 text-[10px]"></i><span><strong>AUs:</strong> FACS activation intensity</span></li>
                </ul>
            </div>
            <div class="bg-white p-4 rounded-lg shadow-sm border border-slate-100">
                <h3 class="font-bold text-slate-700 mb-2 border-b border-slate-100 pb-2">Multimodal Fusion</h3>
                <ul class="text-xs text-slate-600 space-y-1.5">
                    <li class="flex items-start"><i class="fas fa-check text-green-500 mt-0.5 mr-1.5 text-[10px]"></i><span><strong>Conservative:</strong> High precision</span></li>
                    <li class="flex items-start"><i class="fas fa-check text-green-500 mt-0.5 mr-1.5 text-[10px]"></i><span><strong>Strong:</strong> Audio & Video > 0.7</span></li>
                    <li class="flex items-start"><i class="fas fa-check text-green-500 mt-0.5 mr-1.5 text-[10px]"></i><span><strong>Moderate:</strong> Mixed signals</span></li>
                    <li class="flex items-start"><i class="fas fa-check text-green-500 mt-0.5 mr-1.5 text-[10px]"></i><span><strong>Explainable:</strong> Feature attribution</span></li>
                </ul>
            </div>
            <div class="bg-white p-4 rounded-lg shadow-sm border border-slate-100">
                <h3 class="font-bold text-slate-700 mb-2 border-b border-slate-100 pb-2">Scoring System</h3>
                <ul class="text-xs text-slate-600 space-y-1.5">
                    <li class="flex items-start"><i class="fas fa-check text-green-500 mt-0.5 mr-1.5 text-[10px]"></i><span><strong>Indices:</strong> 0-100 scale</span></li>
                    <li class="flex items-start"><i class="fas fa-check text-green-500 mt-0.5 mr-1.5 text-[10px]"></i><span><strong>Baseline:</strong> Normalized Z-scores</span></li>
                    <li class="flex items-start"><i class="fas fa-check text-green-500 mt-0.5 mr-1.5 text-[10px]"></i><span><strong>Context:</strong> Temporal awareness</span></li>
                    <li class="flex items-start"><i class="fas fa-check text-green-500 mt-0.5 mr-1.5 text-[10px]"></i><span><strong>Clinical:</strong> Aux. support tool</span></li>
                </ul>
            </div>
        </div>
        <div class="mt-4 p-3 bg-amber-50 rounded text-xs text-amber-800 border-l-2 border-amber-300 flex items-start">
            <i class="fas fa-exclamation-triangle mt-0.5 mr-2"></i>
            <strong>Note:</strong>&nbsp;All analysis is rule-based and explainable (no black-box classification). Results are behavioral observations, NOT medical diagnoses.
        </div>
    </section>
    """


def _build_footer(report: BehavioralReport) -> str:
    """Build report footer with Tailwind."""
    
    return f"""
    <footer class="mt-12 pt-8 border-t border-slate-200 text-center text-slate-400 text-sm pb-8">
        <p class="font-medium text-slate-500 mb-1">Generated by Behavior Scope v1.0 &bull; {report.timestamp}</p>
        <p>Open-source multimodal behavioral analysis | Non-diagnostic tool</p>
    </footer>
    """


def _get_css_styles() -> str:
    """Get customized styles for Chart.js and specific overrides."""
    return """
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f5f9; 
        }
        ::-webkit-scrollbar-thumb {
            background: #cbd5e1; 
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #94a3b8; 
        }
        
        /* Chart container */
        .chart-container {
            position: relative;
            height: 400px;
            width: 100%;
        }
        
        /* Print optimizations */
        @media print {
            body {
                background: white;
            }
            .no-print {
                display: none;
            }
            .page-break {
                page-break-before: always;
            }
        }
    """


def _embed_image(image_path: str) -> str:
    """Embed image as base64 data URI."""
    try:
        with open(image_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Determine MIME type
        suffix = Path(image_path).suffix.lower()
        mime_type = 'image/png' if suffix == '.png' else 'image/jpeg'
        
        return f"data:{mime_type};base64,{img_data}"
    except Exception as e:
        logger.error(f"Failed to embed image {image_path}: {e}")
        return ""


def _get_score_class(score: float, inverted: bool = False) -> str:
    """Get CSS class based on score value."""
    if inverted:
        # For MAI, higher is worse
        if score >= 70:
            return 'score-low'
        elif score >= 40:
            return 'score-medium'
        else:
            return 'score-high'
    else:
        # For VRI, ASS, RCI, higher is better
        if score >= 70:
            return 'score-high'
        elif score >= 40:
            return 'score-medium'
        else:
            return 'score-low'


def _get_confidence_class(level: str) -> str:
    """Get CSS class for confidence level."""
    return f"conf-{level}"


def _format_time_range(start: float, end: float) -> str:
    """Format time range as string."""
    return f"{_format_seconds(start)} - {_format_seconds(end)}"


def _format_seconds(seconds: float) -> str:
    """Format seconds as MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def _generate_key_findings(report: BehavioralReport) -> str:
    """Generate key findings summary text."""
    findings = []
    
    scores = report.scores
    
    # Vocal regulation
    if 'vocal_regulation' in scores:
        vri = scores['vocal_regulation']['score']
        if vri >= 70:
            findings.append("Speech patterns show good regulation.")
        elif vri < 40:
            findings.append("Vocal instability observed in speech patterns.")
    
    # Motor agitation
    if 'motor_agitation' in scores:
        mai = scores['motor_agitation']['score']
        if mai >= 70:
            findings.append("Elevated motor activity detected.")
        elif mai < 30:
            findings.append("Minimal motor activity observed.")
    
    # Attention
    if 'attention_stability' in scores:
        ass = scores['attention_stability']['score']
        if ass >= 70:
            findings.append("Attention patterns show good stability.")
        elif ass < 40:
            findings.append("Variable attention patterns observed.")
    
    # Consistency
    if 'regulation_consistency' in scores:
        rci = scores['regulation_consistency']
        trend = rci.get('trend', 'stable')
        if trend == 'improving':
            findings.append("Regulation patterns show improvement over session.")
        elif trend == 'worsening':
            findings.append("Regulation patterns show decline over session.")
    
    # High confidence segments
    high_conf = [fe for fe in report.fused_evidence if fe.fused_confidence >= 0.7]
    if len(high_conf) > 5:
        findings.append(f"{len(high_conf)} high-confidence dysregulation segments identified.")
    
    if not findings:
        findings.append("Analysis completed with no significant dysregulation patterns detected.")
    
    return "<br>".join(f"• {finding}" for finding in findings)


def _build_facial_action_units_section(report: BehavioralReport) -> str:
    """Build detailed Facial Action Units analysis section with Tailwind."""
    
    # Check if facial affect data is available
    if 'facial_affect' not in report.scores:
        return ""
    
    fai = report.scores['facial_affect']
    
    # Get dominant AUs with descriptions
    au_descriptions = {
        1: "Eyebrows Raised (inner) - Often seen during concentration, worry, or surprise",
        2: "Eyebrows Raised (outer) - Associated with surprise or questioning",
        4: "Eyebrows Furrowed - Indicates confusion, concern, or cognitive effort",
        5: "Eyes Widened - Surprise, fear, or heightened alertness",
        6: "Cheeks Raised - Genuine positive expressions",
        7: "Eyes Tightened - Intensity or squinting",
        9: "Nose Wrinkled - Confusion, disgust, or negative evaluation",
        10: "Upper Lip Raised - Disgust or skepticism",
        12: "Mouth Corners Pulled Up - Positive expressions, smiling",
        15: "Mouth Corners Pulled Down - Sadness, disappointment, or frowning",
        17: "Chin Raised - Pouting or defiance",
        20: "Lips Stretched Horizontally - Tension, fear, or forced expressions",
        23: "Lips Tightened - Tension or suppressed emotion",
        25: "Lips Parted - Speech preparation or mild surprise",
        26: "Jaw Dropped/Mouth Open - Surprise, shock, or speech"
    }
    
    dominant_aus = fai.get('dominant_aus', [])
    
    # Build dominant AUs list HTML
    dominant_aus_html = ""
    for au in dominant_aus:
        desc = au_descriptions.get(au, f"Action Unit {au}")
        dominant_aus_html += f'<li class="flex items-start p-2 bg-slate-50 rounded border border-slate-100"><span class="flex-shrink-0 w-16 font-bold text-primary">AU{au}:</span> <span class="text-slate-600 text-sm">{desc}</span></li>'
    
    # Interpret scores
    affect_range = fai['affect_range']
    facial_mobility = fai['facial_mobility']
    flat_affect = fai['flat_affect']
    symmetry = fai['facial_symmetry']
    
    # Range interpretation
    if affect_range >= 70:
        range_interp = "<span class='text-green-600 font-medium flex items-center'><i class='fas fa-check mr-1'></i> Wide variety</span>"
    elif affect_range >= 40:
        range_interp = "<span class='text-amber-600 font-medium flex items-center'><i class='fas fa-exclamation-circle mr-1'></i> Moderate range</span>"
    else:
        range_interp = "<span class='text-red-600 font-medium flex items-center'><i class='fas fa-times-circle mr-1'></i> Limited variety</span>"
    
    # Mobility interpretation
    if facial_mobility >= 70:
        mobility_interp = "<span class='text-green-600 font-medium flex items-center'><i class='fas fa-check mr-1'></i> High activity</span>"
    elif facial_mobility >= 40:
        mobility_interp = "<span class='text-amber-600 font-medium flex items-center'><i class='fas fa-exclamation-circle mr-1'></i> Moderate movement</span>"
    else:
        mobility_interp = "<span class='text-red-600 font-medium flex items-center'><i class='fas fa-times-circle mr-1'></i> Reduced movement</span>"
    
    # Flat affect interpretation (lower is better)
    if flat_affect <= 30:
        flat_interp = "<span class='text-green-600 font-medium flex items-center'><i class='fas fa-check mr-1'></i> Expressive</span>"
    elif flat_affect <= 60:
        flat_interp = "<span class='text-amber-600 font-medium flex items-center'><i class='fas fa-exclamation-circle mr-1'></i> Restricted</span>"
    else:
        flat_interp = "<span class='text-red-600 font-medium flex items-center'><i class='fas fa-times-circle mr-1'></i> Flat/Blunted</span>"
    
    # Symmetry interpretation
    if symmetry >= 70:
        symmetry_interp = "<span class='text-green-600 font-medium flex items-center'><i class='fas fa-check mr-1'></i> Symmetric</span>"
    elif symmetry >= 40:
        symmetry_interp = "<span class='text-amber-600 font-medium flex items-center'><i class='fas fa-exclamation-circle mr-1'></i> Moderate asymmetry</span>" 
    else:
        symmetry_interp = "<span class='text-red-600 font-medium flex items-center'><i class='fas fa-times-circle mr-1'></i> Asymmetric</span>"
    
    return f"""
    <section class="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <div class="mb-6 flex flex-col md:flex-row md:items-center md:justify-between">
            <div>
                <h2 class="text-xl font-bold text-slate-800 flex items-center mb-2">
                    <i class="fas fa-smile text-primary mr-2"></i>Facial Action Units Analysis (FACS)
                </h2>
                <p class="text-slate-500 text-sm max-w-2xl">
                    Objective measurement of facial muscle activations based on the Facial Action Coding System (FACS). 
                    This analysis describes <strong>what facial movements occurred</strong>, not emotions.
                </p>
            </div>
            <div class="mt-4 md:mt-0 hidden md:block">
                <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-blue-50 text-blue-700 border border-blue-100">
                    <i class="fas fa-video mr-1.5"></i> Computer Vision Analysis
                </span>
            </div>
        </div>
        
        <div class="bg-gradient-to-r from-blue-500 to-indigo-600 rounded-lg p-6 text-white text-center mb-6 shadow-md relative overflow-hidden">
            <div class="absolute top-0 right-0 opacity-10 transform translate-x-4 -translate-y-4">
                <i class="fas fa-smile text-9xl"></i>
            </div>
            <div class="relative z-10">
                <h3 class="text-3xl font-bold mb-1">{fai['score']:.1f}<span class="text-lg opacity-75 font-normal">/100</span></h3>
                <p class="font-medium opacity-90 text-sm uppercase tracking-wide">Facial Affect Index</p>
                <p class="text-xs opacity-75 mt-2 max-w-lg mx-auto">Composite score reflecting facial expressiveness, mobility, and affect range.</p>
            </div>
        </div>
        
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <!-- Affect Range -->
            <div class="bg-slate-50 rounded-lg p-4 border border-slate-200 hover:border-blue-200 transition-colors">
                <div class="flex justify-between items-start mb-2">
                    <h3 class="font-semibold text-slate-700 text-xs uppercase tracking-wider">Affect Range</h3>
                    <i class="fas fa-theater-masks text-slate-400"></i>
                </div>
                <div class="text-2xl font-bold text-slate-800 mb-1">{affect_range:.1f}</div>
                <div class="text-xs mb-2">{range_interp}</div>
            </div>
            
            <!-- Facial Mobility -->
            <div class="bg-slate-50 rounded-lg p-4 border border-slate-200 hover:border-blue-200 transition-colors">
                <div class="flex justify-between items-start mb-2">
                    <h3 class="font-semibold text-slate-700 text-xs uppercase tracking-wider">Mobility</h3>
                    <i class="fas fa-arrows-alt text-slate-400"></i>
                </div>
                <div class="text-2xl font-bold text-slate-800 mb-1">{facial_mobility:.1f}</div>
                <div class="text-xs mb-2">{mobility_interp}</div>
            </div>
            
            <!-- Flat Affect Indicator -->
            <div class="bg-slate-50 rounded-lg p-4 border border-slate-200 hover:border-blue-200 transition-colors">
                <div class="flex justify-between items-start mb-2">
                    <h3 class="font-semibold text-slate-700 text-xs uppercase tracking-wider">Flat Affect</h3>
                    <i class="fas fa-meh-blank text-slate-400"></i>
                </div>
                <div class="text-2xl font-bold text-slate-800 mb-1">{flat_affect:.1f}</div>
                <div class="text-xs mb-2">{flat_interp}</div>
            </div>
            
            <!-- Facial Symmetry -->
            <div class="bg-slate-50 rounded-lg p-4 border border-slate-200 hover:border-blue-200 transition-colors">
                <div class="flex justify-between items-start mb-2">
                    <h3 class="font-semibold text-slate-700 text-xs uppercase tracking-wider">Symmetry</h3>
                    <i class="fas fa-balance-scale text-slate-400"></i>
                </div>
                <div class="text-2xl font-bold text-slate-800 mb-1">{symmetry:.1f}</div>
                <div class="text-xs mb-2">{symmetry_interp}</div>
            </div>
        </div>
        
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div class="lg:col-span-2 bg-white border rounded-lg overflow-hidden border-slate-200">
                <div class="bg-slate-50 px-4 py-3 border-b border-slate-200">
                    <h3 class="font-bold text-slate-700 flex items-center text-sm">
                        <i class="fas fa-list-ol mr-2 text-primary"></i>Most Frequently Activated Action Units
                    </h3>
                </div>
                <div class="p-4">
                    <ul class="space-y-2">
                        {dominant_aus_html if dominant_aus_html else '<li class="text-slate-500 italic text-sm">No dominant Action Units detected</li>'}
                    </ul>
                </div>
            </div>
            
            <div class="bg-blue-50 border border-blue-100 p-5 rounded-lg h-full">
                <div class="flex items-start">
                    <div class="flex-shrink-0 mt-0.5">
                        <i class="fas fa-info-circle text-blue-500"></i>
                    </div>
                    <div class="ml-3">
                        <h4 class="text-sm font-bold text-blue-800">Clinical Note</h4>
                        <p class="text-xs text-blue-700 mt-2 leading-relaxed">
                            Action Units represent <strong>objective muscle movements</strong>, not emotions. 
                            For example, AU12 (Mouth Corners Pulled Up) happens in both genuine smiling and social masking.
                        </p>
                        <p class="text-xs text-blue-700 mt-2 leading-relaxed">
                            Interpret these patterns within the full clinical context.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </section>
    """


def _build_autism_section(report: BehavioralReport) -> str:
    """Build autism-specific analysis section with Tailwind."""
    
    if not report.autism_results:
        return ""
    
    autism = report.autism_results
    
    # Social Engagement Index card
    sei = autism.get('social_engagement')
    sei_score = sei.social_engagement_index if sei else 0.0
    sei_interp = sei.interpretation if sei else ""
    
    sei_color_class = _get_score_color_class(sei_score, inverted=False)
    
    # Turn-taking card
    turn = autism.get('turn_taking')
    if turn:
        turn_balance = f"{turn.child_percentage:.1f}% child / {100-turn.child_percentage:.1f}% therapist"
        turn_latency = f"{turn.mean_response_latency:.2f}s"
        turn_reciprocity = turn.reciprocity_score
        turn_interp = turn.explanation
    else:
        turn_balance = "N/A"
        turn_latency = "N/A"
        turn_reciprocity = 0.0
        turn_interp = ""
    
    reciprocity_color_class = _get_score_color_class(turn_reciprocity, inverted=False)
    
    # Eye contact card
    eye = autism.get('eye_contact')
    if eye:
        eye_pct = f"{eye.percentage_of_session:.1f}%"
        eye_episodes = eye.episode_count
        eye_score = eye.eye_contact_score
        eye_interp = eye.explanation
    else:
        eye_pct = "N/A"
        eye_episodes = 0
        eye_score = 0.0
        eye_interp = ""
    
    eye_color_class = _get_score_color_class(eye_score, inverted=False)
    
    # Stereotypy card
    stereo = autism.get('stereotypy')
    if stereo:
        stereo_count = stereo.episode_count
        stereo_pct = f"{stereo.percentage_of_session:.1f}%"
        stereo_types = ", ".join([f"{k}: {v}" for k, v in stereo.stereotypy_types.items()]) if stereo.stereotypy_types else "None detected"
        stereo_intensity = stereo.intensity_score
        stereo_interp = stereo.explanation
    else:
        stereo_count = 0
        stereo_pct = "0.0%"
        stereo_types = "N/A"
        stereo_intensity = 0.0
        stereo_interp = ""
    
    stereo_color_class = _get_score_color_class(stereo_intensity, inverted=True)
    
    return f"""
    <section class="bg-white rounded-xl shadow-sm border border-slate-200 p-6 mt-8">
        <h2 class="text-xl font-bold text-slate-800 flex items-center mb-4">
            <i class="fas fa-puzzle-piece text-secondary mr-2"></i>Autism-Specific Analysis
        </h2>
        <p class="text-slate-500 text-sm mb-6">
            Social communication and behavioral markers relevant to autism spectrum observation.
            These metrics provide supplementary information for clinical assessment.
        </p>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <!-- Social Engagement Index -->
            <div class="bg-slate-50 rounded-lg p-5 border border-slate-100">
                <h3 class="font-semibold text-slate-700 mb-2 flex items-center">
                    <i class="fas fa-users text-slate-400 mr-2"></i>Social Engagement Index
                </h3>
                <div class="text-3xl font-bold {sei_color_class} mb-4">{sei_score:.1f}/100</div>
                <div class="space-y-2 mb-4">
                    <p class="text-xs font-semibold text-slate-500 uppercase tracking-wide">Components</p>
                    <ul class="text-sm text-slate-600 space-y-1">
                        <li class="flex justify-between"><span>Eye Contact:</span> <span class="font-mono">{(sei.eye_contact_component if sei else 0.0):.1f}</span></li>
                        <li class="flex justify-between"><span>Turn-Taking:</span> <span class="font-mono">{(sei.turn_taking_component if sei else 0.0):.1f}</span></li>
                        <li class="flex justify-between"><span>Responsiveness:</span> <span class="font-mono">{(sei.responsiveness_component if sei else 0.0):.1f}</span></li>
                        <li class="flex justify-between"><span>Attention:</span> <span class="font-mono">{(sei.attention_component if sei else 0.0):.1f}</span></li>
                    </ul>
                </div>
                <p class="text-sm text-slate-500 italic border-l-2 border-slate-200 pl-3">{sei_interp}</p>
            </div>
            
            <!-- Turn-Taking -->
            <div class="bg-slate-50 rounded-lg p-5 border border-slate-100">
                <h3 class="font-semibold text-slate-700 mb-2 flex items-center">
                    <i class="fas fa-comments text-slate-400 mr-2"></i>Turn-Taking Dynamics
                </h3>
                <div class="text-3xl font-bold {reciprocity_color_class} mb-4">Reciprocity: {turn_reciprocity:.1f}/100</div>
                <div class="space-y-2 mb-4 text-sm text-slate-600">
                    <div class="flex justify-between border-b border-slate-100 pb-1">
                        <span class="text-slate-500">Balance:</span> 
                        <span class="font-medium text-right">{turn_balance}</span>
                    </div>
                    <div class="flex justify-between border-b border-slate-100 pb-1">
                        <span class="text-slate-500">Response Latency:</span> 
                        <span class="font-medium text-right">{turn_latency}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-slate-500">Total Turns:</span> 
                        <span class="font-medium text-right">{turn.total_turns if turn else 0}</span>
                    </div>
                </div>
                <p class="text-sm text-slate-500 italic border-l-2 border-slate-200 pl-3">{turn_interp}</p>
            </div>
            
            <!-- Eye Contact -->
            <div class="bg-slate-50 rounded-lg p-5 border border-slate-100">
                <h3 class="font-semibold text-slate-700 mb-2 flex items-center">
                    <i class="fas fa-eye text-slate-400 mr-2"></i>Eye Contact Patterns
                </h3>
                <div class="text-3xl font-bold {eye_color_class} mb-4">{eye_score:.1f}/100</div>
                <div class="space-y-2 mb-4 text-sm text-slate-600">
                    <div class="flex justify-between border-b border-slate-100 pb-1">
                        <span class="text-slate-500">Session Coverage:</span> 
                        <span class="font-medium text-right">{eye_pct}</span>
                    </div>
                    <div class="flex justify-between border-b border-slate-100 pb-1">
                        <span class="text-slate-500">Episodes:</span> 
                        <span class="font-medium text-right">{eye_episodes}</span>
                    </div>
                    <div class="flex justify-between border-b border-slate-100 pb-1">
                        <span class="text-slate-500">During Speaking:</span> 
                        <span class="font-medium text-right">{(eye.during_speaking_percentage if eye else 0.0):.1f}%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-slate-500">During Listening:</span> 
                        <span class="font-medium text-right">{(eye.during_listening_percentage if eye else 0.0):.1f}%</span>
                    </div>
                </div>
                <p class="text-sm text-slate-500 italic border-l-2 border-slate-200 pl-3">{eye_interp}</p>
            </div>
            
            <!-- Stereotypy -->
            <div class="bg-slate-50 rounded-lg p-5 border border-slate-100">
                <h3 class="font-semibold text-slate-700 mb-2 flex items-center">
                    <i class="fas fa-sync-alt text-slate-400 mr-2"></i>Stereotyped Movements
                </h3>
                <div class="text-3xl font-bold {stereo_color_class} mb-4">Intensity: {stereo_intensity:.1f}/100</div>
                <div class="space-y-2 mb-4 text-sm text-slate-600">
                    <div class="flex justify-between border-b border-slate-100 pb-1">
                        <span class="text-slate-500">Episodes:</span> 
                        <span class="font-medium text-right">{stereo_count}</span>
                    </div>
                    <div class="flex justify-between border-b border-slate-100 pb-1">
                        <span class="text-slate-500">Session Coverage:</span> 
                        <span class="font-medium text-right">{stereo_pct}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-slate-500">Types:</span> 
                        <span class="font-medium text-right truncate max-w-[150px]" title="{stereo_types}">{stereo_types}</span>
                    </div>
                </div>
                <p class="text-sm text-slate-500 italic border-l-2 border-slate-200 pl-3">{stereo_interp}</p>
            </div>
        </div>
        
        <div class="bg-blue-50 border-l-4 border-blue-500 p-4 mt-6 rounded-r-lg">
            <h4 class="text-blue-800 font-bold mb-1 flex items-center">
                <i class="fas fa-info-circle mr-2"></i>Clinical Note:
            </h4>
            <p class="text-blue-700 text-sm">
                These metrics provide observational data on social communication and behavioral patterns.
                They are NOT diagnostic criteria for autism spectrum disorder. Clinical interpretation must
                consider developmental history, context, and multiple assessment tools (e.g., ADOS-2, ADI-R).
                Use these findings as supplementary information within comprehensive clinical evaluation.
            </p>
        </div>
    </section>
    """


def _get_score_color_class(score: float, inverted: bool = False) -> str:
    """
    Get color class for score display.
    
    Args:
        score: Score value (0-100)
        inverted: If True, lower scores are better (e.g., stereotypy intensity)
    """
    if inverted:
        score = 100 - score
    
    if score >= 70:
        return "text-emerald-500 bg-emerald-50 px-2 rounded"  # Green (good)
    elif score >= 50:
        return "text-amber-500 bg-amber-50 px-2 rounded"  # Orange (moderate)
    else:
        return "text-red-500 bg-red-50 px-2 rounded"  # Red (concern)


def _build_clinical_section(report: BehavioralReport) -> str:
    """Build clinical analysis section (stuttering, question-response) with Tailwind."""
    
    if not report.clinical_results:
        return ""
    
    clinical = report.clinical_results
    
    # Stuttering card
    stutter = clinical.get('stuttering')
    if stutter:
        stutter_count = stutter.total_disfluencies
        stutter_rate = stutter.disfluency_rate
        stutter_ssi = stutter.stuttering_severity_index
        stutter_types = ", ".join([f"{k}: {v}" for k, v in stutter.disfluency_types.items()]) if stutter.disfluency_types else "None detected"
        long_block = stutter.longest_block
        stutter_interp = stutter.interpretation
    else:
        stutter_count = 0
        stutter_rate = 0.0
        stutter_ssi = 0.0
        stutter_types = "N/A"
        long_block = 0.0
        stutter_interp = ""
    
    stutter_color_class = _get_score_color_class(stutter_ssi, inverted=True)
    
    stutter_severity = 'Mild' if stutter_ssi < 20 else 'Moderate' if stutter_ssi < 40 else 'Moderately Severe' if stutter_ssi < 60 else 'Severe'
    
    # Question-response card
    qr = clinical.get('question_response')
    if qr:
        qr_total = qr.total_questions
        qr_response_rate = qr.response_rate
        qr_latency = qr.mean_response_latency
        qr_index = qr.responsiveness_index
        qr_answered = qr.answered_questions
        qr_approp_count = qr.appropriate_responses
        qr_approp_rate = qr.appropriateness_rate
        qr_interp = qr.interpretation
    else:
        qr_total = 0
        qr_response_rate = 0.0
        qr_latency = 0.0
        qr_index = 0.0
        qr_answered = 0
        qr_approp_count = 0
        qr_approp_rate = 0.0
        qr_interp = ""
    
    qr_color_class = _get_score_color_class(qr_index, inverted=False)
    
    qr_level = 'Below Expected' if qr_index < 40 else 'Below Typical' if qr_index < 60 else 'Functional'
    
    return f"""
    <section class="bg-white rounded-xl shadow-sm border border-slate-200 p-6 mt-8">
        <h2 class="text-xl font-bold text-slate-800 flex items-center mb-4">
            <i class="fas fa-stethoscope text-secondary mr-2"></i>Clinical Analysis
        </h2>
        <p class="text-slate-500 text-sm mb-6">
            Specialized clinical metrics for speech fluency and communication ability assessment.
        </p>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <!-- Stuttering/Disfluency -->
            <div class="bg-slate-50 rounded-lg p-5 border border-slate-100">
                <h3 class="font-semibold text-slate-700 mb-2 flex items-center">
                    <i class="fas fa-wave-square text-slate-400 mr-2"></i>Stuttering/Disfluency
                </h3>
                <div class="text-3xl font-bold {stutter_color_class} inline-block mb-4">SSI: {stutter_ssi:.1f}/100</div>
                <div class="space-y-2 mb-4 text-sm text-slate-600">
                    <div class="flex justify-between border-b border-slate-100 pb-1">
                        <span class="text-slate-500">Total Disfluencies:</span> 
                        <span class="font-medium text-right">{stutter_count}</span>
                    </div>
                    <div class="flex justify-between border-b border-slate-100 pb-1">
                        <span class="text-slate-500">Disfluency Rate:</span> 
                        <span class="font-medium text-right">{stutter_rate:.1f}% (per 100 syl)</span>
                    </div>
                    <div class="flex justify-between border-b border-slate-100 pb-1">
                         <span class="text-slate-500">Types:</span> 
                         <span class="font-medium text-right truncate max-w-[150px]" title="{stutter_types}">{stutter_types}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-slate-500">Longest Block:</span> 
                        <span class="font-medium text-right">{long_block:.2f}s</span>
                    </div>
                </div>
                <p class="text-sm text-slate-500 italic border-l-2 border-slate-200 pl-3 mb-3">{stutter_interp}</p>
                 <div class="text-xs bg-slate-100 p-2 rounded text-slate-700">
                    <strong>SSI Severity:</strong> {stutter_severity}
                </div>
            </div>
            
            <!-- Question-Response Ability -->
            <div class="bg-slate-50 rounded-lg p-5 border border-slate-100">
                <h3 class="font-semibold text-slate-700 mb-2 flex items-center">
                    <i class="fas fa-question text-slate-400 mr-2"></i>Question-Response Ability
                </h3>
                <div class="text-3xl font-bold {qr_color_class} inline-block mb-4">Responsiveness: {qr_index:.1f}/100</div>
                <div class="space-y-2 mb-4 text-sm text-slate-600">
                    <div class="flex justify-between border-b border-slate-100 pb-1">
                        <span class="text-slate-500">Questions Detected:</span> 
                        <span class="font-medium text-right">{qr_total}</span>
                    </div>
                    <div class="flex justify-between border-b border-slate-100 pb-1">
                        <span class="text-slate-500">Questions Answered:</span> 
                        <span class="font-medium text-right">{qr_answered} ({qr_response_rate:.1f}%)</span>
                    </div>
                    <div class="flex justify-between border-b border-slate-100 pb-1">
                        <span class="text-slate-500">Mean Latency:</span> 
                        <span class="font-medium text-right">{qr_latency:.2f}s</span>
                    </div>
                    <div class="flex justify-between">
                         <span class="text-slate-500">Appropriate:</span> 
                         <span class="font-medium text-right">{qr_approp_count} ({qr_approp_rate:.1f}%)</span>
                    </div>
                </div>
                <p class="text-sm text-slate-500 italic border-l-2 border-slate-200 pl-3 mb-3">{qr_interp}</p>
                <div class="text-xs bg-slate-100 p-2 rounded text-slate-700">
                    <strong>Responsiveness Level:</strong> {qr_level}
                </div>
            </div>
        </div>
        
        <div class="bg-amber-50 border-l-4 border-amber-500 p-4 mt-6 rounded-r-lg">
            <h4 class="text-amber-800 font-bold mb-1 flex items-center">
                <i class="fas fa-notes-medical mr-2"></i>Clinical Note:
            </h4>
            <div class="text-amber-700 text-sm space-y-2">
                <p><strong>Stuttering:</strong> Severity estimates based on SSI-4 framework. Formal stuttering diagnosis requires comprehensive evaluation.</p>
                <p><strong>Question-Response:</strong> Automated detection provides estimates. Response appropriateness based on temporal heuristics.</p>
            </div>
        </div>
    </section>
    """

