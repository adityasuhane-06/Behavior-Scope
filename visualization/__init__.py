"""
Visualization and reporting module.

This package generates interpretable clinical reports:
- Timeline plots: Audio+video feature overlay with dysregulation markers
- Segment exports: Top-N video clips for review
- HTML reports: Comprehensive session summary with scores and explanations

Clinical rationale:
- Visual timelines enable pattern recognition
- Video segments provide concrete examples
- Interpretable reports support clinical decision-making
- Non-diagnostic language throughout
"""

from .timeline_plots import (
    plot_multimodal_timeline,
    plot_behavioral_scores,
    save_timeline_plot
)
from .segment_marker import (
    export_dysregulation_segments,
    create_segment_annotations
)
from .report_generator import (
    generate_html_report,
    BehavioralReport
)

__all__ = [
    'plot_multimodal_timeline',
    'plot_behavioral_scores',
    'save_timeline_plot',
    'export_dysregulation_segments',
    'create_segment_annotations',
    'generate_html_report',
    'BehavioralReport',
]
