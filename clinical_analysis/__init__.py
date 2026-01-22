"""
Clinical analysis modules for specific assessment needs.

This package provides targeted clinical analysis beyond general behavioral regulation:
- Stuttering/disfluency analysis (speech fluency assessment)
- Question-response ability (comprehension and expressive evaluation)
- Facial action units (FACS-based facial pattern analysis)

These modules complement the core pipeline and autism-specific analysis.
"""

from .stuttering import (
    DisfluencyEvent,
    StutteringAnalysis,
    analyze_stuttering,
    get_disfluency_timeline
)

from .question_response import (
    QuestionEvent,
    ResponseEvent,
    QuestionResponseAnalysis,
    analyze_question_response_ability
)

from .facial_action_units import (
    ActionUnit,
    FacialActionUnits,
    FacialActionUnitAnalyzer,
    analyze_facial_action_units
)

__all__ = [
    # Stuttering
    'DisfluencyEvent',
    'StutteringAnalysis',
    'analyze_stuttering',
    'get_disfluency_timeline',
    
    # Question-response
    'QuestionEvent',
    'ResponseEvent',
    'QuestionResponseAnalysis',
    'analyze_question_response_ability',
    
    # Facial Action Units
    'ActionUnit',
    'FacialActionUnits',
    'FacialActionUnitAnalyzer',
    'analyze_facial_action_units',
]
