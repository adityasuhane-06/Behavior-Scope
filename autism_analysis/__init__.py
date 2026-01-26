"""
Autism-specific analysis modules.

This package provides specialized analysis for autism spectrum disorder (ASD)
assessment in speech therapy contexts:
- Turn-taking dynamics (child-therapist balance)
- Eye contact patterns (social communication marker)
- Stereotyped movements (repetitive behaviors)
- Social engagement indices

Clinical rationale:
- Core ASD features: social communication, repetitive behaviors
- Speech therapy targets: turn-taking, joint attention, reciprocity
- Evidence-based markers from ADOS-2, ADI-R protocols
- Non-diagnostic (observation only, requires clinical interpretation)

Engineering approach:
- Builds on Phase 1 foundation (audio, video, fusion)
- Autism-specific feature extraction
- Clinical threshold-based rules
- Longitudinal tracking support
"""

from .turn_taking import (
    TurnTakingEvent,
    TurnTakingAnalysis,
    analyze_turn_taking,
    compute_response_latency_child,
    get_response_latency_distribution
)

# Eye contact analysis is now handled by Enhanced Attention Tracking System
# from .eye_contact import (...)  # Replaced with enhanced system

from .stereotypy import (
    StereotypyEvent,
    StereotypyAnalysis,
    detect_stereotyped_movements,
    classify_stereotypy_type
)

from .social_engagement import (
    compute_social_engagement_index,
    SocialEngagementMetrics
)

__all__ = [
    # Turn-taking
    'TurnTakingEvent',
    'TurnTakingAnalysis',
    'analyze_turn_taking',
    'compute_response_latency_child',
    'get_response_latency_distribution',
    
    # Eye contact - now handled by Enhanced Attention Tracking System
    # 'EyeContactEvent',
    # 'EyeContactAnalysis', 
    # 'analyze_eye_contact',
    # 'compute_eye_contact_during_speaking',
    
    # Stereotypy
    'StereotypyEvent',
    'StereotypyAnalysis',
    'detect_stereotyped_movements',
    'classify_stereotypy_type',
    
    # Social engagement
    'compute_social_engagement_index',
    'SocialEngagementMetrics',
]
