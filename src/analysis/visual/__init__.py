"""
Visual Analysis Module
====================

Visual analysis components including:
- Video processing and frame extraction
- Visual content analysis
- Pose detection and behavioral analysis
"""

from .video_processor import extract_frames
from .opencv_analyzer import OpenCVAnalyzer
from .movenet_analyzer import MoveNetAnalyzer
from .environment_analyzer import EnvironmentAnalyzer
from .emotion_analyzer import analyze_emotions_in_frames, print_emotion_summary, save_emotion_analysis, get_emotion_statistics, detect_emotion_patterns, analyze_emotional_engagement

__all__ = [
    'extract_frames',
    'OpenCVAnalyzer', 
    'MoveNetAnalyzer',
    'EnvironmentAnalyzer',
    'analyze_emotions_in_frames',
    'print_emotion_summary',
    'save_emotion_analysis',
    'get_emotion_statistics',
    'detect_emotion_patterns',
    'analyze_emotional_engagement'
]
