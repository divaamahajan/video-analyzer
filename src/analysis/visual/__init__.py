"""
Visual Analysis Module
====================

Visual analysis components including:
- Video processing and frame extraction
- Visual content analysis
- Pose detection and behavioral analysis
"""

from .video_processor import extract_frames
from .simple_visual_analyzer import SimpleVisualAnalyzer
from .pose_analyzer import PoseAnalyzer

__all__ = [
    'extract_frames',
    'SimpleVisualAnalyzer', 
    'PoseAnalyzer'
]
