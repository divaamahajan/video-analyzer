"""
Audio Analysis Module
===================

Audio analysis components including:
- Direct audio processing and analysis
- Audio feature extraction
- Audio-based emotion detection
"""

from .audio_processor import AudioProcessor
from .audio_analyzer import AudioAnalyzer

__all__ = [
    'AudioProcessor',
    'AudioAnalyzer'
]
