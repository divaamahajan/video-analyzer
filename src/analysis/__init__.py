"""
Analysis Module
==============

Comprehensive video analysis system with modular components:

- Visual Analysis: Video processing, visual content analysis, pose detection
- Audio Analysis: Audio processing, transcription, audio-based analysis  
- Text Analysis: Speech transcription, pace analysis, sentiment analysis
"""

# Import main analyzers for easy access
from .visual import extract_frames, OpenCVAnalyzer, MoveNetAnalyzer
from .audio import AudioProcessor, AudioAnalyzer
from .text import TranscriptionService, TranscriptionAnalyzer, PaceAnalyzer, PronunciationAnalyzer, SentimentAnalyzer, ContentAnalyzer

__all__ = [
    # Visual analysis
    'extract_frames',
    'OpenCVAnalyzer',
    'MoveNetAnalyzer',
    
    # Audio analysis
    'AudioProcessor',
    'AudioAnalyzer',
    
    # Text analysis
    'TranscriptionService',
    'TranscriptionAnalyzer',
    'PaceAnalyzer',
    'PronunciationAnalyzer',
    'SentimentAnalyzer',
    'ContentAnalyzer'
]
