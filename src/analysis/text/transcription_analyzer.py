#!/usr/bin/env python3
"""
Transcription Analyzer Module
============================

Main coordinator for transcription analysis that combines:
- TranscriptionService: Basic transcription functionality
- PaceAnalyzer: Pace and rhythm analysis
- PronunciationAnalyzer: Text-based pronunciation analysis
- SentimentAnalyzer: Text-based emotion and sentiment analysis
"""

import os
from typing import Dict, Any
from .transcription_service import TranscriptionService
from .pace_analyzer import PaceAnalyzer
from .pronunciation_analyzer import PronunciationAnalyzer
from .sentiment_analyzer import SentimentAnalyzer


class TranscriptionAnalyzer:
    """Main coordinator for transcription analysis"""
    
    def __init__(self, api_key: str):
        """Initialize with OpenAI API key"""
        self.transcription_service = TranscriptionService(api_key)
        self.pace_analyzer = PaceAnalyzer()
        self.pronunciation_analyzer = PronunciationAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def transcribe_with_timestamps(self, audio_path: str, prompt: str = None) -> Dict[str, Any]:
        """Transcribe audio with word and segment timestamps"""
        return self.transcription_service.transcribe_with_timestamps(audio_path, prompt)
    
    def analyze_pace_and_rhythm(self, transcription_result: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive pace and rhythm analysis from transcription"""
        return self.pace_analyzer.analyze_pace_and_rhythm(transcription_result)
    
    def analyze_pronunciation_and_articulation_text(self, transcription_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pronunciation patterns from text (repeated words, etc.)"""
        return self.pronunciation_analyzer.analyze_pronunciation_and_articulation_text(transcription_result)
    
    def analyze_emotion_and_sentiment_text(self, transcription_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze emotional intensity and sentiment from text"""
        return self.sentiment_analyzer.analyze_emotion_and_sentiment_text(transcription_result)
    
    def analyze_transcription(self, audio_path: str, prompt: str = None) -> Dict[str, Any]:
        """Complete transcription analysis pipeline"""
        print("\n📝 Starting Transcription Analysis")
        print("="*50)
        
        # Transcribe
        transcription = self.transcribe_with_timestamps(audio_path, prompt)
        
        # Text-based analysis
        pace_analysis = self.analyze_pace_and_rhythm(transcription)
        pronunciation_analysis = self.analyze_pronunciation_and_articulation_text(transcription)
        emotion_analysis = self.analyze_emotion_and_sentiment_text(transcription)
        
        return {
            "transcription": transcription,
            "pace_and_rhythm": pace_analysis,
            "pronunciation_patterns": pronunciation_analysis,
            "emotion_and_sentiment": emotion_analysis
        }
    
    def print_transcription_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of transcription analysis results"""
        print("\n" + "="*60)
        print("📝 TRANSCRIPTION ANALYSIS REPORT")
        print("="*60)
        
        # Transcription Overview
        transcription = results['transcription']
        self.transcription_service.print_transcription_info(transcription)
        
        # Pace & Rhythm Analysis
        pace = results['pace_and_rhythm']
        self.pace_analyzer.print_pace_summary(pace)
        
        # Pronunciation Patterns
        pronunciation = results['pronunciation_patterns']
        self.pronunciation_analyzer.print_pronunciation_summary(pronunciation)
        
        # Emotion & Sentiment Analysis
        emotion = results['emotion_and_sentiment']
        self.sentiment_analyzer.print_sentiment_summary(emotion)
        
        print("\n" + "="*60)
        print("✅ Transcription Analysis Complete!")
        print("="*60)
