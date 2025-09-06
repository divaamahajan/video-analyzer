#!/usr/bin/env python3
"""
Sentiment Analyzer Module
========================

Handles text-based emotion and sentiment analysis including:
- Text sentiment analysis
- Confidence indicators from speech patterns
- Emotional intensity analysis from text
- Word-based emotion detection
"""

import numpy as np
import re
from typing import Dict, List, Any


class SentimentAnalyzer:
    """Analyzes sentiment and emotion from transcribed text"""
    
    def __init__(self):
        """Initialize sentiment analyzer"""
        pass
    
    def analyze_emotion_and_sentiment_text(self, transcription_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze emotional intensity and sentiment from text"""
        print("😊 Analyzing Text Sentiment and Emotion")
        
        text = transcription_result["text"]
        
        # Sentiment analysis from text
        text_sentiment = self._analyze_text_sentiment(text)
        
        # Confidence indicators from text patterns
        confidence_indicators = self._analyze_confidence_indicators_text(transcription_result)
        
        # Emotional intensity from text patterns
        emotional_intensity_text = self._analyze_emotional_intensity_text(text)
        
        return {
            "text_sentiment": text_sentiment,
            "confidence_indicators": confidence_indicators,
            "emotional_intensity_text": emotional_intensity_text
        }
    
    def _analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """Simple sentiment analysis from text"""
        positive_words = ["happy", "good", "great", "excellent", "wonderful", "amazing", "confident", 
                         "excited", "fantastic", "brilliant", "perfect", "love", "enjoy", "awesome",
                         "outstanding", "incredible", "fabulous", "marvelous", "superb", "terrific"]
        negative_words = ["sad", "bad", "terrible", "awful", "worried", "nervous", "hesitant", 
                         "disappointed", "frustrated", "angry", "upset", "hate", "dislike", "horrible",
                         "dreadful", "awful", "terrible", "disgusting", "annoying", "frustrating"]
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_words = len(words)
        sentiment_score = (positive_count - negative_count) / max(total_words, 1)
        
        if sentiment_score > 0.1:
            sentiment = "positive"
        elif sentiment_score < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "positive_words": positive_count,
            "negative_words": negative_count
        }
    
    def _analyze_confidence_indicators_text(self, transcription_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze confidence indicators from text patterns"""
        segments = transcription_result["segments"]
        text = transcription_result["text"].lower()
        
        # Speaking rate consistency
        segment_durations = [seg.end - seg.start for seg in segments]
        duration_variance = np.var(segment_durations) if len(segment_durations) > 1 else 0
        
        # Pause analysis (fewer pauses = more confident)
        pause_count = len(segments) - 1
        total_time = segments[-1].end - segments[0].start if segments else 0
        pause_frequency = pause_count / (total_time / 60) if total_time > 0 else 0
        
        # Confidence words and phrases
        confidence_words = ["definitely", "certainly", "absolutely", "clearly", "obviously", 
                           "sure", "confident", "positive", "know", "understand", "believe",
                           "convinced", "guaranteed", "assured", "determined"]
        uncertainty_words = ["maybe", "perhaps", "might", "could", "possibly", "think", 
                            "believe", "guess", "suppose", "unsure", "uncertain", "doubt",
                            "wonder", "consider", "imagine", "assume"]
        
        confidence_count = sum(1 for word in confidence_words if word in text)
        uncertainty_count = sum(1 for word in uncertainty_words if word in text)
        
        # Overall confidence score from text
        confidence_score = (
            (1 - min(duration_variance, 1)) * 0.3 +
            (1 - min(pause_frequency / 10, 1)) * 0.3 +
            (confidence_count / max(confidence_count + uncertainty_count, 1)) * 0.4
        )
        
        return {
            "confidence_score": confidence_score,
            "speaking_rate_consistency": 1 - min(duration_variance, 1),
            "pause_frequency": pause_frequency,
            "confidence_words": confidence_count,
            "uncertainty_words": uncertainty_count,
            "confidence_level": "high" if confidence_score > 0.7 else 
                              "medium" if confidence_score > 0.4 else "low"
        }
    
    def _analyze_emotional_intensity_text(self, text: str) -> Dict[str, Any]:
        """Analyze emotional intensity from text patterns"""
        # Intensifiers
        intensifiers = ["very", "extremely", "incredibly", "absolutely", "totally", "completely", 
                       "really", "so", "such", "amazing", "fantastic", "incredible", "unbelievable",
                       "outstanding", "remarkable", "extraordinary", "phenomenal"]
        
        # Exclamation marks
        exclamation_count = text.count('!')
        
        # Caps (indicating excitement)
        caps_count = len(re.findall(r'[A-Z]{2,}', text))
        
        # Repetition (indicating emphasis)
        words = text.split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        repetition_count = sum(1 for count in word_counts.values() if count > 2)
        
        # Calculate intensity score
        intensity_score = (
            min(sum(1 for word in intensifiers if word in text.lower()) / 10, 1) * 0.3 +
            min(exclamation_count / 5, 1) * 0.2 +
            min(caps_count / 3, 1) * 0.2 +
            min(repetition_count / 5, 1) * 0.3
        )
        
        # Determine emotion type
        if intensity_score > 0.7:
            emotion = "excited"
        elif intensity_score > 0.5:
            emotion = "engaged"
        elif intensity_score > 0.3:
            emotion = "moderate"
        else:
            emotion = "neutral"
        
        return {
            "intensity": intensity_score,
            "emotion": emotion,
            "intensifiers": sum(1 for word in intensifiers if word in text.lower()),
            "exclamations": exclamation_count,
            "caps": caps_count,
            "repetitions": repetition_count
        }
    
    def print_sentiment_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of sentiment analysis results"""
        print("\n" + "="*50)
        print("😊 SENTIMENT & EMOTION ANALYSIS")
        print("="*50)
        
        # Text sentiment
        sentiment = results['text_sentiment']
        print(f"\n📝 TEXT SENTIMENT")
        print(f"   Sentiment: {sentiment['sentiment']} (score: {sentiment['sentiment_score']:.3f})")
        print(f"   Positive words: {sentiment['positive_words']}")
        print(f"   Negative words: {sentiment['negative_words']}")
        
        # Confidence indicators
        confidence = results['confidence_indicators']
        print(f"\n🎯 CONFIDENCE INDICATORS")
        print(f"   Confidence level: {confidence['confidence_level']} (score: {confidence['confidence_score']:.3f})")
        print(f"   Speaking rate consistency: {confidence['speaking_rate_consistency']:.3f}")
        print(f"   Pause frequency: {confidence['pause_frequency']:.1f} per minute")
        print(f"   Confidence words: {confidence['confidence_words']}")
        print(f"   Uncertainty words: {confidence['uncertainty_words']}")
        
        # Emotional intensity
        intensity = results['emotional_intensity_text']
        print(f"\n⚡ EMOTIONAL INTENSITY")
        print(f"   Intensity: {intensity['intensity']:.3f} ({intensity['emotion']})")
        print(f"   Intensifiers: {intensity['intensifiers']}")
        print(f"   Exclamations: {intensity['exclamations']}")
        print(f"   Caps: {intensity['caps']}")
        print(f"   Repetitions: {intensity['repetitions']}")
        
        print("\n" + "="*50)
        print("✅ Sentiment Analysis Complete!")
        print("="*50)
