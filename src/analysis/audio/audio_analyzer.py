#!/usr/bin/env python3
"""
Audio Analysis Module
====================

Main coordinator for audio analysis that combines:
- AudioProcessor: Direct audio analysis
- TranscriptionAnalyzer: Text-based analysis from transcriptions
"""

import os
from typing import Dict, Any
from .audio_processor import AudioProcessor
from ..text.transcription_analyzer import TranscriptionAnalyzer


class AudioAnalyzer:
    """Comprehensive audio analysis coordinator for video content"""
    
    def __init__(self, api_key: str):
        """Initialize with OpenAI API key"""
        self.audio_processor = AudioProcessor()
        self.transcription_analyzer = TranscriptionAnalyzer(api_key)
    
    def extract_audio_from_video(self, video_path: str, output_path: str = "input_audio.wav") -> str:
        """Extract audio from video file"""
        return self.audio_processor.extract_audio_from_video(video_path, output_path)
    
    def transcribe_with_timestamps(self, audio_path: str, prompt: str = None) -> Dict[str, Any]:
        """Transcribe audio with word and segment timestamps"""
        return self.transcription_analyzer.transcribe_with_timestamps(audio_path, prompt)
    
    def analyze_pace_and_rhythm(self, transcription_result: Dict[str, Any], audio_path: str = None) -> Dict[str, Any]:
        """Comprehensive pace and rhythm analysis"""
        return self.transcription_analyzer.analyze_pace_and_rhythm(transcription_result)
    
    def analyze_pitch_and_tone(self, audio_path: str) -> Dict[str, Any]:
        """Comprehensive pitch and tone analysis"""
        return self.audio_processor.analyze_pitch_and_tone(audio_path)
    
    def analyze_volume_and_clarity(self, audio_path: str) -> Dict[str, Any]:
        """Comprehensive volume and clarity analysis"""
        return self.audio_processor.analyze_volume_and_clarity(audio_path)
    
    def analyze_pronunciation_and_articulation(self, transcription_result: Dict[str, Any], audio_path: str) -> Dict[str, Any]:
        """Analyze pronunciation and articulation quality"""
        # Combine audio-based and text-based analysis
        audio_pronunciation = self.audio_processor.analyze_pronunciation_and_articulation(
            audio_path, transcription_result["segments"]
        )
        text_pronunciation = self.transcription_analyzer.analyze_pronunciation_and_articulation_text(
            transcription_result
        )
        
        return {
            "audio_analysis": audio_pronunciation,
            "text_analysis": text_pronunciation
        }
    
    def analyze_emotion_and_sentiment(self, transcription_result: Dict[str, Any], audio_path: str) -> Dict[str, Any]:
        """Analyze emotional intensity and sentiment"""
        # Get audio-based emotion analysis
        audio_emotion = self.audio_processor.analyze_emotion_and_sentiment_audio(audio_path)
        
        # Get text-based emotion analysis
        text_emotion = self.transcription_analyzer.analyze_emotion_and_sentiment_text(transcription_result)
        
        return {
            "audio_emotion": audio_emotion,
            "text_emotion": text_emotion
        }
    
    def analyze_audio(self, video_path: str, audio_path: str = "input_audio.wav") -> Dict[str, Any]:
        """Complete comprehensive audio analysis pipeline"""
        print("\n🎵 Starting Comprehensive Audio Analysis")
        print("="*50)
        
        # Extract audio
        audio_file = self.extract_audio_from_video(video_path, audio_path)
        
        # Transcribe
        transcription = self.transcribe_with_timestamps(
            audio_file, 
            prompt="English + Hindi conversation"
        )
        
        # Comprehensive analysis
        pace_analysis = self.analyze_pace_and_rhythm(transcription, audio_file)
        pitch_analysis = self.analyze_pitch_and_tone(audio_file)
        volume_analysis = self.analyze_volume_and_clarity(audio_file)
        pronunciation_analysis = self.analyze_pronunciation_and_articulation(transcription, audio_file)
        emotion_analysis = self.analyze_emotion_and_sentiment(transcription, audio_file)
        
        # Combine results
        return {
            "transcription": transcription,
            "pace_and_rhythm": pace_analysis,
            "pitch_and_tone": pitch_analysis,
            "volume_and_clarity": volume_analysis,
            "pronunciation": pronunciation_analysis,
            "emotion_and_sentiment": emotion_analysis
        }
    
    def print_audio_summary(self, results: Dict[str, Any]):
        """Print a comprehensive formatted summary of audio analysis"""
        print("\n" + "="*60)
        print("🎤 COMPREHENSIVE AUDIO & VOCAL ANALYSIS REPORT")
        print("="*60)
        
        # Transcription Overview
        transcription = results['transcription']
        print(f"\n📝 TRANSCRIPTION OVERVIEW")
        print(f"   Full text: {transcription['text']}")
        print(f"   Segments: {len(transcription['segments'])}")
        print(f"   Words: {len(transcription['words'])}")
        
        # Pace & Rhythm Analysis
        pace = results['pace_and_rhythm']
        print(f"\n🏃 PACE & RHYTHM")
        print(f"   Words per minute: {pace['wpm']:.1f}")
        print(f"   Average pause length: {pace['avg_pause_length']:.2f}s")
        print(f"   Pause frequency: {pace['pause_frequency']:.1f} per minute")
        
        # Filler Analysis
        fillers = pace['filler_analysis']
        print(f"   Filler words: {fillers['total_fillers']} ({fillers['fillers_per_min']:.1f}/min)")
        print(f"   Filler distribution: Start={fillers['filler_timing_distribution']['start']}, "
              f"Middle={fillers['filler_timing_distribution']['middle']}, "
              f"End={fillers['filler_timing_distribution']['end']}")
        
        # Pitch & Tone Analysis
        pitch = results['pitch_and_tone']
        if 'error' not in pitch:
            print(f"\n🎵 PITCH & TONE")
            print(f"   Pitch range: {pitch['pitch_min']:.1f} - {pitch['pitch_max']:.1f} Hz")
            print(f"   Pitch variance: {pitch['pitch_variance']:.1f}")
            print(f"   Pitch CV: {pitch['pitch_cv']:.3f}")
            print(f"   Contour trend: {pitch['pitch_contour']['trend']}")
            
            # Emotional cues
            emotion = pitch['emotional_cues']
            print(f"   Primary emotion: {emotion['primary_emotion']}")
            print(f"   Emotional indicators: {emotion['emotional_indicators']}")
            
            # Prosody
            prosody = pitch['prosody']
            print(f"   Rising intonation: {prosody['rising_intonation']:.1%}")
            print(f"   Falling intonation: {prosody['falling_intonation']:.1%}")
            print(f"   Emphasis points: {prosody['emphasis_points']}")
        
        # Volume & Clarity Analysis
        volume = results['volume_and_clarity']
        print(f"\n🔊 VOLUME & CLARITY")
        print(f"   Volume: {volume['volume_db']:.1f} dB")
        print(f"   Volume consistency: {volume['volume_consistency']:.3f}")
        print(f"   Clipping: {volume['clipping_percentage']:.1f}%")
        print(f"   Soft segments: {volume['soft_segments_percentage']:.1f}%")
        print(f"   Dynamic range: {volume['dynamic_range']:.3f}")
        print(f"   Clarity score: {volume['clarity_score']:.3f}")
        
        # Pronunciation Analysis
        pronunciation = results['pronunciation']
        print(f"\n🗣️ PRONUNCIATION & ARTICULATION")
        
        # Audio-based analysis
        if 'audio_analysis' in pronunciation and 'error' not in pronunciation['audio_analysis']:
            audio_pron = pronunciation['audio_analysis']
            print(f"   Audio Articulation score: {audio_pron['avg_articulation_score']:.3f}")
            print(f"   Audio Pronunciation quality: {audio_pron['pronunciation_quality']}")
            print(f"   Potential mispronunciations: {len(audio_pron['potential_mispronunciations'])}")
        
        # Text-based analysis
        if 'text_analysis' in pronunciation and 'error' not in pronunciation['text_analysis']:
            text_pron = pronunciation['text_analysis']
            if text_pron['repeated_phrases']:
                print(f"   Repeated phrases:")
                for phrase in text_pron['repeated_phrases'][:5]:  # Show top 5
                    print(f"     - '{phrase['word']}': {phrase['count']} times ({phrase['frequency']:.1%})")
            
            # Word complexity
            complexity = text_pron['word_complexity']
            print(f"   Vocabulary richness: {complexity['vocabulary_richness']:.3f}")
            print(f"   Average word length: {complexity['avg_word_length']:.1f}")
        
        # Emotion & Sentiment Analysis
        emotion = results['emotion_and_sentiment']
        print(f"\n😊 EMOTION & SENTIMENT")
        
        # Audio emotion
        if 'audio_emotion' in emotion:
            audio_intensity = emotion['audio_emotion']['emotional_intensity']
            print(f"   Audio Emotional intensity: {audio_intensity['intensity']:.3f} ({audio_intensity['emotion']})")
        
        # Text emotion
        if 'text_emotion' in emotion:
            text_sentiment = emotion['text_emotion']['text_sentiment']
            print(f"   Text sentiment: {text_sentiment['sentiment']} (score: {text_sentiment['sentiment_score']:.3f})")
            print(f"   Positive words: {text_sentiment['positive_words']}, Negative: {text_sentiment['negative_words']}")
            
            confidence = emotion['text_emotion']['confidence_indicators']
            print(f"   Confidence level: {confidence['confidence_level']} (score: {confidence['confidence_score']:.3f})")
        
        print("\n" + "="*60)
        print("✅ Analysis Complete!")
        print("="*60)
