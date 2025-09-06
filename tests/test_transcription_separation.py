#!/usr/bin/env python3
"""
Test Transcription Separation
============================

Test script to demonstrate the separated transcription analysis components:
1. TranscriptionService - Basic transcription functionality
2. PaceAnalyzer - Pace and rhythm analysis
3. PronunciationAnalyzer - Text-based pronunciation analysis
4. SentimentAnalyzer - Text-based emotion and sentiment analysis
5. TranscriptionAnalyzer - Coordinator that combines all
"""

import os
import sys
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.analysis.transcription_service import TranscriptionService
from src.analysis.pace_analyzer import PaceAnalyzer
from src.analysis.pronunciation_analyzer import PronunciationAnalyzer
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.analysis.transcription_analyzer import TranscriptionAnalyzer

# Load environment variables
load_dotenv()


def test_transcription_service():
    """Test TranscriptionService (basic transcription)"""
    print("📄 Testing TranscriptionService (Basic Transcription)")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found. Skipping transcription service test.")
        return
    
    service = TranscriptionService(api_key)
    
    # Test with existing audio file if available
    audio_path = "input_audio.wav"
    if os.path.exists(audio_path):
        print(f"📁 Found audio file: {audio_path}")
        
        # Test simple transcription
        print("\n📝 Testing simple transcription...")
        try:
            simple_text = service.transcribe_simple(audio_path)
            print(f"✅ Simple transcription successful")
            print(f"   Text: {simple_text[:100]}...")
        except Exception as e:
            print(f"❌ Simple transcription failed: {e}")
        
        # Test detailed transcription
        print("\n📝 Testing detailed transcription...")
        try:
            detailed_result = service.transcribe_with_timestamps(audio_path)
            print(f"✅ Detailed transcription successful")
            print(f"   Text: {detailed_result['text'][:100]}...")
            print(f"   Segments: {len(detailed_result['segments'])}")
            print(f"   Words: {len(detailed_result['words'])}")
            
            # Print transcription info
            service.print_transcription_info(detailed_result)
            
        except Exception as e:
            print(f"❌ Detailed transcription failed: {e}")
    else:
        print(f"⚠️ Audio file not found: {audio_path}")
        print("   Please run the main analysis first to generate audio file")


def test_pace_analyzer():
    """Test PaceAnalyzer (pace and rhythm analysis)"""
    print("\n\n🏃 Testing PaceAnalyzer (Pace & Rhythm Analysis)")
    print("=" * 50)
    
    analyzer = PaceAnalyzer()
    
    # Create mock transcription data for testing
    mock_transcription = {
        "text": "Hello, this is a test. Um, let me think about this. You know, it's really interesting. So, what do you think?",
        "segments": [
            {"text": "Hello, this is a test.", "start": 0.0, "end": 2.5},
            {"text": "Um, let me think about this.", "start": 3.0, "end": 6.0},
            {"text": "You know, it's really interesting.", "start": 6.5, "end": 9.0},
            {"text": "So, what do you think?", "start": 9.5, "end": 12.0}
        ],
        "words": []
    }
    
    print("📊 Testing pace and rhythm analysis...")
    try:
        pace_result = analyzer.analyze_pace_and_rhythm(mock_transcription)
        print(f"✅ Pace analysis successful")
        print(f"   WPM: {pace_result['wpm']:.1f}")
        print(f"   Fillers: {pace_result['filler_analysis']['total_fillers']}")
        
        # Print detailed summary
        analyzer.print_pace_summary(pace_result)
        
    except Exception as e:
        print(f"❌ Pace analysis failed: {e}")


def test_pronunciation_analyzer():
    """Test PronunciationAnalyzer (text-based pronunciation analysis)"""
    print("\n\n🗣️ Testing PronunciationAnalyzer (Pronunciation Analysis)")
    print("=" * 50)
    
    analyzer = PronunciationAnalyzer()
    
    # Create mock transcription data for testing
    mock_transcription = {
        "text": "This is a really interesting and fascinating presentation. The presentation was absolutely amazing and incredible. I think this is fantastic.",
        "segments": [
            {"text": "This is a really interesting and fascinating presentation.", "start": 0.0, "end": 4.0},
            {"text": "The presentation was absolutely amazing and incredible.", "start": 4.5, "end": 8.0},
            {"text": "I think this is fantastic.", "start": 8.5, "end": 10.5}
        ]
    }
    
    print("📚 Testing pronunciation analysis...")
    try:
        pronunciation_result = analyzer.analyze_pronunciation_and_articulation_text(mock_transcription)
        print(f"✅ Pronunciation analysis successful")
        
        # Print detailed summary
        analyzer.print_pronunciation_summary(pronunciation_result)
        
    except Exception as e:
        print(f"❌ Pronunciation analysis failed: {e}")


def test_sentiment_analyzer():
    """Test SentimentAnalyzer (text-based sentiment analysis)"""
    print("\n\n😊 Testing SentimentAnalyzer (Sentiment Analysis)")
    print("=" * 50)
    
    analyzer = SentimentAnalyzer()
    
    # Create mock transcription data for testing
    mock_transcription = {
        "text": "I'm really excited about this project! It's absolutely fantastic and I'm confident it will be successful. Maybe we should consider some improvements though.",
        "segments": [
            {"text": "I'm really excited about this project!", "start": 0.0, "end": 3.0},
            {"text": "It's absolutely fantastic and I'm confident it will be successful.", "start": 3.5, "end": 7.0},
            {"text": "Maybe we should consider some improvements though.", "start": 7.5, "end": 10.0}
        ]
    }
    
    print("💭 Testing sentiment analysis...")
    try:
        sentiment_result = analyzer.analyze_emotion_and_sentiment_text(mock_transcription)
        print(f"✅ Sentiment analysis successful")
        
        # Print detailed summary
        analyzer.print_sentiment_summary(sentiment_result)
        
    except Exception as e:
        print(f"❌ Sentiment analysis failed: {e}")


def test_combined_analyzer():
    """Test TranscriptionAnalyzer (combined coordinator)"""
    print("\n\n📝 Testing TranscriptionAnalyzer (Combined Coordinator)")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found. Skipping combined analysis.")
        return
    
    analyzer = TranscriptionAnalyzer(api_key)
    
    # Test with existing audio file if available
    audio_path = "input_audio.wav"
    if os.path.exists(audio_path):
        print(f"📁 Found audio file: {audio_path}")
        
        # Test full analysis pipeline
        print("\n📝 Testing full transcription analysis pipeline...")
        try:
            results = analyzer.analyze_transcription(audio_path)
            print(f"✅ Full analysis successful")
            print(f"   Transcription: {len(results['transcription']['text'])} characters")
            print(f"   Pace analysis: {results['pace_and_rhythm']['wpm']:.1f} WPM")
            print(f"   Pronunciation analysis: {'✅' if 'error' not in results['pronunciation_patterns'] else '❌'}")
            print(f"   Sentiment analysis: {'✅' if 'text_sentiment' in results['emotion_and_sentiment'] else '❌'}")
            
            # Print summary
            print("\n📊 Full Analysis Summary:")
            analyzer.print_transcription_summary(results)
            
        except Exception as e:
            print(f"❌ Full analysis failed: {e}")
    else:
        print(f"⚠️ Audio file not found: {audio_path}")
        print("   Please ensure input_audio.wav exists")


def main():
    """Main test function"""
    print("🧪 Transcription Analysis Separation Test")
    print("=" * 60)
    
    # Test individual components
    test_transcription_service()
    test_pace_analyzer()
    test_pronunciation_analyzer()
    test_sentiment_analyzer()
    test_combined_analyzer()
    
    print("\n" + "=" * 60)
    print("✅ Separation test complete!")
    print("\n📋 Summary:")
    print("   • TranscriptionService: Basic transcription functionality")
    print("   • PaceAnalyzer: Pace, rhythm, and filler word analysis")
    print("   • PronunciationAnalyzer: Text-based pronunciation patterns")
    print("   • SentimentAnalyzer: Text-based emotion and sentiment analysis")
    print("   • TranscriptionAnalyzer: Coordinator that combines all approaches")
    print("=" * 60)


if __name__ == "__main__":
    main()
