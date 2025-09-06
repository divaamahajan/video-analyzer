#!/usr/bin/env python3
"""
Test Audio Separation
====================

Test script to demonstrate the separated audio analysis components:
1. AudioProcessor - Direct audio analysis
2. TranscriptionAnalyzer - Text-based analysis
3. AudioAnalyzer - Coordinator that combines both
"""

import os
import sys
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.analysis.audio_processor import AudioProcessor
from src.analysis.transcription_analyzer import TranscriptionAnalyzer
from src.analysis.audio_analyzer import AudioAnalyzer

# Load environment variables
load_dotenv()


def test_audio_processor():
    """Test AudioProcessor (direct audio analysis)"""
    print("🎵 Testing AudioProcessor (Direct Audio Analysis)")
    print("=" * 50)
    
    processor = AudioProcessor()
    
    # Test with existing audio file if available
    audio_path = "input_audio.wav"
    if os.path.exists(audio_path):
        print(f"📁 Found audio file: {audio_path}")
        
        # Test pitch and tone analysis
        print("\n🎵 Testing pitch and tone analysis...")
        try:
            pitch_result = processor.analyze_pitch_and_tone(audio_path)
            if 'error' not in pitch_result:
                print(f"✅ Pitch analysis successful")
                print(f"   Pitch range: {pitch_result['pitch_min']:.1f} - {pitch_result['pitch_max']:.1f} Hz")
                print(f"   Primary emotion: {pitch_result['emotional_cues']['primary_emotion']}")
            else:
                print(f"⚠️ Pitch analysis error: {pitch_result['error']}")
        except Exception as e:
            print(f"❌ Pitch analysis failed: {e}")
        
        # Test volume and clarity analysis
        print("\n🔊 Testing volume and clarity analysis...")
        try:
            volume_result = processor.analyze_volume_and_clarity(audio_path)
            print(f"✅ Volume analysis successful")
            print(f"   Volume: {volume_result['volume_db']:.1f} dB")
            print(f"   Clarity score: {volume_result['clarity_score']:.3f}")
        except Exception as e:
            print(f"❌ Volume analysis failed: {e}")
        
        # Test emotion analysis
        print("\n😊 Testing emotion analysis...")
        try:
            emotion_result = processor.analyze_emotion_and_sentiment_audio(audio_path)
            intensity = emotion_result['emotional_intensity']
            print(f"✅ Emotion analysis successful")
            print(f"   Emotional intensity: {intensity['intensity']:.3f} ({intensity['emotion']})")
        except Exception as e:
            print(f"❌ Emotion analysis failed: {e}")
    else:
        print(f"⚠️ Audio file not found: {audio_path}")
        print("   Please run the main analysis first to generate audio file")


def test_transcription_analyzer():
    """Test TranscriptionAnalyzer (text-based analysis)"""
    print("\n\n📝 Testing TranscriptionAnalyzer (Text-based Analysis)")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found. Skipping transcription analysis.")
        return
    
    analyzer = TranscriptionAnalyzer(api_key)
    
    # Test with existing audio file if available
    audio_path = "input_audio.wav"
    if os.path.exists(audio_path):
        print(f"📁 Found audio file: {audio_path}")
        
        # Test transcription
        print("\n📝 Testing transcription...")
        try:
            transcription_result = analyzer.transcribe_with_timestamps(audio_path)
            print(f"✅ Transcription successful")
            print(f"   Text: {transcription_result['text'][:100]}...")
            print(f"   Segments: {len(transcription_result['segments'])}")
            print(f"   Words: {len(transcription_result['words'])}")
            
            # Test pace and rhythm analysis
            print("\n🏃 Testing pace and rhythm analysis...")
            try:
                pace_result = analyzer.analyze_pace_and_rhythm(transcription_result)
                print(f"✅ Pace analysis successful")
                print(f"   WPM: {pace_result['wpm']:.1f}")
                print(f"   Fillers: {pace_result['filler_analysis']['total_fillers']}")
            except Exception as e:
                print(f"❌ Pace analysis failed: {e}")
            
            # Test text-based emotion analysis
            print("\n😊 Testing text emotion analysis...")
            try:
                emotion_result = analyzer.analyze_emotion_and_sentiment_text(transcription_result)
                sentiment = emotion_result['text_sentiment']
                print(f"✅ Text emotion analysis successful")
                print(f"   Sentiment: {sentiment['sentiment']} (score: {sentiment['sentiment_score']:.3f})")
            except Exception as e:
                print(f"❌ Text emotion analysis failed: {e}")
                
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
    else:
        print(f"⚠️ Audio file not found: {audio_path}")
        print("   Please run the main analysis first to generate audio file")


def test_combined_analyzer():
    """Test AudioAnalyzer (combined coordinator)"""
    print("\n\n🎤 Testing AudioAnalyzer (Combined Coordinator)")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found. Skipping combined analysis.")
        return
    
    analyzer = AudioAnalyzer(api_key)
    
    # Test with existing video file if available
    video_path = "input_video.mp4"
    if os.path.exists(video_path):
        print(f"📁 Found video file: {video_path}")
        
        # Test full analysis pipeline
        print("\n🎤 Testing full analysis pipeline...")
        try:
            results = analyzer.analyze_audio(video_path)
            print(f"✅ Full analysis successful")
            print(f"   Transcription: {len(results['transcription']['text'])} characters")
            print(f"   Pace analysis: {results['pace_and_rhythm']['wpm']:.1f} WPM")
            print(f"   Pitch analysis: {'✅' if 'error' not in results['pitch_and_tone'] else '❌'}")
            print(f"   Volume analysis: {results['volume_and_clarity']['volume_db']:.1f} dB")
            
            # Print summary
            print("\n📊 Analysis Summary:")
            analyzer.print_audio_summary(results)
            
        except Exception as e:
            print(f"❌ Full analysis failed: {e}")
    else:
        print(f"⚠️ Video file not found: {video_path}")
        print("   Please ensure input_video.mp4 exists")


def main():
    """Main test function"""
    print("🧪 Audio Analysis Separation Test")
    print("=" * 60)
    
    # Test individual components
    test_audio_processor()
    test_transcription_analyzer()
    test_combined_analyzer()
    
    print("\n" + "=" * 60)
    print("✅ Separation test complete!")
    print("\n📋 Summary:")
    print("   • AudioProcessor: Direct audio analysis (pitch, volume, emotion)")
    print("   • TranscriptionAnalyzer: Text-based analysis (pace, sentiment, confidence)")
    print("   • AudioAnalyzer: Coordinator that combines both approaches")
    print("=" * 60)


if __name__ == "__main__":
    main()
