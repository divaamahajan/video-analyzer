#!/usr/bin/env python3
"""
Test Content Analyzer
====================

Test script to demonstrate the ContentAnalyzer functionality.
"""

import os
import sys
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.text import ContentAnalyzer

# Load environment variables
load_dotenv()


def test_content_analyzer():
    """Test ContentAnalyzer with sample transcription data"""
    print("📝 Testing ContentAnalyzer")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found. Skipping content analysis test.")
        return
    
    # Create sample transcription data
    sample_transcription = {
        "text": "Hello everyone, I'm really excited to be here today. I think this is going to be a great presentation. Maybe you'll find it interesting, and hopefully it will be useful for you. I want to talk about three main points: first, the importance of communication; second, how to improve your speaking skills; and third, some practical tips you can use. So, let's get started!",
        "segments": [
            {"text": "Hello everyone, I'm really excited to be here today.", "start": 0.0, "end": 3.0},
            {"text": "I think this is going to be a great presentation.", "start": 3.5, "end": 6.0},
            {"text": "Maybe you'll find it interesting, and hopefully it will be useful for you.", "start": 6.5, "end": 10.0},
            {"text": "I want to talk about three main points: first, the importance of communication; second, how to improve your speaking skills; and third, some practical tips you can use.", "start": 10.5, "end": 18.0},
            {"text": "So, let's get started!", "start": 18.5, "end": 20.0}
        ],
        "words": []
    }
    
    try:
        # Initialize analyzer
        analyzer = ContentAnalyzer(api_key)
        
        # Run analysis
        print("🔍 Running content analysis...")
        results = analyzer.analyze_content(sample_transcription)
        
        # Print results
        analyzer.print_content_summary(results)
        
        print("\n✅ Content analysis test completed successfully!")
        
    except Exception as e:
        print(f"❌ Content analysis test failed: {e}")


def test_with_real_audio():
    """Test with real audio file if available"""
    print("\n\n🎤 Testing with Real Audio")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found. Skipping real audio test.")
        return
    
    audio_path = "../input_audio.wav"
    if not os.path.exists(audio_path):
        print(f"⚠️ Audio file not found: {audio_path}")
        print("   Please run the main analysis first to generate audio file")
        return
    
    try:
        from analysis.text import TranscriptionService, ContentAnalyzer
        
        # Transcribe audio
        print("📄 Transcribing audio...")
        transcription_service = TranscriptionService(api_key)
        transcription_result = transcription_service.transcribe_with_timestamps(audio_path)
        
        # Analyze content
        print("📝 Analyzing content...")
        content_analyzer = ContentAnalyzer(api_key)
        content_results = content_analyzer.analyze_content(transcription_result)
        
        # Print results
        content_analyzer.print_content_summary(content_results)
        
        print("\n✅ Real audio content analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Real audio test failed: {e}")


def main():
    """Main test function"""
    print("🧪 Content Analyzer Test Suite")
    print("=" * 60)
    
    # Test with sample data
    test_content_analyzer()
    
    # Test with real audio
    test_with_real_audio()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
