#!/usr/bin/env python3
"""
Test Advanced Content Analyzer
==============================

Test script to demonstrate the advanced ContentAnalyzer functionality.
"""

import os
import sys
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Load environment variables
load_dotenv()


def test_advanced_content_analyzer():
    """Test advanced ContentAnalyzer with sample transcription data"""
    print("📝 Testing Advanced ContentAnalyzer")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found. Skipping advanced content analysis test.")
        return
    
    # Import directly to avoid circular imports
    from analysis.text.content_analyzer import ContentAnalyzer
    
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
        print("🔍 Running advanced content analysis...")
        results = analyzer.analyze_content(sample_transcription)
        
        # Print results
        analyzer.print_content_summary(results)
        
        print("\n✅ Advanced content analysis test completed successfully!")
        
    except Exception as e:
        print(f"❌ Advanced content analysis test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function"""
    print("🧪 Advanced Content Analyzer Test")
    print("=" * 60)
    
    # Test with sample data
    test_advanced_content_analyzer()
    
    print("\n" + "=" * 60)
    print("✅ Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
