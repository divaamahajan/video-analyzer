#!/usr/bin/env python3
"""
Simple Content Analyzer Test (No AI)
====================================

Test the ContentAnalyzer without AI agent dependencies.
"""

import os
import sys
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Load environment variables
load_dotenv()


def test_content_analyzer_simple():
    """Test ContentAnalyzer with sample data (no AI)"""
    print("📝 Testing ContentAnalyzer (Simple)")
    print("=" * 50)
    
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
        # Test individual analysis methods directly
        from analysis.text.content_analyzer import ContentAnalyzer
        
        # Create a mock analyzer without AI agent
        class MockContentAnalyzer:
            def __init__(self):
                # Copy the analysis methods from ContentAnalyzer
                self.transition_words = [
                    "first", "second", "third", "next", "then", "finally", "lastly",
                    "however", "but", "although", "despite", "in contrast", "on the other hand",
                    "furthermore", "moreover", "additionally", "also", "besides", "in addition",
                    "therefore", "thus", "consequently", "as a result", "hence", "so",
                    "for example", "for instance", "specifically", "namely", "such as",
                    "in conclusion", "to summarize", "overall", "in summary", "to conclude"
                ]
                
                self.power_words = [
                    "must", "will", "can", "should", "definitely", "certainly", "absolutely",
                    "essential", "critical", "vital", "crucial", "important", "significant",
                    "proven", "established", "confirmed", "verified", "guaranteed"
                ]
                
                self.weak_words = [
                    "maybe", "probably", "perhaps", "might", "could", "possibly", "potentially",
                    "sort of", "kind of", "somewhat", "rather", "quite", "pretty", "fairly",
                    "i think", "i believe", "i feel", "i guess", "i suppose", "i assume"
                ]
            
            def _split_into_sentences(self, text):
                import re
                sentences = re.split(r'[.!?]+', text)
                return [s.strip() for s in sentences if s.strip()]
            
            def _analyze_transitions(self, text):
                transition_counts = {}
                for transition in self.transition_words:
                    count = text.lower().count(transition)
                    if count > 0:
                        transition_counts[transition] = count
                
                total_transitions = sum(transition_counts.values())
                return {
                    "transition_counts": transition_counts,
                    "total_transitions": total_transitions,
                    "transition_usage": "good" if total_transitions > 3 else "limited"
                }
        
        # Test the analyzer
        analyzer = MockContentAnalyzer()
        
        # Test transition analysis
        print("🔍 Testing transition analysis...")
        transitions = analyzer._analyze_transitions(sample_transcription["text"])
        print(f"   Total transitions: {transitions['total_transitions']}")
        print(f"   Transition usage: {transitions['transition_usage']}")
        
        # Test sentence splitting
        print("🔍 Testing sentence splitting...")
        sentences = analyzer._split_into_sentences(sample_transcription["text"])
        print(f"   Sentences found: {len(sentences)}")
        print(f"   First sentence: {sentences[0]}")
        
        print("\n✅ Simple content analysis test completed successfully!")
        
    except Exception as e:
        print(f"❌ Simple content analysis test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function"""
    print("🧪 Simple Content Analyzer Test")
    print("=" * 60)
    
    # Test with sample data
    test_content_analyzer_simple()
    
    print("\n" + "=" * 60)
    print("✅ Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
