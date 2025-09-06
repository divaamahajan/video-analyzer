#!/usr/bin/env python3
"""
Standalone Content Analyzer Test
===============================

Test the ContentAnalyzer without AI agent dependencies.
"""

import os
import sys
import re
import numpy as np
from collections import Counter

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


class StandaloneContentAnalyzer:
    """Standalone version of ContentAnalyzer for testing"""
    
    def __init__(self):
        """Initialize without AI agent"""
        # Linguistic analysis patterns
        self.filler_words = [
            "um", "uh", "like", "you know", "so", "well", "actually", 
            "basically", "literally", "kind of", "sort of", "right"
        ]
        
        self.weak_words = [
            "maybe", "probably", "perhaps", "might", "could", "possibly",
            "sort of", "kind of", "somewhat", "rather", "quite", "pretty"
        ]
        
        self.confidence_markers = [
            "i will", "i can", "i know", "i believe", "i think", "i feel",
            "definitely", "certainly", "absolutely", "sure", "confident"
        ]
    
    def analyze_content(self, transcription_result):
        """Comprehensive content analysis without AI"""
        print("📝 Analyzing Content and Linguistic Features")
        
        text = transcription_result["text"]
        segments = transcription_result.get("segments", [])
        
        # Basic text preprocessing
        processed_text = self._preprocess_text(text)
        
        # Linguistic analysis
        linguistic_analysis = self._analyze_linguistic_features(processed_text, segments)
        
        # Structure analysis
        structure_analysis = self._analyze_structure(processed_text, segments)
        
        # Vocabulary analysis
        vocabulary_analysis = self._analyze_vocabulary(processed_text)
        
        # Clarity analysis
        clarity_analysis = self._analyze_clarity(processed_text)
        
        # Engagement analysis
        engagement_analysis = self._analyze_engagement(processed_text)
        
        return {
            "ai_analysis": {"ai_insights": "AI analysis disabled for testing", "analysis_complete": False},
            "linguistic_features": linguistic_analysis,
            "structure": structure_analysis,
            "vocabulary": vocabulary_analysis,
            "clarity": clarity_analysis,
            "engagement": engagement_analysis,
            "raw_text": text,
            "processed_text": processed_text
        }
    
    def _preprocess_text(self, text):
        """Clean and preprocess text for analysis"""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        return text
    
    def _analyze_linguistic_features(self, text, segments):
        """Analyze linguistic features and patterns"""
        words = text.lower().split()
        sentences = self._split_into_sentences(text)
        
        # Filler word analysis
        filler_analysis = self._analyze_fillers(text, words)
        
        # Weak word analysis
        weak_word_analysis = self._analyze_weak_words(text, words)
        
        # Confidence markers
        confidence_analysis = self._analyze_confidence_markers(text, words)
        
        # Repetition analysis
        repetition_analysis = self._analyze_repetition(words)
        
        return {
            "filler_words": filler_analysis,
            "weak_words": weak_word_analysis,
            "confidence_markers": confidence_analysis,
            "repetition": repetition_analysis,
            "total_words": len(words),
            "total_sentences": len(sentences)
        }
    
    def _analyze_fillers(self, text, words):
        """Analyze filler word usage"""
        filler_counts = {}
        for filler in self.filler_words:
            count = text.lower().count(filler)
            if count > 0:
                filler_counts[filler] = count
        
        total_fillers = sum(filler_counts.values())
        filler_frequency = (total_fillers / len(words)) * 100 if words else 0
        
        return {
            "filler_counts": filler_counts,
            "total_fillers": total_fillers,
            "filler_frequency_per_100_words": filler_frequency,
            "filler_density": "high" if filler_frequency > 5 else "medium" if filler_frequency > 2 else "low"
        }
    
    def _analyze_weak_words(self, text, words):
        """Analyze weak/modifying word usage"""
        weak_counts = {}
        for weak_word in self.weak_words:
            count = text.lower().count(weak_word)
            if count > 0:
                weak_counts[weak_word] = count
        
        total_weak = sum(weak_counts.values())
        weak_frequency = (total_weak / len(words)) * 100 if words else 0
        
        return {
            "weak_word_counts": weak_counts,
            "total_weak_words": total_weak,
            "weak_word_frequency_per_100_words": weak_frequency,
            "assertiveness_level": "low" if weak_frequency > 10 else "medium" if weak_frequency > 5 else "high"
        }
    
    def _analyze_confidence_markers(self, text, words):
        """Analyze confidence and uncertainty markers"""
        confidence_count = 0
        for marker in self.confidence_markers:
            confidence_count += text.lower().count(marker)
        
        return {
            "confidence_markers": confidence_count,
            "confidence_level": "high" if confidence_count > 3 else "medium" if confidence_count > 1 else "low"
        }
    
    def _analyze_repetition(self, words):
        """Analyze word repetition patterns"""
        word_counts = Counter(words)
        repeated_words = {word: count for word, count in word_counts.items() if count > 1}
        
        total_words = len(words)
        unique_words = len(word_counts)
        repetition_ratio = len(repeated_words) / unique_words if unique_words > 0 else 0
        
        return {
            "repeated_words": repeated_words,
            "repetition_ratio": repetition_ratio,
            "vocabulary_diversity": unique_words / total_words if total_words > 0 else 0,
            "repetition_level": "high" if repetition_ratio > 0.3 else "medium" if repetition_ratio > 0.1 else "low"
        }
    
    def _analyze_structure(self, text, segments):
        """Analyze sentence and paragraph structure"""
        sentences = self._split_into_sentences(text)
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        
        return {
            "sentence_count": len(sentences),
            "avg_sentence_length": np.mean(sentence_lengths) if sentence_lengths else 0,
            "sentence_length_std": np.std(sentence_lengths) if sentence_lengths else 0,
            "sentence_length_distribution": {
                "short": len([l for l in sentence_lengths if l < 10]),
                "medium": len([l for l in sentence_lengths if 10 <= l < 20]),
                "long": len([l for l in sentence_lengths if l >= 20])
            }
        }
    
    def _analyze_vocabulary(self, text):
        """Analyze vocabulary richness and sophistication"""
        words = [word.lower().strip('.,!?;:') for word in text.split()]
        unique_words = set(words)
        
        total_words = len(words)
        unique_count = len(unique_words)
        lexical_richness = unique_count / total_words if total_words > 0 else 0
        
        word_lengths = [len(word) for word in words]
        avg_word_length = np.mean(word_lengths)
        
        return {
            "total_words": total_words,
            "unique_words": unique_count,
            "lexical_richness": lexical_richness,
            "avg_word_length": avg_word_length,
            "vocabulary_level": "advanced" if lexical_richness > 0.8 else "intermediate" if lexical_richness > 0.6 else "basic"
        }
    
    def _analyze_clarity(self, text):
        """Analyze clarity and conciseness"""
        sentences = self._split_into_sentences(text)
        avg_sentence_length = np.mean([len(sentence.split()) for sentence in sentences])
        
        return {
            "avg_sentence_length": avg_sentence_length,
            "clarity_score": 0.8 if avg_sentence_length < 15 else 0.6 if avg_sentence_length < 25 else 0.4
        }
    
    def _analyze_engagement(self, text):
        """Analyze engagement and rhetorical devices"""
        sentences = self._split_into_sentences(text)
        questions = [s for s in sentences if s.strip().endswith('?')]
        exclamations = [s for s in sentences if s.strip().endswith('!')]
        
        question_ratio = len(questions) / len(sentences) if sentences else 0
        exclamation_ratio = len(exclamations) / len(sentences) if sentences else 0
        
        return {
            "questions": len(questions),
            "question_ratio": question_ratio,
            "exclamations": len(exclamations),
            "exclamation_ratio": exclamation_ratio,
            "engagement_score": (question_ratio * 0.5) + (exclamation_ratio * 0.5)
        }
    
    def _split_into_sentences(self, text):
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def print_content_summary(self, results):
        """Print comprehensive content analysis summary"""
        print("\n" + "="*60)
        print("📝 CONTENT & LINGUISTIC ANALYSIS REPORT")
        print("="*60)
        
        # Linguistic Features
        linguistic = results['linguistic_features']
        print(f"\n🗣️ LINGUISTIC FEATURES")
        print(f"   Total words: {linguistic['total_words']}")
        print(f"   Total sentences: {linguistic['total_sentences']}")
        
        # Filler words
        fillers = linguistic['filler_words']
        print(f"   Filler words: {fillers['total_fillers']} ({fillers['filler_frequency_per_100_words']:.1f}/100 words)")
        print(f"   Filler density: {fillers['filler_density']}")
        
        # Weak words
        weak = linguistic['weak_words']
        print(f"   Weak words: {weak['total_weak_words']} ({weak['weak_word_frequency_per_100_words']:.1f}/100 words)")
        print(f"   Assertiveness: {weak['assertiveness_level']}")
        
        # Confidence markers
        confidence = linguistic['confidence_markers']
        print(f"   Confidence markers: {confidence['confidence_markers']}")
        print(f"   Confidence level: {confidence['confidence_level']}")
        
        # Structure Analysis
        structure = results['structure']
        print(f"\n📊 STRUCTURE ANALYSIS")
        print(f"   Average sentence length: {structure['avg_sentence_length']:.1f} words")
        print(f"   Sentence length distribution:")
        dist = structure['sentence_length_distribution']
        print(f"     - Short (<10 words): {dist['short']}")
        print(f"     - Medium (10-20 words): {dist['medium']}")
        print(f"     - Long (>20 words): {dist['long']}")
        
        # Vocabulary Analysis
        vocab = results['vocabulary']
        print(f"\n📚 VOCABULARY ANALYSIS")
        print(f"   Lexical richness: {vocab['lexical_richness']:.3f}")
        print(f"   Average word length: {vocab['avg_word_length']:.1f} characters")
        print(f"   Vocabulary level: {vocab['vocabulary_level']}")
        
        # Clarity Analysis
        clarity = results['clarity']
        print(f"\n🔍 CLARITY ANALYSIS")
        print(f"   Average sentence length: {clarity['avg_sentence_length']:.1f} words")
        print(f"   Clarity score: {clarity['clarity_score']:.3f}")
        
        # Engagement Analysis
        engagement = results['engagement']
        print(f"\n🎯 ENGAGEMENT ANALYSIS")
        print(f"   Questions: {engagement['questions']} ({engagement['question_ratio']:.1%})")
        print(f"   Exclamations: {engagement['exclamations']} ({engagement['exclamation_ratio']:.1%})")
        print(f"   Engagement score: {engagement['engagement_score']:.3f}")
        
        print("\n" + "="*60)
        print("✅ Content Analysis Complete!")
        print("="*60)


def test_content_analyzer():
    """Test ContentAnalyzer with sample transcription data"""
    print("📝 Testing ContentAnalyzer")
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
        # Initialize analyzer
        analyzer = StandaloneContentAnalyzer()
        
        # Run analysis
        print("🔍 Running content analysis...")
        results = analyzer.analyze_content(sample_transcription)
        
        # Print results
        analyzer.print_content_summary(results)
        
        print("\n✅ Content analysis test completed successfully!")
        
    except Exception as e:
        print(f"❌ Content analysis test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function"""
    print("🧪 Standalone Content Analyzer Test")
    print("=" * 60)
    
    # Test with sample data
    test_content_analyzer()
    
    print("\n" + "=" * 60)
    print("✅ Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
