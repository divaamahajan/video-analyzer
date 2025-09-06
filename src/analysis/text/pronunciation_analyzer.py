#!/usr/bin/env python3
"""
Pronunciation Analyzer Module
============================

Handles text-based pronunciation and articulation analysis including:
- Repeated word/phrase detection
- Word complexity analysis
- Sentence structure analysis
- Vocabulary richness assessment
"""

import numpy as np
import re
from typing import Dict, List, Any


class PronunciationAnalyzer:
    """Analyzes pronunciation patterns from transcribed text"""
    
    def __init__(self):
        """Initialize pronunciation analyzer"""
        pass
    
    def analyze_pronunciation_and_articulation_text(self, transcription_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pronunciation patterns from text (repeated words, etc.)"""
        print("🗣️ Analyzing Text-based Pronunciation Patterns")
        
        segments = transcription_result["segments"]
        if not segments:
            return {"error": "No segments to analyze"}
        
        # Detect repeated words/phrases
        repeated_phrases = self._detect_repeated_phrases(segments)
        
        # Analyze word complexity
        word_complexity = self._analyze_word_complexity(transcription_result["text"])
        
        # Analyze sentence structure
        sentence_structure = self._analyze_sentence_structure(segments)
        
        return {
            "repeated_phrases": repeated_phrases,
            "word_complexity": word_complexity,
            "sentence_structure": sentence_structure
        }
    
    def _detect_repeated_phrases(self, segments: List) -> List[Dict[str, Any]]:
        """Detect repeated words or phrases"""
        all_text = " ".join([seg.text.lower() for seg in segments])
        words = re.findall(r'\b\w+\b', all_text)
        
        # Count word frequencies
        word_counts = {}
        for word in words:
            if len(word) > 2:  # Only count words longer than 2 characters
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Find repeated words
        repeated = {word: count for word, count in word_counts.items() if count > 1}
        
        return [
            {"word": word, "count": count, "frequency": count / len(words)}
            for word, count in repeated.items()
        ]
    
    def _analyze_word_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze word complexity and vocabulary richness"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not words:
            return {"complexity_score": 0, "vocabulary_richness": 0}
        
        # Average word length
        avg_word_length = np.mean([len(word) for word in words])
        
        # Unique words ratio
        unique_words = len(set(words))
        vocabulary_richness = unique_words / len(words)
        
        # Complex words (longer than 6 characters)
        complex_words = [word for word in words if len(word) > 6]
        complex_word_ratio = len(complex_words) / len(words)
        
        # Calculate overall complexity score
        complexity_score = (avg_word_length / 10) * 0.4 + vocabulary_richness * 0.3 + complex_word_ratio * 0.3
        
        return {
            "complexity_score": complexity_score,
            "vocabulary_richness": vocabulary_richness,
            "avg_word_length": avg_word_length,
            "complex_word_ratio": complex_word_ratio,
            "unique_words": unique_words,
            "total_words": len(words)
        }
    
    def _analyze_sentence_structure(self, segments: List) -> Dict[str, Any]:
        """Analyze sentence structure and patterns"""
        if not segments:
            return {"sentence_count": 0, "avg_sentence_length": 0}
        
        # Count sentences (rough estimate based on punctuation)
        all_text = " ".join([seg.text for seg in segments])
        sentence_count = len(re.findall(r'[.!?]+', all_text))
        
        # Average sentence length
        words_per_segment = [len(seg.text.split()) for seg in segments]
        avg_sentence_length = np.mean(words_per_segment) if words_per_segment else 0
        
        # Question vs statement ratio
        questions = len(re.findall(r'\?', all_text))
        statements = len(re.findall(r'[.!]', all_text))
        question_ratio = questions / (questions + statements) if (questions + statements) > 0 else 0
        
        return {
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "question_ratio": question_ratio,
            "questions": questions,
            "statements": statements
        }
    
    def print_pronunciation_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of pronunciation analysis results"""
        print("\n" + "="*50)
        print("🗣️ PRONUNCIATION PATTERNS ANALYSIS")
        print("="*50)
        
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return
        
        # Repeated phrases
        if results['repeated_phrases']:
            print(f"\n🔄 REPEATED PHRASES")
            print(f"   Found {len(results['repeated_phrases'])} repeated words/phrases:")
            for phrase in results['repeated_phrases'][:10]:  # Show top 10
                print(f"     - '{phrase['word']}': {phrase['count']} times ({phrase['frequency']:.1%})")
        else:
            print(f"\n🔄 REPEATED PHRASES")
            print(f"   No repeated phrases detected")
        
        # Word complexity
        complexity = results['word_complexity']
        print(f"\n📚 VOCABULARY ANALYSIS")
        print(f"   Complexity score: {complexity['complexity_score']:.3f}")
        print(f"   Vocabulary richness: {complexity['vocabulary_richness']:.3f}")
        print(f"   Average word length: {complexity['avg_word_length']:.1f} characters")
        print(f"   Complex word ratio: {complexity['complex_word_ratio']:.1%}")
        print(f"   Unique words: {complexity['unique_words']}/{complexity['total_words']}")
        
        # Sentence structure
        structure = results['sentence_structure']
        print(f"\n📝 SENTENCE STRUCTURE")
        print(f"   Sentence count: {structure['sentence_count']}")
        print(f"   Average sentence length: {structure['avg_sentence_length']:.1f} words")
        print(f"   Question ratio: {structure['question_ratio']:.1%}")
        print(f"   Questions: {structure['questions']}, Statements: {structure['statements']}")
        
        print("\n" + "="*50)
        print("✅ Pronunciation Analysis Complete!")
        print("="*50)
