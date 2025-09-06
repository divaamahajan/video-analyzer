#!/usr/bin/env python3
"""
Content Analyzer Module
======================

Advanced content and linguistic analysis focusing on communication effectiveness.
Complements existing transcription_analyzer.py by focusing on:

- Structure & Organization (sentence analysis, paragraph flow, transitions)
- Clarity & Conciseness (redundancy, wordiness, ambiguity, directness)
- Advanced Vocabulary Analysis (lexical diversity, technical vs simple, power words)
- Readability & Engagement (readability scores, rhetorical devices)
- Topic & Semantic Flow (main ideas, coherence, topic relevance)
- Cross-referencing with audio (alignment with pauses, emphasis)

Note: This complements existing transcription_analyzer.py which handles:
- Basic pace and rhythm (PaceAnalyzer)
- Basic pronunciation patterns (PronunciationAnalyzer) 
- Basic sentiment analysis (SentimentAnalyzer)
"""

import re
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import Counter
import textstat
from ...utils.ai_agent import BasicAIAgent


class ContentAnalyzer:
    """Advanced content and linguistic analysis focusing on communication effectiveness"""
    
    def __init__(self, api_key: str):
        """Initialize with OpenAI API key for AI analysis"""
        self.ai_agent = BasicAIAgent(api_key)
        
        # Advanced linguistic patterns
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
        
        self.ambiguous_words = [
            "thing", "stuff", "something", "somehow", "somewhat", "somewhere",
            "whatever", "whichever", "whenever", "wherever", "however", "whatever"
        ]
        
        self.redundant_phrases = [
            "each and every", "first and foremost", "null and void", "safe and sound",
            "tried and true", "ways and means", "bits and pieces", "odds and ends"
        ]
        
        self.hesitation_phrases = [
            "i guess", "i suppose", "i think", "i believe", "i feel", "i assume",
            "you know", "i mean", "well", "so", "um", "uh", "like"
        ]
    
    def analyze_content(self, transcription_result: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced content and linguistic analysis focusing on communication effectiveness"""
        print("📝 Analyzing Advanced Content and Linguistic Features")
        
        text = transcription_result["text"]
        segments = transcription_result.get("segments", [])
        
        # Basic text preprocessing
        processed_text = self._preprocess_text(text)
        
        # AI-powered analysis for advanced insights
        ai_analysis = self._analyze_with_ai(processed_text)
        
        # 1. Structure & Organization Analysis
        structure_analysis = self._analyze_structure_and_organization(processed_text, segments)
        
        # 2. Clarity & Conciseness Analysis
        clarity_analysis = self._analyze_clarity_and_conciseness(processed_text)
        
        # 3. Advanced Vocabulary Analysis
        vocabulary_analysis = self._analyze_advanced_vocabulary(processed_text)
        
        # 4. Readability & Engagement Analysis
        readability_analysis = self._analyze_readability_and_engagement(processed_text)
        
        # 5. Topic & Semantic Flow Analysis
        topic_flow_analysis = self._analyze_topic_and_semantic_flow(processed_text, segments)
        
        # 6. Cross-referencing with Audio (if available)
        audio_alignment_analysis = self._analyze_audio_alignment(transcription_result)
        
        return {
            "ai_analysis": ai_analysis,
            "structure_and_organization": structure_analysis,
            "clarity_and_conciseness": clarity_analysis,
            "advanced_vocabulary": vocabulary_analysis,
            "readability_and_engagement": readability_analysis,
            "topic_and_semantic_flow": topic_flow_analysis,
            "audio_alignment": audio_alignment_analysis,
            "raw_text": text,
            "processed_text": processed_text
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        return text
    
    def _analyze_with_ai(self, text: str) -> Dict[str, Any]:
        """Use AI agent for advanced content analysis"""
        print("🤖 Running AI-powered content analysis...")
        
        system_prompt = """You are an expert linguist and communication analyst. 
        You want to capture not just what someone says, but **how effectively, clearly, and engagingly they communicate**.
Analyze the following text for below features:

## **Content / Linguistic Features (Text Analysis)**

### **1. Structure & Organization**

* **Sentence Length**: average words per sentence, max/min, variance
* **Sentence Type**: simple, compound, complex
* **Paragraph/Idea Flow**:

  * Detect logical sequence (intro → point → support → conclusion)
  * Flag abrupt jumps or disconnected ideas
* **Transitions**: frequency of connectors like *“however,” “therefore,” “next”*
* **Opening & Closing Strength**: identify if start/end statements are strong or weak

---

### **2. Clarity & Conciseness**

* **Redundancy**: repeated words, phrases, or ideas
* **Wordiness**: unnecessary modifiers, long sentences that can be simplified
* **Ambiguity**: vague words like *“thing,” “stuff,” “somehow”*
* **Directness**: ratio of active voice vs passive voice

---

### **3. Fillers & Disfluencies**

* **Filler Words**: “um,” “ah,” “like,” “you know”

  * Count per sentence, per 100 words, per minute
* **Hesitation Phrases**: “I guess,” “maybe,” “probably”
* **Repetition**: repeated sentence starters or words

---

### **4. Vocabulary & Word Choice**

* **Lexical Diversity**: unique words / total words
* **Technical vs Simple Words**: detect jargon or overly complex terms
* **Positive/Negative Wording**: sentiment analysis
* **Power Words**: “must,” “will,” “can,” versus weak words like “probably”

---

### **5. Readability & Engagement**

* **Readability Scores**: Flesch-Kincaid, Gunning Fog index
* **Sentence Complexity Distribution**: % simple, compound, complex
* **Questions / Rhetorical Devices**: detect if speaker engages audience via questions
* **Emphasis Words**: frequency of strong emphasis or key-point phrases

---

### **6. Topic & Semantic Flow**

* **Main Ideas**: extract key topics or points
* **Supporting Evidence**: detect if examples/illustrations are provided
* **Topic Relevance**: flag off-topic sentences
* **Coherence**: semantic similarity between consecutive sentences (topic flow)

---

### **7. Sentiment & Confidence**

* **Confidence Markers**: “I believe,” “I am certain,” “we will”
* **Tentative Language**: “maybe,” “probably,” “I guess”
* **Emotion in Text**: tone indicators for excitement, seriousness, humor

---

### **8. Cross-Referencing Features**

* Combine with audio:

  * Are pauses aligned with transitions?
  * Do gestures match emphasis points?

---

### **Practical Output for ML Model**

Each recording could produce a **feature vector**, e.g.:

| Feature                    | Type    | Example / Notes   |
| -------------------------- | ------- | ----------------- |
| Avg. Sentence Length       | numeric | 12 words          |
| % Complex Sentences        | numeric | 35%               |
| Filler Words per 100 words | numeric | 7                 |
| Lexical Diversity          | numeric | 0.65              |
| Readability Score          | numeric | 60 Flesch-Kincaid |
| Sentiment Score            | numeric | +0.2 positive     |
| Topic Coherence            | numeric | 0–1               |
| Confidence Word Frequency  | numeric | 5 occurrences     |
| Redundancy Count           | numeric | 3 repeated ideas  |
Provide specific, actionable insights with examples from the text."""
        
        try:
            response = self.ai_agent.generate_response(
                prompt=f"Analyze this speech content for communication effectiveness: {text}",
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=1000
            )
            
            return {
                "ai_insights": response,
                "analysis_complete": True
            }
        except Exception as e:
            print(f"⚠️ AI analysis failed: {e}")
            return {
                "ai_insights": "AI analysis unavailable",
                "analysis_complete": False,
                "error": str(e)
            }
    
    def _analyze_structure_and_organization(self, text: str, segments: List) -> Dict[str, Any]:
        """Analyze structure and organization patterns"""
        sentences = self._split_into_sentences(text)
        
        # Sentence length analysis
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        
        # Sentence type analysis
        sentence_type_analysis = self._analyze_sentence_types(sentences)
        
        # Paragraph/idea flow analysis
        flow_analysis = self._analyze_idea_flow(sentences, segments)
        
        # Transition analysis
        transition_analysis = self._analyze_transitions(text)
        
        # Opening and closing strength
        opening_closing_analysis = self._analyze_opening_closing_strength(sentences)
        
        return {
            "sentence_length": {
                "average": np.mean(sentence_lengths) if sentence_lengths else 0,
                "max": max(sentence_lengths) if sentence_lengths else 0,
                "min": min(sentence_lengths) if sentence_lengths else 0,
                "variance": np.var(sentence_lengths) if sentence_lengths else 0,
                "distribution": {
                    "short": len([l for l in sentence_lengths if l < 10]),
                    "medium": len([l for l in sentence_lengths if 10 <= l < 20]),
                    "long": len([l for l in sentence_lengths if l >= 20])
                }
            },
            "sentence_types": sentence_type_analysis,
            "idea_flow": flow_analysis,
            "transitions": transition_analysis,
            "opening_closing": opening_closing_analysis,
            "total_sentences": len(sentences)
        }
    
    def _analyze_sentence_types(self, sentences: List[str]) -> Dict[str, Any]:
        """Analyze sentence types: simple, compound, complex"""
        simple_count = 0
        compound_count = 0
        complex_count = 0
        
        conjunctions = ["and", "but", "or", "so", "yet", "for", "nor"]
        subordinating = ["because", "although", "while", "if", "when", "where", "since", "as", "unless", "until"]
        
        for sentence in sentences:
            words = sentence.lower().split()
            
            # Count conjunctions and subordinating conjunctions
            conjunction_count = sum(1 for word in words if word in conjunctions)
            subordinating_count = sum(1 for word in words if word in subordinating)
            
            if subordinating_count > 0:
                complex_count += 1
            elif conjunction_count > 0:
                compound_count += 1
            else:
                simple_count += 1
        
        total = len(sentences)
        return {
            "simple": simple_count,
            "compound": compound_count,
            "complex": complex_count,
            "percentages": {
                "simple": (simple_count / total * 100) if total > 0 else 0,
                "compound": (compound_count / total * 100) if total > 0 else 0,
                "complex": (complex_count / total * 100) if total > 0 else 0
            }
        }
    
    def _analyze_idea_flow(self, sentences: List[str], segments: List) -> Dict[str, Any]:
        """Analyze logical sequence and idea flow"""
        # Basic flow indicators
        intro_indicators = ["first", "let me start", "beginning", "introduction", "overview"]
        transition_indicators = ["next", "then", "furthermore", "additionally", "moreover"]
        conclusion_indicators = ["finally", "in conclusion", "to summarize", "overall", "in summary"]
        
        intro_count = sum(1 for sentence in sentences 
                         for indicator in intro_indicators 
                         if indicator in sentence.lower())
        
        transition_count = sum(1 for sentence in sentences 
                              for indicator in transition_indicators 
                              if indicator in sentence.lower())
        
        conclusion_count = sum(1 for sentence in sentences 
                              for indicator in conclusion_indicators 
                              if indicator in sentence.lower())
        
        # Detect abrupt jumps (sentences that don't connect well)
        abrupt_jumps = self._detect_abrupt_jumps(sentences)
        
        return {
            "intro_indicators": intro_count,
            "transition_indicators": transition_count,
            "conclusion_indicators": conclusion_count,
            "abrupt_jumps": abrupt_jumps,
            "flow_quality": "good" if abrupt_jumps < 2 else "fair" if abrupt_jumps < 4 else "poor"
        }
    
    def _detect_abrupt_jumps(self, sentences: List[str]) -> int:
        """Detect sentences that don't connect well with previous ones"""
        abrupt_jumps = 0
        
        for i in range(1, len(sentences)):
            prev_sentence = sentences[i-1].lower()
            curr_sentence = sentences[i].lower()
            
            # Check for topic changes without transitions
            prev_words = set(prev_sentence.split())
            curr_words = set(curr_sentence.split())
            
            # If very few words overlap and no transition words, it might be abrupt
            overlap = len(prev_words.intersection(curr_words))
            if overlap < 2 and not any(transition in curr_sentence for transition in self.transition_words):
                abrupt_jumps += 1
        
        return abrupt_jumps
    
    def _analyze_opening_closing_strength(self, sentences: List[str]) -> Dict[str, Any]:
        """Analyze strength of opening and closing statements"""
        if not sentences:
            return {"opening_strength": 0, "closing_strength": 0}
        
        opening = sentences[0].lower()
        closing = sentences[-1].lower()
        
        # Opening strength indicators
        opening_strong = ["i will", "we will", "let me", "today i", "i want to", "the purpose"]
        opening_weak = ["maybe", "i think", "i guess", "perhaps", "i suppose"]
        
        opening_strength = sum(1 for indicator in opening_strong if indicator in opening)
        opening_weakness = sum(1 for indicator in opening_weak if indicator in opening)
        
        # Closing strength indicators
        closing_strong = ["in conclusion", "to summarize", "finally", "overall", "in summary"]
        closing_weak = ["i guess", "maybe", "i think", "perhaps", "i suppose"]
        
        closing_strength = sum(1 for indicator in closing_strong if indicator in closing)
        closing_weakness = sum(1 for indicator in closing_weak if indicator in closing)
        
        return {
            "opening_strength": max(0, opening_strength - opening_weakness),
            "closing_strength": max(0, closing_strength - closing_weakness),
            "opening_quality": "strong" if opening_strength > opening_weakness else "weak",
            "closing_quality": "strong" if closing_strength > closing_weakness else "weak"
        }
    
    def _analyze_clarity_and_conciseness(self, text: str) -> Dict[str, Any]:
        """Analyze clarity and conciseness patterns"""
        words = text.lower().split()
        sentences = self._split_into_sentences(text)
        
        # Redundancy analysis
        redundancy_analysis = self._analyze_redundancy(text, words)
        
        # Wordiness analysis
        wordiness_analysis = self._analyze_wordiness(sentences)
        
        # Ambiguity analysis
        ambiguity_analysis = self._analyze_ambiguity(text, words)
        
        # Directness analysis (active vs passive voice)
        directness_analysis = self._analyze_directness(sentences)
        
        return {
            "redundancy": redundancy_analysis,
            "wordiness": wordiness_analysis,
            "ambiguity": ambiguity_analysis,
            "directness": directness_analysis,
            "clarity_score": self._calculate_clarity_score(redundancy_analysis, wordiness_analysis, ambiguity_analysis, directness_analysis)
        }
    
    def _analyze_redundancy(self, text: str, words: List[str]) -> Dict[str, Any]:
        """Analyze redundant words, phrases, and ideas"""
        # Redundant phrases
        redundant_phrase_count = 0
        for phrase in self.redundant_phrases:
            redundant_phrase_count += text.lower().count(phrase)
        
        # Word repetition
        word_counts = Counter(words)
        repeated_words = {word: count for word, count in word_counts.items() if count > 1}
        
        # Idea repetition (basic heuristic)
        sentences = self._split_into_sentences(text)
        idea_repetition = self._detect_idea_repetition(sentences)
        
        return {
            "redundant_phrases": redundant_phrase_count,
            "repeated_words": len(repeated_words),
            "most_repeated_words": sorted(repeated_words.items(), key=lambda x: x[1], reverse=True)[:5],
            "idea_repetition": idea_repetition,
            "redundancy_level": "high" if redundant_phrase_count > 3 or len(repeated_words) > 10 else "medium" if redundant_phrase_count > 1 or len(repeated_words) > 5 else "low"
        }
    
    def _detect_idea_repetition(self, sentences: List[str]) -> int:
        """Detect repeated ideas across sentences"""
        # Simple heuristic: look for similar sentence structures and key words
        idea_repetition = 0
        
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                sentence1 = sentences[i].lower().split()
                sentence2 = sentences[j].lower().split()
                
                # Check for high word overlap
                overlap = len(set(sentence1).intersection(set(sentence2)))
                if overlap > len(sentence1) * 0.6:  # 60% overlap
                    idea_repetition += 1
        
        return idea_repetition
    
    def _analyze_wordiness(self, sentences: List[str]) -> Dict[str, Any]:
        """Analyze wordiness and unnecessary modifiers"""
        wordy_sentences = 0
        unnecessary_modifiers = 0
        
        # Common unnecessary modifiers
        modifiers = ["very", "really", "quite", "rather", "somewhat", "pretty", "fairly", "extremely", "incredibly"]
        
        for sentence in sentences:
            words = sentence.lower().split()
            
            # Count unnecessary modifiers
            modifier_count = sum(1 for word in words if word in modifiers)
            unnecessary_modifiers += modifier_count
            
            # Check for wordy patterns
            if len(words) > 25:  # Long sentences
                wordy_sentences += 1
            
            # Check for redundant phrases within sentence
            for phrase in self.redundant_phrases:
                if phrase in sentence.lower():
                    wordy_sentences += 1
        
        return {
            "wordy_sentences": wordy_sentences,
            "unnecessary_modifiers": unnecessary_modifiers,
            "wordiness_level": "high" if wordy_sentences > 3 or unnecessary_modifiers > 5 else "medium" if wordy_sentences > 1 or unnecessary_modifiers > 2 else "low"
        }
    
    def _analyze_ambiguity(self, text: str, words: List[str]) -> Dict[str, Any]:
        """Analyze ambiguous words and phrases"""
        ambiguous_count = 0
        ambiguous_words_found = []
        
        for word in self.ambiguous_words:
            count = text.lower().count(word)
            if count > 0:
                ambiguous_count += count
                ambiguous_words_found.append((word, count))
        
        return {
            "ambiguous_words": ambiguous_count,
            "ambiguous_words_found": ambiguous_words_found,
            "ambiguity_level": "high" if ambiguous_count > 5 else "medium" if ambiguous_count > 2 else "low"
        }
    
    def _analyze_directness(self, sentences: List[str]) -> Dict[str, Any]:
        """Analyze directness (active vs passive voice)"""
        active_voice_indicators = ["i", "we", "you", "they", "he", "she", "it"]
        passive_voice_indicators = ["was", "were", "been", "being", "is", "are", "am"]
        
        active_sentences = 0
        passive_sentences = 0
        
        for sentence in sentences:
            words = sentence.lower().split()
            
            # Simple heuristic: count active vs passive indicators
            active_count = sum(1 for word in words if word in active_voice_indicators)
            passive_count = sum(1 for word in words if word in passive_voice_indicators)
            
            if passive_count > active_count:
                passive_sentences += 1
            else:
                active_sentences += 1
        
        total = len(sentences)
        return {
            "active_sentences": active_sentences,
            "passive_sentences": passive_sentences,
            "active_ratio": active_sentences / total if total > 0 else 0,
            "passive_ratio": passive_sentences / total if total > 0 else 0,
            "directness_level": "high" if active_sentences > passive_sentences else "low"
        }
    
    def _calculate_clarity_score(self, redundancy: Dict, wordiness: Dict, ambiguity: Dict, directness: Dict) -> float:
        """Calculate overall clarity score"""
        # Start with perfect score
        score = 1.0
        
        # Penalize redundancy
        if redundancy["redundancy_level"] == "high":
            score -= 0.3
        elif redundancy["redundancy_level"] == "medium":
            score -= 0.15
        
        # Penalize wordiness
        if wordiness["wordiness_level"] == "high":
            score -= 0.2
        elif wordiness["wordiness_level"] == "medium":
            score -= 0.1
        
        # Penalize ambiguity
        if ambiguity["ambiguity_level"] == "high":
            score -= 0.2
        elif ambiguity["ambiguity_level"] == "medium":
            score -= 0.1
        
        # Reward directness
        if directness["directness_level"] == "high":
            score += 0.1
        
        return max(0, min(1, score))
    
    def _analyze_advanced_vocabulary(self, text: str) -> Dict[str, Any]:
        """Analyze advanced vocabulary patterns"""
        words = [word.lower().strip('.,!?;:') for word in text.split()]
        unique_words = set(words)
        
        # Basic vocabulary metrics
        total_words = len(words)
        unique_count = len(unique_words)
        lexical_diversity = unique_count / total_words if total_words > 0 else 0
        
        # Technical vs simple words analysis
        technical_analysis = self._analyze_technical_vs_simple(words)
        
        # Power words analysis
        power_word_analysis = self._analyze_power_words(text, words)
        
        # Word length analysis
        word_length_analysis = self._analyze_word_lengths(words)
        
        # Vocabulary sophistication
        sophistication_analysis = self._analyze_vocabulary_sophistication(words)
        
        return {
            "lexical_diversity": lexical_diversity,
            "total_words": total_words,
            "unique_words": unique_count,
            "technical_vs_simple": technical_analysis,
            "power_words": power_word_analysis,
            "word_lengths": word_length_analysis,
            "sophistication": sophistication_analysis,
            "vocabulary_level": self._determine_vocabulary_level(lexical_diversity, technical_analysis, sophistication_analysis)
        }
    
    def _analyze_technical_vs_simple(self, words: List[str]) -> Dict[str, Any]:
        """Analyze technical vs simple word usage"""
        # Technical word indicators (basic heuristic)
        technical_indicators = [
            "analysis", "implementation", "methodology", "framework", "algorithm",
            "optimization", "configuration", "infrastructure", "architecture", "protocol",
            "sophisticated", "comprehensive", "systematic", "strategic", "operational"
        ]
        
        # Simple word indicators
        simple_indicators = [
            "good", "bad", "big", "small", "nice", "easy", "hard", "fun", "cool",
            "thing", "stuff", "way", "time", "place", "person", "people", "work"
        ]
        
        technical_count = sum(1 for word in words if word in technical_indicators)
        simple_count = sum(1 for word in words if word in simple_indicators)
        
        # Word length as complexity indicator
        long_words = [word for word in words if len(word) > 8]
        short_words = [word for word in words if len(word) <= 4]
        
        return {
            "technical_words": technical_count,
            "simple_words": simple_count,
            "technical_ratio": technical_count / len(words) if words else 0,
            "simple_ratio": simple_count / len(words) if words else 0,
            "long_words": len(long_words),
            "short_words": len(short_words),
            "complexity_ratio": len(long_words) / len(words) if words else 0
        }
    
    def _analyze_power_words(self, text: str, words: List[str]) -> Dict[str, Any]:
        """Analyze power words vs weak words"""
        power_count = 0
        weak_count = 0
        
        for word in self.power_words:
            power_count += text.lower().count(word)
        
        for word in self.weak_words:
            weak_count += text.lower().count(word)
        
        return {
            "power_words": power_count,
            "weak_words": weak_count,
            "power_ratio": power_count / len(words) if words else 0,
            "weak_ratio": weak_count / len(words) if words else 0,
            "assertiveness_score": (power_count - weak_count) / max(len(words), 1)
        }
    
    def _analyze_word_lengths(self, words: List[str]) -> Dict[str, Any]:
        """Analyze word length distribution"""
        word_lengths = [len(word) for word in words]
        
        return {
            "average_length": np.mean(word_lengths) if word_lengths else 0,
            "max_length": max(word_lengths) if word_lengths else 0,
            "min_length": min(word_lengths) if word_lengths else 0,
            "length_distribution": {
                "1-3_chars": len([l for l in word_lengths if l <= 3]),
                "4-6_chars": len([l for l in word_lengths if 4 <= l <= 6]),
                "7-9_chars": len([l for l in word_lengths if 7 <= l <= 9]),
                "10+_chars": len([l for l in word_lengths if l >= 10])
            }
        }
    
    def _analyze_vocabulary_sophistication(self, words: List[str]) -> Dict[str, Any]:
        """Analyze vocabulary sophistication"""
        # Sophisticated word patterns
        sophisticated_patterns = [
            # Multi-syllable words
            len([word for word in words if len(word) > 8]),
            # Words with prefixes/suffixes
            len([word for word in words if any(prefix in word for prefix in ['un', 're', 'pre', 'anti', 'pro'])]),
            # Abstract concepts
            len([word for word in words if word in ['concept', 'principle', 'theory', 'philosophy', 'methodology']])
        ]
        
        sophistication_score = sum(sophisticated_patterns) / len(words) if words else 0
        
        return {
            "sophistication_score": sophistication_score,
            "sophistication_level": "high" if sophistication_score > 0.1 else "medium" if sophistication_score > 0.05 else "low"
        }
    
    def _determine_vocabulary_level(self, lexical_diversity: float, technical_analysis: Dict, sophistication_analysis: Dict) -> str:
        """Determine overall vocabulary level"""
        score = 0
        
        # Lexical diversity contribution
        if lexical_diversity > 0.8:
            score += 3
        elif lexical_diversity > 0.6:
            score += 2
        elif lexical_diversity > 0.4:
            score += 1
        
        # Technical word contribution
        if technical_analysis["technical_ratio"] > 0.1:
            score += 2
        elif technical_analysis["technical_ratio"] > 0.05:
            score += 1
        
        # Sophistication contribution
        if sophistication_analysis["sophistication_level"] == "high":
            score += 2
        elif sophistication_analysis["sophistication_level"] == "medium":
            score += 1
        
        if score >= 6:
            return "advanced"
        elif score >= 3:
            return "intermediate"
        else:
            return "basic"
    
    def _analyze_readability_and_engagement(self, text: str) -> Dict[str, Any]:
        """Analyze readability and engagement patterns"""
        sentences = self._split_into_sentences(text)
        
        # Readability scores
        readability_scores = self._calculate_readability_scores(text)
        
        # Sentence complexity distribution
        complexity_distribution = self._analyze_sentence_complexity_distribution(sentences)
        
        # Rhetorical devices
        rhetorical_devices = self._analyze_rhetorical_devices(text)
        
        # Emphasis words
        emphasis_analysis = self._analyze_emphasis_words(text)
        
        return {
            "readability_scores": readability_scores,
            "complexity_distribution": complexity_distribution,
            "rhetorical_devices": rhetorical_devices,
            "emphasis_words": emphasis_analysis,
            "engagement_score": self._calculate_engagement_score(
                rhetorical_devices["questions"] / len(sentences) if sentences else 0,
                rhetorical_devices["exclamations"] / len(sentences) if sentences else 0,
                0
            )
        }
    
    def _calculate_readability_scores(self, text: str) -> Dict[str, Any]:
        """Calculate various readability scores"""
        try:
            flesch_score = textstat.flesch_reading_ease(text)
            flesch_kincaid = textstat.flesch_kincaid_grade(text)
            gunning_fog = textstat.gunning_fog(text)
        except:
            flesch_score = 0
            flesch_kincaid = 0
            gunning_fog = 0
        
        return {
            "flesch_reading_ease": flesch_score,
            "flesch_kincaid_grade": flesch_kincaid,
            "gunning_fog_index": gunning_fog,
            "readability_level": self._get_readability_level(flesch_score)
        }
    
    def _analyze_sentence_complexity_distribution(self, sentences: List[str]) -> Dict[str, Any]:
        """Analyze sentence complexity distribution"""
        simple_count = 0
        compound_count = 0
        complex_count = 0
        
        conjunctions = ["and", "but", "or", "so", "yet", "for", "nor"]
        subordinating = ["because", "although", "while", "if", "when", "where", "since", "as", "unless", "until"]
        
        for sentence in sentences:
            words = sentence.lower().split()
            
            conjunction_count = sum(1 for word in words if word in conjunctions)
            subordinating_count = sum(1 for word in words if word in subordinating)
            
            if subordinating_count > 0:
                complex_count += 1
            elif conjunction_count > 0:
                compound_count += 1
            else:
                simple_count += 1
        
        total = len(sentences)
        return {
            "simple": simple_count,
            "compound": compound_count,
            "complex": complex_count,
            "percentages": {
                "simple": (simple_count / total * 100) if total > 0 else 0,
                "compound": (compound_count / total * 100) if total > 0 else 0,
                "complex": (complex_count / total * 100) if total > 0 else 0
            }
        }
    
    def _analyze_rhetorical_devices(self, text: str) -> Dict[str, Any]:
        """Analyze rhetorical devices"""
        # Questions
        questions = len(re.findall(r'\?', text))
        
        # Exclamations
        exclamations = len(re.findall(r'!', text))
        
        # Repetition patterns
        repetition = len(re.findall(r'\b(\w+)\s+\1\b', text.lower()))
        
        # Alliteration
        alliteration = len(re.findall(r'\b(\w)\w*\s+\1\w*\b', text.lower()))
        
        # Parallel structure
        parallel_structure = len(re.findall(r'\b(\w+)\s+and\s+\1\b', text.lower()))
        
        return {
            "questions": questions,
            "exclamations": exclamations,
            "repetition": repetition,
            "alliteration": alliteration,
            "parallel_structure": parallel_structure,
            "total_devices": questions + exclamations + repetition + alliteration + parallel_structure
        }
    
    def _analyze_emphasis_words(self, text: str) -> Dict[str, Any]:
        """Analyze emphasis words and phrases"""
        emphasis_words = [
            "important", "critical", "essential", "vital", "crucial", "significant",
            "key", "major", "main", "primary", "fundamental", "core", "central"
        ]
        
        emphasis_count = 0
        emphasis_found = []
        
        for word in emphasis_words:
            count = text.lower().count(word)
            if count > 0:
                emphasis_count += count
                emphasis_found.append((word, count))
        
        return {
            "emphasis_count": emphasis_count,
            "emphasis_words_found": emphasis_found,
            "emphasis_frequency": emphasis_count / len(text.split()) if text.split() else 0
        }
    
    def _calculate_engagement_score(self, rhetorical_devices: Dict, emphasis_analysis: Dict) -> float:
        """Calculate overall engagement score"""
        score = 0
        
        # Questions increase engagement
        score += min(rhetorical_devices["questions"] * 0.1, 0.3)
        
        # Exclamations increase engagement
        score += min(rhetorical_devices["exclamations"] * 0.1, 0.2)
        
        # Rhetorical devices increase engagement
        score += min(rhetorical_devices["total_devices"] * 0.05, 0.2)
        
        # Emphasis words increase engagement
        score += min(emphasis_analysis["emphasis_frequency"] * 2, 0.3)
        
        return min(1.0, score)
    
    def _get_readability_level(self, flesch_score: float) -> str:
        """Convert Flesch score to readability level"""
        if flesch_score >= 90:
            return "very_easy"
        elif flesch_score >= 80:
            return "easy"
        elif flesch_score >= 70:
            return "fairly_easy"
        elif flesch_score >= 60:
            return "standard"
        elif flesch_score >= 50:
            return "fairly_difficult"
        elif flesch_score >= 30:
            return "difficult"
        else:
            return "very_difficult"
    
    def _analyze_topic_and_semantic_flow(self, text: str, segments: List) -> Dict[str, Any]:
        """Analyze topic and semantic flow patterns"""
        sentences = self._split_into_sentences(text)
        
        # Main ideas extraction
        main_ideas = self._extract_main_ideas(sentences)
        
        # Supporting evidence detection
        supporting_evidence = self._detect_supporting_evidence(sentences)
        
        # Topic relevance analysis
        topic_relevance = self._analyze_topic_relevance(sentences)
        
        # Coherence analysis
        coherence_analysis = self._analyze_coherence(sentences)
        
        return {
            "main_ideas": main_ideas,
            "supporting_evidence": supporting_evidence,
            "topic_relevance": topic_relevance,
            "coherence": coherence_analysis,
            "flow_quality": self._assess_flow_quality(main_ideas, supporting_evidence, topic_relevance, coherence_analysis)
        }
    
    def _extract_main_ideas(self, sentences: List[str]) -> Dict[str, Any]:
        """Extract main ideas and topics"""
        # Key topic indicators
        topic_indicators = [
            "main", "primary", "key", "important", "essential", "critical",
            "first", "second", "third", "point", "idea", "concept"
        ]
        
        main_idea_sentences = []
        for i, sentence in enumerate(sentences):
            if any(indicator in sentence.lower() for indicator in topic_indicators):
                main_idea_sentences.append((i, sentence))
        
        return {
            "main_idea_sentences": main_idea_sentences,
            "main_idea_count": len(main_idea_sentences),
            "main_idea_ratio": len(main_idea_sentences) / len(sentences) if sentences else 0
        }
    
    def _detect_supporting_evidence(self, sentences: List[str]) -> Dict[str, Any]:
        """Detect supporting evidence and examples"""
        # Evidence indicators
        evidence_indicators = [
            "for example", "for instance", "such as", "like", "including",
            "specifically", "namely", "in particular", "to illustrate"
        ]
        
        # Example indicators
        example_indicators = [
            "example", "instance", "case", "scenario", "situation", "scenario"
        ]
        
        evidence_sentences = []
        example_sentences = []
        
        for i, sentence in enumerate(sentences):
            if any(indicator in sentence.lower() for indicator in evidence_indicators):
                evidence_sentences.append((i, sentence))
            if any(indicator in sentence.lower() for indicator in example_indicators):
                example_sentences.append((i, sentence))
        
        return {
            "evidence_sentences": evidence_sentences,
            "example_sentences": example_sentences,
            "evidence_count": len(evidence_sentences),
            "example_count": len(example_sentences),
            "support_ratio": (len(evidence_sentences) + len(example_sentences)) / len(sentences) if sentences else 0
        }
    
    def _analyze_topic_relevance(self, sentences: List[str]) -> Dict[str, Any]:
        """Analyze topic relevance and detect off-topic sentences"""
        # Calculate word overlap between sentences to detect topic shifts
        topic_shifts = 0
        off_topic_sentences = []
        
        for i in range(1, len(sentences)):
            prev_words = set(sentences[i-1].lower().split())
            curr_words = set(sentences[i].lower().split())
            
            # Calculate word overlap
            overlap = len(prev_words.intersection(curr_words))
            total_words = len(prev_words.union(curr_words))
            
            # If overlap is very low, it might be off-topic
            if total_words > 0 and overlap / total_words < 0.1:
                topic_shifts += 1
                off_topic_sentences.append((i, sentences[i]))
        
        return {
            "topic_shifts": topic_shifts,
            "off_topic_sentences": off_topic_sentences,
            "relevance_score": 1 - (topic_shifts / len(sentences)) if sentences else 1
        }
    
    def _analyze_coherence(self, sentences: List[str]) -> Dict[str, Any]:
        """Analyze semantic coherence between sentences"""
        coherence_scores = []
        
        for i in range(1, len(sentences)):
            prev_words = set(sentences[i-1].lower().split())
            curr_words = set(sentences[i].lower().split())
            
            # Calculate semantic similarity (simple word overlap)
            overlap = len(prev_words.intersection(curr_words))
            total_words = len(prev_words.union(curr_words))
            
            if total_words > 0:
                coherence_score = overlap / total_words
                coherence_scores.append(coherence_score)
        
        return {
            "coherence_scores": coherence_scores,
            "average_coherence": np.mean(coherence_scores) if coherence_scores else 0,
            "coherence_consistency": 1 - np.std(coherence_scores) if coherence_scores else 0
        }
    
    def _assess_flow_quality(self, main_ideas: Dict, supporting_evidence: Dict, topic_relevance: Dict, coherence: Dict) -> str:
        """Assess overall flow quality"""
        score = 0
        
        # Main ideas contribute to flow
        if main_ideas["main_idea_ratio"] > 0.3:
            score += 2
        elif main_ideas["main_idea_ratio"] > 0.1:
            score += 1
        
        # Supporting evidence contributes to flow
        if supporting_evidence["support_ratio"] > 0.2:
            score += 2
        elif supporting_evidence["support_ratio"] > 0.1:
            score += 1
        
        # Topic relevance contributes to flow
        if topic_relevance["relevance_score"] > 0.8:
            score += 2
        elif topic_relevance["relevance_score"] > 0.6:
            score += 1
        
        # Coherence contributes to flow
        if coherence["average_coherence"] > 0.3:
            score += 2
        elif coherence["average_coherence"] > 0.2:
            score += 1
        
        if score >= 6:
            return "excellent"
        elif score >= 4:
            return "good"
        elif score >= 2:
            return "fair"
        else:
            return "poor"
    
    def _analyze_audio_alignment(self, transcription_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze alignment between text and audio features"""
        segments = transcription_result.get("segments", [])
        
        if not segments:
            return {"error": "No segments available for audio alignment analysis"}
        
        # Analyze pause alignment with transitions
        pause_transition_alignment = self._analyze_pause_transition_alignment(segments)
        
        # Analyze emphasis alignment
        emphasis_alignment = self._analyze_emphasis_alignment(segments)
        
        return {
            "pause_transition_alignment": pause_transition_alignment,
            "emphasis_alignment": emphasis_alignment,
            "alignment_quality": self._assess_alignment_quality(pause_transition_alignment, emphasis_alignment)
        }
    
    def _analyze_pause_transition_alignment(self, segments: List) -> Dict[str, Any]:
        """Analyze if pauses align with transitions"""
        transition_aligned_pauses = 0
        total_transitions = 0
        
        for i, segment in enumerate(segments):
            text = segment.text.lower()
            
            # Check if segment contains transition words
            has_transition = any(transition in text for transition in self.transition_words)
            if has_transition:
                total_transitions += 1
                
                # Check if there's a pause after this segment
                if i < len(segments) - 1:
                    next_segment = segments[i + 1]
                    pause_duration = next_segment.start - segment.end
                    
                    # If pause is longer than 0.5 seconds, it's aligned
                    if pause_duration > 0.5:
                        transition_aligned_pauses += 1
        
        return {
            "transition_aligned_pauses": transition_aligned_pauses,
            "total_transitions": total_transitions,
            "alignment_ratio": transition_aligned_pauses / max(total_transitions, 1)
        }
    
    def _analyze_emphasis_alignment(self, segments: List) -> Dict[str, Any]:
        """Analyze if emphasis words align with audio emphasis"""
        # This is a simplified analysis - in a real implementation, you'd analyze audio features
        emphasis_words = [
            "important", "critical", "essential", "vital", "crucial", "significant",
            "key", "major", "main", "primary", "fundamental", "core", "central"
        ]
        
        emphasis_segments = []
        for segment in segments:
            if any(word in segment.text.lower() for word in emphasis_words):
                emphasis_segments.append(segment)
        
        return {
            "emphasis_segments": len(emphasis_segments),
            "emphasis_frequency": len(emphasis_segments) / len(segments) if segments else 0
        }
    
    def _assess_alignment_quality(self, pause_alignment: Dict, emphasis_alignment: Dict) -> str:
        """Assess overall alignment quality"""
        score = 0
        
        # Pause alignment contributes to quality
        if pause_alignment["alignment_ratio"] > 0.7:
            score += 2
        elif pause_alignment["alignment_ratio"] > 0.4:
            score += 1
        
        # Emphasis alignment contributes to quality
        if emphasis_alignment["emphasis_frequency"] > 0.1:
            score += 1
        
        if score >= 3:
            return "excellent"
        elif score >= 2:
            return "good"
        elif score >= 1:
            return "fair"
        else:
            return "poor"
    
    def _analyze_fillers(self, text: str, words: List[str]) -> Dict[str, Any]:
        """Analyze filler word usage"""
        filler_counts = {}
        filler_positions = []
        
        for filler in self.filler_words:
            count = text.lower().count(filler)
            if count > 0:
                filler_counts[filler] = count
                # Find positions
                start = 0
                while True:
                    pos = text.lower().find(filler, start)
                    if pos == -1:
                        break
                    filler_positions.append({
                        "word": filler,
                        "position": pos,
                        "context": text[max(0, pos-20):pos+20]
                    })
                    start = pos + 1
        
        total_fillers = sum(filler_counts.values())
        filler_frequency = (total_fillers / len(words)) * 100 if words else 0
        
        return {
            "filler_counts": filler_counts,
            "total_fillers": total_fillers,
            "filler_frequency_per_100_words": filler_frequency,
            "filler_positions": filler_positions,
            "filler_density": "high" if filler_frequency > 5 else "medium" if filler_frequency > 2 else "low"
        }
    
    def _analyze_weak_words(self, text: str, words: List[str]) -> Dict[str, Any]:
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
    
    def _analyze_confidence_markers(self, text: str, words: List[str]) -> Dict[str, Any]:
        """Analyze confidence and uncertainty markers"""
        confidence_count = 0
        uncertainty_count = 0
        
        for marker in self.confidence_markers:
            confidence_count += text.lower().count(marker)
        
        for marker in self.uncertainty_markers:
            uncertainty_count += text.lower().count(marker)
        
        total_markers = confidence_count + uncertainty_count
        confidence_ratio = confidence_count / max(uncertainty_count, 1) if uncertainty_count > 0 else confidence_count
        
        return {
            "confidence_markers": confidence_count,
            "uncertainty_markers": uncertainty_count,
            "confidence_ratio": confidence_ratio,
            "confidence_level": "high" if confidence_ratio > 2 else "medium" if confidence_ratio > 1 else "low"
        }
    
    def _analyze_repetition(self, words: List[str]) -> Dict[str, Any]:
        """Analyze word repetition patterns"""
        word_counts = Counter(words)
        
        # Find repeated words
        repeated_words = {word: count for word, count in word_counts.items() if count > 1}
        
        # Calculate repetition metrics
        total_words = len(words)
        unique_words = len(word_counts)
        repetition_ratio = len(repeated_words) / unique_words if unique_words > 0 else 0
        
        # Find most repeated words
        most_repeated = sorted(repeated_words.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "repeated_words": repeated_words,
            "most_repeated": most_repeated,
            "repetition_ratio": repetition_ratio,
            "vocabulary_diversity": unique_words / total_words if total_words > 0 else 0,
            "repetition_level": "high" if repetition_ratio > 0.3 else "medium" if repetition_ratio > 0.1 else "low"
        }
    
    def _analyze_transitions(self, text: str) -> Dict[str, Any]:
        """Analyze use of transition words and phrases"""
        transition_words = [
            "first", "second", "third", "next", "then", "finally",
            "however", "but", "although", "despite", "in contrast",
            "furthermore", "moreover", "additionally", "also",
            "therefore", "thus", "consequently", "as a result",
            "for example", "for instance", "specifically",
            "in conclusion", "to summarize", "overall"
        ]
        
        transition_counts = {}
        for transition in transition_words:
            count = text.lower().count(transition)
            if count > 0:
                transition_counts[transition] = count
        
        total_transitions = sum(transition_counts.values())
        
        return {
            "transition_counts": transition_counts,
            "total_transitions": total_transitions,
            "transition_usage": "good" if total_transitions > 3 else "limited"
        }
    
    def _analyze_structure(self, text: str, segments: List) -> Dict[str, Any]:
        """Analyze sentence and paragraph structure"""
        sentences = self._split_into_sentences(text)
        
        # Sentence length analysis
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        
        # Sentence complexity analysis
        complexity_analysis = self._analyze_sentence_complexity(sentences)
        
        # Paragraph structure (using segments as paragraphs)
        paragraph_analysis = self._analyze_paragraph_structure(segments)
        
        return {
            "sentence_count": len(sentences),
            "avg_sentence_length": np.mean(sentence_lengths) if sentence_lengths else 0,
            "sentence_length_std": np.std(sentence_lengths) if sentence_lengths else 0,
            "sentence_length_distribution": {
                "short": len([l for l in sentence_lengths if l < 10]),
                "medium": len([l for l in sentence_lengths if 10 <= l < 20]),
                "long": len([l for l in sentence_lengths if l >= 20])
            },
            "complexity": complexity_analysis,
            "paragraph_structure": paragraph_analysis
        }
    
    def _analyze_sentence_complexity(self, sentences: List[str]) -> Dict[str, Any]:
        """Analyze sentence complexity patterns"""
        simple_sentences = 0
        compound_sentences = 0
        complex_sentences = 0
        
        conjunctions = ["and", "but", "or", "so", "yet", "for", "nor"]
        subordinating = ["because", "although", "while", "if", "when", "where", "since"]
        
        for sentence in sentences:
            words = sentence.lower().split()
            
            # Count conjunctions
            conjunction_count = sum(1 for word in words if word in conjunctions)
            subordinating_count = sum(1 for word in words if word in subordinating)
            
            if conjunction_count > 0 and subordinating_count > 0:
                complex_sentences += 1
            elif conjunction_count > 0:
                compound_sentences += 1
            else:
                simple_sentences += 1
        
        total = len(sentences)
        return {
            "simple_sentences": simple_sentences,
            "compound_sentences": compound_sentences,
            "complex_sentences": complex_sentences,
            "complexity_ratio": {
                "simple": simple_sentences / total if total > 0 else 0,
                "compound": compound_sentences / total if total > 0 else 0,
                "complex": complex_sentences / total if total > 0 else 0
            }
        }
    
    def _analyze_paragraph_structure(self, segments: List) -> Dict[str, Any]:
        """Analyze paragraph structure using segments"""
        if not segments:
            return {"error": "No segments available"}
        
        segment_lengths = [len(seg.text.split()) for seg in segments]
        
        return {
            "paragraph_count": len(segments),
            "avg_paragraph_length": np.mean(segment_lengths),
            "paragraph_length_std": np.std(segment_lengths),
            "paragraph_consistency": 1 - (np.std(segment_lengths) / np.mean(segment_lengths)) if np.mean(segment_lengths) > 0 else 0
        }
    
    def _analyze_vocabulary(self, text: str) -> Dict[str, Any]:
        """Analyze vocabulary richness and sophistication"""
        words = [word.lower().strip('.,!?;:') for word in text.split()]
        unique_words = set(words)
        
        # Basic vocabulary metrics
        total_words = len(words)
        unique_count = len(unique_words)
        lexical_richness = unique_count / total_words if total_words > 0 else 0
        
        # Word length analysis
        word_lengths = [len(word) for word in words]
        avg_word_length = np.mean(word_lengths)
        
        # Technical vs simple words (basic heuristic)
        technical_words = [word for word in unique_words if len(word) > 8]
        simple_words = [word for word in unique_words if len(word) <= 4]
        
        technical_ratio = len(technical_words) / unique_count if unique_count > 0 else 0
        simple_ratio = len(simple_words) / unique_count if unique_count > 0 else 0
        
        return {
            "total_words": total_words,
            "unique_words": unique_count,
            "lexical_richness": lexical_richness,
            "avg_word_length": avg_word_length,
            "technical_words": len(technical_words),
            "simple_words": len(simple_words),
            "technical_ratio": technical_ratio,
            "simple_ratio": simple_ratio,
            "vocabulary_level": "advanced" if technical_ratio > 0.3 else "intermediate" if technical_ratio > 0.1 else "basic"
        }
    
    def _analyze_clarity(self, text: str) -> Dict[str, Any]:
        """Analyze clarity and conciseness"""
        sentences = self._split_into_sentences(text)
        
        # Readability scores
        try:
            flesch_score = textstat.flesch_reading_ease(text)
            flesch_kincaid = textstat.flesch_kincaid_grade(text)
        except:
            flesch_score = 0
            flesch_kincaid = 0
        
        # Passive voice detection (basic)
        passive_indicators = ["was", "were", "been", "being"]
        passive_count = sum(1 for sentence in sentences 
                           for word in sentence.lower().split() 
                           if word in passive_indicators)
        
        # Clarity metrics
        avg_sentence_length = np.mean([len(sentence.split()) for sentence in sentences])
        
        return {
            "flesch_reading_ease": flesch_score,
            "flesch_kincaid_grade": flesch_kincaid,
            "readability_level": self._get_readability_level(flesch_score),
            "passive_voice_indicators": passive_count,
            "avg_sentence_length": avg_sentence_length,
            "clarity_score": self._calculate_clarity_score(flesch_score, avg_sentence_length, passive_count)
        }
    
    def _analyze_engagement(self, text: str) -> Dict[str, Any]:
        """Analyze engagement and rhetorical devices"""
        sentences = self._split_into_sentences(text)
        
        # Question analysis
        questions = [s for s in sentences if s.strip().endswith('?')]
        question_ratio = len(questions) / len(sentences) if sentences else 0
        
        # Exclamation analysis
        exclamations = [s for s in sentences if s.strip().endswith('!')]
        exclamation_ratio = len(exclamations) / len(sentences) if sentences else 0
        
        # Direct address analysis
        direct_address = text.lower().count("you") + text.lower().count("your")
        
        # Rhetorical devices
        rhetorical_devices = self._detect_rhetorical_devices(text)
        
        return {
            "questions": len(questions),
            "question_ratio": question_ratio,
            "exclamations": len(exclamations),
            "exclamation_ratio": exclamation_ratio,
            "direct_address_count": direct_address,
            "rhetorical_devices": rhetorical_devices,
            "engagement_score": self._calculate_engagement_score(question_ratio, exclamation_ratio, direct_address)
        }
    
    def _detect_rhetorical_devices(self, text: str) -> Dict[str, int]:
        """Detect common rhetorical devices"""
        devices = {
            "repetition": len(re.findall(r'\b(\w+)\s+\1\b', text.lower())),
            "alliteration": len(re.findall(r'\b(\w)\w*\s+\1\w*\b', text.lower())),
            "parallel_structure": len(re.findall(r'\b(\w+)\s+and\s+\1\b', text.lower())),
            "rhetorical_questions": len(re.findall(r'\?', text))
        }
        return devices
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_readability_level(self, flesch_score: float) -> str:
        """Convert Flesch score to readability level"""
        if flesch_score >= 90:
            return "very_easy"
        elif flesch_score >= 80:
            return "easy"
        elif flesch_score >= 70:
            return "fairly_easy"
        elif flesch_score >= 60:
            return "standard"
        elif flesch_score >= 50:
            return "fairly_difficult"
        elif flesch_score >= 30:
            return "difficult"
        else:
            return "very_difficult"
    
    
    def _calculate_engagement_score(self, question_ratio: float, exclamation_ratio: float, direct_address: int) -> float:
        """Calculate engagement score"""
        # Questions and exclamations increase engagement
        engagement = (question_ratio * 0.4) + (exclamation_ratio * 0.3)
        
        # Direct address increases engagement
        engagement += min(direct_address * 0.01, 0.3)
        
        return min(1, engagement)
    
    def print_content_summary(self, results: Dict[str, Any]):
        """Print comprehensive content analysis summary"""
        print("\n" + "="*60)
        print("📝 ADVANCED CONTENT & LINGUISTIC ANALYSIS REPORT")
        print("="*60)
        
        # AI Analysis
        ai = results['ai_analysis']
        print(f"\n🤖 AI-POWERED INSIGHTS")
        print(f"   Analysis: {'✅ Complete' if ai['analysis_complete'] else '❌ Failed'}")
        if ai['analysis_complete']:
            print(f"   Insights: {ai['ai_insights']}")
        
        # Structure & Organization
        structure = results['structure_and_organization']
        print(f"\n📊 STRUCTURE & ORGANIZATION")
        print(f"   Total sentences: {structure['total_sentences']}")
        print(f"   Average sentence length: {structure['sentence_length']['average']:.1f} words")
        print(f"   Sentence length variance: {structure['sentence_length']['variance']:.1f}")
        print(f"   Sentence length distribution:")
        dist = structure['sentence_length']['distribution']
        print(f"     - Short (<10 words): {dist['short']}")
        print(f"     - Medium (10-20 words): {dist['medium']}")
        print(f"     - Long (>20 words): {dist['long']}")
        
        # Sentence types
        types = structure['sentence_types']
        print(f"   Sentence types:")
        print(f"     - Simple: {types['simple']} ({types['percentages']['simple']:.1f}%)")
        print(f"     - Compound: {types['compound']} ({types['percentages']['compound']:.1f}%)")
        print(f"     - Complex: {types['complex']} ({types['percentages']['complex']:.1f}%)")
        
        # Idea flow
        flow = structure['idea_flow']
        print(f"   Idea flow quality: {flow['flow_quality']}")
        print(f"   Intro indicators: {flow['intro_indicators']}")
        print(f"   Transition indicators: {flow['transition_indicators']}")
        print(f"   Conclusion indicators: {flow['conclusion_indicators']}")
        print(f"   Abrupt jumps: {flow['abrupt_jumps']}")
        
        # Opening/Closing strength
        opening_closing = structure['opening_closing']
        print(f"   Opening quality: {opening_closing['opening_quality']}")
        print(f"   Closing quality: {opening_closing['closing_quality']}")
        
        # Clarity & Conciseness
        clarity = results['clarity_and_conciseness']
        print(f"\n🔍 CLARITY & CONCISENESS")
        print(f"   Overall clarity score: {clarity['clarity_score']:.3f}")
        
        # Redundancy
        redundancy = clarity['redundancy']
        print(f"   Redundancy level: {redundancy['redundancy_level']}")
        print(f"   Redundant phrases: {redundancy['redundant_phrases']}")
        print(f"   Repeated words: {redundancy['repeated_words']}")
        print(f"   Idea repetition: {redundancy['idea_repetition']}")
        
        # Wordiness
        wordiness = clarity['wordiness']
        print(f"   Wordiness level: {wordiness['wordiness_level']}")
        print(f"   Wordy sentences: {wordiness['wordy_sentences']}")
        print(f"   Unnecessary modifiers: {wordiness['unnecessary_modifiers']}")
        
        # Ambiguity
        ambiguity = clarity['ambiguity']
        print(f"   Ambiguity level: {ambiguity['ambiguity_level']}")
        print(f"   Ambiguous words: {ambiguity['ambiguous_words']}")
        
        # Directness
        directness = clarity['directness']
        print(f"   Directness level: {directness['directness_level']}")
        print(f"   Active sentences: {directness['active_sentences']} ({directness['active_ratio']:.1%})")
        print(f"   Passive sentences: {directness['passive_sentences']} ({directness['passive_ratio']:.1%})")
        
        # Advanced Vocabulary
        vocab = results['advanced_vocabulary']
        print(f"\n📚 ADVANCED VOCABULARY")
        print(f"   Vocabulary level: {vocab['vocabulary_level']}")
        print(f"   Lexical diversity: {vocab['lexical_diversity']:.3f}")
        print(f"   Total words: {vocab['total_words']}")
        print(f"   Unique words: {vocab['unique_words']}")
        
        # Technical vs Simple
        tech_simple = vocab['technical_vs_simple']
        print(f"   Technical words: {tech_simple['technical_words']} ({tech_simple['technical_ratio']:.1%})")
        print(f"   Simple words: {tech_simple['simple_words']} ({tech_simple['simple_ratio']:.1%})")
        print(f"   Complexity ratio: {tech_simple['complexity_ratio']:.1%}")
        
        # Power words
        power = vocab['power_words']
        print(f"   Power words: {power['power_words']}")
        print(f"   Weak words: {power['weak_words']}")
        print(f"   Assertiveness score: {power['assertiveness_score']:.3f}")
        
        # Word lengths
        lengths = vocab['word_lengths']
        print(f"   Average word length: {lengths['average_length']:.1f} characters")
        print(f"   Word length distribution:")
        len_dist = lengths['length_distribution']
        print(f"     - 1-3 chars: {len_dist['1-3_chars']}")
        print(f"     - 4-6 chars: {len_dist['4-6_chars']}")
        print(f"     - 7-9 chars: {len_dist['7-9_chars']}")
        print(f"     - 10+ chars: {len_dist['10+_chars']}")
        
        # Readability & Engagement
        readability = results['readability_and_engagement']
        print(f"\n📖 READABILITY & ENGAGEMENT")
        print(f"   Engagement score: {readability['engagement_score']:.3f}")
        
        # Readability scores
        read_scores = readability['readability_scores']
        print(f"   Flesch Reading Ease: {read_scores['flesch_reading_ease']:.1f}")
        print(f"   Flesch-Kincaid Grade: {read_scores['flesch_kincaid_grade']:.1f}")
        print(f"   Gunning Fog Index: {read_scores['gunning_fog_index']:.1f}")
        print(f"   Readability level: {read_scores['readability_level']}")
        
        # Complexity distribution
        complexity = readability['complexity_distribution']
        print(f"   Sentence complexity:")
        print(f"     - Simple: {complexity['simple']} ({complexity['percentages']['simple']:.1f}%)")
        print(f"     - Compound: {complexity['compound']} ({complexity['percentages']['compound']:.1f}%)")
        print(f"     - Complex: {complexity['complex']} ({complexity['percentages']['complex']:.1f}%)")
        
        # Rhetorical devices
        rhetorical = readability['rhetorical_devices']
        print(f"   Rhetorical devices:")
        print(f"     - Questions: {rhetorical['questions']}")
        print(f"     - Exclamations: {rhetorical['exclamations']}")
        print(f"     - Repetition: {rhetorical['repetition']}")
        print(f"     - Alliteration: {rhetorical['alliteration']}")
        print(f"     - Parallel structure: {rhetorical['parallel_structure']}")
        print(f"     - Total devices: {rhetorical['total_devices']}")
        
        # Emphasis words
        emphasis = readability['emphasis_words']
        print(f"   Emphasis words: {emphasis['emphasis_count']}")
        print(f"   Emphasis frequency: {emphasis['emphasis_frequency']:.3f}")
        
        # Topic & Semantic Flow
        topic_flow = results['topic_and_semantic_flow']
        print(f"\n🌊 TOPIC & SEMANTIC FLOW")
        print(f"   Flow quality: {topic_flow['flow_quality']}")
        
        # Main ideas
        main_ideas = topic_flow['main_ideas']
        print(f"   Main ideas: {main_ideas['main_idea_count']} ({main_ideas['main_idea_ratio']:.1%})")
        
        # Supporting evidence
        evidence = topic_flow['supporting_evidence']
        print(f"   Supporting evidence: {evidence['evidence_count']}")
        print(f"   Examples: {evidence['example_count']}")
        print(f"   Support ratio: {evidence['support_ratio']:.1%}")
        
        # Topic relevance
        relevance = topic_flow['topic_relevance']
        print(f"   Topic relevance score: {relevance['relevance_score']:.3f}")
        print(f"   Topic shifts: {relevance['topic_shifts']}")
        print(f"   Off-topic sentences: {len(relevance['off_topic_sentences'])}")
        
        # Coherence
        coherence = topic_flow['coherence']
        print(f"   Average coherence: {coherence['average_coherence']:.3f}")
        print(f"   Coherence consistency: {coherence['coherence_consistency']:.3f}")
        
        # Audio Alignment
        audio_align = results['audio_alignment']
        print(f"\n🎵 AUDIO ALIGNMENT")
        if 'error' in audio_align:
            print(f"   Error: {audio_align['error']}")
        else:
            print(f"   Alignment quality: {audio_align['alignment_quality']}")
            
            # Pause alignment
            pause_align = audio_align['pause_transition_alignment']
            print(f"   Pause-transition alignment: {pause_align['alignment_ratio']:.1%}")
            print(f"   Transition-aligned pauses: {pause_align['transition_aligned_pauses']}/{pause_align['total_transitions']}")
            
            # Emphasis alignment
            emphasis_align = audio_align['emphasis_alignment']
            print(f"   Emphasis segments: {emphasis_align['emphasis_segments']}")
            print(f"   Emphasis frequency: {emphasis_align['emphasis_frequency']:.1%}")
        
        print("\n" + "="*60)
        print("✅ Advanced Content Analysis Complete!")
        print("="*60)
