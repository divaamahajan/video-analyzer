#!/usr/bin/env python3
"""
Analysis Aggregator
=================

Aggregates results from various analysis components and expert evaluations
to provide a comprehensive view of communication effectiveness.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime


class AnalysisAggregator:
    """Aggregates and normalizes results from various analysis components"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Initialize the aggregator with optional custom weights"""
        # Default weights for different analysis components
        self.weights = weights or {
            "visual": 0.20,
            "emotion": 0.15,
            "audio": 0.20,
            "content": 0.20,
            "environment": 0.15,
            "expert": 0.10
        }
        
        # Ensure weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if weight_sum != 1.0:
            for key in self.weights:
                self.weights[key] /= weight_sum
    
    def aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from various analysis components"""
        aggregated = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {},
            "scores": {},
            "overall_score": 0.0,
            "dimension_scores": {}
        }
        
        # Extract and normalize metrics from different components
        self._extract_visual_metrics(results, aggregated)
        self._extract_emotion_metrics(results, aggregated)
        self._extract_audio_metrics(results, aggregated)
        self._extract_content_metrics(results, aggregated)
        self._extract_environment_metrics(results, aggregated)
        self._extract_expert_evaluations(results, aggregated)
        
        # Calculate dimension scores
        self._calculate_dimension_scores(aggregated)
        
        # Calculate overall score
        self._calculate_overall_score(aggregated)
        
        return aggregated
    
    def _extract_visual_metrics(self, results: Dict[str, Any], aggregated: Dict[str, Any]) -> None:
        """Extract and normalize visual metrics"""
        visual_results = results.get("visual_results", {})
        pose_results = results.get("pose_results", {})
        
        if visual_results or pose_results:
            # Initialize visual metrics
            aggregated["metrics"]["visual"] = {}
            
            # Extract metrics from visual_results
            if visual_results:
                # Extract engagement score
                if "engagement_score" in visual_results:
                    aggregated["metrics"]["visual"]["engagement_score"] = visual_results["engagement_score"]
                
                # Extract eye contact
                if "eye_contact" in visual_results:
                    aggregated["metrics"]["visual"]["eye_contact"] = visual_results["eye_contact"]
                
                # Extract gesture frequency
                if "gesture_frequency" in visual_results:
                    aggregated["metrics"]["visual"]["gesture_frequency"] = visual_results["gesture_frequency"]
                
                # Extract spatial awareness
                if "spatial_awareness" in visual_results:
                    aggregated["metrics"]["visual"]["spatial_awareness"] = visual_results["spatial_awareness"]
                
                # Extract facial expression analysis
                if "facial_expressions" in visual_results:
                    aggregated["metrics"]["visual"]["facial_expressions"] = visual_results["facial_expressions"]
            
            # Extract metrics from pose_results
            if pose_results:
                # Extract posture score
                if "posture_score" in pose_results:
                    aggregated["metrics"]["visual"]["posture_score"] = pose_results["posture_score"]
                
                # Extract movement magnitude
                if "movement_magnitude" in pose_results:
                    aggregated["metrics"]["visual"]["movement_magnitude"] = pose_results["movement_magnitude"]
            
            # Calculate visual score
            visual_scores = []
            for key, value in aggregated["metrics"]["visual"].items():
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    visual_scores.append(value)
            
            if visual_scores:
                aggregated["scores"]["visual"] = sum(visual_scores) / len(visual_scores)
    
    def _extract_audio_metrics(self, results: Dict[str, Any], aggregated: Dict[str, Any]) -> None:
        """Extract and normalize audio metrics"""
        audio_results = results.get("audio_results", {})
        
        if audio_results:
            # Initialize audio metrics
            aggregated["metrics"]["audio"] = {}
            
            # Extract words per minute
            if "words_per_minute" in audio_results:
                aggregated["metrics"]["audio"]["words_per_minute"] = audio_results["words_per_minute"]
                
                # Normalize WPM to a 0-1 score (optimal range: 120-160 WPM)
                wpm = audio_results["words_per_minute"]
                if 120 <= wpm <= 160:
                    wpm_score = 1.0
                elif wpm < 120:
                    wpm_score = max(0, wpm / 120)
                else:  # wpm > 160
                    wpm_score = max(0, 1 - (wpm - 160) / 80)  # Decreases to 0 at 240 WPM
                
                aggregated["metrics"]["audio"]["words_per_minute_score"] = wpm_score
            
            # Extract pitch variance
            if "pitch_variance" in audio_results:
                aggregated["metrics"]["audio"]["pitch_variance"] = audio_results["pitch_variance"]
            
            # Extract clarity score
            if "clarity_score" in audio_results:
                aggregated["metrics"]["audio"]["clarity_score"] = audio_results["clarity_score"]
            
            # Extract articulation score
            if "articulation_score" in audio_results:
                aggregated["metrics"]["audio"]["articulation_score"] = audio_results["articulation_score"]
            
            # Extract pause analysis
            if "pause_analysis" in audio_results:
                aggregated["metrics"]["audio"]["pause_analysis"] = audio_results["pause_analysis"]
            
            # Extract filler word detection
            if "filler_word_count" in audio_results:
                aggregated["metrics"]["audio"]["filler_word_count"] = audio_results["filler_word_count"]
                
                # Normalize filler word count to a 0-1 score (lower is better)
                # Assuming 0 fillers is perfect (1.0) and 30+ fillers is poor (0.0)
                filler_count = audio_results["filler_word_count"]
                filler_score = max(0, 1 - (filler_count / 30))
                
                aggregated["metrics"]["audio"]["filler_word_score"] = filler_score
            
            # Calculate audio score
            audio_scores = []
            for key, value in aggregated["metrics"]["audio"].items():
                if key.endswith("_score") and isinstance(value, (int, float)) and 0 <= value <= 1:
                    audio_scores.append(value)
            
            if audio_scores:
                aggregated["scores"]["audio"] = sum(audio_scores) / len(audio_scores)
    
    def _extract_content_metrics(self, results: Dict[str, Any], aggregated: Dict[str, Any]) -> None:
        """Extract and normalize content metrics"""
        content_results = results.get("content_results", {})
        
        if content_results:
            # Initialize content metrics
            aggregated["metrics"]["content"] = {}
            
            # Extract lexical diversity
            if "lexical_diversity" in content_results:
                aggregated["metrics"]["content"]["lexical_diversity"] = content_results["lexical_diversity"]
            
            # Extract readability score
            if "readability_score" in content_results:
                aggregated["metrics"]["content"]["readability_score"] = content_results["readability_score"]
                
                # Normalize readability score (Flesch-Kincaid) to 0-1
                # Assuming optimal range is 60-80 (8th-10th grade level)
                readability = content_results["readability_score"]
                if 60 <= readability <= 80:
                    readability_score = 1.0
                elif readability < 60:
                    readability_score = max(0, readability / 60)
                else:  # readability > 80
                    readability_score = max(0, 1 - (readability - 80) / 20)  # Decreases to 0 at 100
                
                aggregated["metrics"]["content"]["readability_normalized"] = readability_score
            
            # Extract clarity score
            if "clarity_score" in content_results:
                aggregated["metrics"]["content"]["clarity_score"] = content_results["clarity_score"]
            
            # Extract engagement score
            if "engagement_score" in content_results:
                aggregated["metrics"]["content"]["engagement_score"] = content_results["engagement_score"]
            
            # Extract topic coherence
            if "topic_coherence" in content_results:
                aggregated["metrics"]["content"]["topic_coherence"] = content_results["topic_coherence"]
            
            # Calculate content score
            content_scores = []
            for key, value in aggregated["metrics"]["content"].items():
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    content_scores.append(value)
            
            if content_scores:
                aggregated["scores"]["content"] = sum(content_scores) / len(content_scores)
    
    def _extract_emotion_metrics(self, results: Dict[str, Any], aggregated: Dict[str, Any]) -> None:
        """Extract and normalize emotion metrics from emotion analysis"""
        emotion_results = results.get("emotion_results", {})
        
        if emotion_results:
            # Initialize emotion metrics
            aggregated["metrics"]["emotion"] = {}
            
            # Extract emotion distribution
            if "emotion_distribution" in emotion_results:
                aggregated["metrics"]["emotion"]["emotion_distribution"] = emotion_results["emotion_distribution"]
                
                # Calculate emotion diversity score (0-1)
                # Higher diversity (more balanced emotions) gets a higher score
                distribution = emotion_results["emotion_distribution"]
                if distribution and isinstance(distribution, dict):
                    values = list(distribution.values())
                    if values:
                        # Calculate normalized entropy as diversity measure
                        total = sum(values)
                        if total > 0:
                            probs = [v/total for v in values]
                            entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
                            max_entropy = np.log2(len(values)) if len(values) > 0 else 1
                            diversity_score = entropy / max_entropy if max_entropy > 0 else 0
                            aggregated["metrics"]["emotion"]["emotion_diversity_score"] = diversity_score
            
            # Extract emotional engagement
            if "emotional_engagement" in emotion_results:
                aggregated["metrics"]["emotion"]["emotional_engagement"] = emotion_results["emotional_engagement"]
            
            # Extract emotion patterns
            if "emotion_patterns" in emotion_results:
                patterns = emotion_results["emotion_patterns"]
                
                # Extract emotional stability
                if "emotional_stability" in patterns:
                    aggregated["metrics"]["emotion"]["emotional_stability"] = patterns["emotional_stability"]
                
                # Extract emotional volatility
                if "emotional_volatility" in patterns:
                    volatility = patterns["emotional_volatility"]
                    # Convert to score (lower volatility is better)
                    volatility_score = max(0, 1 - volatility)
                    aggregated["metrics"]["emotion"]["emotional_volatility_score"] = volatility_score
                
                # Extract microexpression indicators
                if "microexpression_indicators" in patterns:
                    aggregated["metrics"]["emotion"]["microexpression_indicators"] = patterns["microexpression_indicators"]
                
                # Extract emotion masking
                if "emotion_masking_indicators" in patterns:
                    masking = patterns["emotion_masking_indicators"]
                    # Convert to score (lower masking is better)
                    masking_score = max(0, 1 - masking)
                    aggregated["metrics"]["emotion"]["authenticity_score"] = masking_score
            
            # Extract average confidence
            if "average_confidence" in emotion_results:
                aggregated["metrics"]["emotion"]["confidence_score"] = emotion_results["average_confidence"]
            
            # Calculate overall emotion score
            emotion_scores = []
            for key, value in aggregated["metrics"]["emotion"].items():
                if key.endswith("_score") and isinstance(value, (int, float)) and 0 <= value <= 1:
                    emotion_scores.append(value)
            
            if emotion_scores:
                aggregated["scores"]["emotion"] = sum(emotion_scores) / len(emotion_scores)
            else:
                aggregated["scores"]["emotion"] = 0.0
    
    def _extract_environment_metrics(self, results: Dict[str, Any], aggregated: Dict[str, Any]) -> None:
        """Extract and normalize environment metrics"""
        # First check for dedicated environment results from EnvironmentAnalyzer
        environment_results = results.get("environment_results", {})
        
        # Fall back to legacy environment metrics if needed
        visual_results = results.get("visual_results", {})
        environment_metrics = visual_results.get("environment_metrics", {})
        
        # Initialize environment metrics
        aggregated["metrics"]["environment"] = {}
        
        if environment_results:
            # Extract metrics from dedicated environment analyzer
            
            # Extract lighting quality
            if "lighting_quality" in environment_results:
                aggregated["metrics"]["environment"]["lighting_quality"] = environment_results["lighting_quality"]
            
            # Extract framing score
            if "framing_score" in environment_results:
                aggregated["metrics"]["environment"]["framing_score"] = environment_results["framing_score"]
            
            # Extract background score
            if "background_score" in environment_results:
                aggregated["metrics"]["environment"]["background_score"] = environment_results["background_score"]
            
            # Extract distance score
            if "distance_score" in environment_results:
                aggregated["metrics"]["environment"]["distance_score"] = environment_results["distance_score"]
                
            # Use the pre-calculated environment score if available
            if "environment_score" in environment_results:
                aggregated["scores"]["environment"] = environment_results["environment_score"]
                return
        
        elif environment_metrics:
            # Extract metrics from legacy environment metrics
            
            # Extract lighting quality
            if "lighting_quality" in environment_metrics:
                aggregated["metrics"]["environment"]["lighting_quality"] = environment_metrics["lighting_quality"]
            
            # Extract framing score
            if "framing_score" in environment_metrics:
                aggregated["metrics"]["environment"]["framing_score"] = environment_metrics["framing_score"]
            
            # Extract background analysis
            if "background_analysis" in environment_metrics:
                aggregated["metrics"]["environment"]["background_score"] = environment_metrics["background_analysis"]
            
            # Extract distance from camera
            if "distance_from_camera" in environment_metrics:
                aggregated["metrics"]["environment"]["distance_score"] = environment_metrics["distance_from_camera"]
        
        # Calculate environment score if not already set
        if "environment" not in aggregated["scores"]:
            environment_scores = []
            for key, value in aggregated["metrics"]["environment"].items():
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    environment_scores.append(value)
            
            if environment_scores:
                aggregated["scores"]["environment"] = sum(environment_scores) / len(environment_scores)
            else:
                aggregated["scores"]["environment"] = 0.0
    
    def _extract_expert_evaluations(self, results: Dict[str, Any], aggregated: Dict[str, Any]) -> None:
        """Extract expert evaluations"""
        expert_results = results.get("expert_results", {})
        
        if expert_results:
            # Initialize expert metrics
            aggregated["metrics"]["expert"] = {}
            aggregated["scores"]["expert"] = {}
            
            # Extract scores from each expert
            for expert_name, evaluation in expert_results.items():
                if isinstance(evaluation, dict) and "score" in evaluation:
                    aggregated["scores"]["expert"][expert_name] = evaluation["score"]
            
            # Calculate average expert score
            expert_scores = list(aggregated["scores"]["expert"].values())
            if expert_scores:
                aggregated["scores"]["expert_average"] = sum(expert_scores) / len(expert_scores)
    
    def _calculate_dimension_scores(self, aggregated: Dict[str, Any]) -> None:
        """Calculate scores for each dimension"""
        for dimension in ["visual", "emotion", "audio", "content", "environment"]:
            if dimension in aggregated["scores"]:
                aggregated["dimension_scores"][dimension] = aggregated["scores"][dimension]
        
        # Add expert dimension score if available
        if "expert_average" in aggregated["scores"]:
            aggregated["dimension_scores"]["expert"] = aggregated["scores"]["expert_average"]
    
    def _calculate_overall_score(self, aggregated: Dict[str, Any]) -> None:
        """Calculate overall weighted score"""
        overall_score = 0.0
        total_weight = 0.0
        
        for dimension, score in aggregated["dimension_scores"].items():
            if dimension in self.weights:
                overall_score += score * self.weights[dimension]
                total_weight += self.weights[dimension]
        
        if total_weight > 0:
            aggregated["overall_score"] = overall_score / total_weight
        else:
            aggregated["overall_score"] = 0.0
    
    def save_aggregated_results(self, aggregated: Dict[str, Any], output_dir: str = "reports") -> str:
        """Save aggregated results to a JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aggregated_results_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(aggregated, f, indent=2)
        
        return filepath