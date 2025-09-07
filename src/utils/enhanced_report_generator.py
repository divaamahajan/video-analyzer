#!/usr/bin/env python3
"""
Enhanced Report Generator
=======================

Generates comprehensive reports with expert evaluations and personalized
practice plans based on multi-dimensional analysis results.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional


class EnhancedReportGenerator:
    """Generates enhanced reports with expert evaluations and practice plans"""
    
    def __init__(self, output_dir: str = "reports"):
        """Initialize the report generator"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_comprehensive_report(
        self,
        video_path: str,
        aggregated_results: Dict[str, Any],
        expert_evaluations: Optional[Dict[str, Any]] = None,
        practice_plan: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a comprehensive markdown report with expert evaluations and practice plan"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = f"enhanced_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self._generate_report_content(
                video_path, aggregated_results, expert_evaluations,
                practice_plan, config, timestamp
            ))
        
        return filepath
    
    def _generate_report_content(
        self,
        video_path: str,
        aggregated_results: Dict[str, Any],
        expert_evaluations: Optional[Dict[str, Any]],
        practice_plan: Optional[str],
        config: Optional[Dict[str, Any]],
        timestamp: str
    ) -> str:
        """Generate the markdown content for the comprehensive report"""
        
        content = []
        
        # Header
        content.append("# 🎬 Communication Effectiveness Analysis Report")
        content.append("")
        content.append(f"**Generated:** {timestamp}")
        content.append(f"**Video File:** {os.path.basename(video_path)}")
        content.append("")
        
        # Overall Score
        overall_score = aggregated_results.get("overall_score", 0.0)
        content.append(f"## 📊 Overall Communication Score: {overall_score:.2f}/1.0")
        content.append("")
        
        # Score visualization
        content.append(self._generate_score_visualization(overall_score))
        content.append("")
        
        # Executive Summary
        content.append("## 📝 Executive Summary")
        content.append("")
        content.append("This comprehensive analysis evaluates your communication effectiveness across multiple dimensions:")
        content.append("")
        
        # Dimension scores
        dimension_scores = aggregated_results.get("dimension_scores", {})
        if dimension_scores:
            content.append("| Dimension | Score | Rating |")
            content.append("|-----------|-------|--------|")
            
            for dimension, score in dimension_scores.items():
                rating = self._get_rating_for_score(score)
                content.append(f"| {dimension.replace('_', ' ').title()} | {score:.2f} | {rating} |")
            
            content.append("")
        
        # Visual Analysis Section
        visual_metrics = aggregated_results.get("metrics", {}).get("visual", {})
        if visual_metrics:
            content.append("## 👁️ Visual Communication Analysis")
            content.append("")
            content.append("Your visual communication includes body language, facial expressions, and overall visual presence.")
            content.append("")
            
            # Add visual metrics table
            content.append("| Metric | Value | Rating |")
            content.append("|--------|-------|--------|")
            
            for metric, value in visual_metrics.items():
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    rating = self._get_rating_for_score(value)
                    content.append(f"| {metric.replace('_', ' ').title()} | {value:.2f} | {rating} |")
            
            content.append("")
            
            # Add expert evaluation if available
            if expert_evaluations and "body_language" in expert_evaluations:
                body_eval = expert_evaluations["body_language"]
                content.append("### 🧠 Expert Analysis: Body Language")
                content.append("")
                content.append(f"**Score:** {body_eval.get('score', 0.0):.2f}/1.0")
                content.append("")
                content.append(f"**Analysis:** {body_eval.get('explanation', 'No analysis available')}")
                content.append("")
                
                # Add evidence points
                if "evidence" in body_eval and body_eval["evidence"]:
                    content.append("**Key Observations:**")
                    content.append("")
                    for evidence in body_eval["evidence"]:
                        content.append(f"- {evidence}")
                    content.append("")
                
                # Add recommendations
                if "recommendations" in body_eval and body_eval["recommendations"]:
                    content.append("**Recommendations:**")
                    content.append("")
                    for rec in body_eval["recommendations"]:
                        content.append(f"- {rec}")
                    content.append("")
        
        # Emotion Analysis Section
        emotion_metrics = aggregated_results.get("metrics", {}).get("emotion", {})
        if emotion_metrics:
            content.append("## 😊 Emotional Expression Analysis")
            content.append("")
            content.append("Your emotional expression includes the range, authenticity, and engagement of emotions displayed.")
            content.append("")
            
            # Add emotion metrics table
            content.append("| Metric | Value | Rating |")
            content.append("|--------|-------|--------|")
            
            for metric, value in emotion_metrics.items():
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    rating = self._get_rating_for_score(value)
                    content.append(f"| {metric.replace('_', ' ').title()} | {value:.2f} | {rating} |")
            
            content.append("")
            
            # Add emotion distribution if available
            if "emotion_distribution" in emotion_metrics and isinstance(emotion_metrics["emotion_distribution"], dict):
                content.append("### 📊 Emotion Distribution")
                content.append("")
                content.append("| Emotion | Percentage |")
                content.append("|---------|------------|")
                
                distribution = emotion_metrics["emotion_distribution"]
                total = sum(distribution.values())
                
                if total > 0:
                    for emotion, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / total) * 100
                        content.append(f"| {emotion.title()} | {percentage:.1f}% |")
                
                content.append("")
            
            # Add expert evaluation if available
            if expert_evaluations and "emotional_expression" in expert_evaluations:
                emotion_eval = expert_evaluations["emotional_expression"]
                content.append("### 🧠 Expert Analysis: Emotional Expression")
                content.append("")
                content.append(f"**Score:** {emotion_eval.get('score', 0.0):.2f}/1.0")
                content.append("")
                content.append(f"**Analysis:** {emotion_eval.get('explanation', 'No analysis available')}")
                content.append("")
                
                # Add evidence points
                if "evidence" in emotion_eval and emotion_eval["evidence"]:
                    content.append("**Key Observations:**")
                    content.append("")
                    for evidence in emotion_eval["evidence"]:
                        content.append(f"- {evidence}")
                    content.append("")
                
                # Add recommendations
                if "recommendations" in emotion_eval and emotion_eval["recommendations"]:
                    content.append("**Recommendations:**")
                    content.append("")
                    for rec in emotion_eval["recommendations"]:
                        content.append(f"- {rec}")
                    content.append("")
        
        # Audio Analysis Section
        audio_metrics = aggregated_results.get("metrics", {}).get("audio", {})
        if audio_metrics:
            content.append("## 🎵 Vocal Expression Analysis")
            content.append("")
            content.append("Your vocal expression includes pace, tone, clarity, and overall audio quality.")
            content.append("")
            
            # Add audio metrics table
            content.append("| Metric | Value | Rating |")
            content.append("|--------|-------|--------|")
            
            for metric, value in audio_metrics.items():
                if isinstance(value, (int, float)):
                    if metric == "words_per_minute":
                        # Special handling for WPM
                        rating = "Optimal" if 120 <= value <= 160 else ("Too Slow" if value < 120 else "Too Fast")
                        content.append(f"| Words Per Minute | {value:.1f} | {rating} |")
                    elif 0 <= value <= 1:
                        # Normal 0-1 scale metrics
                        rating = self._get_rating_for_score(value)
                        content.append(f"| {metric.replace('_', ' ').title()} | {value:.2f} | {rating} |")
            
            content.append("")
            
            # Add expert evaluation if available
            if expert_evaluations and "vocal_expression" in expert_evaluations:
                vocal_eval = expert_evaluations["vocal_expression"]
                content.append("### 🧠 Expert Analysis: Vocal Expression")
                content.append("")
                content.append(f"**Score:** {vocal_eval.get('score', 0.0):.2f}/1.0")
                content.append("")
                content.append(f"**Analysis:** {vocal_eval.get('explanation', 'No analysis available')}")
                content.append("")
                
                # Add evidence points
                if "evidence" in vocal_eval and vocal_eval["evidence"]:
                    content.append("**Key Observations:**")
                    content.append("")
                    for evidence in vocal_eval["evidence"]:
                        content.append(f"- {evidence}")
                    content.append("")
                
                # Add recommendations
                if "recommendations" in vocal_eval and vocal_eval["recommendations"]:
                    content.append("**Recommendations:**")
                    content.append("")
                    for rec in vocal_eval["recommendations"]:
                        content.append(f"- {rec}")
                    content.append("")
        
        # Content Analysis Section
        content_metrics = aggregated_results.get("metrics", {}).get("content", {})
        if content_metrics:
            content.append("## 📝 Content Analysis")
            content.append("")
            content.append("Your content includes the structure, clarity, and effectiveness of your message.")
            content.append("")
            
            # Add content metrics table
            content.append("| Metric | Value | Rating |")
            content.append("|--------|-------|--------|")
            
            for metric, value in content_metrics.items():
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    rating = self._get_rating_for_score(value)
                    content.append(f"| {metric.replace('_', ' ').title()} | {value:.2f} | {rating} |")
            
            content.append("")
            
            # Add expert evaluation if available
            if expert_evaluations and "content" in expert_evaluations:
                content_eval = expert_evaluations["content"]
                content.append("### 🧠 Expert Analysis: Content")
                content.append("")
                content.append(f"**Score:** {content_eval.get('score', 0.0):.2f}/1.0")
                content.append("")
                content.append(f"**Analysis:** {content_eval.get('explanation', 'No analysis available')}")
                content.append("")
                
                # Add evidence points
                if "evidence" in content_eval and content_eval["evidence"]:
                    content.append("**Key Observations:**")
                    content.append("")
                    for evidence in content_eval["evidence"]:
                        content.append(f"- {evidence}")
                    content.append("")
                
                # Add recommendations
                if "recommendations" in content_eval and content_eval["recommendations"]:
                    content.append("**Recommendations:**")
                    content.append("")
                    for rec in content_eval["recommendations"]:
                        content.append(f"- {rec}")
                    content.append("")
        
        # Environment Analysis Section
        env_metrics = aggregated_results.get("metrics", {}).get("environment", {})
        if env_metrics:
            content.append("## 🏠 Environment Analysis")
            content.append("")
            content.append("Your environment includes lighting, framing, background, and overall setup quality.")
            content.append("")
            
            # Add environment metrics table
            content.append("| Metric | Value | Rating |")
            content.append("|--------|-------|--------|")
            
            for metric, value in env_metrics.items():
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    rating = self._get_rating_for_score(value)
                    content.append(f"| {metric.replace('_', ' ').title()} | {value:.2f} | {rating} |")
            
            content.append("")
            
            # Add expert evaluation if available
            if expert_evaluations and "environment" in expert_evaluations:
                env_eval = expert_evaluations["environment"]
                content.append("### 🧠 Expert Analysis: Environment")
                content.append("")
                content.append(f"**Score:** {env_eval.get('score', 0.0):.2f}/1.0")
                content.append("")
                content.append(f"**Analysis:** {env_eval.get('explanation', 'No analysis available')}")
                content.append("")
                
                # Add evidence points
                if "evidence" in env_eval and env_eval["evidence"]:
                    content.append("**Key Observations:**")
                    content.append("")
                    for evidence in env_eval["evidence"]:
                        content.append(f"- {evidence}")
                    content.append("")
                
                # Add recommendations
                if "recommendations" in env_eval and env_eval["recommendations"]:
                    content.append("**Recommendations:**")
                    content.append("")
                    for rec in env_eval["recommendations"]:
                        content.append(f"- {rec}")
                    content.append("")
        
        # Practice Plan Section
        if practice_plan:
            content.append("## 🎯 Your 7-Day Practice Plan")
            content.append("")
            content.append(practice_plan)
            content.append("")
        
        # Footer
        content.append("---")
        content.append("")
        content.append("*Generated by Communication Analysis System*")
        content.append(f"*{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        return "\n".join(content)
    
    def _generate_score_visualization(self, score: float) -> str:
        """Generate a visual representation of the score"""
        filled_blocks = int(round(score * 10))
        empty_blocks = 10 - filled_blocks
        
        visualization = "**Score:** "
        visualization += "█" * filled_blocks
        visualization += "░" * empty_blocks
        visualization += f" {score:.2f}/1.0"
        
        return visualization
    
    def _get_rating_for_score(self, score: float) -> str:
        """Get a text rating based on a score from 0 to 1"""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Very Good"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.6:
            return "Above Average"
        elif score >= 0.5:
            return "Average"
        elif score >= 0.4:
            return "Below Average"
        elif score >= 0.3:
            return "Needs Improvement"
        elif score >= 0.2:
            return "Poor"
        else:
            return "Very Poor"