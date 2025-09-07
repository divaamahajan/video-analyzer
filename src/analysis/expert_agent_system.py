#!/usr/bin/env python3
"""
Expert Agent System
=================

A system of specialized expert agents that evaluate communication across
visual, audio, and content dimensions, then provide structured feedback.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional
from ..utils.ai_agent import BasicAIAgent


def convert_numpy_to_json_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-serializable format"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_json_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Handle custom objects by converting to dict
        return convert_numpy_to_json_serializable(obj.__dict__)
    elif hasattr(obj, '_asdict'):
        # Handle namedtuples
        return convert_numpy_to_json_serializable(obj._asdict())
    else:
        # For other types, try to convert to string
        try:
            return str(obj)
        except:
            return f"<{type(obj).__name__}>"


def summarize_analysis_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize analysis data to reduce token count for API calls"""
    summarized = {}
    
    # Summarize visual results
    if "visual_results" in data:
        visual_data = data["visual_results"]
        if isinstance(visual_data, dict):
            summarized["visual_results"] = {
                "overall_statistics": visual_data.get("overall_statistics", {}),
                "summary": {
                    "total_frames_analyzed": len(visual_data.get("frame_analyses", [])),
                    "analysis_type": "simplified_visual_analysis"
                }
            }
        else:
            summarized["visual_results"] = {
                "overall_statistics": {},
                "summary": {"analysis_type": "visual_analysis_completed"}
            }
    
    # Summarize pose results
    if "pose_results" in data:
        pose_data = data["pose_results"]
        if isinstance(pose_data, dict):
            summarized["pose_results"] = {
                "summary": pose_data.get("summary", {}),
                "total_frames": pose_data.get("total_frames", 0)
            }
        else:
            # If pose_data is a list or other structure, just include basic info
            summarized["pose_results"] = {
                "summary": "pose_analysis_completed",
                "total_frames": len(pose_data) if isinstance(pose_data, list) else 0
            }
    
    # Summarize audio results
    if "audio_results" in data:
        audio_data = data["audio_results"]
        if isinstance(audio_data, dict):
            summarized["audio_results"] = {
                "transcription": audio_data.get("transcription", ""),
                "summary": {
                    "words_per_minute": audio_data.get("pace_analysis", {}).get("words_per_minute", 0),
                    "average_pause_length": audio_data.get("pace_analysis", {}).get("average_pause_length", 0),
                    "pitch_range": audio_data.get("pitch_analysis", {}).get("pitch_range", ""),
                    "volume_db": audio_data.get("volume_analysis", {}).get("volume", 0),
                    "clarity_score": audio_data.get("pronunciation_analysis", {}).get("articulation_score", 0),
                    "emotional_intensity": audio_data.get("emotion_analysis", {}).get("emotional_intensity", 0)
                }
            }
        else:
            summarized["audio_results"] = {
                "transcription": "",
                "summary": {"analysis_type": "audio_analysis_completed"}
            }
    
    # Summarize content results
    if "content_results" in data:
        content_data = data["content_results"]
        if isinstance(content_data, dict):
            summarized["content_results"] = {
                "ai_insights": content_data.get("ai_insights", ""),
                "summary": {
                    "total_sentences": content_data.get("structure_analysis", {}).get("total_sentences", 0),
                    "average_sentence_length": content_data.get("structure_analysis", {}).get("average_sentence_length", 0),
                    "lexical_diversity": content_data.get("vocabulary_analysis", {}).get("lexical_diversity", 0),
                    "readability_score": content_data.get("readability_analysis", {}).get("flesch_reading_ease", 0),
                    "engagement_score": content_data.get("engagement_analysis", {}).get("engagement_score", 0)
                }
            }
        else:
            summarized["content_results"] = {
                "ai_insights": "",
                "summary": {"analysis_type": "content_analysis_completed"}
            }
    
    return summarized


class ExpertAgent:
    """Base class for specialized expert agents"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """Initialize the expert agent"""
        self.ai_agent = BasicAIAgent(api_key, model=model)
        self.system_prompt = """
        You are an expert communication coach specializing in evaluating communication effectiveness.
        Analyze the provided data and metrics, then provide a structured evaluation with scores, 
        evidence-based reasoning, and specific recommendations for improvement.
        
        Your response must be in valid JSON format with the following structure:
        {
            "score": float,  // Overall score from 0.0 to 1.0
            "evidence": [string],  // List of specific evidence points from the data
            "explanation": string,  // Detailed explanation of the evaluation
            "recommendations": [string]  // List of specific, actionable recommendations
        }
        """
    
    def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the data and return structured feedback"""
        raise NotImplementedError("Subclasses must implement evaluate()")


class BodyLanguageExpert(ExpertAgent):
    """Expert agent specializing in body language evaluation"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """Initialize the body language expert"""
        super().__init__(api_key, model)
        self.system_prompt += """
        You are a body language and nonverbal communication expert. Analyze the provided visual metrics
        including posture, gestures, facial expressions, and spatial awareness. Focus on how these
        elements contribute to the overall communication effectiveness.
        
        Consider factors such as:
        - Posture and body positioning
        - Gesture frequency, variety, and appropriateness
        - Facial expressiveness and congruence with content
        - Eye contact and engagement
        - Movement patterns and spatial awareness
        """
    
    def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate body language data and return structured feedback"""
        # Convert numpy arrays to JSON-serializable format
        serializable_data = convert_numpy_to_json_serializable(data)
        
        prompt = f"""Analyze the following body language and visual communication metrics:
        
        {json.dumps(serializable_data, indent=2)}
        
        Provide a comprehensive evaluation of the person's body language and visual communication effectiveness.
        """
        
        response = self.ai_agent.generate_response(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "score": 0.0,
                "evidence": ["Error parsing AI response"],
                "explanation": "Failed to generate valid evaluation",
                "recommendations": ["Please try again with more detailed data"]
            }


class VocalExpressionExpert(ExpertAgent):
    """Expert agent specializing in vocal expression evaluation"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """Initialize the vocal expression expert"""
        super().__init__(api_key, model)
        self.system_prompt += """
        You are a vocal coach and speech expert. Analyze the provided audio metrics
        including pace, pitch, clarity, articulation, and emotional tone. Focus on how these
        elements contribute to the overall communication effectiveness.
        
        Consider factors such as:
        - Speaking pace and rhythm
        - Pitch variation and vocal range
        - Voice clarity and projection
        - Articulation and pronunciation
        - Emotional tone and expressiveness
        - Pause usage and emphasis
        - Filler word frequency
        """
    
    def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate vocal expression data and return structured feedback"""
        # Convert numpy arrays to JSON-serializable format
        serializable_data = convert_numpy_to_json_serializable(data)
        
        prompt = f"""Analyze the following vocal expression and audio communication metrics:
        
        {json.dumps(serializable_data, indent=2)}
        
        Provide a comprehensive evaluation of the person's vocal expression and audio communication effectiveness.
        """
        
        response = self.ai_agent.generate_response(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "score": 0.0,
                "evidence": ["Error parsing AI response"],
                "explanation": "Failed to generate valid evaluation",
                "recommendations": ["Please try again with more detailed data"]
            }


class ContentExpert(ExpertAgent):
    """Expert agent specializing in content evaluation"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """Initialize the content expert"""
        super().__init__(api_key, model)
        self.system_prompt += """
        You are a content and linguistic expert. Analyze the provided content metrics
        including lexical diversity, readability, clarity, and topic coherence. Focus on how these
        elements contribute to the overall communication effectiveness.
        
        Consider factors such as:
        - Vocabulary usage and lexical diversity
        - Sentence structure and complexity
        - Content organization and flow
        - Clarity and conciseness
        - Persuasiveness and impact
        - Use of rhetorical devices and techniques
        - Appropriateness for the intended audience
        """
    
    def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate content data and return structured feedback"""
        # Convert numpy arrays to JSON-serializable format
        serializable_data = convert_numpy_to_json_serializable(data)
        
        prompt = f"""Analyze the following content and linguistic metrics:
        
        {json.dumps(serializable_data, indent=2)}
        
        Provide a comprehensive evaluation of the person's content quality and linguistic effectiveness.
        """
        
        response = self.ai_agent.generate_response(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "score": 0.0,
                "evidence": ["Error parsing AI response"],
                "explanation": "Failed to generate valid evaluation",
                "recommendations": ["Please try again with more detailed data"]
            }


class EnvironmentExpert(ExpertAgent):
    """Expert agent specializing in environment evaluation"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """Initialize the environment expert"""
        super().__init__(api_key, model)
        self.system_prompt += """
        You are an environment and setup expert. Analyze the provided environment metrics
        including lighting, framing, background, and technical quality. Focus on how these
        elements contribute to the overall communication effectiveness.
        
        Consider factors such as:
        - Lighting quality and direction
        - Camera framing and positioning
        - Background appropriateness and distractions
        - Technical quality (resolution, stability)
        - Overall professional appearance
        """
    
    def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate environment data and return structured feedback"""
        # Convert numpy arrays to JSON-serializable format
        serializable_data = convert_numpy_to_json_serializable(data)
        
        prompt = f"""Analyze the following environment and setup metrics:
        
        {json.dumps(serializable_data, indent=2)}
        
        Provide a comprehensive evaluation of the person's environment and technical setup effectiveness.
        """
        
        response = self.ai_agent.generate_response(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "score": 0.0,
                "evidence": ["Error parsing AI response"],
                "explanation": "Failed to generate valid evaluation",
                "recommendations": ["Please try again with more detailed data"]
            }


class ExpertAgentSystem:
    """Coordinates multiple expert agents to provide comprehensive communication evaluation"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """Initialize the expert agent system"""
        self.api_key = api_key
        self.model = model
        self.experts = {
            "body_language": BodyLanguageExpert(api_key, model),
            "vocal_expression": VocalExpressionExpert(api_key, model),
            "content": ContentExpert(api_key, model),
            "environment": EnvironmentExpert(api_key, model)
        }
    
    def evaluate_all(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run evaluation across all expert agents"""
        results = {}
        
        # Summarize data to reduce token count
        summarized_data = summarize_analysis_data(data)
        
        # Extract relevant data for each expert
        body_data = {
            "visual_results": summarized_data.get("visual_results", {}),
            "pose_results": summarized_data.get("pose_results", {})
        }
        
        vocal_data = {
            "audio_results": summarized_data.get("audio_results", {})
        }
        
        content_data = {
            "content_results": summarized_data.get("content_results", {}),
            "transcription": summarized_data.get("audio_results", {}).get("transcription", "")
        }
        
        environment_data = {
            "environment_metrics": summarized_data.get("visual_results", {}).get("overall_statistics", {})
        }
        
        # Run evaluations
        results["body_language"] = self.experts["body_language"].evaluate(body_data)
        results["vocal_expression"] = self.experts["vocal_expression"].evaluate(vocal_data)
        results["content"] = self.experts["content"].evaluate(content_data)
        results["environment"] = self.experts["environment"].evaluate(environment_data)
        
        # Calculate aggregate score
        scores = [results[key]["score"] for key in results]
        results["aggregate_score"] = sum(scores) / len(scores) if scores else 0.0
        
        return results