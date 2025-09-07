#!/usr/bin/env python3
"""
Practice Plan Generator
=====================

Generates personalized 7-day practice plans based on expert evaluations
to help users improve their communication skills.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from ..utils.ai_agent import BasicAIAgent


class PracticePlanGenerator:
    """Generates personalized practice plans based on expert evaluations"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """Initialize the practice plan generator"""
        self.ai_agent = BasicAIAgent(api_key, model=model)
        self.system_prompt = """
        You are an expert communication coach specializing in creating personalized practice plans.
        Based on the provided expert evaluations, create a structured 7-day practice plan that addresses
        the user's specific communication weaknesses and builds on their strengths.
        
        Your practice plan should:
        1. Focus on the most critical areas for improvement first
        2. Include specific, actionable exercises for each day
        3. Gradually build skills over the 7-day period
        4. Include measurable goals and self-assessment techniques
        5. Be realistic and achievable within the timeframe
        
        Format your response as a structured markdown document with clear headings, bullet points,
        and day-by-day instructions.
        """
    
    def generate_practice_plan(self, expert_evaluations: Dict[str, Any], user_info: Optional[Dict[str, Any]] = None) -> str:
        """Generate a personalized 7-day practice plan based on expert evaluations"""
        
        # Extract key weaknesses and strengths from evaluations
        weaknesses = []
        strengths = []
        recommendations = []
        
        for category, evaluation in expert_evaluations.items():
            if category != "aggregate_score" and isinstance(evaluation, dict):
                score = evaluation.get("score", 0.0)
                
                # Extract recommendations
                if "recommendations" in evaluation and isinstance(evaluation["recommendations"], list):
                    recommendations.extend(evaluation["recommendations"])
                
                # Identify strengths and weaknesses based on score
                if score < 0.4:
                    weaknesses.append(f"Low {category} score: {score:.2f}")
                elif score > 0.7:
                    strengths.append(f"Strong {category} score: {score:.2f}")
        
        # Create prompt for AI
        prompt = f"""
        Based on the following expert evaluations, create a personalized 7-day practice plan:
        
        Expert Evaluations:
        {json.dumps(expert_evaluations, indent=2)}
        
        Key Weaknesses:
        {"\n".join(f"- {w}" for w in weaknesses)}
        
        Key Strengths:
        {"\n".join(f"- {s}" for s in strengths)}
        
        Recommendations:
        {"\n".join(f"- {r}" for r in recommendations)}
        """
        
        # Add user info if available
        if user_info:
            prompt += f"""
            
            User Information:
            {json.dumps(user_info, indent=2)}
            """
        
        # Generate practice plan
        practice_plan = self.ai_agent.generate_response(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.5,
            max_tokens=2000
        )
        
        return self._format_practice_plan(practice_plan, expert_evaluations)
    
    def _format_practice_plan(self, plan_content: str, evaluations: Dict[str, Any]) -> str:
        """Format the practice plan with additional metadata"""
        
        # Get today's date
        today = datetime.now()
        
        # Create header
        header = f"""# 🎯 Your Personalized 7-Day Communication Practice Plan

**Generated:** {today.strftime('%Y-%m-%d')}
**Overall Communication Score:** {evaluations.get('aggregate_score', 0.0):.2f}/1.0

## 📊 Evaluation Summary

| Area | Score | Key Finding |
|------|-------|-------------|
"""
        
        # Add evaluation summary table
        for category, evaluation in evaluations.items():
            if category != "aggregate_score" and isinstance(evaluation, dict):
                score = evaluation.get("score", 0.0)
                explanation = evaluation.get("explanation", "")
                summary = explanation[:50] + "..." if len(explanation) > 50 else explanation
                header += f"| {category.replace('_', ' ').title()} | {score:.2f} | {summary} |\n"
        
        header += "\n## 🗓️ Your 7-Day Plan\n\n"
        
        # Add day headers if they don't exist
        plan_with_days = plan_content
        for i in range(1, 8):
            day_date = today + timedelta(days=i-1)
            day_header = f"### Day {i}: {day_date.strftime('%A, %B %d')}"
            
            if f"Day {i}" not in plan_content and f"DAY {i}" not in plan_content:
                if i > 1:
                    plan_with_days += f"\n\n{day_header}\n\nContinue practicing and building on previous day's exercises."
                else:
                    plan_with_days += f"\n\n{day_header}\n\nStart with foundational exercises based on your evaluation."
        
        # Add footer
        footer = "\n\n---\n\n**Next Steps:** Record a new video after completing this 7-day plan and upload it for a progress evaluation."
        
        return header + plan_with_days + footer
    
    def save_practice_plan(self, plan_content: str, output_dir: str = "reports") -> str:
        """Save the practice plan to a file"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"7day_practice_plan_{timestamp}.md"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(plan_content)
        
        return filepath