#!/usr/bin/env python3
"""
Communication Analysis System
===========================

An AI-powered platform that evaluates communication effectiveness across:
- Visual dimensions (body language, facial expressions, gestures)
- Audio dimensions (pace, tone, clarity, articulation)
- Content dimensions (structure, clarity, engagement)
- Environment dimensions (lighting, framing, background)

Then translates insights into actionable recommendations and a personalized 7-day practice plan.
"""

import sys
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config.settings import Config
from src.analysis.visual import extract_frames, MoveNetAnalyzer, OpenCVAnalyzer, EnvironmentAnalyzer
from src.analysis.visual import analyze_emotions_in_frames, print_emotion_summary, save_emotion_analysis, get_emotion_statistics, detect_emotion_patterns, analyze_emotional_engagement
from src.analysis.audio import AudioAnalyzer
from src.analysis.text import ContentAnalyzer
from src.analysis.report_analyzer import ReportAnalyzer
from src.analysis.expert_agent_system import ExpertAgentSystem
from src.analysis.aggregator import AnalysisAggregator
from src.utils.summary_generator import print_pose_summary
from src.utils.report_generator import ComprehensiveReportGenerator
from src.utils.enhanced_report_generator import EnhancedReportGenerator
from src.utils.practice_plan_generator import PracticePlanGenerator

def main():
    """Main analysis pipeline"""
    print("🎬 Starting Communication Analysis System")
    print("="*50)
    
    # Initialize variables to store results
    visual_results = None
    pose_results = None
    emotion_results = None
    audio_results = None
    content_results = None
    environment_results = None
    expert_results = None
    aggregated_results = None
    practice_plan = None
    
    try:
        # (parallel path 1|| Parent= Input_Video|| Branch 1): Extract video frames
        print("\n📹 Step 1: Video Frame Extraction")
        frames = extract_frames(Config.VIDEO_PATH, Config.FRAME_INTERVAL)
        
        if not frames:
            print("❌ No frames extracted. Please check video file.")
            return
        

        # (parallel path 2|| Parent= Extract video frames|| Branch 1): Comprehensive Visual Analysis
        print("\n👁️ Comprehensive Visual Analysis")
        visual_analyzer = OpenCVAnalyzer()
        visual_results = visual_analyzer.analyze_video(frames)
        visual_analyzer.print_visual_summary(visual_results)
        
        # (parallel path 2|| Parent= Extract video frames|| Branch 2): Legacy Pose Analysis
        print("\n🧍 Legacy Pose & Behavioral Analysis")
        pose_analyzer = MoveNetAnalyzer()
        pose_results = pose_analyzer.analyze_frames(frames)
        print_pose_summary(pose_results)
        
        # (parallel path 2|| Parent= Extract video frames|| Branch 3): Enhanced Emotion Analysis
        print("\n🎭 Enhanced Facial Expression Analysis")
        emotion_analysis = analyze_emotions_in_frames(frames, Config.INTENDED_EMOTION)
        print_emotion_summary(emotion_analysis)
        save_emotion_analysis(emotion_analysis, Config.EMOTION_OUTPUT_FILE)
        
        # Get additional emotion insights
        emotion_stats = get_emotion_statistics(emotion_analysis)
        emotion_patterns = detect_emotion_patterns(emotion_analysis)
        emotional_engagement = analyze_emotional_engagement(emotion_analysis['emotion_results'])
        
        # Combine all emotion analysis results
        emotion_results = {
            **emotion_analysis,
            'statistics': emotion_stats,
            'patterns': emotion_patterns,
            'engagement': emotional_engagement
        }
        
        print(f"✅ Enhanced emotion analysis complete - Engagement level: {emotional_engagement['engagement_level']}")
        print(f"   Engagement score: {emotional_engagement['engagement_score']:.2f}/1.0")
        
        # (parallel path 2|| Parent= Extract video frames|| Branch 4): Environment Analysis
        print("\n🏠 Environment Analysis")
        environment_analyzer = EnvironmentAnalyzer()
        environment_results = environment_analyzer.analyze_video_frames(frames)
        print(f"✅ Environment analysis complete - Score: {environment_results.get('environment_score', 0):.2f}/1.0")
        

        # (parallel path 1|| Parent= Input_Video|| Branch 2): Audio Analysis
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OpenAI API key not found. Skipping audio and content analysis.")
        else:
            # Audio Analysis
            print("\n🎵 Audio Analysis")
            print("="*50)
            audio_analyzer = AudioAnalyzer(api_key)
            audio_results = audio_analyzer.analyze_audio(Config.VIDEO_PATH)
            audio_analyzer.print_audio_summary(audio_results)
            
            # Content Analysis
            print("\n📝 Content Analysis")
            print("="*50)
            content_analyzer = ContentAnalyzer(api_key)
            content_results = content_analyzer.analyze_content(audio_results["transcription"])
            content_analyzer.print_content_summary(content_results)
        
        # Aggregate Results
        print("\n" + "="*80)
        print("📊 AGGREGATING ANALYSIS RESULTS")
        print("="*80)
        
        aggregator = AnalysisAggregator()
        
        # Prepare data for aggregation
        all_results = {
            "visual_results": visual_results,
            "pose_results": pose_results,
            "emotion_results": emotion_results,
            "audio_results": audio_results,
            "content_results": content_results,
            "environment_results": environment_results
        }
        
        # Aggregate results
        aggregated_results = aggregator.aggregate_results(all_results)
        
        # Save aggregated results
        aggregated_results_path = aggregator.save_aggregated_results(aggregated_results)
        print(f"✅ Aggregated results saved: {aggregated_results_path}")
        
        # Expert Agent Evaluation
        print("\n" + "="*80)
        print("🧠 EXPERT AGENT EVALUATION")
        print("="*80)
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print("🔍 Running expert agent evaluations...")
            expert_system = ExpertAgentSystem(api_key)
            expert_results = expert_system.evaluate_all(all_results)
            
            print("✅ Expert evaluations completed!")
            
            # Add expert results to aggregated results
            aggregated_results["expert_results"] = expert_results
            
            # Generate Practice Plan
            print("\n" + "="*80)
            print("🎯 GENERATING 7-DAY PRACTICE PLAN")
            print("="*80)
            
            print("🔍 Creating personalized practice plan...")
            practice_plan_generator = PracticePlanGenerator(api_key)
            practice_plan = practice_plan_generator.generate_practice_plan(expert_results)
            
            # Save practice plan
            practice_plan_path = practice_plan_generator.save_practice_plan(practice_plan)
            print(f"✅ Practice plan generated: {practice_plan_path}")
        else:
            print("⚠️ OpenAI API key not found. Skipping expert evaluation and practice plan generation.")
        
        # Generate Enhanced Markdown Report
        print("\n" + "="*80)
        print("📊 GENERATING ENHANCED MARKDOWN REPORT")
        print("="*80)
        
        enhanced_report_generator = EnhancedReportGenerator()
        
        # Prepare configuration data
        config_data = {
            "Video Path": Config.VIDEO_PATH,
            "Frame Interval": Config.FRAME_INTERVAL,
            "Intended Emotion": Config.INTENDED_EMOTION,
            "Total Frames Analyzed": len(frames) if frames else 0,
            "Analysis Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Generate the enhanced report
        report_path = enhanced_report_generator.generate_comprehensive_report(
            video_path=Config.VIDEO_PATH,
            aggregated_results=aggregated_results,
            expert_evaluations=expert_results,
            practice_plan=practice_plan,
            config=config_data
        )
        
        print(f"✅ Enhanced report generated: {report_path}")
        
        # Final Step: AI-Powered Report Analysis (optional additional insights)
        print("\n" + "="*80)
        print("🤖 ADDITIONAL AI-POWERED REPORT ANALYSIS")
        print("="*80)
        
        print("🔍 Analyzing comprehensive report with AI...")
        report_analyzer = ReportAnalyzer(api_key)
        analysis_results = report_analyzer.analyze_report(report_path)
        
        if analysis_results["analysis_complete"]:
            print("✅ Additional AI analysis completed successfully!")
            
            # Save final report with AI insights
            print("📝 Generating final report with additional AI insights...")
            final_report_path = report_analyzer.save_final_report(
                report_path=report_path,
                ai_insights=analysis_results["ai_insights"]
            )
            
            print(f"✅ Final report with additional AI insights saved: {final_report_path}")
        else:
            print(f"❌ Additional AI analysis failed: {analysis_results.get('error', 'Unknown error')}")
        
        print("\n" + "="*80)
        print("✅ COMMUNICATION ANALYSIS COMPLETE!")
        print("="*80)
        print("\n📊 Overall Communication Score: {:.2f}/1.0".format(aggregated_results.get("overall_score", 0.0)))
        print("\n🎯 Your personalized 7-day practice plan is ready!")
        print(f"\n📄 View your full report at: {report_path}")
        print("="*80)




    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()