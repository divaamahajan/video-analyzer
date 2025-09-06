#!/usr/bin/env python3
"""
Video Analysis System
====================

A comprehensive video analysis tool that performs:
- Pose detection and behavioral analysis
- Facial expression recognition
- Movement pattern analysis
- Comprehensive reporting
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
from src.analysis.visual import extract_frames, PoseAnalyzer, SimpleVisualAnalyzer
from src.analysis.audio import AudioAnalyzer
from src.analysis.text import ContentAnalyzer
from src.analysis.report_analyzer import ReportAnalyzer
from src.utils.summary_generator import print_pose_summary
from src.utils.report_generator import ComprehensiveReportGenerator
from legacy.emotion_detection import analyze_emotions_in_frames, print_emotion_summary, save_emotion_analysis

def main():
    """Main analysis pipeline"""
    print("🎬 Starting Video Analysis System")
    print("="*50)
    
    # Initialize variables to store results
    visual_results = None
    pose_results = None
    emotion_results = None
    audio_results = None
    content_results = None
    
    try:
        # (parallel path 1|| Parent= Input_Video|| Branch 1): Extract video frames
        print("\n📹 Step 1: Video Frame Extraction")
        frames = extract_frames(Config.VIDEO_PATH, Config.FRAME_INTERVAL)
        
        if not frames:
            print("❌ No frames extracted. Please check video file.")
            return
        

        # (parallel path 2|| Parent= Extract video frames|| Branch 1): Comprehensive Visual Analysis
        print("\n👁️ Comprehensive Visual Analysis")
        visual_analyzer = SimpleVisualAnalyzer()
        visual_results = visual_analyzer.analyze_video(frames)
        visual_analyzer.print_visual_summary(visual_results)
        
        # (parallel path 2|| Parent= Extract video frames|| Branch 2): Legacy Pose Analysis
        print("\n🧍 Legacy Pose & Behavioral Analysis")
        pose_analyzer = PoseAnalyzer()
        pose_results = pose_analyzer.analyze_frames(frames)
        print_pose_summary(pose_results)
        
        # (parallel path 2|| Parent= Extract video frames|| Branch 3): Legacy Emotion Analysis
        print("\n🎭 Legacy Facial Expression Analysis")
        emotion_analysis = analyze_emotions_in_frames(frames, Config.INTENDED_EMOTION)
        print_emotion_summary(emotion_analysis)
        save_emotion_analysis(emotion_analysis, Config.EMOTION_OUTPUT_FILE)
        emotion_results = emotion_analysis
        

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
        
        # Generate Comprehensive Markdown Report
        print("\n" + "="*80)
        print("📊 GENERATING COMPREHENSIVE MARKDOWN REPORT")
        print("="*80)
        
        report_generator = ComprehensiveReportGenerator()
        
        # Prepare configuration data
        config_data = {
            "Video Path": Config.VIDEO_PATH,
            "Frame Interval": Config.FRAME_INTERVAL,
            "Intended Emotion": Config.INTENDED_EMOTION,
            "Total Frames Analyzed": len(frames) if frames else 0,
            "Analysis Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Generate the comprehensive report
        report_path = report_generator.generate_comprehensive_report(
            video_path=Config.VIDEO_PATH,
            visual_results=visual_results,
            pose_results=pose_results,
            emotion_results=emotion_results,
            audio_results=audio_results,
            content_results=content_results,
            config=config_data
        )
        
        print(f"✅ Comprehensive report generated: {report_path}")
        
        # Final Step: AI-Powered Report Analysis
        if api_key:
            print("\n" + "="*80)
            print("🤖 AI-POWERED REPORT ANALYSIS")
            print("="*80)
            
            print("🔍 Analyzing comprehensive report with AI...")
            report_analyzer = ReportAnalyzer(api_key)
            analysis_results = report_analyzer.analyze_report(report_path)
            
            if analysis_results["analysis_complete"]:
                print("✅ AI analysis completed successfully!")
                
                # Save final report with AI insights
                print("📝 Generating final report with AI insights...")
                final_report_path = report_analyzer.save_final_report(
                    report_path=report_path,
                    ai_insights=analysis_results["ai_insights"]
                )
                
                print(f"✅ Final report with AI insights saved: {final_report_path}")
                
                # Display AI insights summary
                print("\n🤖 AI INSIGHTS SUMMARY:")
                print("="*60)
                ai_insights = analysis_results["ai_insights"]
                print(ai_insights)
                print("="*60)
                
            else:
                print(f"❌ AI analysis failed: {analysis_results.get('error', 'Unknown error')}")
        else:
            print("⚠️ Skipping AI report analysis - OpenAI API key required")
        
        print("\n" + "="*80)
        print("✅ COMPREHENSIVE ANALYSIS COMPLETE!")
        print("="*80)




    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()