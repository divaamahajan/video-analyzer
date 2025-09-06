#!/usr/bin/env python3
"""
Comprehensive Report Generator
=============================

Generates detailed markdown reports from all analysis results.
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional, List


class ComprehensiveReportGenerator:
    """Generates comprehensive markdown reports from analysis results"""
    
    def __init__(self, output_dir: str = "reports"):
        """Initialize the report generator"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_comprehensive_report(
        self,
        video_path: str,
        visual_results: Optional[Dict[str, Any]] = None,
        pose_results: Optional[Dict[str, Any]] = None,
        emotion_results: Optional[Dict[str, Any]] = None,
        audio_results: Optional[Dict[str, Any]] = None,
        content_results: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a comprehensive markdown report from all analysis results"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = f"comprehensive_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self._generate_report_content(
                video_path, visual_results, pose_results, emotion_results,
                audio_results, content_results, config, timestamp
            ))
        
        return filepath
    
    def _generate_report_content(
        self,
        video_path: str,
        visual_results: Optional[Dict[str, Any]],
        pose_results: Optional[Dict[str, Any]],
        emotion_results: Optional[Dict[str, Any]],
        audio_results: Optional[Dict[str, Any]],
        content_results: Optional[Dict[str, Any]],
        config: Optional[Dict[str, Any]],
        timestamp: str
    ) -> str:
        """Generate the markdown content for the comprehensive report"""
        
        content = []
        
        # Header
        content.append("# 🎬 Comprehensive Video Analysis Report")
        content.append("")
        content.append(f"**Generated:** {timestamp}")
        content.append(f"**Video File:** {os.path.basename(video_path)}")
        content.append("")
        content.append("---")
        content.append("")
        
        # Executive Summary
        content.append("## 📊 Executive Summary")
        content.append("")
        content.append("This comprehensive analysis provides insights across multiple dimensions:")
        content.append("")
        content.append("- **Visual Language**: Body posture, facial expressions, gestures, and engagement")
        content.append("- **Audio Characteristics**: Voice quality, tone, pace, and clarity")
        content.append("- **Content Quality**: Linguistic features, readability, and communication effectiveness")
        content.append("")
        
        # Visual Analysis Section
        if visual_results:
            content.append("## 👁️ Visual Language Analysis (Mute & Watch)")
            content.append("")
            content.extend(self._format_visual_analysis(visual_results))
            content.append("")
        
        # Pose Analysis Section
        if pose_results:
            content.append("## 🧍 Body Language & Pose Analysis")
            content.append("")
            content.extend(self._format_pose_analysis(pose_results))
            content.append("")
        
        # Emotion Analysis Section
        if emotion_results:
            content.append("## 🎭 Facial Expression & Emotion Analysis")
            content.append("")
            content.extend(self._format_emotion_analysis(emotion_results))
            content.append("")
        
        # Audio Analysis Section
        if audio_results:
            content.append("## 🎵 Audio Voice, Tone & Clarity Analysis (Listen)")
            content.append("")
            content.extend(self._format_audio_analysis(audio_results))
            content.append("")
        
        # Content Analysis Section
        if content_results:
            content.append("## 📝 Content & Linguistic Analysis")
            content.append("")
            content.extend(self._format_content_analysis(content_results))
            content.append("")
        
        # Recommendations Section
        content.append("## 💡 Key Recommendations")
        content.append("")
        content.extend(self._generate_recommendations(visual_results, audio_results, content_results))
        content.append("")
        
        # Technical Details
        content.append("## 🔧 Technical Details")
        content.append("")
        content.append("### Analysis Configuration")
        if config:
            for key, value in config.items():
                content.append(f"- **{key}**: {value}")
        else:
            content.append("- Configuration details not available")
        content.append("")
        
        content.append("### Analysis Tools Used")
        content.append("- **Visual Analysis**: OpenCV, MediaPipe, TensorFlow")
        content.append("- **Audio Analysis**: Librosa, MoviePy, OpenAI Whisper")
        content.append("- **Content Analysis**: NLTK, Textstat, OpenAI GPT")
        content.append("- **Pose Detection**: MoveNet, MediaPipe")
        content.append("- **Emotion Detection**: DeepFace, TensorFlow")
        content.append("")
        
        # Footer
        content.append("---")
        content.append("")
        content.append("*Report generated by Video Analysis System*")
        content.append(f"*Generated on {timestamp}*")
        
        return "\n".join(content)
    
    def _format_visual_analysis(self, results: Dict[str, Any]) -> list:
        """Format visual analysis results"""
        content = []
        
        # Extract overall statistics
        overall_stats = results.get('overall_statistics', {})
        
        # Engagement Metrics
        content.append("### 🎯 Engagement & Expressiveness")
        content.append("")
        engagement_metrics = overall_stats.get('engagement_metrics', {})
        content.append(f"- **Overall Engagement Score**: {engagement_metrics.get('avg_engagement_score', 'N/A')}")
        content.append(f"- **Energy Level**: {engagement_metrics.get('avg_energy_score', 'N/A')}")
        content.append(f"- **Expressiveness**: {engagement_metrics.get('expressiveness_level', 'N/A')}")
        content.append(f"- **High Engagement Frames**: {engagement_metrics.get('high_engagement_frames', 'N/A')}")
        content.append("")
        
        # Posture Analysis
        posture_metrics = overall_stats.get('posture_metrics', {})
        if posture_metrics:
            content.append("### 🧍 Body Posture & Positioning")
            content.append("")
            content.append(f"- **Average Posture Score**: {posture_metrics.get('avg_posture_score', 'N/A')}")
            content.append(f"- **Posture Consistency**: {posture_metrics.get('posture_consistency', 'N/A')}")
            content.append(f"- **Slouching Frames**: {posture_metrics.get('slouching_frames', 'N/A')}")
            content.append("")
        
        # Facial Analysis
        expression_metrics = overall_stats.get('expression_metrics', {})
        if expression_metrics:
            content.append("### 😊 Facial Expressions & Eye Contact")
            content.append("")
            content.append(f"- **Average Smile Intensity**: {expression_metrics.get('avg_smile_intensity', 'N/A')}")
            content.append(f"- **Average Eye Contact**: {expression_metrics.get('avg_eye_contact', 'N/A')}")
            content.append(f"- **Smile Frequency**: {expression_metrics.get('smile_frequency', 'N/A')}%")
            content.append(f"- **Good Eye Contact Frames**: {expression_metrics.get('good_eye_contact_frames', 'N/A')}")
            content.append("")
        
        # Gesture Analysis
        gesture_metrics = overall_stats.get('gesture_metrics', {})
        if gesture_metrics:
            content.append("### 👋 Gestures & Hand Movements")
            content.append("")
            content.append(f"- **Average Gesture Frequency**: {gesture_metrics.get('avg_gesture_frequency', 'N/A')} per second")
            content.append(f"- **Gesture Consistency**: {gesture_metrics.get('gesture_consistency', 'N/A')}")
            content.append("")
        
        return content
    
    def _format_pose_analysis(self, results: List[Dict[str, Any]]) -> list:
        """Format pose analysis results"""
        content = []
        
        if not results:
            return content
        
        # Calculate statistics from the list of frame results
        total_frames = len(results)
        
        # Extract behavioral data from each frame
        basic_posture = {}
        head_movement = {}
        arm_hand_positions = {}
        energy_movement = {}
        
        for frame_result in results:
            if 'behaviors' in frame_result:
                behaviors = frame_result['behaviors']
                
                # Basic Posture
                for key in ['fidgeting', 'slouching', 'leaning', 'head_tilt', 'forward_head_posture']:
                    if key in behaviors:
                        basic_posture[key] = basic_posture.get(key, 0) + (1 if behaviors[key] else 0)
                
                # Head Movement
                for key in ['head_nod', 'head_shake', 'head_forward', 'head_back']:
                    if key in behaviors:
                        head_movement[key] = head_movement.get(key, 0) + (1 if behaviors[key] else 0)
                
                # Arm & Hand Positions
                for key in ['arms_crossed', 'hands_on_hips', 'hands_behind_head', 'arm_asymmetry', 'hand_touching_face', 'hands_in_pockets', 'gesturing_while_speaking']:
                    if key in behaviors:
                        arm_hand_positions[key] = arm_hand_positions.get(key, 0) + (1 if behaviors[key] else 0)
                
                # Energy & Movement
                if 'energy_level' in behaviors:
                    energy = behaviors['energy_level']
                    energy_movement[energy] = energy_movement.get(energy, 0) + 1
        
        # Basic Posture
        if basic_posture:
            content.append("### 🧍 Basic Posture Analysis")
            content.append("")
            content.append(f"- **Fidgeting**: {basic_posture.get('fidgeting', 0)}/{total_frames} frames ({basic_posture.get('fidgeting', 0)/total_frames*100:.1f}%)")
            content.append(f"- **Slouching**: {basic_posture.get('slouching', 0)}/{total_frames} frames ({basic_posture.get('slouching', 0)/total_frames*100:.1f}%)")
            content.append(f"- **Leaning**: {basic_posture.get('leaning', 0)}/{total_frames} frames ({basic_posture.get('leaning', 0)/total_frames*100:.1f}%)")
            content.append(f"- **Head Tilt**: {basic_posture.get('head_tilt', 0)}/{total_frames} frames ({basic_posture.get('head_tilt', 0)/total_frames*100:.1f}%)")
            content.append(f"- **Forward Head Posture**: {basic_posture.get('forward_head_posture', 0)}/{total_frames} frames ({basic_posture.get('forward_head_posture', 0)/total_frames*100:.1f}%)")
            content.append("")
        
        # Head Movement
        if head_movement:
            content.append("### 👤 Head Movement Patterns")
            content.append("")
            content.append(f"- **Head Nods**: {head_movement.get('head_nod', 0)}/{total_frames} frames ({head_movement.get('head_nod', 0)/total_frames*100:.1f}%)")
            content.append(f"- **Head Shakes**: {head_movement.get('head_shake', 0)}/{total_frames} frames ({head_movement.get('head_shake', 0)/total_frames*100:.1f}%)")
            content.append(f"- **Head Forward**: {head_movement.get('head_forward', 0)}/{total_frames} frames ({head_movement.get('head_forward', 0)/total_frames*100:.1f}%)")
            content.append(f"- **Head Back**: {head_movement.get('head_back', 0)}/{total_frames} frames ({head_movement.get('head_back', 0)/total_frames*100:.1f}%)")
            content.append("")
        
        # Arm & Hand Positions
        if arm_hand_positions:
            content.append("### 🤲 Arm & Hand Positions")
            content.append("")
            content.append(f"- **Arms Crossed**: {arm_hand_positions.get('arms_crossed', 0)}/{total_frames} frames ({arm_hand_positions.get('arms_crossed', 0)/total_frames*100:.1f}%)")
            content.append(f"- **Hands On Hips**: {arm_hand_positions.get('hands_on_hips', 0)}/{total_frames} frames ({arm_hand_positions.get('hands_on_hips', 0)/total_frames*100:.1f}%)")
            content.append(f"- **Hand Touching Face**: {arm_hand_positions.get('hand_touching_face', 0)}/{total_frames} frames ({arm_hand_positions.get('hand_touching_face', 0)/total_frames*100:.1f}%)")
            content.append(f"- **Gesturing While Speaking**: {arm_hand_positions.get('gesturing_while_speaking', 0)}/{total_frames} frames ({arm_hand_positions.get('gesturing_while_speaking', 0)/total_frames*100:.1f}%)")
            content.append("")
        
        # Energy & Movement
        if energy_movement:
            content.append("### ⚡ Energy & Movement Levels")
            content.append("")
            content.append(f"- **Low Energy**: {energy_movement.get('low_energy', 0)}/{total_frames} frames ({energy_movement.get('low_energy', 0)/total_frames*100:.1f}%)")
            content.append(f"- **Medium Energy**: {energy_movement.get('medium_energy', 0)}/{total_frames} frames ({energy_movement.get('medium_energy', 0)/total_frames*100:.1f}%)")
            content.append(f"- **High Energy**: {energy_movement.get('high_energy', 0)}/{total_frames} frames ({energy_movement.get('high_energy', 0)/total_frames*100:.1f}%)")
            
            # Calculate average movement magnitude
            movement_magnitudes = [frame.get('movement_magnitude', 0) for frame in results if 'movement_magnitude' in frame]
            if movement_magnitudes:
                avg_movement = sum(movement_magnitudes) / len(movement_magnitudes)
                content.append(f"- **Average Movement Magnitude**: {avg_movement:.4f}")
            content.append("")
        
        return content
    
    def _format_emotion_analysis(self, results: Dict[str, Any]) -> list:
        """Format emotion analysis results"""
        content = []
        
        if not results:
            return content
        
        emotion_results = results.get('emotion_results', [])
        emotion_mismatches = results.get('emotion_mismatches', [])
        intended_emotion = results.get('intended_emotion', 'N/A')
        
        if not emotion_results:
            return content
        
        # Count emotions
        emotion_counts = {}
        for result in emotion_results:
            emotion = result.get('dominant_emotion', 'unknown')
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total_frames = len(emotion_results)
        
        # Emotion Distribution
        if emotion_counts:
            content.append("### 😊 Emotion Distribution")
            content.append("")
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = count / total_frames * 100
                content.append(f"- **{emotion.title()}**: {count}/{total_frames} frames ({percentage:.1f}%)")
            content.append("")
        
        # Emotion Mismatch Analysis
        if emotion_mismatches:
            content.append("### ⚠️ Emotion Mismatch Analysis")
            content.append("")
            content.append(f"- **Intended Emotion**: {intended_emotion.title()}")
            content.append(f"- **Mismatches Detected**: {len(emotion_mismatches)}/{total_frames} frames ({len(emotion_mismatches)/total_frames*100:.1f}%)")
            
            # Show most common mismatched emotions
            mismatch_emotions = {}
            for mismatch in emotion_mismatches:
                emotion = mismatch.get('detected_emotion', 'unknown')
                mismatch_emotions[emotion] = mismatch_emotions.get(emotion, 0) + 1
            
            if mismatch_emotions:
                content.append("")
                content.append("**Most common mismatched emotions:**")
                for emotion, count in sorted(mismatch_emotions.items(), key=lambda x: x[1], reverse=True):
                    content.append(f"  - {emotion.title()}: {count} frames")
            content.append("")
        
        # Average confidence
        valid_results = [r for r in emotion_results if r.get('confidence', 0) > 0]
        if valid_results:
            avg_confidence = sum(r['confidence'] for r in valid_results) / len(valid_results)
            content.append("### 📊 Analysis Quality")
            content.append("")
            content.append(f"- **Average Confidence**: {avg_confidence:.1f}%")
            content.append(f"- **Valid Detections**: {len(valid_results)}/{total_frames} frames")
            content.append("")
        
        return content
    
    def _format_audio_analysis(self, results: Dict[str, Any]) -> list:
        """Format audio analysis results"""
        content = []
        
        # Transcription Overview
        if 'transcription' in results:
            trans = results['transcription']
            content.append("### 📝 Transcription Overview")
            content.append("")
            content.append(f"- **Full Text**: \"{trans.get('text', 'N/A')}\"")
            content.append(f"- **Total Segments**: {trans.get('segments', 'N/A')}")
            content.append(f"- **Total Words**: {trans.get('words', 'N/A')}")
            content.append("")
        
        # Pace & Rhythm
        if 'pace_and_rhythm' in results:
            pace = results['pace_and_rhythm']
            content.append("### 🏃 Pace & Rhythm")
            content.append("")
            content.append(f"- **Words per Minute**: {pace.get('wpm', 'N/A')}")
            content.append(f"- **Average Pause Length**: {pace.get('avg_pause_length', 'N/A')}s")
            content.append(f"- **Pause Frequency**: {pace.get('pause_frequency', 'N/A')} per minute")
            content.append(f"- **Filler Words**: {pace.get('filler_words', 'N/A')} ({pace.get('filler_rate', 'N/A')}/min)")
            content.append("")
        
        # Pitch & Tone
        if 'pitch_and_tone' in results:
            pitch = results['pitch_and_tone']
            content.append("### 🎵 Pitch & Tone Analysis")
            content.append("")
            content.append(f"- **Pitch Range**: {pitch.get('pitch_range', 'N/A')} Hz")
            content.append(f"- **Pitch Variance**: {pitch.get('pitch_variance', 'N/A')}")
            content.append(f"- **Pitch CV**: {pitch.get('pitch_cv', 'N/A')}")
            content.append(f"- **Contour Trend**: {pitch.get('contour_trend', 'N/A')}")
            content.append(f"- **Primary Emotion**: {pitch.get('primary_emotion', 'N/A')}")
            content.append(f"- **Rising Intonation**: {pitch.get('rising_intonation', 'N/A')}%")
            content.append(f"- **Falling Intonation**: {pitch.get('falling_intonation', 'N/A')}%")
            content.append(f"- **Emphasis Points**: {pitch.get('emphasis_points', 'N/A')}")
            content.append("")
        
        # Volume & Clarity
        if 'volume_and_clarity' in results:
            volume = results['volume_and_clarity']
            content.append("### 🔊 Volume & Clarity")
            content.append("")
            content.append(f"- **Volume Level**: {volume.get('volume', 'N/A')} dB")
            content.append(f"- **Volume Consistency**: {volume.get('volume_consistency', 'N/A')}")
            content.append(f"- **Clipping**: {volume.get('clipping', 'N/A')}%")
            content.append(f"- **Soft Segments**: {volume.get('soft_segments', 'N/A')}%")
            content.append(f"- **Dynamic Range**: {volume.get('dynamic_range', 'N/A')}")
            content.append(f"- **Clarity Score**: {volume.get('clarity_score', 'N/A')}")
            content.append("")
        
        # Pronunciation
        if 'pronunciation_analysis' in results:
            pron = results['pronunciation_analysis']
            content.append("### 🗣️ Pronunciation & Articulation")
            content.append("")
            content.append(f"- **Articulation Score**: {pron.get('articulation_score', 'N/A')}")
            content.append(f"- **Pronunciation Quality**: {pron.get('pronunciation_quality', 'N/A')}")
            content.append(f"- **Potential Mispronunciations**: {pron.get('mispronunciations', 'N/A')}")
            content.append(f"- **Vocabulary Richness**: {pron.get('vocabulary_richness', 'N/A')}")
            content.append(f"- **Average Word Length**: {pron.get('avg_word_length', 'N/A')}")
            content.append("")
        
        # Emotion & Sentiment
        if 'emotion_and_sentiment' in results:
            emotion = results['emotion_and_sentiment']
            content.append("### 😊 Emotion & Sentiment")
            content.append("")
            
            if 'audio_emotion' in emotion:
                audio_em = emotion['audio_emotion']
                content.append(f"- **Audio Emotional Intensity**: {audio_em.get('emotional_intensity', 'N/A')}")
            
            if 'text_emotion' in emotion:
                text_em = emotion['text_emotion']
                if 'text_sentiment' in text_em:
                    sent = text_em['text_sentiment']
                    content.append(f"- **Text Sentiment**: {sent.get('sentiment', 'N/A')} (score: {sent.get('sentiment_score', 'N/A')})")
                    content.append(f"- **Positive Words**: {sent.get('positive_words', 'N/A')}")
                    content.append(f"- **Negative Words**: {sent.get('negative_words', 'N/A')}")
                
                if 'confidence_indicators' in text_em:
                    conf = text_em['confidence_indicators']
                    content.append(f"- **Confidence Level**: {conf.get('confidence_level', 'N/A')} (score: {conf.get('confidence_score', 'N/A')})")
            content.append("")
        
        return content
    
    def _format_content_analysis(self, results: Dict[str, Any]) -> list:
        """Format content analysis results"""
        content = []
        
        # AI Insights
        if 'ai_analysis' in results:
            ai = results['ai_analysis']
            content.append("### 🤖 AI-Powered Insights")
            content.append("")
            content.append(f"**Analysis Status**: {'✅ Complete' if ai.get('analysis_complete', False) else '❌ Failed'}")
            if ai.get('analysis_complete') and 'ai_insights' in ai:
                content.append("")
                content.append("**Key Insights:**")
                content.append("")
                content.append(ai['ai_insights'])
                content.append("")
        
        # Structure & Organization
        if 'structure_and_organization' in results:
            struct = results['structure_and_organization']
            content.append("### 📊 Structure & Organization")
            content.append("")
            content.append(f"- **Total Sentences**: {struct.get('total_sentences', 'N/A')}")
            content.append(f"- **Average Sentence Length**: {struct.get('avg_sentence_length', 'N/A')} words")
            content.append(f"- **Sentence Length Variance**: {struct.get('sentence_length_variance', 'N/A')}")
            content.append(f"- **Idea Flow Quality**: {struct.get('idea_flow_quality', 'N/A')}")
            content.append(f"- **Opening Quality**: {struct.get('opening_quality', 'N/A')}")
            content.append(f"- **Closing Quality**: {struct.get('closing_quality', 'N/A')}")
            content.append("")
        
        # Clarity & Conciseness
        if 'clarity_and_conciseness' in results:
            clarity = results['clarity_and_conciseness']
            content.append("### 🔍 Clarity & Conciseness")
            content.append("")
            content.append(f"- **Overall Clarity Score**: {clarity.get('clarity_score', 'N/A')}")
            content.append(f"- **Redundancy Level**: {clarity.get('redundancy_level', 'N/A')}")
            content.append(f"- **Wordiness Level**: {clarity.get('wordiness_level', 'N/A')}")
            content.append(f"- **Ambiguity Level**: {clarity.get('ambiguity_level', 'N/A')}")
            content.append(f"- **Directness Level**: {clarity.get('directness_level', 'N/A')}")
            content.append("")
        
        # Advanced Vocabulary
        if 'advanced_vocabulary' in results:
            vocab = results['advanced_vocabulary']
            content.append("### 📚 Advanced Vocabulary")
            content.append("")
            content.append(f"- **Vocabulary Level**: {vocab.get('vocabulary_level', 'N/A')}")
            content.append(f"- **Lexical Diversity**: {vocab.get('lexical_diversity', 'N/A')}")
            content.append(f"- **Total Words**: {vocab.get('total_words', 'N/A')}")
            content.append(f"- **Unique Words**: {vocab.get('unique_words', 'N/A')}")
            content.append(f"- **Technical Words**: {vocab.get('technical_words', 'N/A')} ({vocab.get('technical_ratio', 'N/A')}%)")
            content.append(f"- **Simple Words**: {vocab.get('simple_words', 'N/A')} ({vocab.get('simple_ratio', 'N/A')}%)")
            content.append(f"- **Assertiveness Score**: {vocab.get('assertiveness_score', 'N/A')}")
            content.append("")
        
        # Readability & Engagement
        if 'readability_and_engagement' in results:
            read = results['readability_and_engagement']
            content.append("### 📖 Readability & Engagement")
            content.append("")
            content.append(f"- **Engagement Score**: {read.get('engagement_score', 'N/A')}")
            
            if 'readability_scores' in read:
                scores = read['readability_scores']
                content.append(f"- **Flesch Reading Ease**: {scores.get('flesch_reading_ease', 'N/A')}")
                content.append(f"- **Flesch-Kincaid Grade**: {scores.get('flesch_kincaid_grade', 'N/A')}")
                content.append(f"- **Gunning Fog Index**: {scores.get('gunning_fog_index', 'N/A')}")
                content.append(f"- **Readability Level**: {scores.get('readability_level', 'N/A')}")
            content.append("")
        
        # Topic & Semantic Flow
        if 'topic_and_semantic_flow' in results:
            topic = results['topic_and_semantic_flow']
            content.append("### 🌊 Topic & Semantic Flow")
            content.append("")
            content.append(f"- **Flow Quality**: {topic.get('flow_quality', 'N/A')}")
            content.append(f"- **Main Ideas**: {topic.get('main_ideas', 'N/A')} ({topic.get('main_idea_ratio', 'N/A')}%)")
            content.append(f"- **Supporting Evidence**: {topic.get('supporting_evidence', 'N/A')}")
            content.append(f"- **Support Ratio**: {topic.get('support_ratio', 'N/A')}%")
            content.append(f"- **Topic Relevance Score**: {topic.get('topic_relevance_score', 'N/A')}")
            content.append(f"- **Topic Shifts**: {topic.get('topic_shifts', 'N/A')}")
            content.append(f"- **Off-topic Sentences**: {topic.get('off_topic_sentences', 'N/A')}")
            content.append("")
        
        return content
    
    def _generate_recommendations(
        self,
        visual_results: Optional[Dict[str, Any]],
        audio_results: Optional[Dict[str, Any]],
        content_results: Optional[Dict[str, Any]]
    ) -> list:
        """Generate actionable recommendations based on analysis results"""
        recommendations = []
        
        recommendations.append("Based on the comprehensive analysis, here are key recommendations for improvement:")
        recommendations.append("")
        
        # Visual recommendations
        if visual_results:
            recommendations.append("### 👁️ Visual Communication")
            recommendations.append("")
            
            engagement_score = visual_results.get('engagement_score', 0)
            if engagement_score < 0.5:
                recommendations.append("- **Increase Engagement**: Work on maintaining better eye contact and facial expressions")
            
            energy_score = visual_results.get('energy_score', 0)
            if energy_score < 0.5:
                recommendations.append("- **Boost Energy**: Use more dynamic gestures and body movements")
            
            if 'posture_analysis' in visual_results:
                posture_score = visual_results['posture_analysis'].get('average_posture_score', 0)
                if posture_score < 0.7:
                    recommendations.append("- **Improve Posture**: Focus on maintaining upright posture and reducing slouching")
            
            recommendations.append("")
        
        # Audio recommendations
        if audio_results:
            recommendations.append("### 🎵 Voice & Speech")
            recommendations.append("")
            
            if 'pace_and_rhythm' in audio_results:
                wpm = audio_results['pace_and_rhythm'].get('wpm', 0)
                if wpm > 180:
                    recommendations.append("- **Slow Down**: Reduce speaking pace for better clarity and comprehension")
                elif wpm < 120:
                    recommendations.append("- **Speed Up**: Increase speaking pace to maintain audience engagement")
            
            if 'volume_and_clarity' in audio_results:
                clarity_score = audio_results['volume_and_clarity'].get('clarity_score', 0)
                if clarity_score < 0.5:
                    recommendations.append("- **Improve Clarity**: Focus on clearer pronunciation and articulation")
            
            if 'pronunciation_analysis' in audio_results:
                pron_quality = audio_results['pronunciation_analysis'].get('pronunciation_quality', '')
                if pron_quality == 'poor':
                    recommendations.append("- **Pronunciation Practice**: Work on clear pronunciation of key terms")
            
            recommendations.append("")
        
        # Content recommendations
        if content_results:
            recommendations.append("### 📝 Content & Language")
            recommendations.append("")
            
            if 'clarity_and_conciseness' in content_results:
                clarity_score = content_results['clarity_and_conciseness'].get('clarity_score', 0)
                if clarity_score < 0.7:
                    recommendations.append("- **Improve Clarity**: Use simpler, more direct language and reduce redundancy")
            
            if 'readability_and_engagement' in content_results:
                engagement_score = content_results['readability_and_engagement'].get('engagement_score', 0)
                if engagement_score < 0.3:
                    recommendations.append("- **Increase Engagement**: Use more questions, examples, and interactive elements")
            
            if 'advanced_vocabulary' in content_results:
                vocab_level = content_results['advanced_vocabulary'].get('vocabulary_level', '')
                if vocab_level == 'basic':
                    recommendations.append("- **Enhance Vocabulary**: Use more sophisticated and precise language")
                elif vocab_level == 'advanced':
                    recommendations.append("- **Simplify Language**: Use more accessible vocabulary for better audience understanding")
            
            recommendations.append("")
        
        # General recommendations
        recommendations.append("### 🎯 Overall Communication Strategy")
        recommendations.append("")
        recommendations.append("- **Practice Regularly**: Consistent practice improves all aspects of communication")
        recommendations.append("- **Record & Review**: Regular self-assessment helps identify improvement areas")
        recommendations.append("- **Audience Focus**: Tailor your communication style to your specific audience")
        recommendations.append("- **Feedback Integration**: Seek and incorporate feedback from trusted sources")
        
        return recommendations
