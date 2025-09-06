#!/usr/bin/env python3
"""
Test script for separate report generation
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pose_report_generator import generate_pose_analysis_report, save_pose_report
from emotion_report_generator import generate_emotion_analysis_report, save_emotion_report

def test_report_generation():
    """Test the separate report generation functions"""
    print("🧪 Testing Separate Report Generation")
    print("="*50)
    
    # Mock data for testing
    mock_pose_results = [
        {
            'frame': 0,
            'movement_magnitude': 0.05,
            'behaviors': {
                'fidgeting': False,
                'slouching': False,
                'leaning': False,
                'head_tilt': False,
                'forward_head_posture': False,
                'head_nod_shake': {'nod': False, 'shake': False},
                'head_forward_back': {'head_forward': False, 'head_back': False},
                'arms_crossed': False,
                'hands_on_hips': False,
                'hands_behind_head': False,
                'arm_asymmetry': False,
                'hand_touching_face': {'any_hand_touching_face': False},
                'hands_in_pockets': {'any_hand_in_pocket': False},
                'gesturing_while_speaking': False,
                'shoulder_roll': False,
                'twisting': False,
                'legs_crossed': False,
                'sitting': False,
                'energy_level': 'medium_energy',
                'stiffness': False,
                'relaxed_stance': True,
                'gaze_direction': {'looking_at_camera': True, 'looking_away': False},
                'microexpressions': {'detected': False, 'intensity': 0, 'type': 'none'},
                'eye_contact_patterns': {'looking_at_camera': True, 'gaze_confidence': 0.8},
                'mouth_expressions': {'confidence': 0.0},
                'open_vs_closed_posture': {'is_open_posture': True, 'openness_score': 0.7},
                'confidence_indicators': {'confidence_score': 0.6, 'overall_confidence': 'medium'},
                'engagement_level': {'level': 'medium', 'score': 0.5},
                'defensive_gestures': {'defensive_score': 0.2, 'is_defensive': False},
                'interest_indicators': {'interest_level': 'medium', 'interest_score': 0.5},
                'stress_indicators': {'stress_level': 'low', 'stress_score': 0.3}
            }
        }
    ]
    
    mock_emotion_data = {
        'emotion_results': [
            {
                'frame': 0,
                'dominant_emotion': 'happy',
                'confidence': 85.5,
                'all_emotions': {'happy': 85.5, 'sad': 5.2, 'angry': 2.1, 'fear': 1.8, 'surprise': 3.2, 'disgust': 0.8, 'neutral': 1.4},
                'mismatch': {
                    'frame': 0,
                    'detected_emotion': 'happy',
                    'intended_emotion': 'happy',
                    'confidence': 85.5,
                    'is_mismatch': False,
                    'emotion_scores': {'happy': 85.5, 'sad': 5.2, 'angry': 2.1, 'fear': 1.8, 'surprise': 3.2, 'disgust': 0.8, 'neutral': 1.4}
                }
            }
        ],
        'emotion_mismatches': [],
        'intended_emotion': 'happy'
    }
    
    try:
        # Test pose report generation
        print("\n📊 Testing Pose Report Generation...")
        pose_report = generate_pose_analysis_report(
            pose_analysis_results=mock_pose_results,
            video_path="test_video.mp4",
            frame_count=1,
            frame_interval=1
        )
        save_pose_report(pose_report, 'test_pose_report.md')
        print("✅ Pose report generated successfully!")
        
        # Test emotion report generation
        print("\n🎭 Testing Emotion Report Generation...")
        emotion_report = generate_emotion_analysis_report(
            emotion_analysis_data=mock_emotion_data,
            video_path="test_video.mp4",
            frame_count=1,
            frame_interval=1
        )
        save_emotion_report(emotion_report, 'test_emotion_report.md')
        print("✅ Emotion report generated successfully!")
        
        print("\n🎉 All tests passed! Separate report generation is working correctly.")
        print("\nGenerated files:")
        print("  📊 test_pose_report.md")
        print("  🎭 test_emotion_report.md")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    test_report_generation()
