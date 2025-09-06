"""
Configuration settings for Video Analysis System
"""

class Config:
    """Main configuration class"""
    
    # Video settings
    VIDEO_PATH = "input_video.mp4"
    FRAME_INTERVAL = 1  # Process every frame (set to 5 for sampling)
    
    # Pose detection settings
    KEYPOINT_CONFIDENCE_THRESHOLD = 0.3
    KEYPOINT_HISTORY_SIZE = 10
    MOVENET_MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
    
    # Emotion detection settings
    INTENDED_EMOTION = 'happy'  # Expected emotion for mismatch detection
    
    # Analysis settings
    MIN_KEYPOINTS_FOR_ANALYSIS = 5
    
    # Output settings
    EMOTION_OUTPUT_FILE = "emotion_analysis.json"
    REPORT_OUTPUT_FILE = "reports/analysis_report.md"
    
    # Keypoint names for MoveNet
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
