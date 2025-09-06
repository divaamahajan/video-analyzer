"""
Pose analysis module
"""

import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tqdm import tqdm
from src.config.settings import Config
from legacy.pose_detection import analyze_pose_behaviors, calculate_angle, calculate_distance

class PoseAnalyzer:
    """Main pose analysis class"""
    
    def __init__(self):
        self.movenet = None
        self.prev_keypoints = None
        self.keypoints_history = []
        self.analysis_results = []
    
    def load_model(self):
        """Load MoveNet pose detection model"""
        print("🤖 Loading MoveNet pose detection model...")
        model = hub.load(Config.MOVENET_MODEL_URL)
        self.movenet = model.signatures['serving_default']
        print("✅ MoveNet model loaded successfully")
    
    def process_frame(self, frame, frame_idx):
        """Process a single frame for pose analysis"""
        # Resize frame to 192x192 (MoveNet input size)
        input_frame = cv2.resize(frame, (192, 192))
        input_frame = tf.cast(input_frame, dtype=tf.int32)
        input_frame = tf.expand_dims(input_frame, axis=0)
        
        # Get pose keypoints
        outputs = self.movenet(input_frame)
        keypoints = outputs['output_0'].numpy()[0][0]  # Shape: (17, 3)
        
        # Filter keypoints by confidence
        valid_keypoints = keypoints[keypoints[:, 2] > Config.KEYPOINT_CONFIDENCE_THRESHOLD]
        
        # Calculate detailed pose metrics
        frame_analysis = {
            'frame': frame_idx,
            'keypoints': keypoints,
            'valid_keypoints_count': len(valid_keypoints)
        }
        
        # Calculate joint angles
        frame_analysis['angles'] = self._calculate_joint_angles(keypoints)
        
        # Calculate hand positions
        frame_analysis['hand_positions'] = self._calculate_hand_positions(keypoints)
        
        # Calculate movement magnitude
        frame_analysis['movement_magnitude'] = self._calculate_movement_magnitude(keypoints)
        
        # Comprehensive behavioral analysis
        frame_analysis['behaviors'] = analyze_pose_behaviors(keypoints, self.prev_keypoints, self.keypoints_history)
        
        # Additional pose metrics
        frame_analysis['pose_metrics'] = self._calculate_pose_metrics(keypoints)
        
        # Update tracking variables
        self.prev_keypoints = keypoints.copy()
        self.keypoints_history.append(keypoints.copy())
        
        # Maintain history size
        if len(self.keypoints_history) > Config.KEYPOINT_HISTORY_SIZE:
            self.keypoints_history.pop(0)
        
        return frame_analysis
    
    def _calculate_joint_angles(self, keypoints):
        """Calculate joint angles from keypoints"""
        angles = {}
        try:
            # Elbow angles
            if all(keypoints[i][2] > Config.KEYPOINT_CONFIDENCE_THRESHOLD for i in [5, 7, 9]):
                angles['left_elbow'] = calculate_angle(keypoints[5], keypoints[7], keypoints[9])
            if all(keypoints[i][2] > Config.KEYPOINT_CONFIDENCE_THRESHOLD for i in [6, 8, 10]):
                angles['right_elbow'] = calculate_angle(keypoints[6], keypoints[8], keypoints[10])
            
            # Knee angles
            if all(keypoints[i][2] > Config.KEYPOINT_CONFIDENCE_THRESHOLD for i in [11, 13, 15]):
                angles['left_knee'] = calculate_angle(keypoints[11], keypoints[13], keypoints[15])
            if all(keypoints[i][2] > Config.KEYPOINT_CONFIDENCE_THRESHOLD for i in [12, 14, 16]):
                angles['right_knee'] = calculate_angle(keypoints[12], keypoints[14], keypoints[16])
            
            # Shoulder angles
            if all(keypoints[i][2] > Config.KEYPOINT_CONFIDENCE_THRESHOLD for i in [7, 5, 6]):
                angles['left_shoulder'] = calculate_angle(keypoints[7], keypoints[5], keypoints[6])
            if all(keypoints[i][2] > Config.KEYPOINT_CONFIDENCE_THRESHOLD for i in [8, 6, 5]):
                angles['right_shoulder'] = calculate_angle(keypoints[8], keypoints[6], keypoints[5])
                
        except Exception as e:
            print(f"⚠️  Error calculating angles: {e}")
            angles = {}
        
        return angles
    
    def _calculate_hand_positions(self, keypoints):
        """Calculate hand positions from keypoints"""
        hand_positions = {}
        
        if keypoints[9][2] > Config.KEYPOINT_CONFIDENCE_THRESHOLD:  # left wrist
            hand_positions['left_wrist'] = {
                'x': keypoints[9][1],
                'y': keypoints[9][0],
                'confidence': keypoints[9][2]
            }
        
        if keypoints[10][2] > Config.KEYPOINT_CONFIDENCE_THRESHOLD:  # right wrist
            hand_positions['right_wrist'] = {
                'x': keypoints[10][1],
                'y': keypoints[10][0],
                'confidence': keypoints[10][2]
            }
        
        return hand_positions
    
    def _calculate_movement_magnitude(self, keypoints):
        """Calculate movement magnitude between frames"""
        if self.prev_keypoints is None:
            return 0
        
        total_movement = 0
        valid_movements = 0
        
        for j in range(len(keypoints)):
            if (keypoints[j][2] > Config.KEYPOINT_CONFIDENCE_THRESHOLD and 
                self.prev_keypoints[j][2] > Config.KEYPOINT_CONFIDENCE_THRESHOLD):
                movement = calculate_distance(keypoints[j], self.prev_keypoints[j])
                total_movement += movement
                valid_movements += 1
        
        return total_movement / valid_movements if valid_movements > 0 else 0
    
    def _calculate_pose_metrics(self, keypoints):
        """Calculate additional pose metrics"""
        pose_metrics = {}
        
        # Calculate torso length (shoulder to hip distance)
        if all(keypoints[i][2] > Config.KEYPOINT_CONFIDENCE_THRESHOLD for i in [5, 6, 11, 12]):
            shoulder_center = [(keypoints[5][1] + keypoints[6][1]) / 2, 
                               (keypoints[5][0] + keypoints[6][0]) / 2]
            hip_center = [(keypoints[11][1] + keypoints[12][1]) / 2, 
                          (keypoints[11][0] + keypoints[12][0]) / 2]
            pose_metrics['torso_length'] = calculate_distance(shoulder_center, hip_center)
        
        # Calculate arm span (wrist to wrist distance)
        if all(keypoints[i][2] > Config.KEYPOINT_CONFIDENCE_THRESHOLD for i in [9, 10]):
            pose_metrics['arm_span'] = calculate_distance(keypoints[9], keypoints[10])
        
        return pose_metrics
    
    def analyze_frames(self, frames):
        """Analyze all frames for pose detection"""
        print("🧍 Starting pose analysis...")
        
        if not self.movenet:
            self.load_model()
        
        self.analysis_results = []
        
        for i, frame in enumerate(tqdm(frames, desc="Processing frames", unit="frame")):
            result = self.process_frame(frame, i)
            self.analysis_results.append(result)
        
        print(f"✅ Pose analysis complete! Processed {len(self.analysis_results)} frames")
        return self.analysis_results
    
    def get_results(self):
        """Get analysis results"""
        return self.analysis_results
