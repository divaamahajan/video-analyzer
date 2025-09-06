#!/usr/bin/env python3
"""
Simplified Visual Analysis Module
=================================

Uses OpenCV and existing libraries to analyze visual presentation features
without requiring MediaPipe.
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Tuple
import math
from collections import deque
from scipy import stats


class SimpleVisualAnalyzer:
    """Simplified visual analysis using OpenCV and basic computer vision"""
    
    def __init__(self):
        """Initialize visual analysis components"""
        # Initialize face cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Tracking variables
        self.face_history = deque(maxlen=30)
        self.movement_history = deque(maxlen=30)
        self.expression_history = deque(maxlen=30)
        
    def analyze_frame(self, frame: np.ndarray, frame_idx: int) -> Dict[str, Any]:
        """Analyze a single frame for visual features"""
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize frame analysis
        frame_analysis = {
            'frame_idx': frame_idx,
            'timestamp': frame_idx / 30.0,
            'body_posture': {},
            'facial_expressions': {},
            'gestures': {},
            'spatial_awareness': {},
            'engagement_metrics': {}
        }
        
        # 1. Face Detection and Analysis
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Use the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # 2. Facial Expression Analysis
            frame_analysis['facial_expressions'] = self._analyze_facial_expressions(
                frame, gray, face, height, width
            )
            
            # 3. Body Posture Analysis (simplified)
            frame_analysis['body_posture'] = self._analyze_body_posture_simple(
                frame, face, height, width
            )
            
            # 4. Spatial Awareness
            frame_analysis['spatial_awareness'] = self._analyze_spatial_awareness_simple(
                face, height, width
            )
            
            # 5. Gesture Analysis (simplified)
            frame_analysis['gestures'] = self._analyze_gestures_simple(
                frame, face, height, width
            )
            
            # 6. Engagement Metrics
            frame_analysis['engagement_metrics'] = self._calculate_engagement_metrics_simple(
                frame_analysis
            )
            
            # Update tracking
            self._update_histories(frame_analysis, face)
        
        return frame_analysis
    
    def _analyze_facial_expressions(self, frame: np.ndarray, gray: np.ndarray, 
                                  face: Tuple, height: int, width: int) -> Dict[str, Any]:
        """Analyze facial expressions using OpenCV"""
        x, y, w, h = face
        face_roi = gray[y:y+h, x:x+w]
        
        expression_analysis = {
            'smile_intensity': 0.0,
            'eye_contact_quality': 0.0,
            'blink_rate': 0.0,
            'facial_symmetry': 0.0,
            'mouth_tension': 0.0,
            'overall_expression': 'neutral'
        }
        
        # 1. Smile detection
        smiles = self.smile_cascade.detectMultiScale(face_roi, 1.8, 20)
        if len(smiles) > 0:
            expression_analysis['smile_intensity'] = min(1.0, len(smiles) * 0.3)
        
        # 2. Eye detection and contact analysis
        eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3)
        if len(eyes) >= 2:
            # Calculate eye contact quality based on eye position
            eye_centers = []
            for (ex, ey, ew, eh) in eyes:
                eye_center_x = x + ex + ew // 2
                eye_center_y = y + ey + eh // 2
                eye_centers.append((eye_center_x, eye_center_y))
            
            if len(eye_centers) >= 2:
                # Calculate if eyes are looking towards camera center
                frame_center_x = width // 2
                eye_center_x = np.mean([ec[0] for ec in eye_centers])
                eye_offset = abs(eye_center_x - frame_center_x) / (width / 2)
                expression_analysis['eye_contact_quality'] = max(0, 1 - eye_offset)
        
        # 3. Facial symmetry (simplified)
        if len(eyes) >= 2:
            left_eye = min(eyes, key=lambda e: e[0])  # Leftmost eye
            right_eye = max(eyes, key=lambda e: e[0])  # Rightmost eye
            
            face_center_x = x + w // 2
            left_eye_center = left_eye[0] + left_eye[2] // 2
            right_eye_center = right_eye[0] + right_eye[2] // 2
            
            left_dist = abs(left_eye_center - face_center_x)
            right_dist = abs(right_eye_center - face_center_x)
            
            symmetry = 1 - abs(left_dist - right_dist) / (w / 2)
            expression_analysis['facial_symmetry'] = max(0, symmetry)
        
        # 4. Mouth tension (simplified)
        mouth_region = face_roi[int(h*0.6):h, int(w*0.2):int(w*0.8)]
        if mouth_region.size > 0:
            # Use edge detection to measure mouth opening
            edges = cv2.Canny(mouth_region, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            expression_analysis['mouth_tension'] = min(1.0, edge_density * 5)
        
        # 5. Overall expression classification
        expression_analysis['overall_expression'] = self._classify_expression_simple(expression_analysis)
        
        return expression_analysis
    
    def _analyze_body_posture_simple(self, frame: np.ndarray, face: Tuple, 
                                   height: int, width: int) -> Dict[str, Any]:
        """Simplified body posture analysis"""
        x, y, w, h = face
        posture_analysis = {
            'posture_score': 0.0,
            'sitting_standing': 'unknown',
            'leaning_direction': 'neutral',
            'slouching_detected': False,
            'shoulder_alignment': 0.0,
            'head_position': 'neutral'
        }
        
        # 1. Head position analysis
        face_center_x = x + w // 2
        frame_center_x = width // 2
        head_offset = (face_center_x - frame_center_x) / (width / 2)
        
        if abs(head_offset) > 0.1:
            posture_analysis['head_position'] = 'tilted_left' if head_offset < 0 else 'tilted_right'
        
        # 2. Leaning analysis (based on face size and position)
        face_size_ratio = (w * h) / (width * height)
        face_y_ratio = y / height
        
        if face_size_ratio > 0.1:  # Large face = close to camera
            posture_analysis['sitting_standing'] = 'close'
        elif face_y_ratio > 0.3:  # Face high in frame
            posture_analysis['sitting_standing'] = 'sitting'
        else:
            posture_analysis['sitting_standing'] = 'standing'
        
        # 3. Slouching detection (face too high in frame)
        if face_y_ratio > 0.4:
            posture_analysis['slouching_detected'] = True
        
        # 4. Shoulder alignment (simplified - based on face angle)
        posture_analysis['shoulder_alignment'] = max(0, 1 - abs(head_offset) * 2)
        
        # 5. Overall posture score
        posture_analysis['posture_score'] = (
            posture_analysis['shoulder_alignment'] * 0.4 +
            (0.8 if not posture_analysis['slouching_detected'] else 0.2) * 0.6
        )
        
        return posture_analysis
    
    def _analyze_spatial_awareness_simple(self, face: Tuple, height: int, width: int) -> Dict[str, Any]:
        """Simplified spatial awareness analysis"""
        x, y, w, h = face
        spatial_analysis = {
            'distance_from_camera': 'unknown',
            'centering_score': 0.0,
            'space_usage': 'limited',
            'frame_occupancy': 0.0
        }
        
        # 1. Distance from camera (based on face size)
        face_area = w * h
        frame_area = width * height
        face_ratio = face_area / frame_area
        
        if face_ratio > 0.15:
            spatial_analysis['distance_from_camera'] = 'close'
        elif face_ratio < 0.05:
            spatial_analysis['distance_from_camera'] = 'far'
        else:
            spatial_analysis['distance_from_camera'] = 'optimal'
        
        # 2. Centering score
        face_center_x = x + w // 2
        frame_center_x = width // 2
        centering_offset = abs(face_center_x - frame_center_x) / (width / 2)
        spatial_analysis['centering_score'] = max(0, 1 - centering_offset)
        
        # 3. Frame occupancy
        spatial_analysis['frame_occupancy'] = min(1.0, face_ratio * 3)  # Scale up face ratio
        
        # 4. Space usage (based on movement history)
        if len(self.movement_history) > 5:
            recent_positions = list(self.movement_history)[-10:]
            x_positions = [pos['x'] for pos in recent_positions]
            x_range = max(x_positions) - min(x_positions) if x_positions else 0
            
            if x_range > 0.2:
                spatial_analysis['space_usage'] = 'extensive'
            elif x_range > 0.1:
                spatial_analysis['space_usage'] = 'moderate'
            else:
                spatial_analysis['space_usage'] = 'limited'
        
        return spatial_analysis
    
    def _analyze_gestures_simple(self, frame: np.ndarray, face: Tuple, 
                               height: int, width: int) -> Dict[str, Any]:
        """Simplified gesture analysis"""
        gesture_analysis = {
            'hand_gesture_frequency': 0.0,
            'gesture_size': 'small',
            'hand_positions': 'unknown',
            'gesture_smoothness': 0.0
        }
        
        # 1. Movement analysis (based on face movement)
        if len(self.movement_history) > 1:
            movement_list = list(self.movement_history)
            prev_pos = movement_list[-2]
            curr_pos = movement_list[-1]
            
            movement = math.sqrt(
                (curr_pos['x'] - prev_pos['x'])**2 + 
                (curr_pos['y'] - prev_pos['y'])**2
            )
            
            if movement > 0.05:
                gesture_analysis['gesture_size'] = 'large'
            elif movement > 0.02:
                gesture_analysis['gesture_size'] = 'medium'
            else:
                gesture_analysis['gesture_size'] = 'small'
            
            # Gesture frequency
            movements = sum(1 for i in range(1, len(movement_list)) 
                          if math.sqrt((movement_list[i]['x'] - movement_list[i-1]['x'])**2 + 
                                     (movement_list[i]['y'] - movement_list[i-1]['y'])**2) > 0.01)
            gesture_analysis['hand_gesture_frequency'] = movements / len(movement_list) * 30
        
        # 2. Hand position (simplified - based on face position)
        face_y = face[1] + face[3] // 2
        if face_y < height * 0.3:
            gesture_analysis['hand_positions'] = 'high'
        elif face_y > height * 0.7:
            gesture_analysis['hand_positions'] = 'low'
        else:
            gesture_analysis['hand_positions'] = 'middle'
        
        return gesture_analysis
    
    def _calculate_engagement_metrics_simple(self, frame_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate engagement metrics"""
        engagement = {
            'overall_engagement_score': 0.0,
            'expressiveness_level': 'low',
            'energy_score': 0.0,
            'confidence_indicators': []
        }
        
        # 1. Overall engagement score
        posture_score = frame_analysis['body_posture']['posture_score']
        smile_score = frame_analysis['facial_expressions']['smile_intensity']
        eye_contact = frame_analysis['facial_expressions']['eye_contact_quality']
        gesture_freq = frame_analysis['gestures']['hand_gesture_frequency']
        
        engagement['overall_engagement_score'] = (
            posture_score * 0.3 +
            smile_score * 0.3 +
            eye_contact * 0.2 +
            min(gesture_freq / 3, 1) * 0.2
        )
        
        # 2. Expressiveness level
        if engagement['overall_engagement_score'] > 0.7:
            engagement['expressiveness_level'] = 'high'
        elif engagement['overall_engagement_score'] > 0.4:
            engagement['expressiveness_level'] = 'medium'
        else:
            engagement['expressiveness_level'] = 'low'
        
        # 3. Confidence indicators
        if posture_score > 0.6:
            engagement['confidence_indicators'].append('good_posture')
        if eye_contact > 0.6:
            engagement['confidence_indicators'].append('good_eye_contact')
        if not frame_analysis['body_posture']['slouching_detected']:
            engagement['confidence_indicators'].append('upright_position')
        
        # 4. Energy score
        movement_energy = min(gesture_freq / 2, 1)
        expression_energy = smile_score
        engagement['energy_score'] = (movement_energy + expression_energy) / 2
        
        return engagement
    
    def _classify_expression_simple(self, expression_data: Dict[str, Any]) -> str:
        """Classify facial expression"""
        smile = expression_data['smile_intensity']
        mouth_tension = expression_data['mouth_tension']
        
        if smile > 0.5:
            return 'happy'
        elif mouth_tension > 0.7:
            return 'tense'
        else:
            return 'neutral'
    
    def _update_histories(self, frame_analysis: Dict[str, Any], face: Tuple):
        """Update tracking histories"""
        x, y, w, h = face
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        self.movement_history.append({
            'x': face_center_x,
            'y': face_center_y
        })
        
        self.expression_history.append({
            'smile_intensity': frame_analysis['facial_expressions']['smile_intensity'],
            'eye_contact': frame_analysis['facial_expressions']['eye_contact_quality']
        })
    
    def analyze_video(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze entire video for visual features"""
        print("\n👁️ Starting Simplified Visual Analysis")
        print("="*50)
        
        all_analyses = []
        
        for i, frame in enumerate(frames):
            analysis = self.analyze_frame(frame, i)
            all_analyses.append(analysis)
        
        # Calculate overall statistics
        overall_stats = self._calculate_overall_statistics(all_analyses)
        
        return {
            'frame_analyses': all_analyses,
            'overall_statistics': overall_stats
        }
    
    def _calculate_overall_statistics(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall statistics"""
        if not analyses:
            return {}
        
        # Filter out frames without face detection
        valid_analyses = [a for a in analyses if a['facial_expressions']['smile_intensity'] >= 0]
        
        if not valid_analyses:
            return {'error': 'No valid face detections'}
        
        stats = {
            'total_frames': len(analyses),
            'valid_frames': len(valid_analyses),
            'posture_metrics': {},
            'expression_metrics': {},
            'gesture_metrics': {},
            'engagement_metrics': {},
            'recommendations': []
        }
        
        # Posture statistics
        posture_scores = [a['body_posture']['posture_score'] for a in valid_analyses]
        stats['posture_metrics'] = {
            'avg_posture_score': np.mean(posture_scores),
            'posture_consistency': 1 - np.std(posture_scores) if len(posture_scores) > 1 else 0,
            'slouching_frames': sum(1 for a in valid_analyses if a['body_posture']['slouching_detected'])
        }
        
        # Expression statistics
        smile_scores = [a['facial_expressions']['smile_intensity'] for a in valid_analyses]
        eye_contact_scores = [a['facial_expressions']['eye_contact_quality'] for a in valid_analyses]
        stats['expression_metrics'] = {
            'avg_smile_intensity': np.mean(smile_scores),
            'avg_eye_contact': np.mean(eye_contact_scores),
            'smile_frequency': sum(1 for s in smile_scores if s > 0.3) / len(smile_scores),
            'good_eye_contact_frames': sum(1 for e in eye_contact_scores if e > 0.6)
        }
        
        # Gesture statistics
        gesture_freqs = [a['gestures']['hand_gesture_frequency'] for a in valid_analyses]
        stats['gesture_metrics'] = {
            'avg_gesture_frequency': np.mean(gesture_freqs),
            'gesture_consistency': 1 - np.std(gesture_freqs) if len(gesture_freqs) > 1 else 0
        }
        
        # Engagement statistics
        engagement_scores = [a['engagement_metrics']['overall_engagement_score'] for a in valid_analyses]
        energy_scores = [a['engagement_metrics']['energy_score'] for a in valid_analyses]
        stats['engagement_metrics'] = {
            'avg_engagement_score': np.mean(engagement_scores),
            'avg_energy_score': np.mean(energy_scores),
            'high_engagement_frames': sum(1 for e in engagement_scores if e > 0.6),
            'expressiveness_level': 'high' if np.mean(engagement_scores) > 0.6 else 'medium' if np.mean(engagement_scores) > 0.3 else 'low'
        }
        
        # Generate recommendations
        stats['recommendations'] = self._generate_recommendations(stats)
        
        return stats
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Posture recommendations
        if stats['posture_metrics']['avg_posture_score'] < 0.5:
            recommendations.append("Improve posture: Sit/stand straighter with shoulders back")
        
        if stats['posture_metrics']['slouching_frames'] > stats['valid_frames'] * 0.3:
            recommendations.append("Reduce slouching: Maintain upright position throughout presentation")
        
        # Expression recommendations
        if stats['expression_metrics']['avg_smile_intensity'] < 0.2:
            recommendations.append("Increase facial expressiveness: Smile more to appear more engaging")
        
        if stats['expression_metrics']['avg_eye_contact'] < 0.5:
            recommendations.append("Improve eye contact: Look directly at the camera/audience more often")
        
        # Gesture recommendations
        if stats['gesture_metrics']['avg_gesture_frequency'] < 0.5:
            recommendations.append("Use more hand gestures: Add natural hand movements to emphasize points")
        
        # Engagement recommendations
        if stats['engagement_metrics']['avg_engagement_score'] < 0.4:
            recommendations.append("Increase overall engagement: Combine better posture, expressions, and gestures")
        
        return recommendations
    
    def print_visual_summary(self, analysis_results: Dict[str, Any]):
        """Print visual analysis summary"""
        print("\n" + "="*60)
        print("👁️ SIMPLIFIED VISUAL ANALYSIS REPORT")
        print("="*60)
        
        stats = analysis_results['overall_statistics']
        
        if 'error' in stats:
            print(f"❌ {stats['error']}")
            return
        
        # Posture Analysis
        print(f"\n🧍 BODY POSTURE & POSITIONING")
        print(f"   Average posture score: {stats['posture_metrics']['avg_posture_score']:.3f}")
        print(f"   Posture consistency: {stats['posture_metrics']['posture_consistency']:.3f}")
        print(f"   Slouching frames: {stats['posture_metrics']['slouching_frames']}/{stats['valid_frames']}")
        
        # Facial Expression Analysis
        print(f"\n😊 FACIAL EXPRESSIONS & ENGAGEMENT")
        print(f"   Average smile intensity: {stats['expression_metrics']['avg_smile_intensity']:.3f}")
        print(f"   Average eye contact: {stats['expression_metrics']['avg_eye_contact']:.3f}")
        print(f"   Smile frequency: {stats['expression_metrics']['smile_frequency']:.1%} of frames")
        print(f"   Good eye contact: {stats['expression_metrics']['good_eye_contact_frames']}/{stats['valid_frames']} frames")
        
        # Gesture Analysis
        print(f"\n👋 GESTURES & HAND MOVEMENTS")
        print(f"   Average gesture frequency: {stats['gesture_metrics']['avg_gesture_frequency']:.2f} per second")
        print(f"   Gesture consistency: {stats['gesture_metrics']['gesture_consistency']:.3f}")
        
        # Overall Engagement
        print(f"\n🎯 OVERALL ENGAGEMENT & EXPRESSIVENESS")
        print(f"   Engagement score: {stats['engagement_metrics']['avg_engagement_score']:.3f}")
        print(f"   Energy score: {stats['engagement_metrics']['avg_energy_score']:.3f}")
        print(f"   Expressiveness level: {stats['engagement_metrics']['expressiveness_level']}")
        print(f"   High engagement frames: {stats['engagement_metrics']['high_engagement_frames']}/{stats['valid_frames']}")
        
        # Recommendations
        if stats['recommendations']:
            print(f"\n💡 IMPROVEMENT RECOMMENDATIONS")
            for i, rec in enumerate(stats['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "="*60)
        print("✅ Visual Analysis Complete!")
        print("="*60)
