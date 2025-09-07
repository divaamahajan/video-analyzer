#!/usr/bin/env python3
"""
Environment Analyzer Module
=========================

This module analyzes the environment aspects of a video, including:
- Lighting quality
- Framing
- Background analysis
- Distance from camera
"""

import cv2
import numpy as np
from collections import deque

class EnvironmentAnalyzer:
    """Analyzes environment aspects of video frames"""
    
    def __init__(self, history_size=30):
        """Initialize the environment analyzer
        
        Args:
            history_size (int): Number of frames to keep in history for smoothing
        """
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.history_size = history_size
        self.brightness_history = deque(maxlen=history_size)
        self.contrast_history = deque(maxlen=history_size)
        self.face_size_ratio_history = deque(maxlen=history_size)
        self.face_position_history = deque(maxlen=history_size)
        
    def analyze_frame(self, frame):
        """Analyze a single frame for environment metrics
        
        Args:
            frame: The video frame to analyze
            
        Returns:
            dict: Dictionary containing environment metrics
        """
        if frame is None:
            return {}
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Calculate frame metrics
        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_height * frame_width
        
        # Calculate brightness
        brightness = np.mean(gray)
        self.brightness_history.append(brightness)
        
        # Calculate contrast
        contrast = np.std(gray)
        self.contrast_history.append(contrast)
        
        # Initialize metrics
        face_detected = False
        face_size_ratio = 0
        face_position_x = 0.5  # Default to center
        face_position_y = 0.5  # Default to center
        
        # If face detected, calculate face metrics
        if len(faces) > 0:
            face_detected = True
            # Use the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Calculate face size ratio (face area / frame area)
            face_area = w * h
            face_size_ratio = face_area / frame_area
            self.face_size_ratio_history.append(face_size_ratio)
            
            # Calculate face position (normalized 0-1)
            face_center_x = x + w/2
            face_center_y = y + h/2
            face_position_x = face_center_x / frame_width
            face_position_y = face_center_y / frame_height
            self.face_position_history.append((face_position_x, face_position_y))
        
        # Calculate lighting quality (0-1)
        # Ideal brightness around 127 (middle of 0-255)
        # Ideal contrast around 50-70
        avg_brightness = np.mean(self.brightness_history) if self.brightness_history else brightness
        avg_contrast = np.mean(self.contrast_history) if self.contrast_history else contrast
        
        brightness_score = 1.0 - abs(avg_brightness - 127) / 127
        contrast_score = min(avg_contrast / 60, 1.0) if avg_contrast < 60 else 2.0 - (avg_contrast / 60)
        lighting_quality = (brightness_score * 0.6 + contrast_score * 0.4)
        lighting_quality = max(0.0, min(1.0, lighting_quality))  # Clamp between 0 and 1
        
        # Calculate framing score (0-1)
        # Based on face position and size
        framing_score = 0.0
        if face_detected:
            # Ideal face position is center (0.5, 0.33) - slightly above center vertically
            # Ideal face size ratio is around 0.15-0.3 of frame
            position_score = 1.0 - (((face_position_x - 0.5)**2 + (face_position_y - 0.33)**2) ** 0.5)
            
            # Size score - penalize if face is too small or too large
            if face_size_ratio < 0.05:  # Too small
                size_score = face_size_ratio / 0.05
            elif face_size_ratio > 0.4:  # Too large
                size_score = 1.0 - min(1.0, (face_size_ratio - 0.4) / 0.2)
            else:  # Good size
                # Ideal is around 0.15-0.25
                ideal_size = 0.2
                size_score = 1.0 - min(1.0, abs(face_size_ratio - ideal_size) / 0.15)
                
            framing_score = position_score * 0.7 + size_score * 0.3
            framing_score = max(0.0, min(1.0, framing_score))  # Clamp between 0 and 1
        
        # Background analysis
        # Simple version: check variance in areas outside the face
        background_score = 0.5  # Default neutral score
        if face_detected:
            # Create a mask for non-face regions
            mask = np.ones_like(gray, dtype=np.uint8) * 255
            x, y, w, h = largest_face
            mask[y:y+h, x:x+w] = 0
            
            # Calculate background variance
            background = cv2.bitwise_and(gray, gray, mask=mask)
            bg_pixels = background[background > 0]
            if len(bg_pixels) > 0:
                bg_variance = np.var(bg_pixels)
                
                # Lower variance is better (less distracting)
                # But too low might mean a plain/boring background
                if bg_variance < 100:  # Very uniform
                    background_score = 0.9
                elif bg_variance < 500:  # Moderately uniform
                    background_score = 0.8 - (bg_variance - 100) / 500
                elif bg_variance < 2000:  # Somewhat busy
                    background_score = 0.7 - (bg_variance - 500) / 2000
                else:  # Very busy/distracting
                    background_score = 0.3
        
        # Calculate distance from camera score
        # Based on face size ratio
        distance_score = 0.5  # Default neutral score
        if face_detected:
            # Ideal face size ratio is around 0.15-0.25 of frame
            if face_size_ratio < 0.05:  # Too far
                distance_score = 0.3
            elif face_size_ratio < 0.1:  # Somewhat far
                distance_score = 0.6
            elif face_size_ratio < 0.3:  # Good distance
                distance_score = 1.0
            elif face_size_ratio < 0.4:  # Somewhat close
                distance_score = 0.7
            else:  # Too close
                distance_score = 0.4
        
        # Return all metrics
        return {
            "lighting_quality": lighting_quality,
            "framing_score": framing_score,
            "background_score": background_score,
            "distance_score": distance_score,
            "face_detected": face_detected,
            "brightness": avg_brightness,
            "contrast": avg_contrast,
            "face_size_ratio": face_size_ratio if face_detected else 0,
            "face_position_x": face_position_x if face_detected else 0.5,
            "face_position_y": face_position_y if face_detected else 0.5
        }
    
    def analyze_video_frames(self, frames):
        """Analyze a sequence of video frames
        
        Args:
            frames: List of video frames to analyze
            
        Returns:
            dict: Dictionary containing aggregated environment metrics
        """
        if not frames:
            return {}
            
        # Analyze each frame
        frame_results = [self.analyze_frame(frame) for frame in frames]
        
        # Filter out empty results
        valid_results = [r for r in frame_results if r]
        if not valid_results:
            return {}
            
        # Aggregate metrics
        lighting_qualities = [r["lighting_quality"] for r in valid_results if "lighting_quality" in r]
        framing_scores = [r["framing_score"] for r in valid_results if "framing_score" in r]
        background_scores = [r["background_score"] for r in valid_results if "background_score" in r]
        distance_scores = [r["distance_score"] for r in valid_results if "distance_score" in r]
        
        # Calculate average metrics
        avg_lighting_quality = np.mean(lighting_qualities) if lighting_qualities else 0
        avg_framing_score = np.mean(framing_scores) if framing_scores else 0
        avg_background_score = np.mean(background_scores) if background_scores else 0
        avg_distance_score = np.mean(distance_scores) if distance_scores else 0
        
        # Calculate overall environment score
        environment_score = (
            avg_lighting_quality * 0.3 +
            avg_framing_score * 0.3 +
            avg_background_score * 0.2 +
            avg_distance_score * 0.2
        )
        
        # Return aggregated results
        return {
            "environment_score": environment_score,
            "lighting_quality": avg_lighting_quality,
            "framing_score": avg_framing_score,
            "background_score": avg_background_score,
            "distance_score": avg_distance_score,
            "frames_analyzed": len(valid_results),
            "face_detection_rate": sum(1 for r in valid_results if r.get("face_detected", False)) / len(valid_results) if valid_results else 0
        }