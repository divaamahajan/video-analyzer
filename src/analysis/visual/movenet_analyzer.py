#!/usr/bin/env python3
"""
MoveNet-based Pose Analysis Module
=================================

Advanced pose analysis using Google's MoveNet model for comprehensive
body language and behavioral analysis.
"""

import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional
from src.config.settings import Config


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points (p2 is the vertex)"""
    # Convert to numpy arrays for easier calculation
    a = np.array(p1[:2])  # [x, y]
    b = np.array(p2[:2])  # [x, y] - vertex
    c = np.array(p3[:2])  # [x, y]
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Avoid numerical errors
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# ============================================================================
# BEHAVIORAL DETECTION FUNCTIONS
# ============================================================================

def detect_microexpressions(keypoints, prev_keypoints, threshold=0.02):
    """Detect microexpressions based on subtle facial movements"""
    if prev_keypoints is None:
        return []
    
    microexpressions = []
    
    # Check for subtle eyebrow movements (if facial keypoints available)
    # This is a simplified version - in practice, you'd need facial keypoints
    if len(keypoints) >= 17:  # Basic pose keypoints
        # Check for head movements that might indicate microexpressions
        head_movement = calculate_distance(keypoints[0], prev_keypoints[0]) if len(keypoints) > 0 else 0
        if 0.01 < head_movement < threshold:
            microexpressions.append('subtle_head_movement')
    
    return microexpressions

def detect_eye_contact_patterns(keypoints, camera_position=(0.5, 0.5)):
    """Detect eye contact patterns based on head orientation"""
    if len(keypoints) < 5:  # Need at least nose and eyes
        return {'eye_contact_quality': 0.0, 'gaze_direction': 'unknown'}
    
    # Simplified eye contact detection based on head position
    nose_point = keypoints[0] if len(keypoints) > 0 else None
    if nose_point is None or nose_point[2] < 0.5:
        return {'eye_contact_quality': 0.0, 'gaze_direction': 'unknown'}
    
    # Calculate if person is looking towards camera center
    head_center_x = nose_point[1]  # x coordinate
    camera_center_x = camera_position[0]
    
    # Calculate offset from camera center
    offset = abs(head_center_x - camera_center_x)
    eye_contact_quality = max(0, 1 - offset * 2)  # Scale to 0-1
    
    # Determine gaze direction
    if offset < 0.1:
        gaze_direction = 'direct'
    elif head_center_x < camera_center_x - 0.1:
        gaze_direction = 'left'
    elif head_center_x > camera_center_x + 0.1:
        gaze_direction = 'right'
    else:
        gaze_direction = 'center'
    
    return {
        'eye_contact_quality': eye_contact_quality,
        'gaze_direction': gaze_direction
    }

def detect_mouth_expressions(keypoints):
    """Detect mouth expressions (simplified version)"""
    # This is a placeholder - real implementation would need facial keypoints
    return {
        'smile_intensity': 0.0,
        'mouth_openness': 0.0,
        'expression_type': 'neutral'
    }

def detect_open_vs_closed_posture(keypoints):
    """Detect open vs closed body posture"""
    if len(keypoints) < 17:
        return {'posture_type': 'unknown', 'openness_score': 0.0}
    
    openness_indicators = 0
    total_indicators = 0
    
    # Check arm positions (simplified)
    if len(keypoints) >= 11:
        # Check if arms are away from body (open posture)
        left_shoulder = keypoints[5]
        left_elbow = keypoints[7]
        left_wrist = keypoints[9]
        
        if all(kp[2] > 0.5 for kp in [left_shoulder, left_elbow, left_wrist]):
            # Calculate arm extension
            arm_extension = calculate_distance(left_shoulder, left_wrist)
            shoulder_elbow_dist = calculate_distance(left_shoulder, left_elbow)
            
            if arm_extension > shoulder_elbow_dist * 1.5:
                openness_indicators += 1
            total_indicators += 1
    
    openness_score = openness_indicators / max(total_indicators, 1)
    
    if openness_score > 0.6:
        posture_type = 'open'
    elif openness_score < 0.3:
        posture_type = 'closed'
    else:
        posture_type = 'neutral'
    
    return {
        'posture_type': posture_type,
        'openness_score': openness_score
    }

def detect_confidence_indicators(keypoints):
    """Detect confidence indicators in posture"""
    confidence_indicators = []
    
    if len(keypoints) < 17:
        return confidence_indicators
    
    # Check for upright posture
    if len(keypoints) >= 5:
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        
        if all(kp[2] > 0.5 for kp in [nose, left_shoulder, right_shoulder]):
            # Check if shoulders are level and head is upright
            shoulder_level = abs(left_shoulder[0] - right_shoulder[0])
            if shoulder_level < 0.05:  # Shoulders are level
                confidence_indicators.append('level_shoulders')
            
            # Check head position relative to shoulders
            head_above_shoulders = nose[0] < (left_shoulder[0] + right_shoulder[0]) / 2
            if head_above_shoulders:
                confidence_indicators.append('upright_head')
    
    # Check for open arm positions
    posture = detect_open_vs_closed_posture(keypoints)
    if posture['posture_type'] == 'open':
        confidence_indicators.append('open_posture')
    
    return confidence_indicators

def detect_engagement_level(keypoints, prev_keypoints):
    """Detect engagement level based on movement and posture"""
    if prev_keypoints is None:
        return {'engagement_level': 'unknown', 'engagement_score': 0.0}
    
    engagement_indicators = 0
    total_indicators = 0
    
    # Check for active movement
    if len(keypoints) >= 17 and len(prev_keypoints) >= 17:
        movement_sum = 0
        valid_movements = 0
        
        for i in range(min(len(keypoints), len(prev_keypoints))):
            if keypoints[i][2] > 0.5 and prev_keypoints[i][2] > 0.5:
                movement = calculate_distance(keypoints[i], prev_keypoints[i])
                movement_sum += movement
                valid_movements += 1
        
        if valid_movements > 0:
            avg_movement = movement_sum / valid_movements
            if 0.01 < avg_movement < 0.1:  # Moderate movement indicates engagement
                engagement_indicators += 1
            total_indicators += 1
    
    # Check for open posture (engagement indicator)
    posture = detect_open_vs_closed_posture(keypoints)
    if posture['openness_score'] > 0.5:
        engagement_indicators += 1
    total_indicators += 1
    
    # Check for confidence indicators
    confidence = detect_confidence_indicators(keypoints)
    if len(confidence) >= 2:
        engagement_indicators += 1
    total_indicators += 1
    
    engagement_score = engagement_indicators / max(total_indicators, 1)
    
    if engagement_score > 0.7:
        engagement_level = 'high'
    elif engagement_score > 0.4:
        engagement_level = 'medium'
    else:
        engagement_level = 'low'
    
    return {
        'engagement_level': engagement_level,
        'engagement_score': engagement_score
    }

def detect_defensive_gestures(keypoints):
    """Detect defensive body language gestures"""
    defensive_gestures = []
    
    if len(keypoints) < 17:
        return defensive_gestures
    
    # Check for crossed arms
    if all(keypoints[i][2] > 0.5 for i in [5, 6, 7, 8, 9, 10]):  # Shoulders, elbows, wrists
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_elbow = keypoints[7]
        right_elbow = keypoints[8]
        
        # Check if wrists are crossed over body center
        body_center_x = (keypoints[5][1] + keypoints[6][1]) / 2
        
        if (left_wrist[1] > body_center_x and right_wrist[1] < body_center_x) or \
           (right_wrist[1] > body_center_x and left_wrist[1] < body_center_x):
            defensive_gestures.append('arms_crossed')
    
    # Check for closed posture
    posture = detect_open_vs_closed_posture(keypoints)
    if posture['posture_type'] == 'closed':
        defensive_gestures.append('closed_posture')
    
    return defensive_gestures

def detect_interest_indicators(keypoints, prev_keypoints):
    """Detect interest indicators in body language"""
    if prev_keypoints is None:
        return []
    
    interest_indicators = []
    
    # Check for forward lean (interest indicator)
    if len(keypoints) >= 17:
        nose = keypoints[0]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        
        if all(kp[2] > 0.5 for kp in [nose, left_hip, right_hip]):
            hip_center = [(left_hip[1] + right_hip[1]) / 2, (left_hip[0] + right_hip[0]) / 2]
            nose_forward = nose[1] > hip_center[0]  # Nose is forward of hips
            
            if nose_forward:
                interest_indicators.append('forward_lean')
    
    # Check for active listening posture
    confidence = detect_confidence_indicators(keypoints)
    if 'upright_head' in confidence and 'level_shoulders' in confidence:
        interest_indicators.append('attentive_posture')
    
    return interest_indicators

def detect_stress_indicators(keypoints, prev_keypoints):
    """Detect stress indicators in body language"""
    if prev_keypoints is None:
        return []
    
    stress_indicators = []
    
    # Check for fidgeting (rapid small movements)
    if len(keypoints) >= 17 and len(prev_keypoints) >= 17:
        movement_sum = 0
        valid_movements = 0
        
        for i in range(min(len(keypoints), len(prev_keypoints))):
            if keypoints[i][2] > 0.5 and prev_keypoints[i][2] > 0.5:
                movement = calculate_distance(keypoints[i], prev_keypoints[i])
                movement_sum += movement
                valid_movements += 1
        
        if valid_movements > 0:
            avg_movement = movement_sum / valid_movements
            if avg_movement > 0.05:  # High movement indicates fidgeting
                stress_indicators.append('fidgeting')
    
    # Check for defensive posture
    defensive = detect_defensive_gestures(keypoints)
    if defensive:
        stress_indicators.extend(defensive)
    
    return stress_indicators

def detect_fidgeting(keypoints, prev_keypoints, threshold=0.05):
    """Detect fidgeting behavior"""
    if prev_keypoints is None:
        return False
    
    if len(keypoints) < 17 or len(prev_keypoints) < 17:
        return False
    
    total_movement = 0
    valid_movements = 0
    
    for i in range(min(len(keypoints), len(prev_keypoints))):
        if keypoints[i][2] > 0.5 and prev_keypoints[i][2] > 0.5:
            movement = calculate_distance(keypoints[i], prev_keypoints[i])
            total_movement += movement
            valid_movements += 1
    
    if valid_movements == 0:
        return False
    
    avg_movement = total_movement / valid_movements
    return avg_movement > threshold

def detect_slouching(keypoints):
    """Detect slouching posture"""
    if len(keypoints) < 17:
        return False
    
    # Check if head is forward of shoulders (slouching indicator)
    nose = keypoints[0]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    
    if all(kp[2] > 0.5 for kp in [nose, left_shoulder, right_shoulder]):
        shoulder_center_x = (left_shoulder[1] + right_shoulder[1]) / 2
        head_forward = nose[1] > shoulder_center_x + 0.05  # Head significantly forward
        return head_forward
    
    return False

def detect_leaning(keypoints):
    """Detect leaning posture"""
    if len(keypoints) < 17:
        return False
    
    # Check for lateral leaning
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    
    if all(kp[2] > 0.5 for kp in [left_shoulder, right_shoulder, left_hip, right_hip]):
        shoulder_center_x = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_center_x = (left_hip[1] + right_hip[1]) / 2
        
        # Check for significant lateral shift
        lateral_shift = abs(shoulder_center_x - hip_center_x)
        return lateral_shift > 0.05
    
    return False

def detect_head_tilt(keypoints):
    """Detect head tilt"""
    if len(keypoints) < 5:
        return False
    
    # Check for head tilt using ear positions (simplified)
    left_ear = keypoints[3] if len(keypoints) > 3 else None
    right_ear = keypoints[4] if len(keypoints) > 4 else None
    
    if left_ear is not None and right_ear is not None and left_ear[2] > 0.5 and right_ear[2] > 0.5:
        # Check for vertical difference between ears
        vertical_diff = abs(left_ear[0] - right_ear[0])
        return vertical_diff > 0.02
    
    return False

def detect_arms_crossed(keypoints):
    """Detect crossed arms"""
    if len(keypoints) < 11:
        return False
    
    # Check if wrists are crossed over body center
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    
    if all(kp[2] > 0.5 for kp in [left_wrist, right_wrist, left_shoulder, right_shoulder]):
        body_center_x = (left_shoulder[1] + right_shoulder[1]) / 2
        
        # Check if wrists are on opposite sides of body center
        left_crossed = left_wrist[1] > body_center_x
        right_crossed = right_wrist[1] < body_center_x
        
        return left_crossed and right_crossed
    
    return False

def detect_hands_on_hips(keypoints):
    """Detect hands on hips posture"""
    if len(keypoints) < 17:
        return False
    
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    
    if all(kp[2] > 0.5 for kp in [left_wrist, right_wrist, left_hip, right_hip]):
        # Check if wrists are near hips
        left_distance = calculate_distance(left_wrist, left_hip)
        right_distance = calculate_distance(right_wrist, right_hip)
        
        return left_distance < 0.1 and right_distance < 0.1
    
    return False

def detect_hands_behind_head(keypoints):
    """Detect hands behind head posture"""
    if len(keypoints) < 11:
        return False
    
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    nose = keypoints[0]
    
    if all(kp[2] > 0.5 for kp in [left_wrist, right_wrist, nose]):
        # Check if wrists are behind head (higher than nose)
        left_behind = left_wrist[0] < nose[0]
        right_behind = right_wrist[0] < nose[0]
        
        return left_behind and right_behind
    
    return False

def detect_legs_crossed(keypoints):
    """Detect crossed legs"""
    if len(keypoints) < 17:
        return False
    
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    
    if all(kp[2] > 0.5 for kp in [left_ankle, right_ankle, left_hip, right_hip]):
        # Check for crossed leg position
        hip_center_x = (left_hip[1] + right_hip[1]) / 2
        
        # Check if ankles are on opposite sides of hip center
        left_crossed = left_ankle[1] > hip_center_x
        right_crossed = right_ankle[1] < hip_center_x
        
        return left_crossed and right_crossed
    
    return False

def detect_standing_vs_sitting(keypoints):
    """Detect if person is standing or sitting"""
    if len(keypoints) < 17:
        return 'unknown'
    
    # Use hip and knee positions to determine standing vs sitting
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]
    
    if all(kp[2] > 0.5 for kp in [left_hip, right_hip, left_knee, right_knee]):
        # Calculate hip-knee distance
        left_hip_knee_dist = calculate_distance(left_hip, left_knee)
        right_hip_knee_dist = calculate_distance(right_hip, right_knee)
        avg_hip_knee_dist = (left_hip_knee_dist + right_hip_knee_dist) / 2
        
        # Standing typically has larger hip-knee distance
        if avg_hip_knee_dist > 0.15:
            return 'standing'
        else:
            return 'sitting'
    
    return 'unknown'

def detect_forward_head_posture(keypoints):
    """Detect forward head posture (text neck)"""
    if len(keypoints) < 7:
        return False
    
    nose = keypoints[0]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    
    if all(kp[2] > 0.5 for kp in [nose, left_shoulder, right_shoulder]):
        shoulder_center_x = (left_shoulder[1] + right_shoulder[1]) / 2
        head_forward = nose[1] > shoulder_center_x + 0.03  # Head forward of shoulders
        return head_forward
    
    return False

def detect_head_nod_shake(keypoints, prev_keypoints):
    """Detect head nodding or shaking"""
    if prev_keypoints is None or len(keypoints) < 5 or len(prev_keypoints) < 5:
        return 'none'
    
    nose = keypoints[0]
    prev_nose = prev_keypoints[0]
    
    if nose[2] > 0.5 and prev_nose[2] > 0.5:
        # Check for vertical movement (nodding)
        vertical_movement = abs(nose[0] - prev_nose[0])
        # Check for horizontal movement (shaking)
        horizontal_movement = abs(nose[1] - prev_nose[1])
        
        if vertical_movement > 0.02 and vertical_movement > horizontal_movement:
            return 'nod'
        elif horizontal_movement > 0.02 and horizontal_movement > vertical_movement:
            return 'shake'
    
    return 'none'

def detect_head_forward_back(keypoints):
    """Detect head moving forward or back"""
    if len(keypoints) < 7:
        return 'neutral'
    
    nose = keypoints[0]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    
    if all(kp[2] > 0.5 for kp in [nose, left_shoulder, right_shoulder]):
        shoulder_center_x = (left_shoulder[1] + right_shoulder[1]) / 2
        
        if nose[1] > shoulder_center_x + 0.05:
            return 'forward'
        elif nose[1] < shoulder_center_x - 0.05:
            return 'back'
    
    return 'neutral'

def detect_arm_asymmetry(keypoints):
    """Detect arm asymmetry"""
    if len(keypoints) < 11:
        return False
    
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    
    if all(kp[2] > 0.5 for kp in [left_shoulder, right_shoulder, left_wrist, right_wrist]):
        # Calculate arm positions relative to shoulders
        left_arm_pos = left_wrist[1] - left_shoulder[1]
        right_arm_pos = right_wrist[1] - right_shoulder[1]
        
        # Check for significant asymmetry
        asymmetry = abs(left_arm_pos - right_arm_pos)
        return asymmetry > 0.1
    
    return False

def detect_hand_touching_face(keypoints):
    """Detect hand touching face"""
    if len(keypoints) < 11:
        return False
    
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    nose = keypoints[0]
    
    if all(kp[2] > 0.5 for kp in [left_wrist, right_wrist, nose]):
        # Check if wrists are near face
        left_distance = calculate_distance(left_wrist, nose)
        right_distance = calculate_distance(right_wrist, nose)
        
        return left_distance < 0.08 or right_distance < 0.08
    
    return False

def detect_hands_in_pockets(keypoints):
    """Detect hands in pockets (simplified)"""
    if len(keypoints) < 17:
        return False
    
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    
    if all(kp[2] > 0.5 for kp in [left_wrist, right_wrist, left_hip, right_hip]):
        # Check if wrists are near hips and lower than hips
        left_near_hip = calculate_distance(left_wrist, left_hip) < 0.08
        right_near_hip = calculate_distance(right_wrist, right_hip) < 0.08
        
        left_lower = left_wrist[0] > left_hip[0]
        right_lower = right_wrist[0] > right_hip[0]
        
        return (left_near_hip and left_lower) or (right_near_hip and right_lower)
    
    return False

def detect_gesturing_while_speaking(keypoints, prev_keypoints):
    """Detect gesturing while speaking (hand movement)"""
    if prev_keypoints is None or len(keypoints) < 11 or len(prev_keypoints) < 11:
        return False
    
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    prev_left_wrist = prev_keypoints[9]
    prev_right_wrist = prev_keypoints[10]
    
    if all(kp[2] > 0.5 for kp in [left_wrist, right_wrist, prev_left_wrist, prev_right_wrist]):
        # Check for hand movement
        left_movement = calculate_distance(left_wrist, prev_left_wrist)
        right_movement = calculate_distance(right_wrist, prev_right_wrist)
        
        # Gesturing involves moderate hand movement
        return left_movement > 0.02 or right_movement > 0.02
    
    return False

def detect_shoulder_roll(keypoints):
    """Detect shoulder rolling or tension"""
    if len(keypoints) < 7:
        return False
    
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    
    if left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5:
        # Check for shoulder asymmetry or tension
        shoulder_diff = abs(left_shoulder[0] - right_shoulder[0])
        return shoulder_diff > 0.05
    
    return False

def detect_twisting(keypoints):
    """Detect body twisting"""
    if len(keypoints) < 17:
        return False
    
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    
    if all(kp[2] > 0.5 for kp in [left_shoulder, right_shoulder, left_hip, right_hip]):
        # Check for shoulder-hip misalignment (twisting)
        shoulder_center_x = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_center_x = (left_hip[1] + right_hip[1]) / 2
        
        twist_amount = abs(shoulder_center_x - hip_center_x)
        return twist_amount > 0.05
    
    return False

def detect_stiffness(keypoints):
    """Detect stiffness in posture"""
    if len(keypoints) < 17:
        return False
    
    # Check for lack of natural curvature in spine
    # This is simplified - real implementation would need more keypoints
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    
    if all(kp[2] > 0.5 for kp in [left_shoulder, right_shoulder, left_hip, right_hip]):
        # Check for very straight posture (stiffness indicator)
        shoulder_center = [(left_shoulder[1] + right_shoulder[1]) / 2, 
                          (left_shoulder[0] + right_shoulder[0]) / 2]
        hip_center = [(left_hip[1] + right_hip[1]) / 2, 
                     (left_hip[0] + right_hip[0]) / 2]
        
        # Very straight line indicates stiffness
        vertical_alignment = abs(shoulder_center[0] - hip_center[0])
        return vertical_alignment < 0.02
    
    return False

def detect_relaxed_stance(keypoints):
    """Detect relaxed stance"""
    if len(keypoints) < 17:
        return False
    
    # Check for natural, relaxed posture
    stiffness = detect_stiffness(keypoints)
    tension = detect_shoulder_roll(keypoints)
    
    # Relaxed if not stiff and not tense
    return not stiffness and not tension

def analyze_pose_behaviors(keypoints, prev_keypoints, keypoints_history=None):
    """Analyze all pose behaviors and return comprehensive results"""
    behaviors = {
        # Basic posture
        'fidgeting': detect_fidgeting(keypoints, prev_keypoints),
        'slouching': detect_slouching(keypoints),
        'leaning': detect_leaning(keypoints),
        
        # Head and neck posture
        'head_tilt': detect_head_tilt(keypoints),
        'forward_head_posture': detect_forward_head_posture(keypoints),
        'head_nod_shake': detect_head_nod_shake(keypoints, prev_keypoints),
        'head_forward_back': detect_head_forward_back(keypoints),
        
        # Arm and hand positions
        'arms_crossed': detect_arms_crossed(keypoints),
        'hands_on_hips': detect_hands_on_hips(keypoints),
        'hands_behind_head': detect_hands_behind_head(keypoints),
        'arm_asymmetry': detect_arm_asymmetry(keypoints),
        'hand_touching_face': detect_hand_touching_face(keypoints),
        'hands_in_pockets': detect_hands_in_pockets(keypoints),
        'gesturing_while_speaking': detect_gesturing_while_speaking(keypoints, prev_keypoints),
        
        # Body posture
        'shoulder_roll': detect_shoulder_roll(keypoints),
        'twisting': detect_twisting(keypoints),
        'legs_crossed': detect_legs_crossed(keypoints),
        'sitting': detect_standing_vs_sitting(keypoints) == 'sitting',
        'stiffness': detect_stiffness(keypoints),
        'relaxed_stance': detect_relaxed_stance(keypoints),
        
        # Advanced behaviors
        'microexpressions': detect_microexpressions(keypoints, prev_keypoints),
        'eye_contact': detect_eye_contact_patterns(keypoints),
        'mouth_expressions': detect_mouth_expressions(keypoints),
        'posture_type': detect_open_vs_closed_posture(keypoints),
        'confidence_indicators': detect_confidence_indicators(keypoints),
        'engagement_level': detect_engagement_level(keypoints, prev_keypoints),
        'defensive_gestures': detect_defensive_gestures(keypoints),
        'interest_indicators': detect_interest_indicators(keypoints, prev_keypoints),
        'stress_indicators': detect_stress_indicators(keypoints, prev_keypoints)
    }
    
    return behaviors


# ============================================================================
# MAIN MOVENET ANALYZER CLASS
# ============================================================================

class MoveNetAnalyzer:
    """Advanced pose analysis using Google's MoveNet model"""
    
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
    
    def print_summary(self):
        """Print analysis summary"""
        if not self.analysis_results:
            print("No analysis results available")
            return
        
        print("\n" + "="*50)
        print("🧍 COMPREHENSIVE BEHAVIORAL ANALYSIS SUMMARY")
        print("="*50)
        
        # Aggregate behaviors across all frames
        total_frames = len(self.analysis_results)
        behavior_counts = {}
        
        for result in self.analysis_results:
            behaviors = result.get('behaviors', {})
            for behavior, value in behaviors.items():
                if isinstance(value, bool) and value:
                    behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
                elif isinstance(value, str) and value != 'none' and value != 'neutral':
                    behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
        
        # Print behavior summary
        categories = {
            'Basic Posture': ['fidgeting', 'slouching', 'leaning', 'head_tilt', 'forward_head_posture'],
            'Head Movement': ['head_nod_shake', 'head_forward_back'],
            'Arm & Hand Positions': ['arms_crossed', 'hands_on_hips', 'hands_behind_head', 'arm_asymmetry', 
                                   'hand_touching_face', 'hands_in_pockets', 'gesturing_while_speaking'],
            'Body Posture': ['shoulder_roll', 'twisting', 'legs_crossed', 'sitting', 'stiffness', 'relaxed_stance']
        }
        
        for category, behaviors in categories.items():
            print(f"\n--- {category} ---")
            for behavior in behaviors:
                count = behavior_counts.get(behavior, 0)
                percentage = (count / total_frames) * 100
                print(f"{behavior.replace('_', ' ').title()}: {count}/{total_frames} frames ({percentage:.1f}%)")
        
        # Calculate average movement magnitude
        movement_magnitudes = [r.get('movement_magnitude', 0) for r in self.analysis_results]
        avg_movement = np.mean(movement_magnitudes) if movement_magnitudes else 0
        print(f"\nAverage movement magnitude: {avg_movement:.4f}")
        
        print("\n" + "="*50)
