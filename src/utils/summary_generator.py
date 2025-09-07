"""
Summary generation utilities
"""

import numpy as np

def print_pose_summary(analysis_results):
    """Print comprehensive pose analysis summary"""
    print(f"\n{'='*50}")
    print("🧍 COMPREHENSIVE BEHAVIORAL ANALYSIS SUMMARY")
    print(f"{'='*50}")
    
    # Basic posture flags
    basic_flags = ['fidgeting', 'slouching', 'leaning', 'head_tilt', 'forward_head_posture']
    print("\n--- Basic Posture ---")
    for flag in basic_flags:
        count = sum(1 for result in analysis_results if result['behaviors'][flag])
        percentage = count/len(analysis_results)*100 if analysis_results else 0
        print(f"{flag.replace('_', ' ').title()}: {count}/{len(analysis_results)} frames ({percentage:.1f}%)")
    
    # Head movement flags
    print("\n--- Head Movement ---")
    nod_count = sum(1 for result in analysis_results if result['behaviors']['head_nod_shake'] == 'nod')
    shake_count = sum(1 for result in analysis_results if result['behaviors']['head_nod_shake'] == 'shake')
    forward_count = sum(1 for result in analysis_results if result['behaviors']['head_forward_back'] == 'forward')
    back_count = sum(1 for result in analysis_results if result['behaviors']['head_forward_back'] == 'back')
    
    print(f"Head Nod: {nod_count}/{len(analysis_results)} frames ({nod_count/len(analysis_results)*100:.1f}%)")
    print(f"Head Shake: {shake_count}/{len(analysis_results)} frames ({shake_count/len(analysis_results)*100:.1f}%)")
    print(f"Head Forward: {forward_count}/{len(analysis_results)} frames ({forward_count/len(analysis_results)*100:.1f}%)")
    print(f"Head Back: {back_count}/{len(analysis_results)} frames ({back_count/len(analysis_results)*100:.1f}%)")
    
    # Arm and hand flags
    arm_flags = ['arms_crossed', 'hands_on_hips', 'hands_behind_head', 'arm_asymmetry', 
                 'hand_touching_face', 'hands_in_pockets', 'gesturing_while_speaking']
    print("\n--- Arm & Hand Positions ---")
    for flag in arm_flags:
        count = sum(1 for result in analysis_results if result['behaviors'][flag])
        percentage = count/len(analysis_results)*100 if analysis_results else 0
        print(f"{flag.replace('_', ' ').title()}: {count}/{len(analysis_results)} frames ({percentage:.1f}%)")
    
    # Body posture flags
    body_flags = ['shoulder_roll', 'twisting', 'legs_crossed', 'sitting', 'stiffness', 'relaxed_stance']
    print("\n--- Body Posture ---")
    for flag in body_flags:
        count = sum(1 for result in analysis_results if result['behaviors'][flag])
        percentage = count/len(analysis_results)*100 if analysis_results else 0
        print(f"{flag.replace('_', ' ').title()}: {count}/{len(analysis_results)} frames ({percentage:.1f}%)")
    
    # Energy and movement
    print("\n--- Energy & Movement ---")
    energy_levels = {}
    for result in analysis_results:
        engagement = result['behaviors'].get('engagement_level', {})
        if isinstance(engagement, dict):
            level = engagement.get('engagement_level', 'unknown')
        else:
            level = 'unknown'
        energy_levels[level] = energy_levels.get(level, 0) + 1
    
    for level, count in energy_levels.items():
        percentage = count/len(analysis_results)*100 if analysis_results else 0
        print(f"{level.replace('_', ' ').title()}: {count}/{len(analysis_results)} frames ({percentage:.1f}%)")
    
    # Gaze direction
    print("\n--- Gaze Direction ---")
    looking_at_camera = sum(1 for result in analysis_results 
                          if result['behaviors'].get('eye_contact', {}).get('gaze_direction') == 'direct')
    looking_away = sum(1 for result in analysis_results 
                      if result['behaviors'].get('eye_contact', {}).get('gaze_direction') in ['left', 'right'])
    print(f"Looking At Camera: {looking_at_camera}/{len(analysis_results)} frames ({looking_at_camera/len(analysis_results)*100:.1f}%)")
    print(f"Looking Away: {looking_away}/{len(analysis_results)} frames ({looking_away/len(analysis_results)*100:.1f}%)")
    
    # Repetitive patterns (using fidgeting as proxy)
    repetitive_count = sum(1 for result in analysis_results if result['behaviors'].get('fidgeting', False))
    print(f"\nRepetitive Fidgeting: {repetitive_count}/{len(analysis_results)} frames ({repetitive_count/len(analysis_results)*100:.1f}%)")
    
    # Average movement magnitude
    avg_movement = np.mean([result['movement_magnitude'] for result in analysis_results])
    print(f"Average movement magnitude: {avg_movement:.4f}")
