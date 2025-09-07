import json
import numpy as np
from deepface import DeepFace
from tqdm import tqdm

# Enhanced emotion mapping for behavioral analysis
EMOTION_MEANINGS = {
    'happy': 'Positive engagement, friendliness, confidence, openness to communication',
    'sad': 'Low energy, disengagement, possible discouragement, withdrawal', 
    'angry': 'Frustration, aggression, strong negative emotion, defensive posture',
    'fear': 'Anxiety, discomfort, uncertainty, stress indicators',
    'surprise': 'Attention, curiosity, shock (positive or negative), alertness',
    'disgust': 'Dislike, disagreement, discomfort, rejection',
    'neutral': 'Baseline, lack of emotional expression, potential masking'
}

# Microexpression analysis
MICROEXPRESSION_INDICATORS = {
    'rapid_emotion_changes': 'Potential emotional volatility or stress',
    'emotion_masking': 'Attempting to hide true feelings',
    'emotional_consistency': 'Stable emotional state',
    'emotional_volatility': 'Unstable emotional state, possible stress'
}

# Facial expression behavioral patterns
FACIAL_PATTERNS = {
    'eye_contact_confidence': 'Direct eye contact indicates confidence and engagement',
    'eye_contact_avoidance': 'Looking away may indicate discomfort or deception',
    'facial_symmetry': 'Asymmetric expressions may indicate forced or fake emotions',
    'microexpression_frequency': 'Frequent microexpressions may indicate emotional stress'
}

def analyze_emotions_in_frames(frames, intended_emotion='happy'):
    """
    Analyze emotions in video frames using DeepFace
    
    Args:
        frames: List of video frames (numpy arrays)
        intended_emotion: Expected emotion for mismatch detection
    
    Returns:
        dict: Analysis results including emotion distribution and mismatches
    """
    print("\nStarting facial expression analysis...")
    
    emotion_results = []
    emotion_mismatches = []
    
    for i, frame in enumerate(tqdm(frames, desc="Analyzing emotions", unit="frame")):
        try:
            # Analyze emotions in the frame
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            
            if isinstance(result, list):
                result = result[0]  # Take first face if multiple detected
            
            # Extract emotion scores
            emotions = result['emotion']
            dominant_emotion = max(emotions, key=emotions.get)
            confidence = emotions[dominant_emotion]
            
            # Check for emotion mismatch
            emotion_mismatch = {
                'frame': i,
                'detected_emotion': dominant_emotion,
                'intended_emotion': intended_emotion,
                'confidence': confidence,
                'is_mismatch': dominant_emotion != intended_emotion,
                'emotion_scores': emotions
            }
            
            emotion_results.append({
                'frame': i,
                'dominant_emotion': dominant_emotion,
                'confidence': confidence,
                'all_emotions': emotions,
                'mismatch': emotion_mismatch
            })
            
            if emotion_mismatch['is_mismatch']:
                emotion_mismatches.append(emotion_mismatch)
                
        except Exception as e:
            # Handle cases where no face is detected or analysis fails
            emotion_results.append({
                'frame': i,
                'dominant_emotion': 'unknown',
                'confidence': 0.0,
                'all_emotions': {},
                'mismatch': None,
                'error': str(e)
            })
    
    print(f"Facial expression analysis complete! Processed {len(emotion_results)} frames.")
    
    return {
        'emotion_results': emotion_results,
        'emotion_mismatches': emotion_mismatches,
        'intended_emotion': intended_emotion
    }

def print_emotion_summary(analysis_data):
    """Print comprehensive emotion analysis summary"""
    emotion_results = analysis_data['emotion_results']
    emotion_mismatches = analysis_data['emotion_mismatches']
    intended_emotion = analysis_data['intended_emotion']
    
    print(f"\n=== FACIAL EXPRESSION ANALYSIS SUMMARY ===")
    
    # Count emotions
    emotion_counts = {}
    for result in emotion_results:
        emotion = result['dominant_emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    print(f"\n--- Emotion Distribution ---")
    for emotion, count in emotion_counts.items():
        percentage = count/len(emotion_results)*100
        print(f"{emotion.title()}: {count}/{len(emotion_results)} frames ({percentage:.1f}%)")
        if emotion in EMOTION_MEANINGS:
            print(f"  → {EMOTION_MEANINGS[emotion]}")
    
    # Emotion mismatch analysis
    if emotion_mismatches:
        print(f"\n--- Emotion Mismatch Analysis ---")
        print(f"Intended emotion: {intended_emotion.title()}")
        print(f"Mismatches detected: {len(emotion_mismatches)}/{len(emotion_results)} frames ({len(emotion_mismatches)/len(emotion_results)*100:.1f}%)")
        
        # Show most common mismatched emotions
        mismatch_emotions = {}
        for mismatch in emotion_mismatches:
            emotion = mismatch['detected_emotion']
            mismatch_emotions[emotion] = mismatch_emotions.get(emotion, 0) + 1
        
        print(f"\nMost common mismatched emotions:")
        for emotion, count in sorted(mismatch_emotions.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emotion.title()}: {count} frames")
    else:
        print(f"\n--- Emotion Mismatch Analysis ---")
        print(f"✅ No emotion mismatches detected! All emotions match intended: {intended_emotion.title()}")
    
    # Average confidence
    valid_results = [r for r in emotion_results if r['confidence'] > 0]
    if valid_results:
        avg_confidence = sum(r['confidence'] for r in valid_results) / len(valid_results)
        print(f"\nAverage emotion confidence: {avg_confidence:.1f}%")

def save_emotion_analysis(analysis_data, filename='emotion_analysis.json'):
    """Save detailed emotion analysis to JSON file"""
    emotion_results = analysis_data['emotion_results']
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        else:
            return obj
    
    # Convert all numpy types to Python native types
    serializable_results = convert_numpy_types(emotion_results)
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nDetailed emotion analysis saved to '{filename}'")

def get_emotion_statistics(analysis_data):
    """Get emotion statistics for further analysis"""
    emotion_results = analysis_data['emotion_results']
    emotion_mismatches = analysis_data['emotion_mismatches']
    
    # Calculate statistics
    stats = {
        'total_frames': len(emotion_results),
        'mismatch_count': len(emotion_mismatches),
        'mismatch_percentage': len(emotion_mismatches) / len(emotion_results) * 100 if emotion_results else 0,
        'emotion_distribution': {},
        'average_confidence': 0,
        'most_common_emotion': None,
        'most_common_mismatch': None
    }
    
    # Emotion distribution
    emotion_counts = {}
    for result in emotion_results:
        emotion = result['dominant_emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    stats['emotion_distribution'] = emotion_counts
    stats['most_common_emotion'] = max(emotion_counts, key=emotion_counts.get) if emotion_counts else None
    
    # Average confidence
    valid_results = [r for r in emotion_results if r['confidence'] > 0]
    if valid_results:
        stats['average_confidence'] = sum(r['confidence'] for r in valid_results) / len(valid_results)
    
    # Most common mismatch
    if emotion_mismatches:
        mismatch_emotions = {}
        for mismatch in emotion_mismatches:
            emotion = mismatch['detected_emotion']
            mismatch_emotions[emotion] = mismatch_emotions.get(emotion, 0) + 1
        stats['most_common_mismatch'] = max(mismatch_emotions, key=mismatch_emotions.get)
    
    return stats

def detect_emotion_patterns(analysis_data, window_size=10):
    """Detect patterns in emotion changes over time"""
    emotion_results = analysis_data['emotion_results']
    
    patterns = {
        'emotion_changes': 0,
        'stability_periods': [],
        'volatile_periods': [],
        'dominant_emotion_consistency': 0,
        'emotion_distribution': {},
        'microexpression_analysis': {},
        'emotional_volatility': 0,
        'emotion_masking_indicators': []
    }
    
    if len(emotion_results) < 2:
        return patterns
    
    # Count emotion changes
    emotion_changes = 0
    for i in range(1, len(emotion_results)):
        if emotion_results[i]['dominant_emotion'] != emotion_results[i-1]['dominant_emotion']:
            emotion_changes += 1
    
    patterns['emotion_changes'] = emotion_changes
    
    # Calculate emotion distribution
    emotion_counts = {}
    for result in emotion_results:
        emotion = result['dominant_emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    patterns['emotion_distribution'] = emotion_counts
    
    # Analyze stability in windows
    for i in range(0, len(emotion_results) - window_size + 1, window_size):
        window_emotions = [r['dominant_emotion'] for r in emotion_results[i:i+window_size]]
        unique_emotions = len(set(window_emotions))
        
        if unique_emotions == 1:
            patterns['stability_periods'].append(i)
        elif unique_emotions > window_size // 2:
            patterns['volatile_periods'].append(i)
    
    # Calculate consistency
    if emotion_results and patterns['emotion_distribution']:
        most_common = max(patterns['emotion_distribution'], key=patterns['emotion_distribution'].get)
        patterns['dominant_emotion_consistency'] = patterns['emotion_distribution'][most_common] / len(emotion_results) * 100
    
    # Enhanced microexpression analysis
    patterns['microexpression_analysis'] = analyze_microexpressions(emotion_results)
    patterns['emotional_volatility'] = calculate_emotional_volatility(emotion_results)
    patterns['emotion_masking_indicators'] = detect_emotion_masking(emotion_results)
    
    return patterns

def analyze_microexpressions(emotion_results):
    """Analyze microexpressions and subtle emotional changes"""
    if len(emotion_results) < 3:
        return {'detected': False, 'frequency': 0, 'intensity': 0}
    
    microexpression_count = 0
    intensity_scores = []
    
    for i in range(1, len(emotion_results) - 1):
        prev_emotion = emotion_results[i-1]['dominant_emotion']
        curr_emotion = emotion_results[i]['dominant_emotion']
        next_emotion = emotion_results[i+1]['dominant_emotion']
        
        # Detect rapid emotion changes (microexpressions)
        if prev_emotion == next_emotion and curr_emotion != prev_emotion:
            microexpression_count += 1
            # Calculate intensity based on confidence scores
            intensity = emotion_results[i]['confidence']
            intensity_scores.append(intensity)
    
    avg_intensity = sum(intensity_scores) / len(intensity_scores) if intensity_scores else 0
    frequency = microexpression_count / len(emotion_results) * 100
    
    return {
        'detected': microexpression_count > 0,
        'frequency': frequency,
        'intensity': avg_intensity,
        'count': microexpression_count
    }

def calculate_emotional_volatility(emotion_results):
    """Calculate emotional volatility over time"""
    if len(emotion_results) < 2:
        return 0
    
    emotion_changes = 0
    confidence_variance = []
    
    for i in range(1, len(emotion_results)):
        if emotion_results[i]['dominant_emotion'] != emotion_results[i-1]['dominant_emotion']:
            emotion_changes += 1
        confidence_variance.append(emotion_results[i]['confidence'])
    
    # Calculate volatility as combination of emotion changes and confidence variance
    emotion_volatility = emotion_changes / len(emotion_results) * 100
    confidence_volatility = np.var(confidence_variance) if confidence_variance else 0
    
    return {
        'emotion_changes_rate': emotion_volatility,
        'confidence_variance': confidence_volatility,
        'overall_volatility': (emotion_volatility + confidence_volatility * 10) / 2
    }

def detect_emotion_masking(emotion_results):
    """Detect potential emotion masking based on confidence patterns"""
    if len(emotion_results) < 5:
        return []
    
    masking_indicators = []
    
    # Check for consistently low confidence (possible masking)
    low_confidence_frames = [r for r in emotion_results if r['confidence'] < 30]
    if len(low_confidence_frames) > len(emotion_results) * 0.3:
        masking_indicators.append('consistently_low_confidence')
    
    # Check for rapid emotion changes with low confidence (inconsistent emotions)
    rapid_changes = 0
    for i in range(1, len(emotion_results)):
        if (emotion_results[i]['dominant_emotion'] != emotion_results[i-1]['dominant_emotion'] and
            emotion_results[i]['confidence'] < 40):
            rapid_changes += 1
    
    if rapid_changes > len(emotion_results) * 0.2:
        masking_indicators.append('rapid_low_confidence_changes')
    
    # Check for neutral emotion dominance (possible emotional suppression)
    neutral_count = sum(1 for r in emotion_results if r['dominant_emotion'] == 'neutral')
    if neutral_count > len(emotion_results) * 0.6:
        masking_indicators.append('excessive_neutral_emotions')
    
    return masking_indicators

def analyze_emotional_engagement(emotion_results):
    """Analyze emotional engagement patterns"""
    if not emotion_results:
        return {'engagement_level': 'unknown', 'indicators': []}
    
    engagement_indicators = []
    engagement_score = 0
    
    # Count positive emotions
    positive_emotions = ['happy', 'surprise']
    positive_count = sum(1 for r in emotion_results if r['dominant_emotion'] in positive_emotions)
    positive_ratio = positive_count / len(emotion_results)
    
    if positive_ratio > 0.4:
        engagement_indicators.append('high_positive_emotions')
        engagement_score += 0.4
    elif positive_ratio > 0.2:
        engagement_indicators.append('moderate_positive_emotions')
        engagement_score += 0.2
    
    # Check for emotional variety (not just neutral)
    unique_emotions = len(set(r['dominant_emotion'] for r in emotion_results))
    if unique_emotions > 3:
        engagement_indicators.append('emotional_variety')
        engagement_score += 0.2
    
    # Check for high confidence emotions (genuine expressions)
    high_confidence_count = sum(1 for r in emotion_results if r['confidence'] > 70)
    if high_confidence_count > len(emotion_results) * 0.5:
        engagement_indicators.append('high_confidence_expressions')
        engagement_score += 0.2
    
    # Check for emotional stability (not too volatile)
    emotion_changes = sum(1 for i in range(1, len(emotion_results)) 
                         if emotion_results[i]['dominant_emotion'] != emotion_results[i-1]['dominant_emotion'])
    change_rate = emotion_changes / len(emotion_results)
    
    if 0.1 < change_rate < 0.3:  # Moderate change rate indicates engagement
        engagement_indicators.append('balanced_emotional_expression')
        engagement_score += 0.2
    
    # Determine engagement level
    if engagement_score > 0.7:
        level = 'high'
    elif engagement_score > 0.4:
        level = 'medium'
    else:
        level = 'low'
    
    return {
        'engagement_level': level,
        'engagement_score': engagement_score,
        'indicators': engagement_indicators,
        'positive_emotion_ratio': positive_ratio,
        'emotional_variety': unique_emotions,
        'high_confidence_ratio': high_confidence_count / len(emotion_results)
    }
