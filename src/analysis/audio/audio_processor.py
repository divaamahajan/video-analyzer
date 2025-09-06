#!/usr/bin/env python3
"""
Audio Processor Module
=====================

Handles direct audio processing and analysis including:
- Audio extraction from video
- Pitch and tone analysis
- Volume and clarity analysis
- Audio feature extraction
"""

import os
import numpy as np
import librosa
from moviepy import VideoFileClip
from typing import Dict, Any
from scipy import stats
from scipy.signal import find_peaks


class AudioProcessor:
    """Direct audio processing and analysis"""
    
    def __init__(self):
        """Initialize audio processor"""
        pass
    
    def extract_audio_from_video(self, video_path: str, output_path: str = "input_audio.wav") -> str:
        """Extract audio from video file"""
        print(f"📄 Extracting Audio from Video: {video_path}")
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(output_path)
        clip.close()
        return output_path
    
    def analyze_pitch_and_tone(self, audio_path: str) -> Dict[str, Any]:
        """Comprehensive pitch and tone analysis"""
        print("🎵 Analyzing Pitch and Tone")
        
        # Load audio
        y, sr = librosa.load(audio_path)
        
        # Extract pitch using multiple methods
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        
        if len(pitch_values) == 0:
            return {"error": "No pitch detected"}
        
        # Pitch range analysis
        pitch_min = np.min(pitch_values)
        pitch_max = np.max(pitch_values)
        pitch_range = pitch_max - pitch_min
        pitch_mean = np.mean(pitch_values)
        pitch_std = np.std(pitch_values)
        
        # Pitch variance (monotone vs expressive)
        pitch_variance = np.var(pitch_values)
        pitch_cv = pitch_std / pitch_mean if pitch_mean > 0 else 0  # Coefficient of variation
        
        # Pitch contour analysis
        pitch_contour = self._analyze_pitch_contour(pitch_values)
        
        # Emotional tone detection
        emotional_cues = self._detect_emotional_cues(pitch_values, y, sr)
        
        # Prosody analysis
        prosody = self._analyze_prosody(y, sr, pitch_values)
        
        return {
            "pitch_range": pitch_range,
            "pitch_min": pitch_min,
            "pitch_max": pitch_max,
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "pitch_variance": pitch_variance,
            "pitch_cv": pitch_cv,
            "pitch_contour": pitch_contour,
            "emotional_cues": emotional_cues,
            "prosody": prosody
        }
    
    def _analyze_pitch_contour(self, pitch_values: np.ndarray) -> Dict[str, Any]:
        """Analyze pitch contour patterns"""
        if len(pitch_values) < 10:
            return {"trend": "insufficient_data"}
        
        # Linear trend
        x = np.arange(len(pitch_values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, pitch_values)
        
        # Rising/falling patterns
        trend = "rising" if slope > 0.1 else "falling" if slope < -0.1 else "stable"
        
        # Pitch jumps (large changes)
        pitch_diffs = np.diff(pitch_values)
        large_jumps = np.sum(np.abs(pitch_diffs) > 2 * np.std(pitch_diffs))
        
        return {
            "trend": trend,
            "slope": slope,
            "r_squared": r_value ** 2,
            "large_jumps": large_jumps,
            "jump_frequency": large_jumps / len(pitch_values)
        }
    
    def _detect_emotional_cues(self, pitch_values: np.ndarray, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Detect emotional cues from pitch and audio features"""
        # Energy analysis
        energy = librosa.feature.rms(y=y)[0]
        energy_mean = np.mean(energy)
        energy_var = np.var(energy)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Zero crossing rate (indicator of voice quality)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr)
        
        # Calculate pitch statistics
        pitch_variance = np.var(pitch_values) if len(pitch_values) > 0 else 0
        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        
        # Emotional indicators
        emotional_indicators = {
            "excited": pitch_variance > 100 and energy_var > 0.01,
            "confident": pitch_mean > 200 and pitch_variance < 50,
            "hesitant": pitch_variance < 30 and energy_var < 0.005,
            "stressed": zcr_mean > 0.1 and pitch_variance > 80,
            "calm": pitch_variance < 40 and energy_var < 0.003
        }
        
        # Determine primary emotion
        primary_emotion = max(emotional_indicators.items(), key=lambda x: x[1])[0]
        
        return {
            "primary_emotion": primary_emotion,
            "emotional_indicators": emotional_indicators,
            "energy_mean": energy_mean,
            "energy_variance": energy_var,
            "spectral_centroid_mean": np.mean(spectral_centroids),
            "zero_crossing_rate": zcr_mean
        }
    
    def _analyze_prosody(self, y: np.ndarray, sr: int, pitch_values: np.ndarray) -> Dict[str, Any]:
        """Analyze prosodic features"""
        # Intonation patterns
        if len(pitch_values) > 0:
            # Rising intonation (questions, uncertainty)
            rising_patterns = np.sum(np.diff(pitch_values) > 0) / len(pitch_values)
            
            # Falling intonation (statements, confidence)
            falling_patterns = np.sum(np.diff(pitch_values) < 0) / len(pitch_values)
        else:
            rising_patterns = 0
            falling_patterns = 0
        
        # Stress patterns using spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        stress_variance = np.var(spectral_centroids)
        
        # Emphasis detection (high energy + high pitch)
        energy = librosa.feature.rms(y=y)[0]
        
        # Ensure arrays have compatible shapes for comparison
        if len(energy) != len(pitch_values):
            # Resample energy to match pitch_values length
            energy_resampled = np.interp(
                np.linspace(0, 1, len(pitch_values)), 
                np.linspace(0, 1, len(energy)), 
                energy
            )
        else:
            energy_resampled = energy
            
        emphasis_points = np.sum((energy_resampled > np.mean(energy_resampled) + np.std(energy_resampled)) & 
                               (pitch_values > np.mean(pitch_values) + np.std(pitch_values)))
        
        return {
            "rising_intonation": rising_patterns,
            "falling_intonation": falling_patterns,
            "stress_variance": stress_variance,
            "emphasis_points": emphasis_points,
            "prosody_richness": stress_variance * emphasis_points
        }
    
    def analyze_volume_and_clarity(self, audio_path: str) -> Dict[str, Any]:
        """Comprehensive volume and clarity analysis"""
        print("🔊 Analyzing Volume and Clarity")
        
        # Load audio
        y, sr = librosa.load(audio_path)
        
        # Volume analysis
        rms = np.sqrt(np.mean(y**2))
        volume_db = 20 * np.log10(rms) if rms > 0 else -np.inf
        
        # Volume variance (consistency)
        frame_length = 2048
        hop_length = 512
        rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        volume_variance = np.var(rms_frames)
        volume_consistency = 1 / (1 + volume_variance)  # Higher = more consistent
        
        # Detect clipping (samples at maximum amplitude)
        clipping_threshold = 0.99
        clipped_samples = np.sum(np.abs(y) > clipping_threshold)
        clipping_percentage = (clipped_samples / len(y)) * 100
        
        # Detect too soft segments
        soft_threshold = np.percentile(rms_frames, 10)  # Bottom 10%
        soft_segments = np.sum(rms_frames < soft_threshold)
        soft_percentage = (soft_segments / len(rms_frames)) * 100
        
        # Dynamic range
        dynamic_range = np.max(rms_frames) - np.min(rms_frames)
        
        # Clarity analysis using spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        
        clarity_score = self._calculate_clarity_score(spectral_centroids, spectral_bandwidth)
        
        return {
            "volume_db": volume_db,
            "volume_variance": volume_variance,
            "volume_consistency": volume_consistency,
            "clipping_percentage": clipping_percentage,
            "soft_segments_percentage": soft_percentage,
            "dynamic_range": dynamic_range,
            "clarity_score": clarity_score,
            "spectral_centroid_mean": np.mean(spectral_centroids),
            "spectral_bandwidth_mean": np.mean(spectral_bandwidth)
        }
    
    def _calculate_clarity_score(self, spectral_centroids: np.ndarray, spectral_bandwidth: np.ndarray) -> float:
        """Calculate audio clarity score based on spectral features"""
        # Higher spectral centroid = brighter sound
        # Lower spectral bandwidth = more focused sound
        centroid_score = np.mean(spectral_centroids) / 2000  # Normalize
        bandwidth_score = 1 - (np.mean(spectral_bandwidth) / 2000)  # Invert and normalize
        
        clarity_score = (centroid_score + bandwidth_score) / 2
        return min(max(clarity_score, 0), 1)  # Clamp between 0 and 1
    
    def analyze_pronunciation_and_articulation(self, audio_path: str, segments: list) -> Dict[str, Any]:
        """Analyze pronunciation and articulation quality from audio"""
        print("🗣️ Analyzing Pronunciation and Articulation")
        
        if not segments:
            return {"error": "No segments to analyze"}
        
        # Load audio for analysis
        y, sr = librosa.load(audio_path)
        
        # Analyze each segment for clarity
        segment_analyses = []
        for seg in segments:
            # Extract segment audio
            start_frame = int(seg.start * sr)
            end_frame = int(seg.end * sr)
            segment_audio = y[start_frame:end_frame]
            
            if len(segment_audio) == 0:
                continue
                
            # Spectral features for articulation
            spectral_centroids = librosa.feature.spectral_centroid(y=segment_audio, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment_audio, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(segment_audio)[0]
            
            # Articulation score (higher = clearer)
            articulation_score = self._calculate_articulation_score(
                spectral_centroids, spectral_bandwidth, zero_crossing_rate
            )
            
            segment_analyses.append({
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
                "articulation_score": articulation_score,
                "spectral_centroid": np.mean(spectral_centroids),
                "spectral_bandwidth": np.mean(spectral_bandwidth),
                "zero_crossing_rate": np.mean(zero_crossing_rate)
            })
        
        # Overall pronunciation metrics
        avg_articulation = np.mean([seg["articulation_score"] for seg in segment_analyses])
        
        # Detect potential mispronunciations (very low articulation scores)
        mispronunciation_threshold = 0.3
        potential_mispronunciations = [
            seg for seg in segment_analyses 
            if seg["articulation_score"] < mispronunciation_threshold
        ]
        
        return {
            "avg_articulation_score": avg_articulation,
            "segment_analyses": segment_analyses,
            "potential_mispronunciations": potential_mispronunciations,
            "pronunciation_quality": "excellent" if avg_articulation > 0.7 else 
                                   "good" if avg_articulation > 0.5 else 
                                   "fair" if avg_articulation > 0.3 else "poor"
        }
    
    def _calculate_articulation_score(self, spectral_centroids: np.ndarray, 
                                    spectral_bandwidth: np.ndarray, 
                                    zero_crossing_rate: np.ndarray) -> float:
        """Calculate articulation clarity score"""
        # Higher spectral centroid = clearer consonants
        centroid_score = np.mean(spectral_centroids) / 2000
        
        # Lower spectral bandwidth = more focused sound
        bandwidth_score = 1 - (np.mean(spectral_bandwidth) / 2000)
        
        # Moderate zero crossing rate = good articulation
        zcr_score = 1 - abs(np.mean(zero_crossing_rate) - 0.05) * 10
        
        # Combine scores
        articulation_score = (centroid_score + bandwidth_score + zcr_score) / 3
        return min(max(articulation_score, 0), 1)
    
    def analyze_emotion_and_sentiment_audio(self, audio_path: str) -> Dict[str, Any]:
        """Analyze emotional intensity from audio features only"""
        print("😊 Analyzing Emotion from Audio")
        
        # Load audio
        y, sr = librosa.load(audio_path)
        
        # Get pitch and energy for emotion analysis
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        energy = librosa.feature.rms(y=y)[0]
        
        # Emotional intensity based on audio features
        emotional_intensity = self._calculate_emotional_intensity(pitch_values, energy, y, sr)
        
        return {
            "emotional_intensity": emotional_intensity
        }
    
    def _calculate_emotional_intensity(self, pitch_values: np.ndarray, energy: np.ndarray, 
                                     y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Calculate emotional intensity from audio features"""
        if len(pitch_values) == 0:
            return {"intensity": 0, "emotion": "neutral"}
        
        # Pitch variance (higher = more emotional)
        pitch_variance = np.var(pitch_values)
        
        # Energy variance (higher = more dynamic)
        energy_variance = np.var(energy)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Calculate intensity score
        intensity_score = (
            min(pitch_variance / 100, 1) * 0.4 +
            min(energy_variance * 100, 1) * 0.3 +
            min(np.var(spectral_centroids) / 1000, 1) * 0.3
        )
        
        # Determine emotion type
        if intensity_score > 0.7:
            emotion = "excited"
        elif intensity_score > 0.5:
            emotion = "engaged"
        elif intensity_score > 0.3:
            emotion = "moderate"
        else:
            emotion = "neutral"
        
        return {
            "intensity": intensity_score,
            "emotion": emotion,
            "pitch_variance": pitch_variance,
            "energy_variance": energy_variance
        }
    
    def process_audio(self, video_path: str, audio_path: str = "input_audio.wav") -> Dict[str, Any]:
        """Complete audio processing pipeline"""
        print("\n🎵 Starting Audio Processing")
        print("="*50)
        
        # Extract audio
        audio_file = self.extract_audio_from_video(video_path, audio_path)
        
        # Audio analysis
        pitch_analysis = self.analyze_pitch_and_tone(audio_file)
        volume_analysis = self.analyze_volume_and_clarity(audio_file)
        emotion_analysis = self.analyze_emotion_and_sentiment_audio(audio_file)
        
        return {
            "audio_file": audio_file,
            "pitch_and_tone": pitch_analysis,
            "volume_and_clarity": volume_analysis,
            "emotion_intensity": emotion_analysis
        }
    
    def print_audio_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of audio processing results"""
        print("\n" + "="*60)
        print("🎤 AUDIO PROCESSING REPORT")
        print("="*60)
        
        # Pitch & Tone Analysis
        pitch = results['pitch_and_tone']
        if 'error' not in pitch:
            print(f"\n🎵 PITCH & TONE")
            print(f"   Pitch range: {pitch['pitch_min']:.1f} - {pitch['pitch_max']:.1f} Hz")
            print(f"   Pitch variance: {pitch['pitch_variance']:.1f}")
            print(f"   Pitch CV: {pitch['pitch_cv']:.3f}")
            print(f"   Contour trend: {pitch['pitch_contour']['trend']}")
            
            # Emotional cues
            emotion = pitch['emotional_cues']
            print(f"   Primary emotion: {emotion['primary_emotion']}")
            print(f"   Emotional indicators: {emotion['emotional_indicators']}")
            
            # Prosody
            prosody = pitch['prosody']
            print(f"   Rising intonation: {prosody['rising_intonation']:.1%}")
            print(f"   Falling intonation: {prosody['falling_intonation']:.1%}")
            print(f"   Emphasis points: {prosody['emphasis_points']}")
        
        # Volume & Clarity Analysis
        volume = results['volume_and_clarity']
        print(f"\n🔊 VOLUME & CLARITY")
        print(f"   Volume: {volume['volume_db']:.1f} dB")
        print(f"   Volume consistency: {volume['volume_consistency']:.3f}")
        print(f"   Clipping: {volume['clipping_percentage']:.1f}%")
        print(f"   Soft segments: {volume['soft_segments_percentage']:.1f}%")
        print(f"   Dynamic range: {volume['dynamic_range']:.3f}")
        print(f"   Clarity score: {volume['clarity_score']:.3f}")
        
        # Emotion Analysis
        emotion = results['emotion_intensity']
        intensity = emotion['emotional_intensity']
        print(f"\n😊 EMOTION INTENSITY")
        print(f"   Emotional intensity: {intensity['intensity']:.3f} ({intensity['emotion']})")
        print(f"   Pitch variance: {intensity['pitch_variance']:.1f}")
        print(f"   Energy variance: {intensity['energy_variance']:.3f}")
        
        print("\n" + "="*60)
        print("✅ Audio Processing Complete!")
        print("="*60)
