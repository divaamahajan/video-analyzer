#!/usr/bin/env python3
"""
Pace Analyzer Module
===================

Handles pace and rhythm analysis from transcribed text including:
- Words per minute calculation
- Pause analysis
- Filler word detection and analysis
- Speaking rhythm patterns
"""

import numpy as np
from typing import Dict, List, Any


class PaceAnalyzer:
    """Analyzes speaking pace and rhythm from transcription data"""
    
    def __init__(self):
        """Initialize pace analyzer"""
        pass
    
    def analyze_pace_and_rhythm(self, transcription_result: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive pace and rhythm analysis from transcription"""
        segments = transcription_result["segments"]
        
        if not segments:
            return {"wpm": 0, "fillers_per_min": 0, "pause_analysis": {}}
        
        # Basic pace metrics
        total_words = sum(len(seg.text.split()) for seg in segments)
        total_time = segments[-1].end - segments[0].start
        wpm = total_words / (total_time / 60) if total_time > 0 else 0
        
        # Pause analysis
        pause_lengths = []
        pause_frequencies = []
        
        for i in range(len(segments) - 1):
            current_end = segments[i].end
            next_start = segments[i + 1].start
            pause_duration = next_start - current_end
            
            if pause_duration > 0.1:  # Only count pauses > 100ms
                pause_lengths.append(pause_duration)
                pause_frequencies.append(pause_duration)
        
        avg_pause_length = np.mean(pause_lengths) if pause_lengths else 0
        pause_frequency = len(pause_lengths) / (total_time / 60) if total_time > 0 else 0
        
        # Filler word analysis with timing
        fillers = ["um", "uh", "like", "you know", "so", "well", "actually", "basically", "literally"]
        filler_analysis = self._analyze_fillers(segments, fillers, total_time)
        
        return {
            "wpm": wpm,
            "total_words": total_words,
            "total_time": total_time,
            "avg_pause_length": avg_pause_length,
            "pause_frequency": pause_frequency,
            "filler_analysis": filler_analysis
        }
    
    def _analyze_fillers(self, segments: List, fillers: List[str], total_time: float) -> Dict[str, Any]:
        """Detailed filler word analysis"""
        filler_count = 0
        filler_positions = []
        filler_timing = {"start": 0, "middle": 0, "end": 0}
        
        for i, seg in enumerate(segments):
            words = seg.text.split()
            segment_duration = seg.end - seg.start
            
            for j, word in enumerate(words):
                if word.lower().strip('.,!?') in fillers:
                    filler_count += 1
                    filler_positions.append({
                        "word": word,
                        "time": seg.start + (j / len(words)) * segment_duration,
                        "position": "start" if j < len(words) * 0.3 else "end" if j > len(words) * 0.7 else "middle"
                    })
                    
                    # Categorize by position in sentence
                    if j < len(words) * 0.3:
                        filler_timing["start"] += 1
                    elif j > len(words) * 0.7:
                        filler_timing["end"] += 1
                    else:
                        filler_timing["middle"] += 1
        
        return {
            "total_fillers": filler_count,
            "fillers_per_min": filler_count / (total_time / 60) if total_time > 0 else 0,
            "filler_positions": filler_positions,
            "filler_timing_distribution": filler_timing
        }
    
    def print_pace_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of pace analysis results"""
        print("\n" + "="*50)
        print("🏃 PACE & RHYTHM ANALYSIS")
        print("="*50)
        
        print(f"\n📊 SPEAKING METRICS")
        print(f"   Words per minute: {results['wpm']:.1f}")
        print(f"   Total words: {results['total_words']}")
        print(f"   Total time: {results['total_time']:.1f}s")
        
        print(f"\n⏸️ PAUSE ANALYSIS")
        print(f"   Average pause length: {results['avg_pause_length']:.2f}s")
        print(f"   Pause frequency: {results['pause_frequency']:.1f} per minute")
        
        # Filler Analysis
        fillers = results['filler_analysis']
        print(f"\n🗣️ FILLER WORDS")
        print(f"   Total fillers: {fillers['total_fillers']} ({fillers['fillers_per_min']:.1f}/min)")
        print(f"   Distribution: Start={fillers['filler_timing_distribution']['start']}, "
              f"Middle={fillers['filler_timing_distribution']['middle']}, "
              f"End={fillers['filler_timing_distribution']['end']}")
        
        if fillers['filler_positions']:
            print(f"   Filler positions:")
            for filler in fillers['filler_positions'][:10]:  # Show first 10
                print(f"     - '{filler['word']}' at {filler['time']:.1f}s ({filler['position']})")
        
        print("\n" + "="*50)
        print("✅ Pace Analysis Complete!")
        print("="*50)
