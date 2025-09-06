#!/usr/bin/env python3
"""
Transcription Service Module
===========================

Handles basic transcription functionality including:
- Audio transcription with timestamps
- Word and segment extraction
- Basic transcription utilities
"""

from openai import OpenAI
from typing import Dict, Any


class TranscriptionService:
    """Basic transcription service using OpenAI Whisper"""
    
    def __init__(self, api_key: str):
        """Initialize with OpenAI API key"""
        self.client = OpenAI(api_key=api_key)
    
    def transcribe_with_timestamps(self, audio_path: str, prompt: str = None) -> Dict[str, Any]:
        """Transcribe audio with word and segment timestamps"""
        print("📄 Extracting Text from Audio")
        
        with open(audio_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                temperature=0,
                prompt=prompt,
                timestamp_granularities=["word", "segment"],
            )

        # Combine all segment texts
        segments = getattr(transcription, "segments", [])
        words = getattr(transcription, "words", [])

        full_text = " ".join([seg.text.strip() for seg in segments]) if segments else transcription.text

        return {
            "text": full_text,
            "segments": segments,
            "words": words
        }
    
    def transcribe_simple(self, audio_path: str, prompt: str = None) -> str:
        """Simple transcription without timestamps"""
        print("📄 Transcribing Audio (Simple)")
        
        with open(audio_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                temperature=0,
                prompt=prompt
            )
        
        return transcription.text
    
    def print_transcription_info(self, transcription_result: Dict[str, Any]):
        """Print basic transcription information"""
        print("\n" + "="*50)
        print("📝 TRANSCRIPTION INFORMATION")
        print("="*50)
        
        print(f"\n📄 TRANSCRIPT")
        print(f"   Text: {transcription_result['text']}")
        print(f"   Length: {len(transcription_result['text'])} characters")
        
        print(f"\n📊 SEGMENTS")
        print(f"   Total segments: {len(transcription_result['segments'])}")
        print(f"   Total words: {len(transcription_result['words'])}")
        
        if transcription_result['segments']:
            first_seg = transcription_result['segments'][0]
            last_seg = transcription_result['segments'][-1]
            duration = last_seg.end - first_seg.start
            print(f"   Duration: {duration:.1f} seconds")
            print(f"   Start time: {first_seg.start:.1f}s")
            print(f"   End time: {last_seg.end:.1f}s")
        
        print("\n" + "="*50)
        print("✅ Transcription Complete!")
        print("="*50)
