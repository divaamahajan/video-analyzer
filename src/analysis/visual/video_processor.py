"""
Video processing utilities
"""

import cv2
from tqdm import tqdm
from src.config.settings import Config

def extract_frames(video_path, frame_interval=1):
    """
    Extract frames from video file
    
    Args:
        video_path: Path to video file
        frame_interval: Extract every Nth frame (1 = all frames)
    
    Returns:
        list: List of video frames (numpy arrays)
    """
    print(f"📹 Extracting frames from: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1
    
    cap.release()
    print(f"✅ Extracted {len(frames)} frames (every {frame_interval}th frame)")
    return frames

def process_video_batch(frames, processor_func, desc="Processing"):
    """
    Process a batch of frames with progress tracking
    
    Args:
        frames: List of frames to process
        processor_func: Function to process each frame
        desc: Description for progress bar
    
    Returns:
        list: Results from processing
    """
    results = []
    
    for i, frame in enumerate(tqdm(frames, desc=desc, unit="frame")):
        result = processor_func(frame, i)
        results.append(result)
    
    return results
