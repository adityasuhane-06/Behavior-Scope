#!/usr/bin/env python3
"""
Debug script to test MediaPipe face detection on frames from video.
"""

import cv2
import mediapipe as mp
import numpy as np
from utils.video_io import extract_frames_from_segment

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3
)

# Extract frames
video_path = "data/raw/test_short.mp4"
frames = extract_frames_from_segment(video_path, 0, 24, target_fps=6)

print(f"Extracted {len(frames)} frames")

if len(frames) > 0:
    frame = frames[0]
    print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
    print(f"Frame min/max: {frame.min()}/{frame.max()}")
    
    # Try processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(f"RGB frame shape: {frame_rgb.shape}")
    
    results = mp_face_mesh.process(frame_rgb)
    
    if results.multi_face_landmarks:
        print(f"✓ SUCCESS: Detected {len(results.multi_face_landmarks)} face(s)")
        print(f"  Landmarks: {len(results.multi_face_landmarks[0].landmark)}")
    else:
        print("✗ FAILED: No faces detected")
        
        # Try with the original frame (BGR)
        print("\nTrying with BGR frame directly...")
        results2 = mp_face_mesh.process(frame)
        if results2.multi_face_landmarks:
            print(f"✓ SUCCESS with BGR: Detected {len(results2.multi_face_landmarks)} face(s)")
        else:
            print("✗ Still no detection with BGR")
            
            # Save frame for manual inspection
            cv2.imwrite("debug_frame.jpg", frame)
            print("\nSaved debug_frame.jpg for manual inspection")

mp_face_mesh.close()
