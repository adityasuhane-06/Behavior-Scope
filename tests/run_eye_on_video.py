import sys
import os
import cv2
from pathlib import Path
import numpy as np
import time

# Add root directory to path so imports work
root = Path(os.getcwd())
if str(root) not in sys.path:
    sys.path.append(str(root))

try:
    from enhanced_attention_tracking.detection.detection_engine import DetectionEngineImpl
    from enhanced_eye_contact_audit import EyeContactAuditGenerator
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def main():
    print("=== Testing Deep Learning Gaze Engine ===")
    
    # 1. Verify Model Weights
    model_path = root / "data" / "models" / "L2CSNet_gaze360.pkl"
    if model_path.exists():
        print(f"[OK] Weights found at: {model_path}")
    else:
        print(f"[WARNING] Weights NOT found at: {model_path}")
        print("Please ensure the .pkl file is named 'L2CSNet_gaze360.pkl'")
        # Search for any .pkl in data/models to help user
        found = list((root / "data" / "models").glob("*.pkl"))
        if found:
            print(f"Found other models: {[f.name for f in found]}")

    # 2. Initialize Engine
    print("\nInitializing Detection Engine...")
    try:
        engine = DetectionEngineImpl()
    except Exception as e:
        print(f"Failed to init engine: {e}")
        return

    # Check Deep Learning Status
    if hasattr(engine, '_l2cs_estimator') and engine._l2cs_estimator and engine._l2cs_estimator.active:
        print("\n[SUCCESS] L2CS Deep Learning Model is ACTIVE!")
        print(f"Device: {engine._l2cs_estimator.device}")
    else:
        print("\n[INFO] L2CS Deep Learning Model is INACTIVE.")
        print("Detailed Status:")
        if not hasattr(engine, '_l2cs_estimator') or not engine._l2cs_estimator:
            print(" - Wrapper not initialized (Check PyTorch installation)")
        elif not engine._l2cs_estimator.active:
            print(" - Loading failed (Check weights file compatibility)")
        print("Falling back to MediaPipe Iris Tracking (Transformer-grade geometric tracking).")

    # 3. Run Video
    video_path = r"data/raw/Doctor_s_Voice_Video_Ready.mp4"
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    
    print(f"\nOpening Video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video")
        return
        
    frame_results = []
    frame_idx = 0
    max_frames = 60  
    
    print(f"Processing first {max_frames} frames...")
    start_time = time.time()
    
    # Init debug gen
    gen = EyeContactAuditGenerator()
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break
            
        timestamp = frame_idx / 24.0 
        
        # Process Frame through Core Engine
        result = engine.process_frame(frame, timestamp)
        frame_results.append(result)
        
        if frame_idx % 10 == 0:
            gv = result.gaze_vector
            
            # Identify Source
            source = "Head/Iris"
            if gv and gv.confidence >= 0.95 and engine._l2cs_estimator and engine._l2cs_estimator.active:
                source = "L2CS-Net"
            elif gv is None:
                source = "Failed"
            
            vec_str = f"x={gv.x:.2f}, y={gv.y:.2f}" if gv else "None"
            
            # Debug Classification
            cls_dir = "N/A"
            if gv:
                 # Audit gen expects tuple and confidence
                 try:
                     cls_dir = gen.classify_gaze_direction((gv.x, gv.y, gv.z), result.confidence_score)
                 except Exception as e: cls_dir = f"Err: {e}"
            
            print(f"Frame {frame_idx}: Source={source} | Gaze={vec_str} | Class={cls_dir}")
            
        frame_idx += 1
        
    cap.release()
    elapsed = time.time() - start_time
    print(f"\nProcessed {len(frame_results)} frames in {elapsed:.2f}s ({len(frame_results)/elapsed:.1f} FPS)")
    
    # 4. Run Audit Generator (To Check Metric)
    print("\nRunning Audit Generator...")
    gen = EyeContactAuditGenerator()
    gaze_vectors = [f.gaze_vector for f in frame_results if f.gaze_vector]
    
    enhanced_results_payload = {
        'frame_results': frame_results,
        'gaze_vectors': gaze_vectors
    }
    
    session_info = {'session_id': 'test_dl', 'duration': frame_idx/24.0, 'video_path': video_path}
    
    try:
        report = gen.generate_audit_report(enhanced_results_payload, session_info)
        print(f"\n=== Audit Result ===")
        print(f"Calculated Metric: {report.eye_contact_percentage:.1f}%")
        
        # Check directions details
        summary = report.gaze_direction_summary
        for d, data in summary.items():
            if data.percentage_of_session > 0:
                num_episodes = len(data.episodes)
                total_dur = data.total_duration
                avg_dur = total_dur / num_episodes if num_episodes > 0 else 0
                print(f" - {d:20}: {data.percentage_of_session:5.1f}% | Frames: {data.frame_count:3} | Freq: {num_episodes} | Avg Dur: {avg_dur:.2f}s")
                
    except Exception as e:
        print(f"Audit generation failed: {e}")

if __name__ == "__main__":
    main()
