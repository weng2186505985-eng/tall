import cv2
import shutil
import numpy as np
from pathlib import Path

def process_single_video(video_path, output_dir, target_fps=2, max_frames=20, target_size=(224, 224)):
    """
    Standalone function for processing a single video (used by preprocessor and inference).
    Returns basic metadata if successful.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    video_name = video_path.stem
    
    # Paths
    raw_frames_dir = output_dir / f".temp_{video_name}"
    processed_frames_dir = output_dir / video_name
    processed_frames_dir.mkdir(parents=True, exist_ok=True)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 1. Extract Frames
    raw_frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0: fps = 25 
    if total_frames <= 0: total_frames = 300 
    
    interval = max(1, total_frames // max_frames)
    
    count = 0
    success, image = cap.read()
    frame_idx = 0
    while success:
        if count % interval == 0:
            frame_name = f"frame_{frame_idx:04d}.png"
            cv2.imwrite(str(raw_frames_dir / frame_name), image)
            frame_idx += 1
            if max_frames and frame_idx >= max_frames:
                break
        success, image = cap.read()
        count += 1
    cap.release()
    
    # 2. Align & Crop Face
    frame_paths = []
    for frame_file in sorted(raw_frames_dir.glob("*.png")):
        img = cv2.imread(str(frame_file))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                # Keep largest face
                x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
                face_img = img[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, target_size)
                
                # Save to processed dir
                save_path = processed_frames_dir / frame_file.name
                cv2.imwrite(str(save_path), face_img)
                frame_paths.append(str(save_path))
    
    # 3. Cleanup temp
    if raw_frames_dir.exists():
        shutil.rmtree(raw_frames_dir)
    
    if not frame_paths:
        if processed_frames_dir.exists() and not any(processed_frames_dir.iterdir()):
            processed_frames_dir.rmdir()
        return None

    return {
        "video_name": video_name,
        "frame_paths": [str(p) for p in frame_paths],
        "processed_dir": str(processed_frames_dir)
    }
