import os
import cv2
import json
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import logging
import numpy as np

# ==========================================
# Phase: Data Preprocessing & Few-Shot Setup (OpenCV Version)
# Description: FaceForensics++ video processing,
#              face alignment, and N-way K-shot construction.
#              Using cv2.VideoCapture due to missing FFmpeg.
# ==========================================

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class FFPlusPlusPreprocessor:
    def __init__(self, root_dir, output_dir, n_way=5, k_shot=5, q_query=15, limit_videos=None):
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.limit_videos = limit_videos
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.fake_types = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
        self.classes = ["Original"] + self.fake_types
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_frames_opencv(self, video_path, output_folder, target_fps=10):
        """
        Extract frames using OpenCV instead of FFmpeg.
        """
        output_folder.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 25 # Fallback
        
        interval = max(1, int(fps / target_fps))
        
        count = 0
        success, image = cap.read()
        frame_idx = 0
        while success:
            if count % interval == 0:
                frame_name = f"frame_{frame_idx:04d}.png"
                cv2.imwrite(str(output_folder / frame_name), image)
                frame_idx += 1
            success, image = cap.read()
            count += 1
        cap.release()

    def align_and_crop_face(self, img_path, target_size=(224, 224)):
        img = cv2.imread(str(img_path))
        if img is None: return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0: return None 
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, target_size)
        return face_img

    def process_video(self, video_path, label, fake_type):
        video_name = video_path.stem
        raw_frames_dir = self.output_dir / "temp_frames" / video_name
        processed_frames_dir = self.output_dir / "processed" / video_name
        processed_frames_dir.mkdir(parents=True, exist_ok=True)
        
        self.extract_frames_opencv(video_path, raw_frames_dir)
        
        frame_paths = []
        for frame_file in sorted(raw_frames_dir.glob("*.png")):
            face = self.align_and_crop_face(frame_file)
            if face is not None:
                save_path = processed_frames_dir / frame_file.name
                cv2.imwrite(str(save_path), face)
                frame_paths.append(str(save_path))
        
        if raw_frames_dir.exists():
            shutil.rmtree(raw_frames_dir)
        
        return {
            "video_name": video_name,
            "label": label,
            "fake_type": fake_type,
            "frame_paths": frame_paths
        }

    def construct_meta_learning_dataset(self, video_data_list):
        class_groups = {}
        for item in video_data_list:
            cls = item['fake_type'] if item['label'] == 1 else "Original"
            if cls not in class_groups:
                class_groups[cls] = []
            class_groups[cls].append(item)

        for cls in self.classes:
            if cls not in class_groups:
                logger.warning(f"Insufficient data for class: {cls}")
                continue

        manifest = {
            "config": {"n_way": self.n_way, "k_shot": self.k_shot, "q_query": self.q_query},
            "all_preprocessed": video_data_list
        }
        
        with open(self.output_dir / "dataset_manifest.json", "w") as f:
            json.dump(manifest, f, indent=4)
        logger.info(f"Manifest saved to {self.output_dir / 'dataset_manifest.json'}")

    def run(self):
        video_tasks = []
        # Original
        original_dir = self.root_dir / "original"
        if original_dir.exists():
            vids = list(original_dir.glob("*.mp4"))
            if self.limit_videos: vids = vids[:self.limit_videos]
            for v in vids: video_tasks.append((v, 0, "Original"))
        # Fake
        for ftype in self.fake_types:
            fake_dir = self.root_dir / ftype
            if fake_dir.exists():
                vids = list(fake_dir.glob("*.mp4"))
                if self.limit_videos: vids = vids[:self.limit_videos]
                for v in vids: video_tasks.append((v, 1, ftype))

        if not video_tasks:
            logger.error(f"No videos found in {self.root_dir}!")
            return

        logger.info(f"Processing {len(video_tasks)} videos...")
        all_video_results = []
        for v_path, label, ftype in tqdm(video_tasks, desc="Overall Progress"):
            res = self.process_video(v_path, label, ftype)
            if res["frame_paths"]: all_video_results.append(res)

        self.construct_meta_learning_dataset(all_video_results)

if __name__ == "__main__":
    DATA_ROOT = "d:/bishe/Yuhonghai/deepfake_detection/data/raw/FaceForensics++_C23" 
    OUTPUT_ROOT = "d:/bishe/Yuhonghai/deepfake_detection/data/processed"
    
    # Run subset for testing
    preprocessor = FFPlusPlusPreprocessor(DATA_ROOT, OUTPUT_ROOT, limit_videos=10)
    preprocessor.run()
