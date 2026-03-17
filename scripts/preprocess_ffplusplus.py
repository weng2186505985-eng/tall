import cv2
import json
import shutil
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# Phase: Data Preprocessing & Few-Shot Setup (OpenCV Version)
# Description: FaceForensics++ video processing,
#              face alignment, and N-way K-shot construction.
#              Using cv2.VideoCapture due to missing FFmpeg.
# ==========================================

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def process_video_worker(video_path, label, fake_type, output_dir, target_fps, max_frames):
    """
    Worker function for parallel processing. 
    Instantiates its own cascade classifier to avoid pickling issues.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    video_name = video_path.stem
    category = fake_type if label == 1 else "Original"
    
    raw_frames_dir = output_dir / ".temp_frames" / video_name
    processed_frames_dir = output_dir / category / video_name
    processed_frames_dir.mkdir(parents=True, exist_ok=True)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # --- extract_frames_opencv logic ---
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
    
    # --- align_and_crop_face logic ---
    frame_paths = []
    for frame_file in sorted(raw_frames_dir.glob("*.png")):
        img = cv2.imread(str(frame_file))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
                face_img = img[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (224, 224))
                save_path = processed_frames_dir / frame_file.name
                cv2.imwrite(str(save_path), face_img)
                frame_paths.append(str(save_path))
    
    if raw_frames_dir.exists():
        shutil.rmtree(raw_frames_dir)
    
    if not frame_paths:
        if processed_frames_dir.exists() and not any(processed_frames_dir.iterdir()):
            processed_frames_dir.rmdir()
        return None

    return {
        "video_name": video_name,
        "label": label,
        "fake_type": fake_type,
        "frame_paths": frame_paths
    }

class FFPlusPlusPreprocessor:
    def __init__(self, root_dir, output_dir, n_way=5, k_shot=5, q_query=15, limit_videos=None, target_fps=2, max_frames=20, num_workers=2):
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.limit_videos = limit_videos
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.num_workers = num_workers
        
        self.fake_types = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter", "DeepFakeDetection"]
        self.classes = ["Original"] + self.fake_types
        
        self.output_dir.mkdir(parents=True, exist_ok=True)


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

        logger.info(f"Processing {len(video_tasks)} videos using {self.num_workers} workers...")
        all_video_results = []
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(process_video_worker, v_path, label, ftype, self.output_dir, self.target_fps, self.max_frames)
                for v_path, label, ftype in video_tasks
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Overall Progress"):
                res = future.result()
                if res and res["frame_paths"]:
                    all_video_results.append(res)

        self.construct_meta_learning_dataset(all_video_results)

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    
    # Use relative paths from project root
    DATA_ROOT = project_root / "deepfake_detection/data/raw/FaceForensics++_C23" 
    OUTPUT_ROOT = project_root / "deepfake_detection/data/processed"
    
    # Run subset for testing
    # Run full dataset processing (limit_videos=None for full run)
    preprocessor = FFPlusPlusPreprocessor(DATA_ROOT, OUTPUT_ROOT, limit_videos=None, target_fps=2, max_frames=20, num_workers=2)
    preprocessor.run()
