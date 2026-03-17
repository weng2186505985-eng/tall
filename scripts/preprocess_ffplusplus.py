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

from utils.preprocessing import process_single_video

def process_video_worker(video_path, label, fake_type, output_dir, target_fps, max_frames):
    """
    Wrapper for the shared processing logic. 
    Category is determined here and passed to the utility.
    """
    category = fake_type if label == 1 else "Original"
    res = process_single_video(video_path, output_dir / category, target_fps, max_frames)
    if res:
        res["label"] = label
        res["fake_type"] = fake_type
    return res

class FFPlusPlusPreprocessor:
    def __init__(self, root_dir, output_dir, n_way=5, k_shot=5, q_query=15, limit_videos=None, target_fps=2, max_frames=20, num_workers=None):
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.limit_videos = limit_videos
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.num_workers = num_workers if num_workers else os.cpu_count() or 2
        
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
