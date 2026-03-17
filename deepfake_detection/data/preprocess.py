import cv2
import subprocess

def extract_frames(video_path, output_dir, fps=10):
    """FFmpeg frame extraction."""
    raise NotImplementedError("Standard FFmpeg extraction is not configured. Use OpenCV version in scripts/ instead.")

def align_face(image, target_size=(112, 112)):
    """OpenCV face alignment and cropping."""
    raise NotImplementedError("Face alignment in data/preprocess.py is a placeholder. Use scripts/preprocess_ffplusplus.py.")
