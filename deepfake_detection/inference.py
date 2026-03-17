import torch
import cv2
from torchvision import transforms
from PIL import Image
from models.tall_swin import TALLSwin
from models.syncnet import SyncNet
from models.fusion import WeightedFusion
from utils.preprocessing import process_single_video
import shutil
import os

class InferencePipeline:
    def __init__(self, visual_ckpt, audio_ckpt=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Visual Model
        self.visual_model = TALLSwin(n_segment=8, pretrained=False).to(self.device)
        self.visual_model.load_state_dict(torch.load(visual_ckpt, map_location=self.device))
        self.visual_model.eval()
        
        # Load Audio Model (Optional)
        self.audio_model = None
        if audio_ckpt:
            self.audio_model = SyncNet().to(self.device)
            self.audio_model.load_state_dict(torch.load(audio_ckpt, map_location=self.device))
            self.audio_model.eval()
            
        self.fusion = WeightedFusion()

    def predict(self, video_path):
        temp_dir = Path("temp_inference")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Real Preprocessing
        
        processed_meta = process_single_video(video_path, temp_dir, target_fps=2, max_frames=8)
        if not processed_meta or not processed_meta["frame_paths"]:
            if temp_dir.exists(): shutil.rmtree(temp_dir)
            return {"error": "No faces detected in video."}

        # 2. Load & Transform Frames
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        frames = []
        # Sample exactly 8 frames if possible, or repeat
        paths = sorted(processed_meta["frame_paths"])[:8]
        for p in paths:
            img = Image.open(p).convert('RGB')
            frames.append(transform(img))
            
        # Pad if insufficient frames
        while len(frames) < 8:
            frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))
            
        x = torch.stack(frames).unsqueeze(0).to(self.device) # (1, 8, 3, 224, 224)
        
        # 3. Model Forward
        with torch.no_grad():
            v_logits, _ = self.visual_model(x)
            
        # 4. Decision
        prob = torch.softmax(v_logits, dim=-1)
        fake_prob = prob[0, 1].item()
        
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            
        return {
            "fake_probability": fake_prob,
            "verdict": "Fake" if fake_prob > 0.5 else "Real"
        }

if __name__ == "__main__":
    # pipeline = InferencePipeline("checkpoints/best_visual.pth")
    # print(pipeline.predict("sample_video.mp4"))
    print("Inference pipeline initialized.")
