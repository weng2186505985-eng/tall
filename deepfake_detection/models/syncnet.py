import torch
import torch.nn as nn
import torch.nn.functional as F

class SyncNet(nn.Module):
    """
    SyncNet architecture for measuring Audio-Visual Synchronization (Lip-Sync).
    Detects inconsistencies between the mouth movements and speech audio.
    """
    def __init__(self):
        super(SyncNet, self).__init__()
        
        # Visual Stream (CNN for mouth region frames)
        # Input: (B, 5, 1, 112, 112) - 5 grayscale frames of mouth region
        self.visual_stream = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # Audio Stream (CNN for Mel-Spectrogram)
        # Input: (B, 1, 80, 20) - Mel-spec segments
        self.audio_stream = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.v_fc = nn.Linear(512, 128)
        self.a_fc = nn.Linear(512, 128)

    def forward(self, visual_input, audio_input):
        """
        Args:
            visual_input: (B, 1, T, H, W) - Grayscale mouth frames
            audio_input: (B, 1, Freq, Time) - Mel-spectrogram
        """
        # Feature extraction
        v_feat = self.visual_stream(visual_input).view(visual_input.size(0), -1)
        a_feat = self.audio_stream(audio_input).view(audio_input.size(0), -1)
        
        # Projection to common embedding space
        v_emb = self.v_fc(v_feat)
        a_emb = self.a_fc(a_feat)
        
        # Normalize for cosine similarity calculation
        v_emb = F.normalize(v_emb, p=2, dim=1)
        a_emb = F.normalize(a_emb, p=2, dim=1)
        
        return v_emb, a_emb

    def get_sync_distance(self, v_emb, a_emb):
        """Distance between visual and audio embeddings (lower is more synced)"""
        return F.pairwise_distance(v_emb, a_emb)

if __name__ == "__main__":
    model = SyncNet()
    v_in = torch.randn(2, 1, 5, 112, 112)
    a_in = torch.randn(2, 1, 80, 20)
    v_emb, a_emb = model(v_in, a_in)
    print(f"Visual Embed shape: {v_emb.shape}")
    print(f"Audio Embed shape: {a_emb.shape}")
    print("SyncNet backbone initialized successfully.")
