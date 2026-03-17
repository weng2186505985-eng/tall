import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.swin_transformer import SwinTransformer

# ==========================================
# Temporal Shift Module (TSM)
# ==========================================
class TSM(nn.Module):
    """
    Temporal Shift Module: Shifts a portion of channels along the temporal axis.
    """
    def __init__(self, n_segment=16, n_div=8):
        super(TSM, self).__init__()
        self.n_segment = n_segment
        self.n_div = n_div

    def forward(self, x):
        # x: (Batch*Time, C, H, W)
        bt, c, h, w = x.size()
        b = bt // self.n_segment
        t = self.n_segment
        
        # Reshape to (Batch, Time, C, H, W)
        x = x.view(b, t, c, h, w)
        
        # Split channels
        fold = c // self.n_div
        out = torch.zeros_like(x)
        
        # Shift forward
        out[:, 1:, :fold] = x[:, :-1, :fold]
        # Shift backward
        out[:, :-1, fold:2*fold] = x[:, 1:, fold:2*fold]
        # Rest remain same
        out[:, :, 2*fold:] = x[:, :, 2*fold:]
        
        # Reshape back to (Batch*Time, C, H, W)
        return out.view(bt, c, h, w)

# ==========================================
# Prototypical Head
# ==========================================
class PrototypicalHead(nn.Module):
    """
    Computes class prototypes and distances for Few-Shot Learning.
    """
    def __init__(self, feature_dim):
        super(PrototypicalHead, self).__init__()
        self.feature_dim = feature_dim

    def forward(self, features, n_way, k_shot):
        """
        Args:
            features: (N*K + N*Q, D)
        """
        # Normalize features for stability (unit hypersphere)
        features = F.normalize(features, p=2, dim=1)
        
        n_support = n_way * k_shot
        support_features = features[:n_support] # (N*K, D)
        query_features = features[n_support:]     # (N*Q, D)
        
        # Compute prototypes: average over K samples
        prototypes = support_features.view(n_way, k_shot, -1).mean(1) # (N_way, D)
        prototypes = F.normalize(prototypes, p=2, dim=1) # Re-normalize prototypes
        
        # Compute Euclidean distance
        dists = torch.pow(query_features.unsqueeze(1) - prototypes.unsqueeze(0), 2).sum(2)
        
        # Scale distances to prevent vanishing gradients if needed
        return -dists * 10 

# ==========================================
# TALL-Swin Model
# ==========================================
class TALLSwin(nn.Module):
    def __init__(self, num_classes=2, n_segment=16, window_size=8, stride=4, pretrained=True):
        super(TALLSwin, self).__init__()
        self.n_segment = n_segment
        self.window_size = window_size
        self.stride = stride
        
        # 1. Swin-Transformer-Base backbone
        # We use timm to load swin_base_patch4_window7_224.ms_in22k
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224.ms_in22k', 
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=0.1,
            grad_checkpointing=True
        )
        
        # 2. Insert TSM after Patch Merging / Swin Blocks
        # In timm Swin, we can wrap the blocks or use hooks. 
        # A common TSM approach for Swin is to apply it between stages.
        self.tsm = TSM(n_segment=n_segment, n_div=8)
        
        # Feature dimension for Swin-B is 1024
        self.feature_dim = 1024
        
        # 3. Prototypical Head for Few-Shot
        self.proto_head = PrototypicalHead(self.feature_dim)
        
        # 4. Standard Classifier (for non-few-shot inference)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x):
        """
        Forward through backbone with TSM logic.
        Input x: (Batch, Time, C, H, W)
        """
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        
        # Note: Ideally TSM is integrated inside Swin blocks to affect attention.
        # Here we apply TSM before the backbone or between layers if modified.
        # Simplified: Apply TSM to feature maps inside stages if we had full surgery.
        # Baseline TSM-Swin: Move temporal info before feature extraction.
        x = self.tsm(x)
        
        # Extract features (B*T, D)
        feats = self.backbone.forward_features(x)
        
        # Swin usually returns (B, L, D) or (B, C, H, W) or (B, D*L)
        if len(feats.shape) == 3:
            # (Batch, Tokens, Dim) -> (Batch, Dim)
            feats = feats.mean(1)
        elif len(feats.shape) == 4:
            # (Batch, Dim, H, W) -> (Batch, Dim)
            feats = feats.mean((2, 3))
        elif len(feats.shape) == 2 and feats.shape[1] > self.feature_dim:
            # Handle flattened tokens if necessary (e.g., shape is B, 50176)
            feats = feats.view(feats.shape[0], -1, self.feature_dim).mean(1)
            
        return feats.view(b, t, -1) # (B, T, D)

    def sliding_window_analysis(self, temporal_feats):
        """
        逐段输出异常分数 (window_size=8, stride=4)
        temporal_feats: (B, T, D)
        """
        b, t, d = temporal_feats.shape
        scores = []
        
        for i in range(0, t - self.window_size + 1, self.stride):
            window = temporal_feats[:, i : i + self.window_size].mean(1) # (B, D)
            logits = self.classifier(window) # (B, num_classes)
            scores.append(logits)
            
        return torch.stack(scores, dim=1) # (B, WinCount, num_classes)

    def forward(self, x, n_way=None, k_shot=None, mode='inference'):
        """
        x: (B, T, C, H, W)
        """
        # 1. Feature Extraction
        temporal_feats = self.forward_features(x) # (B, T, D)
        
        # 2. Few-Shot ProtoNet Path
        if mode == 'few_shot':
            # Flatten all samples (N*K + N*Q) from B dimension
            # Assuming B already contains Support + Query for an episode
            avg_feats = temporal_feats.mean(1) # Temporal pooling (B, D)
            return self.proto_head(avg_feats, n_way, k_shot)
            
        # 3. Standard Sliding Window Inference Path
        else:
            window_logits = self.sliding_window_analysis(temporal_feats)
            # Final output is the average or max of window scores
            final_logits = window_logits.mean(1) 
            return final_logits, window_logits

# ==========================================
# Unit Test
# ==========================================
if __name__ == "__main__":
    print("Initializing TALL-Swin (Swin-B + TSM)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate model (no pretrained weights to save time/mem in test)
    model = TALLSwin(n_segment=16, pretrained=False).to(device)
    
    # Input: (Batch, Time=16, C=3, H=224, W=224)
    dummy_input = torch.randn(2, 16, 3, 224, 224).to(device)
    
    # 1. Test standard inference
    model.eval()
    with torch.no_grad():
        final_out, win_scores = model(dummy_input)
        print(f"Standard Output shape (Final): {final_out.shape}")
        print(f"Standard Output shape (Windows): {win_scores.shape}") # Expect 3 windows (0-8, 4-12, 8-16)

    # 2. Test few-shot mode (5-way 5-shot)
    n_way, k_shot, q_query = 5, 5, 1
    fs_input = torch.randn(n_way * (k_shot + q_query), 16, 3, 224, 224).to(device)
    proto_logits = model(fs_input, n_way=n_way, k_shot=k_shot, mode='few_shot')
    print(f"Few-Shot Logits shape: {proto_logits.shape}") # (N*Q, N_way) -> (5*1, 5)
    
    print("\nUnit Test Passed!")
