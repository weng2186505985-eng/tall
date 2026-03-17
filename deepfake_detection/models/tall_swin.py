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
        # x can be (Batch*Time, C, H, W) or (Batch*Time, L, C)
        if len(x.shape) == 3:
            # Token format: (BT, L, C)
            bt, l, c = x.size()
            b = bt // self.n_segment
            t = self.n_segment
            
            x = x.view(b, t, l, c)
            fold = c // self.n_div
            out = torch.zeros_like(x)
            
            # Shift along Time axis (dim=1)
            out[:, 1:, :, :fold] = x[:, :-1, :, :fold]
            out[:, :-1, :, fold:2*fold] = x[:, 1:, :, fold:2*fold]
            out[:, :, :, 2*fold:] = x[:, :, :, 2*fold:]
            
            return out.view(bt, l, c)
        else:
            # Spatial format: (BT, C, H, W)
            bt, c, h, w = x.size()
            b = bt // self.n_segment
            t = self.n_segment
            
            x = x.view(b, t, c, h, w)
            fold = c // self.n_div
            out = torch.zeros_like(x)
            
            out[:, 1:, :fold, :, :] = x[:, :-1, :fold, :, :]
            out[:, :-1, fold:2*fold, :, :] = x[:, 1:, fold:2*fold, :, :]
            out[:, :, 2*fold:, :, :] = x[:, :, 2*fold:, :, :]
            
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
        # Learnable temperature scaling
        self.scale = nn.Parameter(torch.tensor(10.0))

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
        
        # Scale distances to prevent vanishing gradients
        return -dists * self.scale

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
        
        # 1. Swin Patch Embedding
        x = self.backbone.patch_embed(x)
        
        # 2. Apply TSM after patch embedding (on feature tokens)
        # Position: After patch_embed, before first stage
        x = self.tsm(x)
        
        # 3. Positional Dropping and Swin Blocks
        if self.backbone.absolute_pos_embed is not None:
             x = x + self.backbone.absolute_pos_embed
        x = self.backbone.pos_drop(x)
        
        # Stages
        x = self.backbone.layers(x)
        x = self.backbone.norm(x)
        
        # 4. Final Head / Pooling
        # Using timm's built-in head logic if available, or standard pooling
        if hasattr(self.backbone, 'forward_head'):
            x = self.backbone.forward_head(x, pre_logits=True)
        else:
            if x.dim() == 3: x = x.mean(dim=1)
            elif x.dim() == 4: x = x.mean(dim=(2, 3))
            
        return x.view(b, t, -1) # (B, T, D)

    def sliding_window_analysis(self, temporal_feats):
        """
        逐段输出异常分数 (window_size=8, stride=4)
        temporal_feats: (B, T, D)
        """
        b, t, d = temporal_feats.shape
        scores = []
        
        for i in range(0, t - self.window_size + 1, self.stride):
            window = temporal_feats[:, i : i + self.window_size].mean(1)
            logits = self.classifier(window)
            scores.append(logits)
            
        # Boundary handling: Ensure the very last frames are included if not captured by stride
        last_start = t - self.window_size
        if last_start > 0 and (t - self.window_size) % self.stride != 0:
            window = temporal_feats[:, last_start:].mean(1)
            logits = self.classifier(window)
            scores.append(logits)
            
        return torch.stack(scores, dim=1)

    def forward(self, x, n_way=None, k_shot=None, mode='inference'):
        """
        x: (B, T, C, H, W)
        """
        # 1. Feature Extraction
        temporal_feats = self.forward_features(x) # (B, T, D)
        
        # 2. Few-Shot ProtoNet Path
        if mode == 'few_shot':
            assert n_way is not None and k_shot is not None, "Few-shot mode requires n_way and k_shot"
            
            # Temporal pooling: (B, T, D) -> (B, D)
            avg_feats = temporal_feats.mean(1)
            
            # Verify batch layout: (N*K + N*Q)
            # The EpisodicSampler yields support set then query set.
            # We enforce this expectation by passing the features to the head.
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
        dummy_input = torch.randn(1, 8, 3, 224, 224).to(device)
        final_out, win_scores = model(dummy_input)
        print(f"Standard Output shape (Final): {final_out.shape}")
        print(f"Standard Output shape (Windows): {win_scores.shape}")

    # 2. Test few-shot mode (Reduced for memory)
    n_way, k_shot, q_query = 3, 1, 1
    fs_input = torch.randn(n_way * (k_shot + q_query), 8, 3, 224, 224).to(device)
    proto_logits = model(fs_input, n_way=n_way, k_shot=k_shot, mode='few_shot')
    print(f"Few-Shot Logits shape: {proto_logits.shape}") # (N*Q, N_way) -> (3*1, 3)
    
    print("\nUnit Test Passed!")
