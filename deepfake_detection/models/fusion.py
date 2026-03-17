import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFusion(nn.Module):
    """
    Weighted fusion of visual and audio stream probabilities.
    Learns or uses fixed weights to combine different modalities.
    """
    def __init__(self, weight_v=0.6, weight_a=0.4, trainable=False):
        super(WeightedFusion, self).__init__()
        if trainable:
            self.weights = nn.Parameter(torch.tensor([weight_v, weight_a]))
        else:
            self.register_buffer('weights', torch.tensor([weight_v, weight_a]))
            
        self.trainable = trainable

    def forward(self, visual_logits, audio_logits):
        """
        Merge logits or probabilities.
        """
        v_prob = F.softmax(visual_logits, dim=-1)
        # Assuming audio_logits are scores from SyncNet distance
        # Convert distance to probability (lower distance -> higher sync/higher confidence)
        a_prob = F.softmax(audio_logits, dim=-1) # Placeholder
        
        if self.trainable:
            normalized_weights = F.softmax(self.weights, dim=0)
        else:
            # Force normalization for fixed weights
            normalized_weights = self.weights / (self.weights.sum() + 1e-7)
        
        final_prob = normalized_weights[0] * v_prob + normalized_weights[1] * a_prob
        return final_prob

if __name__ == "__main__":
    fusion = WeightedFusion(trainable=True)
    v = torch.randn(4, 2)
    a = torch.randn(4, 2)
    out = fusion(v, a)
    print(f"Fused output shape: {out.shape}")
    print(f"Weights: {fusion.weights.data}")
