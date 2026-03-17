import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network implementation for Few-Shot Learning.
    Computes class prototypes and uses Euclidean distance for classification.
    """
    def __init__(self, encoder):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = encoder

    def forward(self, support, query, support_labels, n_way, k_shot):
        """
        Args:
            support: Support set images (N*K, C, H, W) or (N*K, T, C, H, W)
            query: Query set images (N*Q, C, H, W) or (N*Q, T, C, H, W)
            support_labels: Labels for support set (N*K,)
            n_way: Number of classes
            k_shot: Samples per class in support set
        """
        # 1. Extract features
        support_embeddings = self.encoder(support) # (N*K, D)
        query_embeddings = self.encoder(query)     # (N*Q, D)
        
        # 2. Compute prototypes (Mean of embeddings for each class)
        # Expected support shape: (N_way, K_shot, D)
        feature_dim = support_embeddings.size(-1)
        prototypes = support_embeddings.view(n_way, k_shot, feature_dim).mean(dim=1) # (N_way, D)
        
        # 3. Compute Euclidean distances to prototypes
        # query_embeddings: (N*Q, D) -> (N*Q, 1, D)
        # prototypes: (N_way, D) -> (1, N_way, D)
        distances = torch.pow(query_embeddings.unsqueeze(1) - prototypes.unsqueeze(0), 2).sum(dim=2) # (N*Q, N_way)
        
        # 4. Probabilities (Log-Softmax of negative distances)
        log_p_y = F.log_softmax(-distances, dim=1)
        
        return log_p_y

    def compute_loss(self, log_p_y, target):
        """
        Standard Cross Entropy for ProtoNet.
        target: (N*Q,)
        """
        return F.nll_loss(log_p_y, target)

if __name__ == "__main__":
    # Test with dummy encoder
    class SimpleEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)
        def forward(self, x): return self.fc(x)

    encoder = SimpleEncoder()
    model = PrototypicalNetwork(encoder)
    
    n_way, k_shot, n_query = 5, 5, 2
    support = torch.randn(n_way * k_shot, 10)
    query = torch.randn(n_way * n_query, 10)
    labels = torch.arange(n_way).repeat_interleave(k_shot)
    target = torch.arange(n_way).repeat_interleave(n_query)
    
    log_p = model(support, query, labels, n_way, k_shot)
    loss = model.compute_loss(log_p, target)
    print(f"Log probabilities shape: {log_p.shape}")
    print(f"Loss: {loss.item():.4f}")
