import sys
import os
from pathlib import Path

# Add project root to path for absolute imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
from deepfake_detection.models.tall_swin import PrototypicalHead, TALLSwin

def test_learnable_scale():
    print("Testing Learnable Scale...")
    head = PrototypicalHead(feature_dim=1024)
    # Check if scale is in parameters
    params = list(head.parameters())
    param_names = [n for n, p in head.named_parameters()]
    print(f"Parameters: {param_names}")
    assert "scale" in param_names, "Scale parameter not found!"
    assert head.scale.requires_grad, "Scale should be learnable!"
    print("Learnable Scale Test Passed!")

def test_sliding_window_boundary():
    print("Testing Sliding Window Boundary...")
    # T=10, Win=8, Stride=4
    # Windows: [0:8], [4:10] (last one should be t-win:t)
    model = TALLSwin(n_segment=10, window_size=8, stride=4, pretrained=False).to('cpu')
    temporal_feats = torch.randn(1, 10, 1024)
    
    with torch.no_grad():
        scores = model.sliding_window_analysis(temporal_feats)
        print(f"Input T=10, Window=8, Stride=4 -> Output shape: {scores.shape}")
        # Range(0, 10-8+1, 4) -> range(0, 3, 4) -> only 0.
        # Boundary logic: last_start = 10-8 = 2. 
        # (10-8)%4 = 2 != 0. So it should add a window at index 2.
        # Total windows: 2.
        assert scores.shape[1] == 2, f"Expected 2 windows, got {scores.shape[1]}"
    print("Sliding Window Boundary Test Passed!")

def test_robust_pooling():
    print("Testing Robust Pooling...")
    model = TALLSwin(pretrained=False).to('cpu')
    
    # Test 3D (BT, L, D)
    x3d = torch.randn(2, 49, 1024)
    out3d = model.forward_features(torch.randn(1, 2, 3, 224, 224).to('cpu')) # uses forward_features logic
    print(f"Output shape: {out3d.shape}")
    assert out3d.shape == (1, 2, 1024), f"Expected (1, 2, 1024), got {out3d.shape}"
    print("Robust Pooling Test Passed!")

if __name__ == "__main__":
    test_learnable_scale()
    test_sliding_window_boundary()
    test_robust_pooling()
