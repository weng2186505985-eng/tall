import torch
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import os
import numpy as np
from tqdm import tqdm

from data.dataset import FewShotFakeVideoDataset, EpisodeSampler
from models.tall_swin import TALLSwin

def evaluate(checkpoint_path=None):
    # 1. Load Config
    config_path = Path(__file__).parent / "configs/default.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Setup Device
    device_name = config['common'].get('device', 'cuda')
    if device_name == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(device_name)
    print(f"Evaluating on {device}")

    # 3. Setup Dataset
    manifest_path = config['paths']['manifest_path']
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return

    n_segment = config['model']['n_segment']
    dataset = FewShotFakeVideoDataset(manifest_path, num_frames=n_segment)
    
    n_way = config['meta_learning']['n_way']
    k_shot = config['meta_learning']['k_shot']
    q_query = config['meta_learning']['q_query']
    iterations = config['meta_learning'].get('eval_iterations', 50)
    
    sampler = EpisodeSampler(dataset, n_way, k_shot, q_query, iterations=iterations)
    dataloader = DataLoader(
        dataset, 
        batch_sampler=sampler, 
        num_workers=config['common'].get('num_workers', 0)
    )

    # 4. Load Model
    model = TALLSwin(
        num_classes=n_way, 
        n_segment=n_segment, 
        pretrained=False
    ).to(device)
    
    if checkpoint_path is None:
        # Try to find latest checkpoint in config-defined dir
        checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        if checkpoint_dir.exists():
            checkpoints = sorted(list(checkpoint_dir.glob("*.pth")))
            if checkpoints:
                checkpoint_path = str(checkpoints[-1])

    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint found. Evaluating untrained model.")

    model.eval()

    # 5. Evaluation Loop
    all_accuracies = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images, _ = batch
            images = images.to(device)
            
            # Forward (ProtoNet mode)
            logits = model(images, n_way=n_way, k_shot=k_shot, mode='few_shot')
            
            # Target for query
            target = torch.arange(n_way).repeat_interleave(q_query).to(device)
            
            preds = logits.argmax(dim=1)
            acc = (preds == target).float().mean().item()
            all_accuracies.append(acc)

    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)
    
    print(f"\nEvaluation Results ({n_way}-way {k_shot}-shot):")
    print(f"Mean Accuracy over {iterations} episodes: {mean_acc:.4f} (+/- {std_acc:.4f})")
    
    return mean_acc

if __name__ == "__main__":
    evaluate()
