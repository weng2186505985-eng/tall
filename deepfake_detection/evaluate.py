import torch
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import os
import numpy as np
from tqdm import tqdm

from data.dataset import FewShotFakeVideoDataset, EpisodeSampler
from models.tall_swin import TALLSwin

from utils.metrics import calculate_eer, calculate_auc

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
        checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        if checkpoint_dir.exists():
            # Prioritize best model for evaluation
            best_pth = checkpoint_dir / "tall_swin_best.pth"
            if best_pth.exists():
                checkpoint_path = str(best_pth)
            else:
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
    all_y_true = []
    all_y_scores = []
    
    # Identify the index of the "Original" class for EER calculation
    try:
        orig_class_idx = dataset.class_list.index("Original")
    except ValueError:
        orig_class_idx = -1

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images, _ = batch
            images = images.to(device)
            
            # Forward (ProtoNet mode)
            logits = model(images, n_way=n_way, k_shot=k_shot, mode='few_shot') # (N_way * Q_query, N_way)
            
            # Target for query (0..n_way-1 relative to selected classes)
            target = torch.arange(n_way).repeat_interleave(q_query).to(device)
            
            preds = logits.argmax(dim=1)
            acc = (preds == target).float().mean().item()
            all_accuracies.append(acc)
            
            # --- EER/AUC Collection ---
            # To compute a general EER, we map the multi-way problem to binary (Real vs Fake)
            # if the "Original" class is present in the current task.
            probs = torch.softmax(logits, dim=-1) # (N*Q, N_way)
            
            # We treat the task as binary: Original (0) vs everything else (1)
            for i in range(logits.size(0)):
                # Get the ground truth class name from the sampler's selection
                # But PrototypicalHead's output order is fixed: cls_0, cls_1...
                # The 'target[i]' is the local class index in the current episode.
                
                # We need to know which local index corresponds to "Original"
                local_class_idx = target[i].item()
                
                # In many-way fake classification, if we want a single EER, 
                # we usually define 'Real' as class 0.
                # Here we check if the selected class is Original.
                is_fake = 0 if local_class_idx == orig_class_idx else 1
                
                # Score for being 'Fake': sum of probabilities of all non-Original classes
                if orig_class_idx != -1:
                    fake_score = 1.0 - probs[i, orig_class_idx].item()
                else:
                    # If Original isn't in this episode, we use the max non-target probability 
                    # or skip for binary EER. For now, we contribute anyway to get a general metric.
                    fake_score = probs[i].max().item()
                
                all_y_true.append(is_fake)
                all_y_scores.append(fake_score)

    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)
    
    print(f"\nEvaluation Results ({n_way}-way {k_shot}-shot):")
    print(f"Mean Accuracy over {iterations} episodes: {mean_acc:.4f} (+/- {std_acc:.4f})")
    
    if len(set(all_y_true)) > 1:
        eer = calculate_eer(all_y_true, all_y_scores)
        auc = calculate_auc(all_y_true, all_y_scores)
        print(f"Equal Error Rate (EER): {eer:.4f}")
        print(f"Area Under Curve (AUC): {auc:.4f}")
    
    return mean_acc

if __name__ == "__main__":
    evaluate()
