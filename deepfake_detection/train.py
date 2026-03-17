import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F

from data.dataset import FewShotFakeVideoDataset, EpisodeSampler
from models.tall_swin import TALLSwin

def train():
    # 1. Resolve Project Root and Load Config
    project_root = Path(__file__).resolve().parent.parent
    config_path = Path(__file__).resolve().parent / "configs/default.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Setup Device
    device_name = config['common'].get('device', 'cuda')
    if device_name == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(device_name)
    
    # 3. Resolve Relative Paths
    for key in ['dataset_root', 'manifest_path', 'output_dir', 'checkpoint_dir']:
        if key in config['paths']:
            val = config['paths'][key]
            if not Path(val).is_absolute():
                config['paths'][key] = str(project_root / val)
    
    print(f"Using device: {device}")

    # 3. Initialize Dataset and Sampler
    manifest_path = config['paths']['manifest_path']
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest {manifest_path} not found. Ensure preprocessing is complete.")
        return

    n_segment = config['model']['n_segment']
    dataset = FewShotFakeVideoDataset(manifest_path, num_frames=n_segment)
    
    n_way = config['meta_learning']['n_way']
    k_shot = config['meta_learning']['k_shot']
    q_query = config['meta_learning']['q_query']
    iterations = config['meta_learning']['iterations']
    
    sampler = EpisodeSampler(dataset, n_way, k_shot, q_query, iterations=iterations)
    dataloader = DataLoader(
        dataset, 
        batch_sampler=sampler, 
        num_workers=config['common'].get('num_workers', 0)
    )

    # 4. Initialize Model
    model = TALLSwin(
        num_classes=n_way, 
        n_segment=n_segment, 
        pretrained=config['model'].get('pretrained', True)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config['training']['step_size'], 
        gamma=config['training']['gamma']
    )
    
    # Enable Mixed Precision
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # 5. Training Loop
    model.train()
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(config['training']['epochs']):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            # batch: images (B, T, C, H, W), labels (B)
            images, labels = batch
            images = images.to(device)
            
            # Target labels for query set (0..n_way-1)
            target = torch.arange(n_way).repeat_interleave(q_query).to(device)
            
            # Forward in few-shot mode with AMP
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits = model(images, n_way=n_way, k_shot=k_shot, mode='few_shot')
                loss = F.cross_entropy(logits, target)
            
            # Scaled Backward and Step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        scheduler.step()
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Save Checkpoint (Keep best and latest to save 30GB+ disk space)
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            torch.save(model.state_dict(), checkpoint_dir / "tall_swin_best.pth")
            print(f"New Best Model Saved (Loss: {best_loss:.4f})")
            
        # Always save latest for resumption
        torch.save(model.state_dict(), checkpoint_dir / "tall_swin_latest.pth")
        
        # 6. Periodic Evaluation (e.g., every 10 epochs)
        if (epoch + 1) % 10 == 0:
            print(f"\n--- Running Intermediate Evaluation (Epoch {epoch+1}) ---")
            from evaluate import evaluate
            model.eval()
            evaluate(checkpoint_path=str(checkpoint_dir / "tall_swin_latest.pth"))
            model.train()

if __name__ == "__main__":
    train()
