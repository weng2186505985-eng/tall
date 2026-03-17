import torch
from torch.utils.data import Dataset, Sampler
import json
import cv2
import numpy as np
from pathlib import Path

class FewShotFakeVideoDataset(Dataset):
    """
    Dataset for loading video frames organized by meta-learning tasks.
    """
    def __init__(self, manifest_path, transform=None, num_frames=8):
        with open(manifest_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transform
        self.num_frames = num_frames
        
        # Flatten manifest for random access
        self.all_samples = self.data.get('all_preprocessed', [])
        if not self.all_samples:
             self.all_samples = self.data.get('support_set', []) + self.data.get('query_set', [])
        
        # Build category map
        self.categories = {}
        for idx, item in enumerate(self.all_samples):
            cls = item['fake_type'] if item['label'] == 1 else "Original"
            if cls not in self.categories:
                self.categories[cls] = []
            self.categories[cls].append(idx)
        
        self.class_list = sorted(list(self.categories.keys()))

    def __len__(self):
        return len(self.all_samples)

    def load_video_frames(self, frame_paths):
        frames = []
        # Sample or loop to get exactly num_frames
        indices = np.linspace(0, len(frame_paths)-1, self.num_frames).astype(int)
        sampled_paths = [frame_paths[i] for i in indices]
        
        for p in sampled_paths:
            img = cv2.imread(p)
            if img is None:
                img = np.zeros((112, 112, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            else:
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            frames.append(img)
            
        return torch.stack(frames) # (T, C, H, W)

    def __getitem__(self, idx):
        item = self.all_samples[idx]
        frames = self.load_video_frames(item['frame_paths'])
        label = item['label']
        # Map class name to numeric index for n-way classification
        cls_name = item['fake_type'] if label == 1 else "Original"
        cls_idx = self.class_list.index(cls_name)
        
        return frames, cls_idx

class EpisodeSampler(Sampler):
    """
    Samples episodes (support set + query set) for N-way K-shot learning.
    """
    def __init__(self, dataset, n_way, k_shot, q_query, iterations):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.iterations = iterations

    def __iter__(self):
        for _ in range(self.iterations):
            # 1. Randomly select N classes
            selected_classes = np.random.choice(self.dataset.class_list, self.n_way, replace=False)
            
            support_indices = []
            query_indices = []
            
            for cls in selected_classes:
                all_indices_for_cls = self.dataset.categories[cls]
                # 2. Select K+Q samples for each class
                population_size = len(all_indices_for_cls)
                num_to_sample = self.k_shot + self.q_query
                
                if population_size < num_to_sample:
                    # Fallback: sample with replacement if population is too small, 
                    # or just take what's available
                    sampled_indices = np.random.choice(all_indices_for_cls, num_to_sample, replace=True)
                else:
                    sampled_indices = np.random.choice(all_indices_for_cls, num_to_sample, replace=False)
                
                support_indices.extend(sampled_indices[:self.k_shot])
                query_indices.extend(sampled_indices[self.k_shot:])
            
            # Yield episode (batch of support followed by batch of query)
            yield support_indices + query_indices

    def __len__(self):
        return self.iterations
