"""
Master training script - Run this to train the entire pipeline
"""

import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from data.data_loader import MedicalImageDataset
from graph.graph_builder import GraphBuilder
from graph.gat_model import GraphAttentionNetwork
from models.simple_vae import GNN_VAE, vae_loss
from training.config import TRAINING_CONFIG, DATA_CONFIG, MODEL_CONFIG, PATHS
from training.train_gnn_vae import TrainerGNNVAE
from torch_geometric.data import Batch

def create_dummy_dataset(num_samples=50, save_path="./data/dummy"):
    """
    Create dummy medical image dataset for testing
    """
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Creating {num_samples} dummy medical images...")
    
    class DummyDataset:
        def __init__(self, num_samples):
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Create random image
            img = torch.rand(1, 256, 256)
            
            # Create random mask with 2-3 regions
            mask = torch.zeros(1, 256, 256)
            mask[:, 50:150, 50:150] = 1
            mask[:, 180:230, 180:230] = 2
            if np.random.rand() > 0.5:
                mask[:, 20:70, 150:200] = 3
            
            return {
                'image': img,
                'mask': mask,
                'index': idx
            }
    
    return DummyDataset(num_samples)


def main():
    print("\n" + "="*60)
    print("GNN-GUIDED MEDICAL IMAGE AUGMENTATION")
    print("Training Pipeline")
    print("="*60 + "\n")
    
    # Setup
    DEVICE = TRAINING_CONFIG['device'] if torch.cuda.is_available() else 'cpu'
    print(f"Device: {DEVICE}")
    
    # Create directories
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)
    
    # Data
    print("\n[1/4] Loading data...")
    try:
        dataset = MedicalImageDataset(
            DATA_CONFIG['data_dir'],
            max_slices=DATA_CONFIG['max_slices']
        )
        print(f"✓ Loaded {len(dataset)} real images")
    except:
        print("⚠ Could not load real data. Creating dummy dataset...")
        dataset = create_dummy_dataset(num_samples=50)
        print(f"✓ Created dummy dataset with {len(dataset)} images")
    
    # Create DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    # Initialize trainer
    print("\n[2/4] Initializing models...")
    trainer = TrainerGNNVAE(device=DEVICE, learning_rate=TRAINING_CONFIG['learning_rate'])
    graph_builder = GraphBuilder()
    print("✓ Models initialized")
    
    # Training loop
    print("\n[3/4] Training...")
    print(f"Training for {TRAINING_CONFIG['epochs']} epochs\n")
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(TRAINING_CONFIG['epochs']):
        loss = trainer.train_epoch(train_loader, graph_builder)
        
        # Early stopping
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
            # Save best model
            trainer.save_models(
                os.path.join(PATHS['models_dir'], 'gat_best.pth'),
                os.path.join(PATHS['models_dir'], 'vae_best.pth')
            )
        else:
            patience_counter += 1
        
        if patience_counter >= TRAINING_CONFIG['early_stopping_patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        # Checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            trainer.save_models(
                os.path.join(PATHS['models_dir'], f'gat_epoch_{epoch+1}.pth'),
                os.path.join(PATHS['models_dir'], f'vae_epoch_{epoch+1}.pth')
            )
    
    # Save final models
    print("\n[4/4] Saving final models...")
    trainer.save_models(
        os.path.join(PATHS['models_dir'], 'gat_final.pth'),
        os.path.join(PATHS['models_dir'], 'vae_final.pth')
    )
    print("✓ Models saved!")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Generate synthetic images:")
    print("   python generate_synthetic.py")
    print("\n2. Evaluate results:")
    print("   python evaluate_pipeline.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()