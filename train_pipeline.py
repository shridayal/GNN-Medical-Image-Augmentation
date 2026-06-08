
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


def main():
    print("\n" + "="*70)
    print("GNN-GUIDED MEDICAL IMAGE AUGMENTATION")
    print("Training Pipeline")
    print("="*70 + "\n")
    
    # Setup
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {DEVICE}\n")
    
    # Create directories
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)
    
    # ===== STEP 1: Load Data =====
    print("[1/4] Loading data...")
    
    try:
        # Try to load from brain_mri folder directly
        dataset = MedicalImageDataset(
            data_dir="./data/brain_mri",
            max_slices=2000
        )
        print(f"✓ Loaded {len(dataset)} real brain images\n")
        
    except Exception as e:
        print(f"✗ Error loading brain data: {e}")
        print("Trying fallback data locations...\n")
        
        try:
            # Try default data folder
            dataset = MedicalImageDataset(
                data_dir="./data",
                max_slices=2000
            )
            print(f"✓ Loaded {len(dataset)} images from ./data\n")
        except:
            print("✗ No real data found!")
            print("Creating synthetic brain data...\n")
            dataset = create_dummy_brain_dataset(num_samples=500)
            print(f"✓ Created {len(dataset)} synthetic brain images\n")
    
    # Create DataLoader
    print("Creating DataLoader...")
    train_loader = DataLoader(
        dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=2
    )
    print(f"Batch size: {TRAINING_CONFIG['batch_size']}\n")
    
    # ===== STEP 2: Initialize Models =====
    print("[2/4] Initializing models...")
    
    trainer = TrainerGNNVAE(
        device=DEVICE,
        learning_rate=TRAINING_CONFIG['learning_rate']
    )
    graph_builder = GraphBuilder()
    
    print("✓ Graph Attention Network (GAT) initialized")
    print("✓ VAE generative model initialized\n")
    
    # ===== STEP 3: Training =====
    print("[3/4] Training...")
    print(f"Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"Learning rate: {TRAINING_CONFIG['learning_rate']}\n")
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(TRAINING_CONFIG['epochs']):
        print(f"Epoch {epoch+1}/{TRAINING_CONFIG['epochs']}")
        
        loss = trainer.train_epoch(train_loader, graph_builder)
        
        print(f"  Loss: {loss:.4f}\n")
        
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
            print(f"\n✓ Early stopping at epoch {epoch+1}")
            break
        
        # Checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            trainer.save_models(
                os.path.join(PATHS['models_dir'], f'gat_epoch_{epoch+1}.pth'),
                os.path.join(PATHS['models_dir'], f'vae_epoch_{epoch+1}.pth')
            )
            print(f"✓ Checkpoint saved at epoch {epoch+1}\n")
    
    # ===== STEP 4: Save Final Models =====
    print("[4/4] Saving final models...")
    
    trainer.save_models(
        os.path.join(PATHS['models_dir'], 'gat_final.pth'),
        os.path.join(PATHS['models_dir'], 'vae_final.pth')
    )
    
    print("✓ GAT model saved")
    print("✓ VAE model saved\n")
    
    # Summary
    print("="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModels saved to: {PATHS['models_dir']}")
    print(f"  - gat_final.pth")
    print(f"  - vae_final.pth")
    print(f"\nNext steps:")
    print("="*70)
    print("\n1. Generate synthetic images (5 minutes):")
    print("   python generate_synthetic.py")
    print("\n2. Evaluate results (5 minutes):")
    print("   python evaluate_pipeline.py")
    print("\n3. View results:")
    print(f"   - Synthetic images: {PATHS['results_dir']}/synthetic_images/")
    print(f"   - Metrics: {PATHS['results_dir']}/metrics/")
    print("\n" + "="*70 + "\n")


def create_dummy_brain_dataset(num_samples=500):
    """Create dummy brain dataset as fallback"""
    
    class DummyBrainDataset:
        def __init__(self, num_samples):
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            img = np.zeros((256, 256), dtype=np.float32)
            mask = np.zeros((256, 256), dtype=np.uint8)
            
            y, x = np.ogrid[:256, :256]
            
            # Brain
            brain = (x - 128)**2 + (y - 128)**2 <= 95**2
            img[brain] = 100 + np.random.randint(-30, 40)
            mask[brain] = 1
            
            # Ventricles
            vent = ((x-100)**2 + (y-128)**2 <= 18**2) | ((x-156)**2 + (y-128)**2 <= 18**2)
            img[vent] = 170
            mask[vent] = 2
            
            # Tumor (70% chance)
            if np.random.rand() > 0.3:
                tx = np.random.randint(100, 156)
                ty = np.random.randint(80, 176)
                tumor = ((x - tx)**2 + (y - ty)**2) <= 20**2
                img[tumor & brain] = 200
                mask[tumor & brain] = 3
            
            # Add noise
            img = img + np.random.normal(0, 15, img.shape)
            img = np.clip(img, 0, 255)
            
            return {
                'image': torch.from_numpy(img).unsqueeze(0) / 255.0,
                'mask': torch.from_numpy(mask).unsqueeze(0),
                'index': idx
            }
    
    return DummyBrainDataset(num_samples)


if __name__ == "__main__":
    main()