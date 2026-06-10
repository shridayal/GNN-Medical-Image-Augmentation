"""
GNN-Guided Medical Image Augmentation - Training Pipeline
Uses REAL Brain MRI data from Kaggle
"""

import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

#  CHANGED: Use real data loader
from data.data_loader import get_dataloader
from graph.graph_builder import GraphBuilder
from graph.gat_model import GraphAttentionNetwork
from models.simple_vae import GNN_VAE, vae_loss
from training.config import TRAINING_CONFIG, DATA_CONFIG, MODEL_CONFIG, PATHS
from training.train_gnn_vae import TrainerGNNVAE
from torch_geometric.data import Batch
import torch.nn as nn


def main():
    print("\n" + "="*70)
    print(" GNN-GUIDED MEDICAL IMAGE AUGMENTATION")
    print("Training Pipeline with REAL Brain MRI Data")
    print("="*70 + "\n")
    
    # ===== SETUP =====
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {DEVICE}\n")
    
    # Create directories
    for path_name, path_val in PATHS.items():
        os.makedirs(path_val, exist_ok=True)
    print(f" Created output directories\n")
    
    # ===== STEP 1: Load REAL Data =====
    print("[1/4] Loading REAL Brain MRI Data from Kaggle...")
    print(f"      Data path: {DATA_CONFIG['data_dir']}")
    print(f"      Tumor types: {DATA_CONFIG['tumor_types']}\n")
    
    try:
        #  LOAD REAL DATA
        train_loader, dataset = get_dataloader(
            data_dir=DATA_CONFIG['data_dir'],
            batch_size=TRAINING_CONFIG['batch_size'],
            tumor_types=DATA_CONFIG['tumor_types'],
            max_images=DATA_CONFIG['max_images']
        )
        
        print(f" Loaded {len(dataset)} real brain MRI images")
        print(f"Created DataLoader: {len(train_loader)} batches\n")
        
    except Exception as e:
        print(f" Error loading real data: {e}")
        print("Please check:")
        print(f"  - Data directory exists: {DATA_CONFIG['data_dir']}")
        print(f"  - Tumor folders exist: {DATA_CONFIG['tumor_types']}")
        print(f"  - Images are .jpg format\n")
        return
    
    # ===== STEP 2: Initialize Models =====
    print("[2/4] Initializing Models...")
    
    try:
        trainer = TrainerGNNVAE(
            device=DEVICE,
            learning_rate=TRAINING_CONFIG['learning_rate']
        )
        graph_builder = GraphBuilder()
        
        print(" Graph Attention Network (GAT) initialized")
        print(" VAE generative model initialized")
        print(f" Learning rate: {TRAINING_CONFIG['learning_rate']}\n")
        
    except Exception as e:
        print(f" Error initializing models: {e}\n")
        return
    
    # ===== STEP 3: Training Loop =====
    print("[3/4] Training Models...")
    print(f"      Total epochs: {TRAINING_CONFIG['epochs']}")
    print(f"      Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"      Early stopping patience: {TRAINING_CONFIG['early_stopping_patience']}\n")
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(TRAINING_CONFIG['epochs']):
        print(f"\n{'='*70}")
        print(f" EPOCH {epoch+1}/{TRAINING_CONFIG['epochs']}")
        print(f"{'='*70}")
        
        epoch_loss = 0.0
        num_batches = 0
        
        try:
            for batch_idx, batch in enumerate(train_loader):
                #  UPDATED: Unpack batch with real data info
                images = batch['image'].to(DEVICE)      # [B, 1, 256, 256]
                masks = batch['mask'].to(DEVICE)        # [B, 1, 256, 256]
                tumor_types = batch['tumor_type']       # List of strings
                
                # Convert masks to graphs
                try:
                    graphs = _masks_to_graphs(masks, graph_builder)
                except:
                    print(f"    Skipping batch {batch_idx+1} (graph conversion error)")
                    continue
                
                # GNN encoding
                gnn_features_list = []
                for graph in graphs:
                    try:
                        graph = graph.to(DEVICE)
                        gnn_feat = trainer.gat(graph)
                        gnn_features_list.append(gnn_feat.mean(dim=0))
                    except:
                        # Fallback: zero features
                        gnn_features_list.append(torch.zeros(MODEL_CONFIG['gat']['output_dim']).to(DEVICE))
                
                gnn_features = torch.stack(gnn_features_list)
                
                # VAE forward pass
                try:
                    reconstructed, mu, logvar, struct_code = trainer.vae(images, gnn_features)
                    loss = vae_loss(images, reconstructed, mu, logvar)
                    
                    # Backward
                    trainer.gat_optimizer.zero_grad()
                    trainer.vae_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainer.gat.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(trainer.vae.parameters(), 1.0)
                    trainer.gat_optimizer.step()
                    trainer.vae_optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    if (batch_idx + 1) % 5 == 0:
                        print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | "
                              f"Types: {tumor_types}")
                
                except Exception as e:
                    print(f"    Error in batch {batch_idx+1}: {str(e)[:50]}")
                    continue
        
        except Exception as e:
            print(f" Error in epoch {epoch+1}: {e}")
            continue
        
        # Compute average loss
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            print(f"\n Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")
        else:
            print(f"\n  Epoch {epoch+1}: No successful batches")
            continue
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # Save best model
            trainer.save_models(
                os.path.join(PATHS['models_dir'], 'gat_best.pth'),
                os.path.join(PATHS['models_dir'], 'vae_best.pth')
            )
            print(f"    New best loss! Saved checkpoint.")
        else:
            patience_counter += 1
            print(f"     No improvement ({patience_counter}/{TRAINING_CONFIG['early_stopping_patience']})")
        
        # Early stopping trigger
        if patience_counter >= TRAINING_CONFIG['early_stopping_patience']:
            print(f"\n  Early stopping triggered at epoch {epoch+1}")
            break
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            trainer.save_models(
                os.path.join(PATHS['models_dir'], f'gat_epoch_{epoch+1}.pth'),
                os.path.join(PATHS['models_dir'], f'vae_epoch_{epoch+1}.pth')
            )
            print(f"    Checkpoint saved at epoch {epoch+1}")
    
    # ===== STEP 4: Save Final Models =====
    print("\n[4/4] Saving Final Models...")
    
    trainer.save_models(
        os.path.join(PATHS['models_dir'], 'gat_final.pth'),
        os.path.join(PATHS['models_dir'], 'vae_final.pth')
    )
    
    print("GAT model saved: gat_final.pth")
    print("VAE model saved: vae_final.pth\n")
    
    # ===== Summary =====
    print("="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\n Results Summary:")
    print(f"   Models saved to: {PATHS['models_dir']}")
    print(f"   Final loss: {best_loss:.4f}")
    print(f"   Total epochs: {epoch+1}")
    print(f"\n Next Steps:")
    print("="*70)
    print("\n1. Generate synthetic images:")
    print("   python inference/generate_synthetic.py")
    print("\n2. Evaluate results:")
    print("   python evaluate_pipeline.py")
    print("\n3. View results:")
    print(f"   - Synthetic images: {PATHS['results_dir']}/synthetic_images/")
    print(f"   - Metrics: {PATHS['results_dir']}/metrics/")
    print("\n" + "="*70 + "\n")


def _masks_to_graphs(masks, graph_builder):
    """
    Convert masks to graph objects
    
    Args:
        masks: [B, 1, 256, 256] tensor
        graph_builder: GraphBuilder instance
    
    Returns:
        List of torch_geometric.data.Data objects
    """
    from torch_geometric.data import Data
    from scipy import ndimage
    
    graphs = []
    batch_size = masks.shape[0]
    
    for b in range(batch_size):
        mask = masks[b].squeeze().cpu().numpy()
        
        # Connected component labeling
        labeled, n_components = ndimage.label(mask > 0.5)
        
        # Extract nodes
        nodes = []
        for i in range(1, min(n_components + 1, 20)):  # Limit to 20 nodes
            component = (labeled == i)
            if component.sum() < 10:  # Skip tiny components
                continue
            
            y, x = ndimage.center_of_mass(component)
            area = component.sum()
            
            node_feat = [y/256, x/256, area/256**2, 1.0]
            nodes.append(node_feat)
        
        if len(nodes) == 0:
            nodes = [[0.5, 0.5, 0.01, 1.0]]
        
        nodes = torch.tensor(nodes, dtype=torch.float32)
        
        # Build edges (connect nearby nodes)
        edges = []
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                dist = torch.norm(nodes[i, :2] - nodes[j, :2])
                if dist < 0.5:  # Proximity threshold
                    edges.append([i, j])
                    edges.append([j, i])
        
        if len(edges) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        graph = Data(x=nodes, edge_index=edge_index)
        graphs.append(graph)
    
    return graphs


if __name__ == "__main__":
    main()
