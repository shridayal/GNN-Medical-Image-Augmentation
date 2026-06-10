"""
Train GNN-VAE model with REAL Brain MRI data from Kaggle
Graph Neural Network + Variational Autoencoder
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from pathlib import Path
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

#  CHANGED: Use real data loader
from data.data_loader import get_dataloader
from graph.graph_builder import GraphBuilder
from graph.gat_model import GraphAttentionNetwork
from models.simple_vae import GNN_VAE, vae_loss
from training.config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG
from torch_geometric.data import Data, Batch
from scipy import ndimage


class TrainerGNNVAE:
    """Train GNN + VAE with real anatomical data"""
    
    def __init__(self, device='cuda', learning_rate=1e-3):
        self.device = device
        self.lr = learning_rate
        
        print(f"\n{'='*60}")
        print(" Initializing GNN-VAE Trainer")
        print(f"{'='*60}")
        
        #  Initialize models using config
        self.gat = GraphAttentionNetwork(
            input_dim=MODEL_CONFIG['gat']['input_dim'],
            output_dim=MODEL_CONFIG['gat']['output_dim']
        )
        
        self.vae = GNN_VAE(
            latent_dim=MODEL_CONFIG['vae']['latent_dim'],
            struct_code_dim=MODEL_CONFIG['vae']['struct_code_dim']
        )
        
        self.gat = self.gat.to(device)
        self.vae = self.vae.to(device)
        
        #  Separate optimizers for GAT and VAE
        self.gat_optimizer = optim.Adam(self.gat.parameters(), lr=learning_rate)
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=learning_rate)
        
        # Count parameters
        gat_params = sum(p.numel() for p in self.gat.parameters() if p.requires_grad)
        vae_params = sum(p.numel() for p in self.vae.parameters() if p.requires_grad)
        
        print(f"\n Model Parameters:")
        print(f"   GAT:  {gat_params:,}")
        print(f"   VAE:  {vae_params:,}")
        print(f"   Total: {gat_params + vae_params:,}")
        print(f"\n Trainer initialized on {device}\n")
    
    def train_epoch(self, train_loader, epoch=None):
        """
        Train for one epoch with REAL data
        
        Args:
            train_loader: DataLoader with real brain MRI data
            epoch: Current epoch number (for logging)
        """
        self.gat.train()
        self.vae.train()
        
        total_loss = 0.0
        num_batches = 0
        num_failed = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                #  UPDATED: Unpack batch with real data info
                images = batch['image'].float().to(self.device)           # [B, 1, 256, 256]
                masks = batch['mask'].float().to(self.device)             # [B, 1, 256, 256]
                tumor_types = batch['tumor_type']                         # List of strings
                
                batch_size = images.size(0)
                
                # ===== Convert masks to graphs =====
                graphs = []
                valid_indices = []
                
                for idx, mask in enumerate(masks):
                    try:
                        graph = self._mask_to_graph(mask.squeeze().cpu().numpy())
                        
                        if graph is not None and graph.num_nodes > 0:
                            graphs.append(graph)
                            valid_indices.append(idx)
                    except:
                        continue
                
                # Skip batch if no valid graphs
                if len(graphs) == 0:
                    num_failed += 1
                    continue
                
                # ===== GAT forward pass =====
                try:
                    # Batch graphs
                    graph_batch = Batch.from_data_list(graphs).to(self.device)
                    
                    # Forward through GAT
                    struct_codes = self.gat(graph_batch)  # [N_nodes, output_dim]
                    
                    # Pool graph-level features
                    # Group by graph index
                    num_graphs = len(graphs)
                    graph_features = []
                    
                    for g_idx in range(num_graphs):
                        # Get indices for this graph
                        start_idx = sum(graphs[i].num_nodes for i in range(g_idx))
                        end_idx = start_idx + graphs[g_idx].num_nodes
                        
                        # Mean pooling
                        feat = struct_codes[start_idx:end_idx].mean(dim=0)
                        graph_features.append(feat)
                    
                    struct_codes_pooled = torch.stack(graph_features)  # [num_graphs, output_dim]
                    
                except Exception as e:
                    print(f"    GAT error: {str(e)[:50]}")
                    num_failed += 1
                    continue
                
                # ===== VAE forward pass =====
                try:
                    # Select corresponding images
                    valid_images = images[valid_indices]
                    
                    # Ensure matching batch sizes
                    if len(struct_codes_pooled) != len(valid_images):
                        min_len = min(len(struct_codes_pooled), len(valid_images))
                        struct_codes_pooled = struct_codes_pooled[:min_len]
                        valid_images = valid_images[:min_len]
                    
                    # VAE forward
                    recon, mu, logvar = self.vae(valid_images, struct_codes_pooled)
                    
                    # Compute loss
                    loss = vae_loss(recon, valid_images, mu, logvar)
                    
                except Exception as e:
                    print(f"    VAE error: {str(e)[:50]}")
                    num_failed += 1
                    continue
                
                # ===== Backward pass =====
                try:
                    self.gat_optimizer.zero_grad()
                    self.vae_optimizer.zero_grad()
                    
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.gat.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                    
                    self.gat_optimizer.step()
                    self.vae_optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"    Backprop error: {str(e)[:50]}")
                    num_failed += 1
                    continue
            
            except Exception as e:
                print(f"   Batch {batch_idx+1} error: {str(e)[:50]}")
                num_failed += 1
                continue
        
        # Compute average loss
        avg_loss = total_loss / max(num_batches, 1) if num_batches > 0 else float('inf')
        
        return avg_loss
    
    def _mask_to_graph(self, mask_np):
        """
        Convert binary mask to graph object
        
        Args:
            mask_np: [256, 256] binary mask array
        
        Returns:
            torch_geometric.data.Data object
        """
        try:
            # Connected component labeling
            labeled, n_components = ndimage.label(mask_np > 0.5)
            
            if n_components == 0:
                return None
            
            # Extract nodes from components
            nodes = []
            
            for comp_idx in range(1, min(n_components + 1, 30)):  # Limit to 30 regions
                component = (labeled == comp_idx)
                area = component.sum()
                
                # Skip tiny components
                if area < 10:
                    continue
                
                # Compute centroid
                y, x = ndimage.center_of_mass(component)
                
                # Node features: [y_norm, x_norm, area_norm, node_type]
                node_feat = [
                    y / 256.0,              # Normalized Y
                    x / 256.0,              # Normalized X
                    np.log(area + 1) / 12,  # Log area (normalized)
                    1.0                     # Node type (organ/structure)
                ]
                nodes.append(node_feat)
            
            if len(nodes) == 0:
                return None
            
            nodes_tensor = torch.tensor(nodes, dtype=torch.float32)
            
            # ===== Build edges (spatial adjacency) =====
            edges = []
            
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    # Euclidean distance between centroids
                    dist = torch.norm(nodes_tensor[i, :2] - nodes_tensor[j, :2])
                    
                    # Connect if within threshold (normalized distance)
                    if dist < 0.5:
                        edges.append([i, j])
                        edges.append([j, i])  # Bidirectional
            
            if len(edges) == 0:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edges, dtype=torch.long).t()
            
            # Create graph
            graph = Data(x=nodes_tensor, edge_index=edge_index)
            
            return graph
        
        except Exception as e:
            return None
    
    def save_models(self, gat_path, vae_path):
        """Save trained models"""
        try:
            torch.save(self.gat.state_dict(), gat_path)
            torch.save(self.vae.state_dict(), vae_path)
            print(f"    Saved: {Path(gat_path).name}")
            print(f"    Saved: {Path(vae_path).name}")
        except Exception as e:
            print(f"    Error saving models: {e}")
    
    def load_models(self, gat_path, vae_path):
        """Load pre-trained models"""
        try:
            self.gat.load_state_dict(
                torch.load(gat_path, map_location=self.device)
            )
            self.vae.load_state_dict(
                torch.load(vae_path, map_location=self.device)
            )
            print(f"    Loaded GAT: {Path(gat_path).name}")
            print(f"    Loaded VAE: {Path(vae_path).name}")
        except Exception as e:
            print(f"    Error loading models: {e}")


def main():
    """Test trainer with real data"""
    import numpy as np
    
    print("\n" + "="*60)
    print(" Testing GNN-VAE Trainer with REAL Data")
    print("="*60 + "\n")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ===== Initialize trainer =====
    trainer = TrainerGNNVAE(
        device=DEVICE,
        learning_rate=TRAINING_CONFIG['learning_rate']
    )
    
    # ===== Load real data =====
    print("Loading real brain MRI data...")
    try:
        train_loader, dataset = get_dataloader(
            data_dir=DATA_CONFIG['data_dir'],
            batch_size=TRAINING_CONFIG['batch_size'],
            tumor_types=DATA_CONFIG['tumor_types'],
            max_images=20  # Small batch for testing
        )
        print(f" Loaded {len(dataset)} images\n")
    except Exception as e:
        print(f" Error loading data: {e}\n")
        return
    
    # ===== Test single epoch =====
    print("Testing single training epoch...\n")
    try:
        loss = trainer.train_epoch(train_loader, epoch=1)
        print(f"\n Training test successful!")
        print(f"   Epoch loss: {loss:.4f}")
    except Exception as e:
        print(f"\n❌ Training error: {e}")


if __name__ == "__main__":
    main()
