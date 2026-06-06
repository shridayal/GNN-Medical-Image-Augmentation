import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from pathlib import Path
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import MedicalImageDataset
from graph.graph_builder import GraphBuilder
from graph.gat_model import GraphAttentionNetwork
from models.simple_vae import GNN_VAE, vae_loss
from torch_geometric.data import Batch

class TrainerGNNVAE:
    """Train GNN + VAE"""
    
    def __init__(self, device='cuda', learning_rate=1e-3):
        self.device = device
        self.lr = learning_rate
        
        # Initialize models
        self.gat = GraphAttentionNetwork(input_dim=5, output_dim=32)
        self.vae = GNN_VAE(latent_dim=32, struct_code_dim=32)
        
        self.gat = self.gat.to(device)
        self.vae = self.vae.to(device)
        
        # Optimizer - FIX: Define params properly
        self.params = list(self.gat.parameters()) + list(self.vae.parameters())
        self.optimizer = optim.Adam(self.params, lr=learning_rate)
        
        print(f"Total trainable parameters: {sum(p.numel() for p in self.params if p.requires_grad)}")
    
    def train_epoch(self, train_loader, graph_builder):
        """Train for one epoch"""
        self.gat.train()
        self.vae.train()
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            try:
                # Build graphs from masks
                graphs = []
                for mask in masks:
                    mask_np = mask.cpu().numpy()
                    graph = graph_builder.mask_to_graph(mask_np)
                    if graph is not None:
                        graphs.append(graph)
                
                if len(graphs) == 0:
                    continue
                
                # Move graphs to device
                graph_batch = Batch.from_data_list(graphs).to(self.device)
                
                # Forward pass through GAT
                struct_codes = self.gat(graph_batch)
                
                # Adjust batch size if needed
                batch_size = min(len(graphs), images.size(0))
                images = images[:batch_size]
                struct_codes = struct_codes[:batch_size]
                
                # VAE forward
                recon, mu, logvar = self.vae(images, struct_codes)
                
                # Loss
                loss = vae_loss(recon, images, mu, logvar)
                
                # Backward - FIX: Use self.params
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{total_loss / max(num_batches, 1):.4f}'})
                
            except Exception as e:
                print(f"\nError in batch: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def save_models(self, gat_path, vae_path):
        """Save trained models"""
        torch.save(self.gat.state_dict(), gat_path)
        torch.save(self.vae.state_dict(), vae_path)
        print(f"✓ Models saved: {Path(gat_path).name}, {Path(vae_path).name}")
    
    def load_models(self, gat_path, vae_path):
        """Load pre-trained models"""
        self.gat.load_state_dict(torch.load(gat_path, map_location=self.device))
        self.vae.load_state_dict(torch.load(vae_path, map_location=self.device))
        print("✓ Models loaded!")


def main():
    """Test trainer"""
    from training.config import TRAINING_CONFIG
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {DEVICE}\n")
    
    # Create trainer
    trainer = TrainerGNNVAE(device=DEVICE, learning_rate=TRAINING_CONFIG['learning_rate'])
    
    print("✓ Trainer initialized successfully!")


if __name__ == "__main__":
    main()