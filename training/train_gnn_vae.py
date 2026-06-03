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
    """Train GNN-guided VAE"""
    
    def __init__(self, device='cuda', learning_rate=1e-3):
        self.device = device
        self.lr = learning_rate
        
        # Initialize models
        self.gat = GraphAttentionNetwork(input_dim=5, output_dim=32)
        self.vae = GNN_VAE(latent_dim=32, struct_code_dim=32)
        
        self.gat = self.gat.to(device)
        self.vae = self.vae.to(device)
        
        # Optimizer
        params = list(self.gat.parameters()) + list(self.vae.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)
    
    def train_epoch(self, train_loader, graph_builder):
        """Train for one epoch"""
        self.gat.train()
        self.vae.train()
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Build graphs from masks
            graphs = []
            for mask in masks:
                graph = graph_builder.mask_to_graph(mask.cpu().numpy())
                if graph is not None:
                    graphs.append(graph)
            
            if len(graphs) == 0:
                continue
            
            # Move graphs to device
            graph_batch = Batch.from_data_list(graphs).to(self.device)
            
            # Forward pass
            struct_codes = self.gat(graph_batch)
            
            # Adjust batch size if needed
            batch_size = min(len(graphs), images.size(0))
            images = images[:batch_size]
            struct_codes = struct_codes[:batch_size]
            
            # VAE forward
            recon, mu, logvar = self.vae(images, struct_codes)
            
            # Loss
            loss = vae_loss(recon, images, mu, logvar)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': total_loss / num_batches})
        
        return total_loss / num_batches
    
    def save_models(self, gat_path, vae_path):
        """Save trained models"""
        torch.save(self.gat.state_dict(), gat_path)
        torch.save(self.vae.state_dict(), vae_path)
        print(f"Models saved:\n  GAT: {gat_path}\n  VAE: {vae_path}")
    
    def load_models(self, gat_path, vae_path):
        """Load pre-trained models"""
        self.gat.load_state_dict(torch.load(gat_path, map_location=self.device))
        self.vae.load_state_dict(torch.load(vae_path, map_location=self.device))
        print("Models loaded!")


def main():
    # Configuration
    BATCH_SIZE = 8
    EPOCHS = 20
    DATA_DIR = "./data/BraTS"  # Change to your data directory
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # Create data loader
    dataset = MedicalImageDataset(DATA_DIR, max_slices=1000)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Create trainer
    trainer = TrainerGNNVAE(device=DEVICE)
    graph_builder = GraphBuilder()
    
    # Training loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        loss = trainer.train_epoch(train_loader, graph_builder)
        print(f"Loss: {loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            trainer.save_models(
                f"./models/gat_epoch_{epoch+1}.pth",
                f"./models/vae_epoch_{epoch+1}.pth"
            )
    
    print("\nTraining complete!")
    trainer.save_models("./models/gat_final.pth", "./models/vae_final.pth")


if __name__ == "__main__":
    main()