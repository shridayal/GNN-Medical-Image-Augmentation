import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.gat_model import GraphAttentionNetwork
from models.simple_vae import GNN_VAE
from graph.graph_builder import GraphBuilder

class SyntheticImageGenerator:
    """Generate synthetic medical images"""
    
    def __init__(self, gat_path, vae_path, device='cuda'):
        self.device = device
        
        # Load models
        self.gat = GraphAttentionNetwork(input_dim=5, output_dim=32)
        self.vae = GNN_VAE(latent_dim=32, struct_code_dim=32)
        
        self.gat.load_state_dict(torch.load(gat_path, map_location=device))
        self.vae.load_state_dict(torch.load(vae_path, map_location=device))
        
        self.gat = self.gat.to(device).eval()
        self.vae = self.vae.to(device).eval()
        
        print("Models loaded!")
    
    def generate_from_mask(self, mask, num_samples=4):
        """
        Generate synthetic images guided by anatomical structure
        
        Args:
            mask: Segmentation mask (numpy array)
            num_samples: Number of images to generate
            
        Returns:
            Synthetic images (torch tensor)
        """
        with torch.no_grad():
            # Build graph from mask
            builder = GraphBuilder()
            graph = builder.mask_to_graph(mask)
            
            if graph is None:
                print("Error: Could not build graph from mask")
                return None
            
            # Get structural code
            graph = graph.to(self.device)
            struct_code = self.gat(graph)
            
            # Expand for multiple samples
            struct_code = struct_code.repeat(num_samples, 1)
            
            # Generate images
            synthetic_images = self.vae.generate(
                num_samples=num_samples,
                struct_code=struct_code,
                device=self.device
            )
        
        return synthetic_images
    
    def generate_random(self, num_samples=4):
        """Generate completely random images (no structural guidance)"""
        with torch.no_grad():
            synthetic_images = self.vae.generate(
                num_samples=num_samples,
                struct_code=None,
                device=self.device
            )
        return synthetic_images
    
    @staticmethod
    def save_images(images, output_dir, prefix="synthetic"):
        """Save generated images"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, img in enumerate(images):
            img_np = img.squeeze().cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            
            img_pil = Image.fromarray(img_np)
            save_path = output_dir / f"{prefix}_{idx}.png"
            img_pil.save(save_path)
            print(f"Saved: {save_path}")
    
    @staticmethod
    def visualize_images(images, real_image=None, title="Generated Images"):
        """Visualize generated images"""
        num_images = len(images)
        fig, axes = plt.subplots(1, num_images + (1 if real_image is not None else 0), 
                                 figsize=(12, 3))
        
        if real_image is not None:
            axes[0].imshow(real_image.squeeze().cpu().numpy(), cmap='gray')
            axes[0].set_title("Real")
            axes[0].axis('off')
            start_idx = 1
        else:
            start_idx = 0
        
        for i, img in enumerate(images):
            axes[start_idx + i].imshow(img.squeeze().cpu().numpy(), cmap='gray')
            axes[start_idx + i].set_title(f"Synthetic {i+1}")
            axes[start_idx + i].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig


def main():
    # Paths
    GAT_MODEL = "./models/gat_final.pth"
    VAE_MODEL = "./models/vae_final.pth"
    OUTPUT_DIR = "./results/synthetic_images"
    
    # Create generator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = SyntheticImageGenerator(GAT_MODEL, VAE_MODEL, device=device)
    
    # Example: Generate from random mask
    print("\n=== Generating Synthetic Images ===\n")
    
    # Create dummy mask
    dummy_mask = np.zeros((256, 256))
    dummy_mask[50:150, 50:150] = 1
    dummy_mask[180:230, 180:230] = 2
    
    # Generate images
    synthetic_images = generator.generate_from_mask(dummy_mask, num_samples=4)
    
    if synthetic_images is not None:
        print(f"Generated {synthetic_images.shape[0]} images")
        print(f"Image shape: {synthetic_images.shape}")
        
        # Save images
        generator.save_images(synthetic_images, OUTPUT_DIR)
        
        # Visualize
        fig = generator.visualize_images(synthetic_images, title="GNN-Guided Synthetic Images")
        plt.savefig("./results/synthetic_comparison.png", dpi=150, bbox_inches='tight')
        print("Visualization saved!")
        
        plt.show()


if __name__ == "__main__":
    main()