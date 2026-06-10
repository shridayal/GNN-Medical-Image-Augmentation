"""
Generate synthetic medical images using trained GNN-VAE
Uses real brain MRI masks as conditioning
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import sys
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.gat_model import GraphAttentionNetwork
from models.simple_vae import GNN_VAE
from training.config import MODEL_CONFIG, PATHS
from torch_geometric.data import Data


class SyntheticImageGenerator:
    """
    Generate synthetic medical images conditioned on anatomical structure
    Uses trained GAT + VAE models
    """
    
    def __init__(self, gat_path, vae_path, device='cuda'):
        """
        Initialize generator with trained models
        
        Args:
            gat_path: Path to trained GAT model
            vae_path: Path to trained VAE model
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device
        
        print(f"\n{'='*60}")
        print("🎨 Loading Synthetic Image Generator")
        print(f"{'='*60}\n")
        
        # ===== Load GAT Model =====
        print(f"📂 Loading GAT from: {gat_path}")
        try:
            self.gat = GraphAttentionNetwork(
                input_dim=MODEL_CONFIG['gat']['input_dim'],
                output_dim=MODEL_CONFIG['gat']['output_dim']
            )
            self.gat.load_state_dict(torch.load(gat_path, map_location=device))
            self.gat = self.gat.to(device).eval()
            print("✅ GAT loaded\n")
        except Exception as e:
            print(f"❌ Error loading GAT: {e}\n")
            raise
        
        # ===== Load VAE Model =====
        print(f"📂 Loading VAE from: {vae_path}")
        try:
            self.vae = GNN_VAE(
                latent_dim=MODEL_CONFIG['vae']['latent_dim'],
                struct_code_dim=MODEL_CONFIG['vae']['struct_code_dim']
            )
            self.vae.load_state_dict(torch.load(vae_path, map_location=device))
            self.vae = self.vae.to(device).eval()
            print("✅ VAE loaded\n")
        except Exception as e:
            print(f"❌ Error loading VAE: {e}\n")
            raise
        
        print(f"✅ Generator ready on {device}\n")
    
    def generate_from_mask(self, mask, num_samples=4, temperature=1.0):
        """
        Generate synthetic images guided by anatomical mask
        
        Args:
            mask: Segmentation mask [1, 256, 256] or [256, 256]
            num_samples: Number of images to generate
            temperature: Sampling temperature (higher = more diverse)
        
        Returns:
            synthetic_images: Generated images [num_samples, 1, 256, 256]
        """
        with torch.no_grad():
            try:
                # Handle tensor/numpy conversion
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.squeeze().cpu().numpy()
                else:
                    mask_np = mask.squeeze()
                
                # ===== Build Graph from Mask =====
                graph = self._mask_to_graph(mask_np)
                
                if graph is None or graph.num_nodes == 0:
                    print("  ⚠️  Warning: Could not build valid graph from mask")
                    return None
                
                # ===== Get Structural Code from GAT =====
                graph = graph.to(self.device)
                
                # Forward through GAT
                node_features = self.gat(graph)  # [num_nodes, output_dim]
                
                # Pool to graph-level features (mean pooling)
                struct_code = node_features.mean(dim=0, keepdim=True)  # [1, output_dim]
                
                # Expand for multiple samples
                struct_code = struct_code.repeat(num_samples, 1)
                
                # ===== Generate Images =====
                synthetic_images = self.vae.generate(
                    num_samples=num_samples,
                    struct_code=struct_code,
                    device=self.device,
                    temperature=temperature
                )
                
                return synthetic_images
            
            except Exception as e:
                print(f"  ❌ Generation error: {str(e)[:60]}")
                return None
    
    def generate_random(self, num_samples=4, temperature=1.0):
        """
        Generate completely random images (no structural guidance)
        
        Args:
            num_samples: Number of images to generate
            temperature: Sampling temperature
        
        Returns:
            synthetic_images: Generated images [num_samples, 1, 256, 256]
        """
        with torch.no_grad():
            try:
                synthetic_images = self.vae.generate(
                    num_samples=num_samples,
                    struct_code=None,
                    device=self.device,
                    temperature=temperature
                )
                return synthetic_images
            except Exception as e:
                print(f"❌ Random generation error: {e}")
                return None
    
    def _mask_to_graph(self, mask_np):
        """
        Convert binary mask to graph object
        
        Args:
            mask_np: [256, 256] binary mask
        
        Returns:
            torch_geometric.data.Data object
        """
        try:
            # Connected component labeling
            labeled, n_components = ndimage.label(mask_np > 0.5)
            
            if n_components == 0:
                return None
            
            # Extract nodes
            nodes = []
            
            for comp_idx in range(1, min(n_components + 1, 30)):
                component = (labeled == comp_idx)
                area = component.sum()
                
                if area < 10:  # Skip tiny components
                    continue
                
                # Centroid
                y, x = ndimage.center_of_mass(component)
                
                # Node features
                node_feat = [
                    y / 256.0,
                    x / 256.0,
                    np.log(area + 1) / 12,
                    1.0
                ]
                nodes.append(node_feat)
            
            if len(nodes) == 0:
                return None
            
            nodes_tensor = torch.tensor(nodes, dtype=torch.float32)
            
            # Build edges
            edges = []
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    dist = torch.norm(nodes_tensor[i, :2] - nodes_tensor[j, :2])
                    if dist < 0.5:
                        edges.append([i, j])
                        edges.append([j, i])
            
            if len(edges) == 0:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edges, dtype=torch.long).t()
            
            graph = Data(x=nodes_tensor, edge_index=edge_index)
            return graph
        
        except Exception as e:
            return None
    
    @staticmethod
    def save_images(images, output_dir, prefix="synthetic", verbose=True):
        """
        Save generated images
        
        Args:
            images: Generated images [num_samples, 1, 256, 256]
            output_dir: Directory to save to
            prefix: Filename prefix
            verbose: Print save paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for idx, img in enumerate(images):
            img_np = img.squeeze().cpu().numpy()
            img_np = np.clip(img_np, 0, 1)
            img_np = (img_np * 255).astype(np.uint8)
            
            img_pil = Image.fromarray(img_np)
            save_path = output_dir / f"{prefix}_{idx:03d}.png"
            img_pil.save(save_path)
            
            saved_paths.append(save_path)
            
            if verbose:
                print(f"  💾 Saved: {save_path.name}")
        
        return saved_paths
    
    @staticmethod
    def visualize_comparison(real_img, mask, synthetic, title="Image Comparison", 
                            save_path=None):
        """
        Visualize real vs mask vs synthetic
        
        Args:
            real_img: Real image [1, 256, 256]
            mask: Anatomical mask [1, 256, 256]
            synthetic: Synthetic image [1, 256, 256]
            title: Figure title
            save_path: Path to save figure
        
        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Real image
        real_np = real_img.squeeze().cpu().numpy() if isinstance(real_img, torch.Tensor) else real_img.squeeze()
        axes[0].imshow(real_np, cmap='gray')
        axes[0].set_title('Real MRI Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Mask
        mask_np = mask.squeeze().cpu().numpy() if isinstance(mask, torch.Tensor) else mask.squeeze()
        axes[1].imshow(mask_np, cmap='jet', alpha=0.8)
        axes[1].set_title('Anatomical Mask', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Synthetic
        synth_np = synthetic.squeeze().cpu().numpy() if isinstance(synthetic, torch.Tensor) else synthetic.squeeze()
        axes[2].imshow(synth_np, cmap='gray')
        axes[2].set_title('Generated Synthetic', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def visualize_batch(images, title="Generated Images", cols=4, save_path=None):
        """
        Visualize batch of images in grid
        
        Args:
            images: Generated images [num_samples, 1, 256, 256]
            title: Figure title
            cols: Number of columns in grid
            save_path: Path to save figure
        
        Returns:
            matplotlib figure
        """
        num_images = len(images)
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        
        if num_images == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, img in enumerate(images):
            img_np = img.squeeze().cpu().numpy() if isinstance(img, torch.Tensor) else img.squeeze()
            img_np = np.clip(img_np, 0, 1)
            
            axes[idx].imshow(img_np, cmap='gray')
            axes[idx].set_title(f'Sample {idx+1}', fontsize=10, fontweight='bold')
            axes[idx].axis('off')
        
        # Hide extra axes
        for idx in range(num_images, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def main():
    """Test image generation"""
    print("\n" + "="*60)
    print("Testing Synthetic Image Generator")
    print("="*60 + "\n")
    
    # ===== Setup =====
    GAT_MODEL = Path(PATHS['models_dir']) / 'gat_final.pth'
    VAE_MODEL = Path(PATHS['models_dir']) / 'vae_final.pth'
    OUTPUT_DIR = Path(PATHS['results_dir']) / 'synthetic_images'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ===== Check Models =====
    if not GAT_MODEL.exists() or not VAE_MODEL.exists():
        print("❌ Models not found!")
        print(f"   Expected: {GAT_MODEL}")
        print(f"   Expected: {VAE_MODEL}")
        print("\n⚠️  First run: python train_pipeline.py\n")
        return
    
    # ===== Initialize Generator =====
    try:
        generator = SyntheticImageGenerator(
            str(GAT_MODEL),
            str(VAE_MODEL),
            device=device
        )
    except Exception as e:
        print(f"❌ Failed to initialize generator: {e}\n")
        return
    
    # ===== Test 1: Generate from mask =====
    print("="*60)
    print("Test 1: Generate from anatomical mask")
    print("="*60 + "\n")
    
    # Create dummy mask
    dummy_mask = np.zeros((256, 256))
    dummy_mask[50:150, 50:150] = 1      # Main structure
    dummy_mask[180:220, 180:220] = 2    # Secondary structure
    
    print("Generating 4 images from mask...")
    synthetic = generator.generate_from_mask(dummy_mask, num_samples=4)
    
    if synthetic is not None:
        print(f"✅ Generated: {synthetic.shape}\n")
        
        # Save
        generator.save_images(synthetic, OUTPUT_DIR, prefix="from_mask")
        print()
    else:
        print("❌ Generation failed\n")
    
    # ===== Test 2: Generate random =====
    print("="*60)
    print("Test 2: Generate unconditional random images")
    print("="*60 + "\n")
    
    print("Generating 4 random images...")
    random_synthetic = generator.generate_random(num_samples=4)
    
    if random_synthetic is not None:
        print(f"✅ Generated: {random_synthetic.shape}\n")
        
        # Save
        generator.save_images(random_synthetic, OUTPUT_DIR, prefix="random")
        print()
    else:
        print("❌ Generation failed\n")
    
    # ===== Test 3: Visualization =====
    if synthetic is not None:
        print("="*60)
        print("Test 3: Visualizations")
        print("="*60 + "\n")
        
        # Batch visualization
        fig = generator.visualize_batch(
            synthetic[:4],
            title="GNN-Guided Synthetic Images",
            save_path=OUTPUT_DIR / "batch_visualization.png"
        )
        print(f"✅ Saved batch visualization\n")
        
        plt.close(fig)
    
    print("="*60)
    print("✅ Generator test complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
