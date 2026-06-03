"""
Generate synthetic medical images using trained GNN-VAE
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent))

from inference.generate_synthetic import SyntheticImageGenerator
from training.config import PATHS

def main():
    print("\n" + "="*60)
    print("SYNTHETIC IMAGE GENERATION")
    print("="*60 + "\n")
    
    # Paths
    GAT_MODEL = Path(PATHS['models_dir']) / 'gat_final.pth'
    VAE_MODEL = Path(PATHS['models_dir']) / 'vae_final.pth'
    OUTPUT_DIR = Path(PATHS['results_dir']) / 'synthetic_images'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if models exist
    if not GAT_MODEL.exists() or not VAE_MODEL.exists():
        print("❌ Models not found!")
        print(f"Expected: {GAT_MODEL} and {VAE_MODEL}")
        print("\nFirst run: python train_pipeline.py")
        return
    
    # Load generator
    print("Loading trained models...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = SyntheticImageGenerator(
        str(GAT_MODEL),
        str(VAE_MODEL),
        device=device
    )
    print(f"✓ Models loaded on {device}\n")
    
    # Create example masks
    print("Generating synthetic images from anatomical masks...\n")
    
    # Example 1: Simple cardiac-like structure
    mask1 = np.zeros((256, 256))
    mask1[80:176, 80:176] = 1  # Main chamber
    mask1[120:176, 100:136] = 2  # Ventricle
    
    # Example 2: Brain-like structure
    mask2 = np.zeros((256, 256))
    mask2[40:216, 60:196] = 1  # Brain
    mask2[100:140, 100:140] = 2  # Tumor
    
    # Example 3: Multi-organ
    mask3 = np.zeros((256, 256))
    mask3[50:100, 50:100] = 1
    mask3[150:200, 150:200] = 2
    mask3[100:150, 150:200] = 3
    
    masks = [mask1, mask2, mask3]
    mask_names = ["cardiac", "brain_tumor", "multi_organ"]
    
    # Generate for each mask
    for mask, name in zip(masks, mask_names):
        print(f"Generating from {name} mask...")
        
        synthetic = generator.generate_from_mask(mask, num_samples=4)
        
        if synthetic is not None:
            # Save individual images
            for idx, img in enumerate(synthetic):
                img_np = img.squeeze().cpu().numpy()
                img_np = np.clip(img_np, 0, 1)
                img_np = (img_np * 255).astype(np.uint8)
                
                img_pil = Image.fromarray(img_np)
                save_path = OUTPUT_DIR / f"{name}_{idx}.png"
                img_pil.save(save_path)
            
            # Create visualization
            fig = generator.visualize_images(
                synthetic,
                title=f"GNN-Guided Synthetic Images ({name})"
            )
            
            viz_path = Path(PATHS['results_dir']) / 'visualizations' / f"{name}_synthetic.png"
            viz_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  ✓ Generated 4 images")
            print(f"  ✓ Saved to {OUTPUT_DIR}")
            print(f"  ✓ Visualization saved to {viz_path}\n")
    
    # Generate random images (unconditional)
    print("Generating unconditional random images...")
    random_images = generator.generate_random(num_samples=4)
    
    fig = generator.visualize_images(
        random_images,
        title="Unconditional Synthetic Images"
    )
    viz_path = Path(PATHS['results_dir']) / 'visualizations' / 'random_unconditional.png'
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Generated 4 unconditional images")
    print(f"✓ Saved visualization to {viz_path}\n")
    
    print("="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    print(f"\nSynthetic images saved to: {OUTPUT_DIR}")
    print(f"Visualizations saved to: {Path(PATHS['results_dir']) / 'visualizations'}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()