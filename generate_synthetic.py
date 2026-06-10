"""
Generate synthetic medical images using trained GNN-VAE
Uses REAL Brain MRI masks from Kaggle data
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os

sys.path.insert(0, str(Path(__file__).parent))

# ✅ CHANGED: Use real data loader
from data.data_loader import get_dataloader
from inference.generate_synthetic import SyntheticImageGenerator
from training.config import PATHS, DATA_CONFIG


def main():
    print("\n" + "="*70)
    print("🎨 SYNTHETIC IMAGE GENERATION")
    print("Using REAL Brain MRI Masks from Kaggle")
    print("="*70 + "\n")
    
    # ===== STEP 1: Check Models =====
    print("[1/4] Checking trained models...")
    
    GAT_MODEL = Path(PATHS['models_dir']) / 'gat_final.pth'
    VAE_MODEL = Path(PATHS['models_dir']) / 'vae_final.pth'
    OUTPUT_DIR = Path(PATHS['results_dir']) / 'synthetic_images'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not GAT_MODEL.exists() or not VAE_MODEL.exists():
        print("❌ Models not found!")
        print(f"   Expected: {GAT_MODEL}")
        print(f"   Expected: {VAE_MODEL}")
        print("\n⚠️  First run: python train_pipeline.py")
        return
    
    print(f"✅ Models found!")
    print(f"   - {GAT_MODEL.name}")
    print(f"   - {VAE_MODEL.name}\n")
    
    # ===== STEP 2: Load Models =====
    print("[2/4] Loading trained models...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"      Device: {device}\n")
    
    try:
        generator = SyntheticImageGenerator(
            str(GAT_MODEL),
            str(VAE_MODEL),
            device=device
        )
        print(f"✅ Models loaded successfully\n")
    except Exception as e:
        print(f"❌ Error loading models: {e}\n")
        return
    
    # ===== STEP 3: Load Real Data & Generate =====
    print("[3/4] Loading real brain MRI data...")
    print(f"      Data path: {DATA_CONFIG['data_dir']}")
    print(f"      Tumor types: {DATA_CONFIG['tumor_types']}\n")
    
    try:
        # ✅ Load real data
        data_loader, dataset = get_dataloader(
            data_dir=DATA_CONFIG['data_dir'],
            batch_size=1,
            tumor_types=DATA_CONFIG['tumor_types'],
            max_images=20  # Generate synthetics for 20 real images
        )
        
        print(f"✅ Loaded {len(dataset)} real images\n")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}\n")
        return
    
    # ===== STEP 4: Generate Synthetics from Real Masks =====
    print("[4/4] Generating synthetic images...\n")
    print("="*70)
    print("GENERATING FROM REAL MASKS")
    print("="*70 + "\n")
    
    num_generated = 0
    num_failed = 0
    tumor_type_counts = {}
    
    for batch_idx, batch in enumerate(data_loader):
        try:
            real_img = batch['image']              # [1, 1, 256, 256]
            mask = batch['mask']                   # [1, 1, 256, 256]
            tumor_type = batch['tumor_type'][0]   # String
            
            # Convert mask to numpy
            mask_np = mask.squeeze().cpu().numpy()
            
            print(f"Sample {batch_idx+1}/20: {tumor_type:12s} | ", end="")
            
            try:
                # Generate synthetic from real mask
                synthetic = generator.generate_from_mask(mask_np, num_samples=1)
                
                if synthetic is None or len(synthetic) == 0:
                    print("❌ Generation failed")
                    num_failed += 1
                    continue
                
                # Extract synthetic image
                synthetic_img = synthetic[0]  # [1, 256, 256]
                
                # ===== Save Synthetic Image =====
                img_np = synthetic_img.squeeze().cpu().numpy()
                img_np = np.clip(img_np, 0, 1)
                img_np = (img_np * 255).astype(np.uint8)
                
                img_pil = Image.fromarray(img_np)
                save_path = OUTPUT_DIR / f"synthetic_{tumor_type}_{batch_idx:03d}.png"
                img_pil.save(save_path)
                
                # ===== Save Comparison Visualization =====
                fig = _create_comparison_figure(
                    real_img.squeeze().cpu().numpy(),
                    mask_np,
                    synthetic_img.squeeze().cpu().numpy(),
                    tumor_type,
                    batch_idx
                )
                
                viz_path = Path(PATHS['results_dir']) / 'visualizations' / f"comparison_{tumor_type}_{batch_idx:03d}.png"
                viz_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(viz_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                num_generated += 1
                tumor_type_counts[tumor_type] = tumor_type_counts.get(tumor_type, 0) + 1
                
                print(f"✅ Generated | Saved: {save_path.name}")
                
            except Exception as e:
                print(f"⚠️  Generation error: {str(e)[:30]}")
                num_failed += 1
                continue
        
        except Exception as e:
            print(f"❌ Batch error: {str(e)[:30]}")
            num_failed += 1
            continue
    
    # ===== Generate Unconditional Random Images =====
    print("\n" + "="*70)
    print("GENERATING UNCONDITIONAL RANDOM IMAGES")
    print("="*70 + "\n")
    
    try:
        random_images = generator.generate_random(num_samples=4)
        
        if random_images is not None:
            for idx, img in enumerate(random_images):
                img_np = img.squeeze().cpu().numpy()
                img_np = np.clip(img_np, 0, 1)
                img_np = (img_np * 255).astype(np.uint8)
                
                img_pil = Image.fromarray(img_np)
                save_path = OUTPUT_DIR / f"random_unconditional_{idx:02d}.png"
                img_pil.save(save_path)
            
            # Visualize
            fig = _visualize_batch(random_images, "Unconditional Random Images")
            viz_path = Path(PATHS['results_dir']) / 'visualizations' / 'random_unconditional.png'
            viz_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✅ Generated 4 unconditional images")
            print(f"✅ Visualization saved\n")
    except Exception as e:
        print(f"⚠️  Unconditional generation error: {e}\n")
    
    # ===== Summary =====
    print("="*70)
    print("✅ GENERATION COMPLETE!")
    print("="*70 + "\n")
    
    print(f"📊 Summary:")
    print(f"   Total generated: {num_generated}")
    print(f"   Total failed: {num_failed}")
    print(f"   Success rate: {100*num_generated/(num_generated+num_failed):.1f}%\n")
    
    if tumor_type_counts:
        print(f"📋 Generated by tumor type:")
        for tumor_type, count in sorted(tumor_type_counts.items()):
            print(f"   - {tumor_type}: {count} images")
    
    print(f"\n📁 Output Locations:")
    print(f"   Synthetic images: {OUTPUT_DIR}")
    print(f"   Comparisons:      {Path(PATHS['results_dir']) / 'visualizations'}")
    print(f"   Total files:      {len(list(OUTPUT_DIR.glob('*.png')))}\n")
    
    print("🚀 Next steps:")
    print("   1. View synthetic images: ls results/synthetic_images/")
    print("   2. Evaluate quality: python evaluate_pipeline.py")
    print("\n" + "="*70 + "\n")


def _create_comparison_figure(real_img, mask, synthetic, tumor_type, idx):
    """
    Create comparison figure: Real | Mask | Synthetic
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Real image
    axes[0].imshow(real_img, cmap='gray')
    axes[0].set_title('Real MRI Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Mask
    axes[1].imshow(mask, cmap='jet', alpha=0.8)
    axes[1].set_title('Anatomical Mask', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Synthetic
    axes[2].imshow(synthetic, cmap='gray')
    axes[2].set_title('Generated Synthetic', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Main title
    fig.suptitle(f'{tumor_type.upper()} - Sample {idx+1}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    return fig


def _visualize_batch(images, title):
    """
    Visualize batch of images in grid
    """
    num_images = len(images)
    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    if num_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, img in enumerate(images):
        img_np = img.squeeze().cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        
        axes[idx].imshow(img_np, cmap='gray')
        axes[idx].set_title(f'Sample {idx+1}', fontsize=10, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide extra axes
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    main()
