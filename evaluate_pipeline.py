"""
Evaluate the quality of generated synthetic images
Compute metrics and compare with real data
"""

import torch
import numpy as np
from pathlib import Path
import json
import sys
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from data.data_loader import MedicalImageDataset
from inference.generate_synthetic import SyntheticImageGenerator
from evaluation.metrics import MedicalImageMetrics
from graph.graph_builder import GraphBuilder
from training.config import PATHS

def main():
    print("\n" + "="*70)
    print("EVALUATION PIPELINE")
    print("="*70 + "\n")
    
    # Paths
    GAT_MODEL = Path(PATHS['models_dir']) / 'gat_final.pth'
    VAE_MODEL = Path(PATHS['models_dir']) / 'vae_final.pth'
    
    # Check if models exist
    if not GAT_MODEL.exists() or not VAE_MODEL.exists():
        print("❌ Models not found!")
        print(f"Expected: {GAT_MODEL} and {VAE_MODEL}")
        print("\nFirst run: python train_pipeline.py")
        return
    
    print(f"Models found!")
    print(f"  - {GAT_MODEL.name}")
    print(f"  - {VAE_MODEL.name}\n")
    
    # Load generator
    print("Loading trained models...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        generator = SyntheticImageGenerator(
            str(GAT_MODEL),
            str(VAE_MODEL),
            device=device
        )
        print(f"✓ Models loaded on {device}\n")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return
    
    # Load real data
    print("Loading real data for evaluation...")
    try:
        dataset = MedicalImageDataset(
            data_dir="./data/brain_mri",
            max_slices=500
        )
        print(f"✓ Loaded {len(dataset)} real images\n")
    except Exception as e:
        print(f"⚠ Could not load real data: {e}")
        print("Skipping comparison metrics\n")
        dataset = None
    
    # Evaluation metrics
    metrics_results = {
        'dice': [],
        'ssim': [],
        'mse': [],
        'psnr': []
    }
    
    graph_builder = GraphBuilder()
    
    print("="*70)
    print("EVALUATING SYNTHETIC IMAGES")
    print("="*70 + "\n")
    
    if dataset:
        # Evaluate on subset
        num_eval = min(20, len(dataset))
        print(f"Evaluating on {num_eval} images...\n")
        
        for idx in range(num_eval):
            try:
                batch = dataset[idx]
                real_img = batch['image'].unsqueeze(0).to(device)
                mask = batch['mask']
                
                # Resize real image if needed
                if real_img.shape != torch.Size([1, 1, 256, 256]):
                    import torch.nn.functional as F
                    real_img = F.interpolate(real_img, size=(256, 256), mode='bilinear')
                
                # Generate synthetic
                mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
                synthetic = generator.generate_from_mask(mask_np, num_samples=1)
                
                if synthetic is None:
                    continue
                
                # Compute metrics
                dice = MedicalImageMetrics.dice_coefficient(synthetic[0], real_img[0])
                ssim = MedicalImageMetrics.ssim_score(synthetic[0], real_img[0])
                mse = MedicalImageMetrics.mse(synthetic[0], real_img[0])
                psnr = MedicalImageMetrics.psnr(synthetic[0], real_img[0])
                
                metrics_results['dice'].append(dice)
                metrics_results['ssim'].append(ssim)
                metrics_results['mse'].append(mse)
                metrics_results['psnr'].append(psnr)
                
                print(f"Image {idx+1}/{num_eval}: Dice={dice:.3f}, SSIM={ssim:.3f}, PSNR={psnr:.2f}")
                
            except Exception as e:
                print(f"Error evaluating image {idx}: {e}")
                continue
    
    # Summary
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70 + "\n")
    
    if len(metrics_results['dice']) > 0:
        summary = {
            'num_samples': len(metrics_results['dice']),
            'dice_mean': float(np.mean(metrics_results['dice'])),
            'dice_std': float(np.std(metrics_results['dice'])),
            'ssim_mean': float(np.mean(metrics_results['ssim'])),
            'ssim_std': float(np.std(metrics_results['ssim'])),
            'mse_mean': float(np.mean(metrics_results['mse'])),
            'mse_std': float(np.std(metrics_results['mse'])),
            'psnr_mean': float(np.mean(metrics_results['psnr'])),
            'psnr_std': float(np.std(metrics_results['psnr'])),
        }
        
        print(f"Samples Evaluated: {summary['num_samples']}\n")
        print(f"Dice Coefficient:  {summary['dice_mean']:.4f} ± {summary['dice_std']:.4f}")
        print(f"SSIM Score:        {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f}")
        print(f"MSE:               {summary['mse_mean']:.6f} ± {summary['mse_std']:.6f}")
        print(f"PSNR (dB):         {summary['psnr_mean']:.2f} ± {summary['psnr_std']:.2f}\n")
        
        # Save results
        results_path = Path(PATHS['results_dir']) / 'metrics' / 'evaluation_results.json'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"✓ Results saved to: {results_path}\n")
    else:
        print("No evaluation metrics computed\n")
    
    # Check synthetic images
    print("="*70)
    print("SYNTHETIC IMAGES")
    print("="*70 + "\n")
    
    synthetic_dir = Path(PATHS['results_dir']) / 'synthetic_images'
    if synthetic_dir.exists():
        synthetic_files = list(synthetic_dir.glob('*.png'))
        print(f"✓ Found {len(synthetic_files)} synthetic images")
        print(f"  Location: {synthetic_dir}\n")
    else:
        print("⚠ No synthetic images found")
        print("  Run: python generate_synthetic.py\n")
    
    # Final summary
    print("="*70)
    print("✓ EVALUATION COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. View synthetic images: ./results/synthetic_images/")
    print("  2. View metrics: ./results/metrics/evaluation_results.json")
    print("  3. View visualizations: ./results/visualizations/")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()