"""
Evaluate the quality of generated synthetic images
Compute metrics and compare with real data (REAL Brain MRI)
"""

import torch
import numpy as np
from pathlib import Path
import json
import sys
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

#  CHANGED: Use real data loader
from data.data_loader import get_dataloader
from inference.generate_synthetic import SyntheticImageGenerator
from evaluation.metrics import MedicalImageMetrics
from training.config import PATHS, DATA_CONFIG, MODEL_CONFIG


def main():
    print("\n" + "="*70)
    print("EVALUATION PIPELINE - Real Brain MRI Synthetic Images")
    print("="*70 + "\n")
    
    # ===== STEP 1: Check Models Exist =====
    print("[1/4] Checking trained models...")
    
    GAT_MODEL = Path(PATHS['models_dir']) / 'gat_final.pth'
    VAE_MODEL = Path(PATHS['models_dir']) / 'vae_final.pth'
    
    if not GAT_MODEL.exists() or not VAE_MODEL.exists():
        print(" Models not found!")
        print(f"   Expected: {GAT_MODEL}")
        print(f"   Expected: {VAE_MODEL}")
        print("\n⚠️  First run: python train_pipeline.py")
        return
    
    print(f" Models found!")
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
        print(f" Models loaded successfully\n")
    except Exception as e:
        print(f" Error loading models: {e}\n")
        return
    
    # ===== STEP 3: Load Real Data =====
    print("[3/4] Loading REAL Brain MRI data for evaluation...")
    print(f"      Data path: {DATA_CONFIG['data_dir']}")
    print(f"      Tumor types: {DATA_CONFIG['tumor_types']}\n")
    
    try:
        #  Load real data
        eval_loader, eval_dataset = get_dataloader(
            data_dir=DATA_CONFIG['data_dir'],
            batch_size=1,  # Evaluate one at a time
            tumor_types=DATA_CONFIG['tumor_types'],
            max_images=50  # Evaluate on 50 samples
        )
        
        print(f" Loaded {len(eval_dataset)} real images")
        print(f" Created evaluation loader: {len(eval_loader)} batches\n")
        
    except Exception as e:
        print(f" Error loading real data: {e}")
        print("\n   Please check:")
        print(f"   - Data directory: {DATA_CONFIG['data_dir']}")
        print(f"   - Tumor folders exist")
        return
    
    # ===== STEP 4: Evaluation Loop =====
    print("[4/4] Evaluating synthetic images...\n")
    print("="*70)
    print("COMPUTING METRICS (Synthetic vs Real)")
    print("="*70 + "\n")
    
    metrics_results = {
        'dice': [],
        'ssim': [],
        'mse': [],
        'psnr': [],
        'tumor_types': [],
    }
    
    num_evaluated = 0
    num_failed = 0
    
    for batch_idx, batch in enumerate(eval_loader):
        try:
            #  UPDATED: Unpack batch with real data info
            real_img = batch['image'].to(device)              # [1, 1, 256, 256]
            mask = batch['mask'].to(device)                  # [1, 1, 256, 256]
            tumor_type = batch['tumor_type'][0]              # String
            
            # Generate synthetic image from mask
            try:
                synthetic = generator.generate_from_mask(
                    mask.cpu().numpy(),
                    num_samples=1
                )
                
                if synthetic is None or len(synthetic) == 0:
                    print(f"    Batch {batch_idx+1}: Failed to generate synthetic image")
                    num_failed += 1
                    continue
                
                synthetic_tensor = torch.from_numpy(synthetic[0]).unsqueeze(0).to(device)
                
            except Exception as e:
                print(f"   Batch {batch_idx+1}: Generation error: {str(e)[:40]}")
                num_failed += 1
                continue
            
            # Compute metrics
            try:
                dice = MedicalImageMetrics.dice_coefficient(synthetic_tensor, real_img)
                ssim = MedicalImageMetrics.ssim_score(synthetic_tensor, real_img)
                mse = MedicalImageMetrics.mse(synthetic_tensor, real_img)
                psnr = MedicalImageMetrics.psnr(synthetic_tensor, real_img)
                
                metrics_results['dice'].append(float(dice))
                metrics_results['ssim'].append(float(ssim))
                metrics_results['mse'].append(float(mse))
                metrics_results['psnr'].append(float(psnr))
                metrics_results['tumor_types'].append(tumor_type)
                
                num_evaluated += 1
                
                print(f"   Sample {num_evaluated}: {tumor_type:12s} | "
                      f"Dice: {dice:.3f} | SSIM: {ssim:.3f} | PSNR: {psnr:.2f} dB")
                
            except Exception as e:
                print(f"   Batch {batch_idx+1}: Metric error: {str(e)[:40]}")
                num_failed += 1
                continue
        
        except Exception as e:
            print(f"   Batch {batch_idx+1}: Unexpected error: {str(e)[:40]}")
            num_failed += 1
            continue
    
    # ===== Results Summary =====
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70 + "\n")
    
    if num_evaluated > 0:
        # Compute statistics
        summary = {
            'num_samples_evaluated': num_evaluated,
            'num_samples_failed': num_failed,
            'dice': {
                'mean': float(np.mean(metrics_results['dice'])),
                'std': float(np.std(metrics_results['dice'])),
                'min': float(np.min(metrics_results['dice'])),
                'max': float(np.max(metrics_results['dice'])),
            },
            'ssim': {
                'mean': float(np.mean(metrics_results['ssim'])),
                'std': float(np.std(metrics_results['ssim'])),
                'min': float(np.min(metrics_results['ssim'])),
                'max': float(np.max(metrics_results['ssim'])),
            },
            'mse': {
                'mean': float(np.mean(metrics_results['mse'])),
                'std': float(np.std(metrics_results['mse'])),
                'min': float(np.min(metrics_results['mse'])),
                'max': float(np.max(metrics_results['mse'])),
            },
            'psnr': {
                'mean': float(np.mean(metrics_results['psnr'])),
                'std': float(np.std(metrics_results['psnr'])),
                'min': float(np.min(metrics_results['psnr'])),
                'max': float(np.max(metrics_results['psnr'])),
            },
        }
        
        print(f"Samples Evaluated: {summary['num_samples_evaluated']}")
        print(f"Samples Failed:    {summary['num_samples_failed']}\n")
        
        print("📈 Metric Summary:")
        print("-" * 70)
        print(f"Dice Coefficient:")
        print(f"  Mean: {summary['dice']['mean']:.4f} ± {summary['dice']['std']:.4f}")
        print(f"  Range: [{summary['dice']['min']:.4f}, {summary['dice']['max']:.4f}]")
        print()
        print(f"SSIM Score:")
        print(f"  Mean: {summary['ssim']['mean']:.4f} ± {summary['ssim']['std']:.4f}")
        print(f"  Range: [{summary['ssim']['min']:.4f}, {summary['ssim']['max']:.4f}]")
        print()
        print(f"MSE (Mean Squared Error):")
        print(f"  Mean: {summary['mse']['mean']:.6f} ± {summary['mse']['std']:.6f}")
        print(f"  Range: [{summary['mse']['min']:.6f}, {summary['mse']['max']:.6f}]")
        print()
        print(f"PSNR (Peak Signal-to-Noise Ratio):")
        print(f"  Mean: {summary['psnr']['mean']:.2f} ± {summary['psnr']['std']:.2f} dB")
        print(f"  Range: [{summary['psnr']['min']:.2f}, {summary['psnr']['max']:.2f}] dB")
        print()
        
        # Metrics by tumor type
        print(" Metrics by Tumor Type:")
        print("-" * 70)
        tumor_types_unique = set(metrics_results['tumor_types'])
        
        for tumor_type in sorted(tumor_types_unique):
            indices = [i for i, t in enumerate(metrics_results['tumor_types']) if t == tumor_type]
            
            dice_vals = [metrics_results['dice'][i] for i in indices]
            ssim_vals = [metrics_results['ssim'][i] for i in indices]
            
            print(f"{tumor_type:12s}: Dice={np.mean(dice_vals):.3f}, SSIM={np.mean(ssim_vals):.3f} (n={len(indices)})")
        
        print()
        
        # Save results
        results_path = Path(PATHS['results_dir']) / 'metrics' / 'evaluation_results.json'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f" Results saved to: {results_path}\n")
        
        # Interpretation
        print("="*70)
        print("📋 INTERPRETATION")
        print("="*70)
        print()
        print(" Good values indicate:")
        print("  - Dice > 0.7:  Good structure preservation")
        print("  - SSIM > 0.8:  High perceptual quality")
        print("  - PSNR > 25:   Good image fidelity")
        print()
        
    else:
        print("❌ No metrics computed - evaluation failed\n")
    
    # ===== Check Generated Images =====
    print("="*70)
    print("  SYNTHETIC IMAGES")
    print("="*70 + "\n")
    
    synthetic_dir = Path(PATHS['results_dir']) / 'synthetic_images'
    if synthetic_dir.exists():
        synthetic_files = list(synthetic_dir.glob('*.png'))
        print(f" Found {len(synthetic_files)} synthetic images")
        print(f"   Location: {synthetic_dir}\n")
        
        if len(synthetic_files) > 0:
            print("   First 5 synthetic images:")
            for img_file in sorted(synthetic_files)[:5]:
                print(f"     - {img_file.name}")
    else:
        print("  No synthetic images found")
        print("   Run: python inference/generate_synthetic.py\n")
    
    # ===== Final Summary =====
    print("\n" + "="*70)
    print(" EVALUATION COMPLETE!")
    print("="*70)
    print("\n Output Locations:")
    print(f"  - Metrics:    {PATHS['results_dir']}/metrics/")
    print(f"  - Images:     {PATHS['results_dir']}/synthetic_images/")
    print(f"  - Visualizations: {PATHS['results_dir']}/visualizations/")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()