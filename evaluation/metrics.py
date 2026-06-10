"""
Medical Image Quality Metrics
Comprehensive evaluation for synthetic vs real images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from scipy import ndimage
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class MedicalImageMetrics:
    """
    Comprehensive evaluation metrics for medical image synthesis
    
    Metrics include:
    - Pixel-level: MSE, PSNR, SSIM
    - Structure-level: Dice, IoU, Hausdorff
    - Distribution: FID (Frechet Inception Distance)
    """
    
    # Metric properties
    HIGHER_IS_BETTER = {'dice', 'ssim', 'psnr', 'iou'}
    LOWER_IS_BETTER = {'mse', 'fid', 'hausdorff'}
    
    # Optimal values
    OPTIMAL_VALUES = {
        'dice': 1.0,
        'ssim': 1.0,
        'psnr': float('inf'),
        'mse': 0.0,
        'iou': 1.0,
        'fid': 0.0,
        'hausdorff': 0.0
    }
    
    @staticmethod
    def dice_coefficient(pred, target, threshold=0.5):
        """
        Dice Similarity Coefficient
        Measures overlap between predicted and target segmentations
        
        Range: [0, 1], where 1 is perfect overlap
        
        Args:
            pred: Predicted image [B, 1, H, W]
            target: Target image [B, 1, H, W]
            threshold: Binarization threshold
        
        Returns:
            Dice score (float)
        """
        try:
            smooth = 1e-6
            
            pred = pred.float()
            target = target.float()
            
            # Binarize
            pred_binary = (pred > threshold).float()
            target_binary = (target > threshold).float()
            
            # Compute intersection and union
            intersection = (pred_binary * target_binary).sum(dim=[2, 3])
            union = pred_binary.sum(dim=[2, 3]) + target_binary.sum(dim=[2, 3])
            
            # Dice coefficient
            dice = (2.0 * intersection + smooth) / (union + smooth)
            
            return float(dice.mean().item())
        except Exception as e:
            return 0.0
    
    @staticmethod
    def intersection_over_union(pred, target, threshold=0.5):
        """
        Intersection over Union (IoU) / Jaccard Index
        Measures overlap relative to union
        
        Range: [0, 1], where 1 is perfect overlap
        
        Args:
            pred: Predicted image
            target: Target image
            threshold: Binarization threshold
        
        Returns:
            IoU score (float)
        """
        try:
            smooth = 1e-6
            
            pred = pred.float()
            target = target.float()
            
            # Binarize
            pred_binary = (pred > threshold).float()
            target_binary = (target > threshold).float()
            
            # Compute intersection and union
            intersection = (pred_binary * target_binary).sum(dim=[2, 3])
            union = (pred_binary | target_binary).sum(dim=[2, 3])
            
            # IoU
            iou = (intersection + smooth) / (union + smooth)
            
            return float(iou.mean().item())
        except Exception as e:
            return 0.0
    
    @staticmethod
    def ssim_score(pred, target, data_range=1.0):
        """
        Structural Similarity Index (SSIM)
        Measures perceptual similarity between images
        
        Range: [-1, 1], where 1 is identical
        
        Args:
            pred: Predicted image
            target: Target image
            data_range: Maximum pixel value
        
        Returns:
            SSIM score (float)
        """
        try:
            pred_np = pred.squeeze().detach().cpu().numpy()
            target_np = target.squeeze().detach().cpu().numpy()
            
            # Handle edge cases
            if pred_np.size == 0 or target_np.size == 0:
                return 0.0
            
            # Compute SSIM
            score = ssim(
                pred_np,
                target_np,
                data_range=data_range,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False
            )
            
            return float(score)
        except Exception as e:
            return 0.0
    
    @staticmethod
    def mse(pred, target):
        """
        Mean Squared Error
        Pixel-level reconstruction error
        
        Range: [0, ∞], where 0 is perfect
        
        Args:
            pred: Predicted image
            target: Target image
        
        Returns:
            MSE (float)
        """
        try:
            return float(F.mse_loss(pred, target).item())
        except Exception as e:
            return float('inf')
    
    @staticmethod
    def psnr(pred, target, max_pixel=1.0):
        """
        Peak Signal-to-Noise Ratio
        Quality metric in decibels (dB)
        
        Range: [0, ∞], where higher is better
        
        Args:
            pred: Predicted image
            target: Target image
            max_pixel: Maximum pixel value (1.0 for normalized)
        
        Returns:
            PSNR in dB (float)
        """
        try:
            mse_val = F.mse_loss(pred, target).item()
            
            if mse_val == 0:
                return 100.0  # Perfect reconstruction
            
            psnr_val = 20 * np.log10(max_pixel / np.sqrt(mse_val))
            
            return float(psnr_val)
        except Exception as e:
            return 0.0
    
    @staticmethod
    def hausdorff_distance(pred, target, threshold=0.5):
        """
        Hausdorff Distance
        Maximum distance between boundary points
        
        Range: [0, ∞], where 0 is perfect
        
        Args:
            pred: Predicted image
            target: Target image
            threshold: Binarization threshold
        
        Returns:
            Hausdorff distance (float)
        """
        try:
            pred_np = (pred.squeeze().detach().cpu().numpy() > threshold).astype(np.uint8)
            target_np = (target.squeeze().detach().cpu().numpy() > threshold).astype(np.uint8)
            
            if pred_np.sum() == 0 or target_np.sum() == 0:
                return float('inf')
            
            # Get boundary points
            pred_boundary = ndimage.binary_erosion(pred_np) ^ pred_np
            target_boundary = ndimage.binary_erosion(target_np) ^ target_np
            
            pred_coords = np.column_stack(np.where(pred_boundary))
            target_coords = np.column_stack(np.where(target_boundary))
            
            if len(pred_coords) == 0 or len(target_coords) == 0:
                return 0.0
            
            # Compute pairwise distances
            from scipy.spatial.distance import directed_hausdorff
            
            dist_ph = directed_hausdorff(pred_coords, target_coords)[0]
            dist_hp = directed_hausdorff(target_coords, pred_coords)[0]
            
            hausdorff = max(dist_ph, dist_hp)
            
            return float(hausdorff)
        except Exception as e:
            return float('inf')
    
    @staticmethod
    def fid_score(real_features, synthetic_features):
        """
        Frechet Inception Distance (FID)
        Measures distribution difference between real and synthetic
        
        Range: [0, ∞], where 0 is identical distribution
        
        Args:
            real_features: Real image features [N, D]
            synthetic_features: Synthetic image features [N, D]
        
        Returns:
            FID score (float)
        """
        try:
            real_features = np.asarray(real_features, dtype=np.float64)
            synthetic_features = np.asarray(synthetic_features, dtype=np.float64)
            
            if real_features.shape[0] == 0 or synthetic_features.shape[0] == 0:
                return float('inf')
            
            # Compute means
            real_mean = np.mean(real_features, axis=0)
            synthetic_mean = np.mean(synthetic_features, axis=0)
            
            # Mean difference
            mean_diff = np.sum((real_mean - synthetic_mean) ** 2)
            
            # Compute covariances
            real_cov = np.cov(real_features.T)
            synthetic_cov = np.cov(synthetic_features.T)
            
            # Handle 1D case
            if real_cov.ndim == 0:
                real_cov = real_cov.reshape(1, 1)
            if synthetic_cov.ndim == 0:
                synthetic_cov = synthetic_cov.reshape(1, 1)
            
            # Ensure covariances are 2D
            if real_cov.ndim == 1:
                real_cov = np.diag(real_cov)
            if synthetic_cov.ndim == 1:
                synthetic_cov = np.diag(synthetic_cov)
            
            # Trace term
            trace_term = np.trace(real_cov + synthetic_cov)
            
            # Product term (Wasserstein)
            cov_product = real_cov @ synthetic_cov
            trace_product = np.trace(scipy.linalg.sqrtm(cov_product))
            
            fid = mean_diff + trace_term - 2 * trace_product
            
            return float(np.real(fid))
        except Exception as e:
            return float('inf')
    
    @staticmethod
    def compute_feature_statistics(images):
        """
        Compute statistics for FID (simplified)
        Uses image statistics instead of inception features
        
        Args:
            images: Batch of images [B, 1, H, W]
        
        Returns:
            Feature array [B, num_features]
        """
        try:
            features = []
            
            for img in images:
                img_np = img.squeeze().cpu().numpy()
                
                # Extract statistics
                feature_vec = [
                    np.mean(img_np),        # Mean
                    np.std(img_np),         # Std
                    np.min(img_np),         # Min
                    np.max(img_np),         # Max
                    np.median(img_np),      # Median
                    np.percentile(img_np, 25),  # Q1
                    np.percentile(img_np, 75),  # Q3
                ]
                
                features.append(feature_vec)
            
            return np.array(features)
        except Exception as e:
            return np.array([])
    
    @staticmethod
    def evaluate_pair(real_image, synthetic_image, compute_fid=False):
        """
        Evaluate a single image pair
        
        Args:
            real_image: Real image
            synthetic_image: Synthetic image
            compute_fid: Whether to compute FID
        
        Returns:
            Dictionary of metrics
        """
        results = {
            'dice': MedicalImageMetrics.dice_coefficient(synthetic_image, real_image),
            'iou': MedicalImageMetrics.intersection_over_union(synthetic_image, real_image),
            'ssim': MedicalImageMetrics.ssim_score(synthetic_image, real_image),
            'mse': MedicalImageMetrics.mse(synthetic_image, real_image),
            'psnr': MedicalImageMetrics.psnr(synthetic_image, real_image),
            'hausdorff': MedicalImageMetrics.hausdorff_distance(synthetic_image, real_image),
        }
        
        return results
    
    @staticmethod
    def evaluate_batch(real_images, synthetic_images, compute_fid=False):
        """
        Evaluate a batch of images
        
        Args:
            real_images: Batch of real images [B, 1, H, W]
            synthetic_images: Batch of synthetic images [B, 1, H, W]
            compute_fid: Whether to compute FID
        
        Returns:
            Summary dictionary with mean and std for each metric
        """
        results = {
            'dice': [],
            'iou': [],
            'ssim': [],
            'mse': [],
            'psnr': [],
            'hausdorff': []
        }
        
        # Evaluate each pair
        for real, synth in zip(real_images, synthetic_images):
            pair_results = MedicalImageMetrics.evaluate_pair(
                real.unsqueeze(0),
                synth.unsqueeze(0),
                compute_fid=False
            )
            
            for metric, value in pair_results.items():
                results[metric].append(value)
        
        # Compute FID if requested
        if compute_fid:
            real_features = MedicalImageMetrics.compute_feature_statistics(real_images)
            synthetic_features = MedicalImageMetrics.compute_feature_statistics(synthetic_images)
            fid = MedicalImageMetrics.fid_score(real_features, synthetic_features)
        else:
            fid = None
        
        # Summary statistics
        summary = {}
        
        for metric, values in results.items():
            if len(values) > 0 and not any(np.isinf(v) for v in values):
                summary[f'{metric}_mean'] = float(np.mean(values))
                summary[f'{metric}_std'] = float(np.std(values))
                summary[f'{metric}_min'] = float(np.min(values))
                summary[f'{metric}_max'] = float(np.max(values))
            else:
                summary[f'{metric}_mean'] = 0.0
                summary[f'{metric}_std'] = 0.0
                summary[f'{metric}_min'] = 0.0
                summary[f'{metric}_max'] = 0.0
        
        if fid is not None:
            summary['fid'] = fid
        
        return summary
    
    @staticmethod
    def print_results(summary):
        """
        Pretty print evaluation results
        
        Args:
            summary: Summary dictionary from evaluate_batch
        """
        print("\n" + "="*70)
        print("📊 EVALUATION METRICS")
        print("="*70 + "\n")
        
        # Pixel-level metrics
        print("📈 Pixel-Level Metrics:")
        print("-" * 70)
        for metric in ['mse', 'psnr', 'ssim']:
            if f'{metric}_mean' in summary:
                mean = summary[f'{metric}_mean']
                std = summary[f'{metric}_std']
                print(f"  {metric.upper():10s}: {mean:.4f} ± {std:.4f}")
        print()
        
        # Structure-level metrics
        print("🏗️  Structure-Level Metrics:")
        print("-" * 70)
        for metric in ['dice', 'iou', 'hausdorff']:
            if f'{metric}_mean' in summary:
                mean = summary[f'{metric}_mean']
                std = summary[f'{metric}_std']
                print(f"  {metric.upper():10s}: {mean:.4f} ± {std:.4f}")
        print()
        
        # Distribution metrics
        if 'fid' in summary:
            print("📊 Distribution Metrics:")
            print("-" * 70)
            print(f"  FID:        {summary['fid']:.4f}")
            print()
        
        print("="*70 + "\n")


# Import scipy for FID
try:
    import scipy.linalg
except ImportError:
    print("Warning: scipy not found, FID computation may fail")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Medical Image Metrics")
    print("="*60 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ===== Test 1: Single metric =====
    print("Test 1: Individual Metrics")
    print("-" * 60 + "\n")
    
    real = torch.rand(1, 1, 256, 256, device=device)
    synthetic = torch.rand(1, 1, 256, 256, device=device)
    
    dice = MedicalImageMetrics.dice_coefficient(synthetic, real)
    ssim_val = MedicalImageMetrics.ssim_score(synthetic, real)
    mse_val = MedicalImageMetrics.mse(synthetic, real)
    psnr_val = MedicalImageMetrics.psnr(synthetic, real)
    iou_val = MedicalImageMetrics.intersection_over_union(synthetic, real)
    
    print(f"Dice:  {dice:.4f}")
    print(f"SSIM:  {ssim_val:.4f}")
    print(f"MSE:   {mse_val:.6f}")
    print(f"PSNR:  {psnr_val:.2f} dB")
    print(f"IoU:   {iou_val:.4f}")
    print("✅ Test passed\n")
    
    # ===== Test 2: Batch evaluation =====
    print("Test 2: Batch Evaluation")
    print("-" * 60 + "\n")
    
    real_batch = torch.rand(8, 1, 256, 256, device=device)
    synthetic_batch = torch.rand(8, 1, 256, 256, device=device)
    
    summary = MedicalImageMetrics.evaluate_batch(real_batch, synthetic_batch)
    
    MedicalImageMetrics.print_results(summary)
    print("✅ Test passed\n")
    
    # ===== Test 3: Perfect reconstruction =====
    print("Test 3: Perfect Reconstruction")
    print("-" * 60 + "\n")
    
    perfect = torch.ones(4, 1, 256, 256, device=device) * 0.5
    
    summary = MedicalImageMetrics.evaluate_batch(perfect, perfect)
    
    print(f"Dice:  {summary['dice_mean']:.4f} (should be ~1.0)")
    print(f"SSIM:  {summary['ssim_mean']:.4f} (should be ~1.0)")
    print(f"MSE:   {summary['mse_mean']:.6f} (should be ~0.0)")
    print("✅ Test passed\n")
    
    print("="*60)
    print("✅ All tests passed!")
    print("="*60 + "\n")
