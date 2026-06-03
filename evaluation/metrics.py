import torch
import numpy as np
from scipy import stats
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import dice_score
import torch.nn.functional as F

class MedicalImageMetrics:
    """Evaluation metrics for medical image quality"""
    
    @staticmethod
    def dice_coefficient(pred, target):
        """
        Dice similarity coefficient
        """
        smooth = 1e-5
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2 * intersection + smooth) / (union + smooth)
        return dice.item() if isinstance(dice, torch.Tensor) else float(dice)
    
    @staticmethod
    def ssim_score(pred, target):
        """Structural Similarity Index"""
        pred_np = pred.squeeze().cpu().numpy()
        target_np = target.squeeze().cpu().numpy()
        return ssim(pred_np, target_np, data_range=1.0)
    
    @staticmethod
    def mse(pred, target):
        """Mean Squared Error"""
        return F.mse_loss(pred, target).item()
    
    @staticmethod
    def psnr(pred, target):
        """Peak Signal-to-Noise Ratio"""
        mse_val = F.mse_loss(pred, target).item()
        if mse_val == 0:
            return float('inf')
        max_pixel = 1.0
        psnr_val = 20 * np.log10(max_pixel / np.sqrt(mse_val))
        return psnr_val
    
    @staticmethod
    def fid_score(real_features, synthetic_features):
        """
        Frechet Inception Distance (simplified)
        """
        real_mean = np.mean(real_features, axis=0)
        real_cov = np.cov(real_features.T)
        
        synthetic_mean = np.mean(synthetic_features, axis=0)
        synthetic_cov = np.cov(synthetic_features.T)
        
        mean_diff = np.sum((real_mean - synthetic_mean) ** 2)
        
        # Compute matrix sqrt
        cov_sqrt = np.linalg.inv(np.linalg.cholesky(real_cov))
        fid = mean_diff + np.trace(real_cov + synthetic_cov - 2 * cov_sqrt @ synthetic_cov @ cov_sqrt.T)
        
        return float(fid)
    
    @staticmethod
    def evaluate_batch(real_images, synthetic_images):
        """Evaluate a batch of images"""
        results = {
            'dice': [],
            'ssim': [],
            'mse': [],
            'psnr': []
        }
        
        for real, synth in zip(real_images, synthetic_images):
            results['dice'].append(MedicalImageMetrics.dice_coefficient(synth, real))
            results['ssim'].append(MedicalImageMetrics.ssim_score(synth, real))
            results['mse'].append(MedicalImageMetrics.mse(synth, real))
            results['psnr'].append(MedicalImageMetrics.psnr(synth, real))
        
        # Compute statistics
        summary = {
            'dice_mean': np.mean(results['dice']),
            'dice_std': np.std(results['dice']),
            'ssim_mean': np.mean(results['ssim']),
            'ssim_std': np.std(results['ssim']),
            'mse_mean': np.mean(results['mse']),
            'mse_std': np.std(results['mse']),
            'psnr_mean': np.mean(results['psnr']),
            'psnr_std': np.std(results['psnr'])
        }
        
        return summary


if __name__ == "__main__":
    # Test
    real = torch.randn(4, 1, 256, 256)
    synthetic = torch.randn(4, 1, 256, 256)
    
    metrics = MedicalImageMetrics.evaluate_batch(real, synthetic)
    for key, val in metrics.items():
        print(f"{key}: {val:.4f}")