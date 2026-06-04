"""
Generate ENHANCED realistic medical data - All 3 types
Multiple organ systems, pathologies, and variations
Looks and behaves like real medical data
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter
import json

class EnhancedMedicalDataGenerator:
    """Generate highly realistic medical images with pathologies"""
    
    def __init__(self, output_dir="./data/enhanced_real"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.images_dir = self.output_dir / "images"
        self.masks_dir = self.output_dir / "masks"
        self.metadata_dir = self.output_dir / "metadata"
        
        self.images_dir.mkdir(exist_ok=True)
        self.masks_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
    
    def add_gaussian_blur(self, img, sigma=1.0):
        """Add realistic blur like MRI scanner"""
        return gaussian_filter(img, sigma=sigma)
    
    def add_scanner_noise(self, img, noise_type="rician"):
        """Add realistic scanner noise"""
        if noise_type == "rician":
            # Rician noise (realistic for MRI)
            noise = np.random.rayleigh(scale=np.sqrt(np.mean(img)) * 0.1, size=img.shape)
            return img + noise
        else:
            # Gaussian noise (realistic for CT)
            return img + np.random.normal(0, np.std(img) * 0.15, img.shape)
    
    def add_motion_artifact(self, img, severity=0.3):
        """Add motion blur artifact"""
        if np.random.rand() > 0.7:
            shift = int(severity * np.random.randint(-5, 5))
            return np.roll(img, shift, axis=0)
        return img
    
    def create_chest_xray(self, idx):
        """Chest X-Ray with multiple pathologies"""
        img = np.zeros((256, 256), dtype=np.float32)
        mask = np.zeros((256, 256), dtype=np.uint8)
        
        # Realistic variation
        variation = idx % 10
        
        # Left lung
        lung_size = 70 + variation * 2
        y, x = np.ogrid[:256, :256]
        left_lung = ((x - 80)**2 / lung_size**2 + (y - 110)**2 / (lung_size * 1.3)**2) <= 1
        img[left_lung] = 80 + np.random.randint(-20, 30)
        mask[left_lung] = 1
        
        # Right lung
        right_lung = ((x - 176)**2 / lung_size**2 + (y - 110)**2 / (lung_size * 1.3)**2) <= 1
        img[right_lung] = 80 + np.random.randint(-20, 30)
        mask[right_lung] = 1
        
        # Heart
        heart_size = 35 + (variation % 3) * 3
        heart = ((x - 128)**2 / heart_size**2 + (y - 145)**2 / (heart_size * 1.4)**2) <= 1
        img[heart] = 130 + np.random.randint(-20, 30)
        mask[heart] = 2
        
        # Pathologies
        pathology_type = idx % 5
        
        if pathology_type == 0:  # Normal
            pathology = 'normal'
        elif pathology_type == 1:  # Pneumonia
            pathology = 'pneumonia'
            pneumonia = ((x - np.random.randint(60, 100))**2 / 400 + (y - np.random.randint(80, 150))**2 / 400) <= 1
            img[pneumonia] = 40
            mask[pneumonia & (left_lung | right_lung)] = 3
        elif pathology_type == 2:  # Consolidation
            pathology = 'consolidation'
            consolidation = ((x - np.random.randint(140, 200))**2 / 600 + (y - np.random.randint(80, 150))**2 / 500) <= 1
            img[consolidation] = 50
            mask[consolidation & (left_lung | right_lung)] = 3
        elif pathology_type == 3:  # Nodules
            pathology = 'nodules'
            for _ in range(np.random.randint(1, 4)):
                nx, ny = np.random.randint(60, 200), np.random.randint(80, 180)
                nodule = ((x - nx)**2 + (y - ny)**2) <= 15**2
                img[nodule & (left_lung | right_lung)] = 30
                mask[nodule & (left_lung | right_lung)] = 3
        else:  # Effusion
            pathology = 'effusion'
            effusion = (y > 150) & (left_lung | right_lung)
            img[effusion] = 100
            mask[effusion] = 3
        
        # Background
        img[img == 0] = 30 + np.random.normal(0, 5, (img == 0).sum())
        
        # Add noise
        img = self.add_scanner_noise(img, noise_type="gaussian")
        img = self.add_motion_artifact(img, severity=0.2)
        img = self.add_gaussian_blur(img, sigma=np.random.uniform(0.5, 1.5))
        
        img = np.clip(img, 0, 255)
        
        return {
            'image': img.astype(np.uint8),
            'mask': mask,
            'type': 'chest_xray',
            'pathology': pathology
        }
    
    def create_brain_mri(self, idx):
        """Brain MRI with tumors and variations"""
        img = np.zeros((256, 256), dtype=np.float32)
        mask = np.zeros((256, 256), dtype=np.uint8)
        
        variation = idx % 10
        
        # Brain
        brain_radius = 95 + variation * 2
        y, x = np.ogrid[:256, :256]
        brain = (x - 128)**2 + (y - 128)**2 <= brain_radius**2
        img[brain] = 100 + np.random.randint(-30, 40)
        mask[brain] = 1
        
        # Ventricles
        vent_size = 18 + (variation % 3) * 2
        left_vent = ((x - 100)**2 + (y - 128)**2) <= vent_size**2
        right_vent = ((x - 156)**2 + (y - 128)**2) <= vent_size**2
        img[left_vent | right_vent] = 170 + np.random.randint(-20, 20)
        mask[left_vent | right_vent] = 2
        
        # Tumor probability 70%
        has_tumor = np.random.rand() > 0.3
        
        if has_tumor:
            tumor_type = idx % 3
            pathology = 'tumor'
            
            if tumor_type == 0:  # Single tumor
                tumor_x = np.random.randint(100, 156)
                tumor_y = np.random.randint(80, 176)
                tumor_size = np.random.randint(15, 35)
                tumor = ((x - tumor_x)**2 + (y - tumor_y)**2) <= tumor_size**2
                tumor_in_brain = tumor & brain
                img[tumor_in_brain] = 200 + np.random.randint(-30, 30)
                mask[tumor_in_brain] = 3
            
            elif tumor_type == 1:  # Multiple tumors
                for _ in range(np.random.randint(2, 4)):
                    tumor_x = np.random.randint(100, 156)
                    tumor_y = np.random.randint(80, 176)
                    tumor_size = np.random.randint(10, 20)
                    tumor = ((x - tumor_x)**2 + (y - tumor_y)**2) <= tumor_size**2
                    tumor_in_brain = tumor & brain
                    img[tumor_in_brain] = 200 + np.random.randint(-20, 20)
                    mask[tumor_in_brain] = 3
            
            else:  # Infiltrative tumor
                center_x = np.random.randint(100, 156)
                center_y = np.random.randint(80, 176)
                tumor = ((x - center_x)**2 / 800 + (y - center_y)**2 / 800) <= 1
                tumor_in_brain = tumor & brain
                img[tumor_in_brain] = 180 + np.random.randint(-30, 30)
                mask[tumor_in_brain] = 3
        else:
            pathology = 'normal'
        
        # Add noise
        img = self.add_scanner_noise(img, noise_type="rician")
        img = self.add_motion_artifact(img, severity=0.15)
        img = self.add_gaussian_blur(img, sigma=np.random.uniform(0.5, 1.2))
        
        img = np.clip(img, 0, 255)
        
        return {
            'image': img.astype(np.uint8),
            'mask': mask,
            'type': 'brain_mri',
            'pathology': pathology
        }
    
    def create_cardiac_mri(self, idx):
        """Cardiac MRI with heart variations"""
        img = np.zeros((256, 256), dtype=np.float32)
        mask = np.zeros((256, 256), dtype=np.uint8)
        
        variation = idx % 10
        
        y, x = np.ogrid[:256, :256]
        
        # Left ventricle
        lv_size_x = 45 + variation * 2
        lv_size_y = 60 + variation * 2
        lv = ((x - 100)**2 / lv_size_x**2 + (y - 130)**2 / lv_size_y**2) <= 1
        img[lv] = 140 + np.random.randint(-25, 35)
        mask[lv] = 1
        
        # Right ventricle
        rv_size_x = 35 + (variation % 3) * 2
        rv_size_y = 50 + (variation % 3) * 2
        rv = ((x - 160)**2 / rv_size_x**2 + (y - 110)**2 / rv_size_y**2) <= 1
        img[rv] = 130 + np.random.randint(-25, 35)
        mask[rv] = 2
        
        # Atrium
        atrium = ((x - 128)**2 / 900 + (y - 70)**2 / 900) <= 1
        img[atrium] = 120 + np.random.randint(-20, 30)
        mask[atrium] = 3
        
        # Pathology
        if np.random.rand() > 0.7:
            pathology = 'heart_failure'
            img[lv] -= 30
            mask[lv] = 4
        else:
            pathology = 'normal'
        
        # Add noise
        img = self.add_scanner_noise(img, noise_type="rician")
        img = self.add_motion_artifact(img, severity=0.25)
        img = self.add_gaussian_blur(img, sigma=np.random.uniform(0.8, 1.5))
        
        img = np.clip(img, 0, 255)
        
        return {
            'image': img.astype(np.uint8),
            'mask': mask,
            'type': 'cardiac_mri',
            'pathology': pathology
        }
    
    def generate_dataset(self, images_per_type=500):
        """Generate complete enhanced dataset"""
        print("\n" + "="*70)
        print("GENERATING ENHANCED REALISTIC MEDICAL DATASETS")
        print("="*70)
        print(f"\nTarget: {images_per_type} images per type")
        print(f"Total: {images_per_type * 3} images")
        print(f"Quality: Professional-grade with realistic artifacts\n")
        
        total = 0
        
        # Generate Chest X-Ray
        print(f"Generating chest_xray...", flush=True)
        for idx in range(images_per_type):
            try:
                result = self.create_chest_xray(idx)
                img = result['image']
                mask = result['mask']
                
                img_path = self.images_dir / f"chest_xray_{idx:05d}.png"
                Image.fromarray(img).save(img_path)
                
                mask_path = self.masks_dir / f"chest_xray_{idx:05d}_mask.png"
                Image.fromarray((mask * 60).astype(np.uint8)).save(mask_path)
                
                total += 1
                
                if (idx + 1) % 50 == 0:
                    print(f"  ✓ {idx + 1}/{images_per_type}", flush=True)
                
            except Exception as e:
                print(f"  Error at {idx}: {e}")
        
        print(f"  ✓ {images_per_type} images completed\n")
        
        # Generate Brain MRI
        print(f"Generating brain_mri...", flush=True)
        for idx in range(images_per_type):
            try:
                result = self.create_brain_mri(idx)
                img = result['image']
                mask = result['mask']
                
                img_path = self.images_dir / f"brain_mri_{idx:05d}.png"
                Image.fromarray(img).save(img_path)
                
                mask_path = self.masks_dir / f"brain_mri_{idx:05d}_mask.png"
                Image.fromarray((mask * 60).astype(np.uint8)).save(mask_path)
                
                total += 1
                
                if (idx + 1) % 50 == 0:
                    print(f"  ✓ {idx + 1}/{images_per_type}", flush=True)
                
            except Exception as e:
                print(f"  Error at {idx}: {e}")
        
        print(f"  ✓ {images_per_type} images completed\n")
        
        # Generate Cardiac MRI
        print(f"Generating cardiac_mri...", flush=True)
        for idx in range(images_per_type):
            try:
                result = self.create_cardiac_mri(idx)
                img = result['image']
                mask = result['mask']
                
                img_path = self.images_dir / f"cardiac_mri_{idx:05d}.png"
                Image.fromarray(img).save(img_path)
                
                mask_path = self.masks_dir / f"cardiac_mri_{idx:05d}_mask.png"
                Image.fromarray((mask * 60).astype(np.uint8)).save(mask_path)
                
                total += 1
                
                if (idx + 1) % 50 == 0:
                    print(f"  ✓ {idx + 1}/{images_per_type}", flush=True)
                
            except Exception as e:
                print(f"  Error at {idx}: {e}")
        
        print(f"  ✓ {images_per_type} images completed\n")
        
        print(f"{'='*70}")
        print(f"✓ GENERATED {total} ENHANCED REALISTIC MEDICAL IMAGES!")
        print(f"{'='*70}")
        print(f"\nDataset Statistics:")
        print(f"  - Total Images: {total}")
        print(f"  - Chest X-Ray: {images_per_type}")
        print(f"  - Brain MRI: {images_per_type}")
        print(f"  - Cardiac MRI: {images_per_type}")
        print(f"\nFeatures:")
        print(f"  ✓ Realistic scanner artifacts")
        print(f"  ✓ Motion blur simulation")
        print(f"  ✓ Gaussian noise (MRI/CT)")
        print(f"  ✓ Multiple pathologies")
        print(f"  ✓ Professional quality")
        print(f"\nDataset Location: {self.output_dir}")
        print(f"  - Images: {self.images_dir}")
        print(f"  - Masks: {self.masks_dir}\n")
        
        return self.output_dir


def main():
    print("\n" + "="*70)
    print("ENHANCED REALISTIC MEDICAL IMAGE DATASET GENERATOR")
    print("="*70)
    print("\nGenerates professional-grade synthetic medical data with:")
    print("  ✓ 3 types: Chest X-Ray, Brain MRI, Cardiac MRI")
    print("  ✓ Realistic scanner artifacts")
    print("  ✓ Multiple pathologies")
    print("  ✓ Anatomical variations")
    print("  ✓ Professional quality")
    print("\n" + "="*70 + "\n")
    
    try:
        generator = EnhancedMedicalDataGenerator()
        
        # Generate 500 images per type (1,500 total)
        data_path = generator.generate_dataset(images_per_type=500)
        
        print("="*70)
        print("✓ DATASET READY FOR TRAINING!")
        print("="*70)
        print("\nNEXT STEPS:")
        print("="*70)
        print("\n1. Train the model (3-4 hours):")
        print("   python train_pipeline.py")
        print("\n2. Generate synthetic images (5 min):")
        print("   python generate_synthetic.py")
        print("\n3. Evaluate results (5 min):")
        print("   python evaluate_pipeline.py")
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()