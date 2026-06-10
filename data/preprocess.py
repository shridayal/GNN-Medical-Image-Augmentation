import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
from skimage import exposure, filters
from scipy import ndimage

class BrainMRIDataset(Dataset):
    """Load brain MRI tumor images from Kaggle structure"""
    
    def __init__(self, data_dir="./data/brain_mri/Training", 
                 tumor_types=["glioma", "meningioma", "pituitary"],
                 img_size=256, max_images=None):
        
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.tumor_types = tumor_types
        
        self.image_files = []
        self.labels = []
        
        print(f"\n📂 Loading images from {self.data_dir}...")
        self._load_image_paths(max_images)
        
        if len(self.image_files) == 0:
            raise ValueError(f"❌ No images found in {self.data_dir}")
        
        print(f"✅ Loaded {len(self.image_files)} images")
        for tumor_type in self.tumor_types:
            count = sum(1 for label in self.labels if label == tumor_type)
            print(f"   - {tumor_type}: {count}")
    
    def _load_image_paths(self, max_images=None):
        """Find all JPG files in tumor class folders"""
        
        for tumor_type in self.tumor_types:
            tumor_dir = self.data_dir / tumor_type
            
            if not tumor_dir.exists():
                print(f"⚠️  {tumor_type} folder not found: {tumor_dir}")
                continue
            
            jpg_files = sorted(list(tumor_dir.glob("*.jpg")))
            
            if max_images:
                jpg_files = jpg_files[:max_images]
            
            print(f"   📷 {tumor_type}: Found {len(jpg_files)} images")
            
            for img_path in jpg_files:
                self.image_files.append(img_path)
                self.labels.append(tumor_type)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Load image and generate mask"""
        
        try:
            img_path = self.image_files[idx]
            tumor_type = self.labels[idx]
            
            # Load image
            img = Image.open(img_path).convert('L')
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Generate anatomical mask
            mask_array = self._generate_mask(img_array)
            
            # Resize to target size
            img_array = self._resize(img_array, self.img_size)
            mask_array = self._resize(mask_array, self.img_size, mode='nearest')
            
            # Normalize image (z-score)
            img_array = (img_array - img_array.mean()) / (img_array.std() + 1e-6)
            img_array = np.clip(img_array, -3, 3)
            
            # Convert to tensors [1, H, W]
            img_tensor = torch.from_numpy(img_array).unsqueeze(0).float()
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).float()
            
            return {
                'image': img_tensor,
                'mask': mask_tensor,
                'tumor_type': tumor_type,
                'path': str(img_path)
            }
        
        except Exception as e:
            print(f"❌ Error loading {self.image_files[idx]}: {e}")
            return {
                'image': torch.zeros(1, self.img_size, self.img_size),
                'mask': torch.zeros(1, self.img_size, self.img_size),
                'tumor_type': 'error',
                'path': str(self.image_files[idx])
            }
    
    def _generate_mask(self, img_array):
        """Generate anatomical mask from image using Otsu thresholding"""
        
        # Enhance contrast
        img_enhanced = exposure.equalize_adapthist(img_array)
        
        # Apply Gaussian blur
        img_blurred = filters.gaussian(img_enhanced, sigma=1.0)
        
        # Otsu thresholding
        from skimage.filters import threshold_otsu
        threshold = threshold_otsu(img_blurred)
        mask = (img_blurred > threshold).astype(np.float32)
        
        # Remove small artifacts
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
        mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))
        
        # Keep only largest connected component
        labeled, num_features = ndimage.label(mask)
        if num_features > 0:
            sizes = ndimage.sum(mask, labeled, range(num_features + 1))
            largest_label = np.argmax(sizes[1:]) + 1
            mask = (labeled == largest_label).astype(np.float32)
        
        return mask
    
    def _resize(self, img, size, mode='bilinear'):
        """Resize image to target size"""
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        
        if mode == 'nearest':
            resized = F.interpolate(
                img_tensor, size=(size, size), mode='nearest'
            )
        else:
            resized = F.interpolate(
                img_tensor, size=(size, size), mode='bilinear', align_corners=False
            )
        
        return resized.squeeze().numpy()


def get_dataloader(data_dir="./data/brain_mri/Training", 
                   batch_size=4, 
                   num_workers=0,
                   tumor_types=["glioma", "meningioma", "pituitary"],
                   max_images=None):
    """Create DataLoader"""
    
    dataset = BrainMRIDataset(
        data_dir=data_dir,
        tumor_types=tumor_types,
        img_size=256,
        max_images=max_images
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return loader, dataset


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧠 Brain MRI Data Loader Test")
    print("="*60)
    
    # Load data
    loader, dataset = get_dataloader(
        data_dir="./data/brain_mri/Training",
        batch_size=4,
        tumor_types=["glioma", "meningioma", "pituitary"],
        max_images=50
    )
    
    # Test batch
    batch = next(iter(loader))
    print(f"\n✅ Batch loaded successfully!")
    print(f"   Image shape: {batch['image'].shape}")
    print(f"   Mask shape:  {batch['mask'].shape}")
    print(f"   Tumor types: {batch['tumor_type']}")
    print(f"\n   Image range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
    print(f"   Mask range:  [{batch['mask'].min():.3f}, {batch['mask'].max():.3f}]")
    
    # Show some statistics
    print(f"\n   Image stats: mean={batch['image'].mean():.3f}, std={batch['image'].std():.3f}")
    print(f"   Mask stats:  mean={batch['mask'].mean():.3f}, std={batch['mask'].std():.3f}")
    print("\n" + "="*60)