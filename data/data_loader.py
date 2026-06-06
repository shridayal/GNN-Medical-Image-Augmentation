import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torch.nn.functional as F

class MedicalImageDataset(Dataset):
    """
    Load medical images on-the-fly (streaming)
    Does NOT load all images into memory at once
    """
    
    def __init__(self, data_dir="./data/brain_mri", max_slices=5000):
        self.data_dir = Path(data_dir)
        self.max_slices = max_slices
        
        # Just index the files, don't load them yet
        self.image_files = []
        self.mask_files = []
        
        print(f"Loading image list from {self.data_dir}...")
        self._index_images()
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.data_dir}")
        
        print(f"✓ Indexed {len(self.image_files)} images (not loaded into memory)")
        print(f"✓ Images will be loaded on-the-fly during training\n")
    
    def _index_images(self):
        """Just create index of image files, don't load them"""
        
        img_folder = self.data_dir / "images"
        mask_folder = self.data_dir / "masks"
        
        if not img_folder.exists():
            print(f"✗ Image folder not found: {img_folder}")
            return
        
        # Get PNG file paths (sorted)
        png_files = sorted(list(img_folder.glob("*.png")))
        
        # Limit to max_slices
        for idx, img_path in enumerate(png_files[:self.max_slices]):
            self.image_files.append(img_path)
            
            # Find corresponding mask
            mask_filename = img_path.name.replace(".png", "_mask.png")
            mask_path = mask_folder / mask_filename
            
            if mask_path.exists():
                self.mask_files.append(mask_path)
            else:
                self.mask_files.append(None)  # No mask, will create simple one
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Load image and mask from disk (on-the-fly)"""
        
        try:
            # Load image from disk
            img_path = self.image_files[idx]
            img = Image.open(img_path)
            img = img.convert('L')  # Grayscale
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Load mask from disk
            mask_path = self.mask_files[idx]
            
            if mask_path and mask_path.exists():
                mask = Image.open(mask_path)
                mask = mask.convert('L')
                mask_array = np.array(mask, dtype=np.float32) / 255.0
            else:
                # Create simple mask
                mask_array = (img_array > 0.3).astype(np.float32)
            
            # Convert to tensors
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
            
            # Ensure 256x256
            if img_tensor.shape[-1] != 256 or img_tensor.shape[-2] != 256:
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0),
                    size=(256, 256),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                
                mask_tensor = F.interpolate(
                    mask_tensor.unsqueeze(0),
                    size=(256, 256),
                    mode='nearest'
                ).squeeze(0)
            
            return {
                'image': img_tensor,
                'mask': mask_tensor,
                'index': idx
            }
        
        except Exception as e:
            print(f"Error loading {self.image_files[idx]}: {e}")
            # Return zeros as fallback
            return {
                'image': torch.zeros(1, 256, 256),
                'mask': torch.zeros(1, 256, 256),
                'index': idx
            }


def get_dataloader(data_dir="./data/brain_mri", batch_size=8, num_workers=0):
    """Create DataLoader with memory-efficient settings"""
    
    dataset = MedicalImageDataset(data_dir, max_slices=5000)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # Keep at 0 for stability
        pin_memory=False,  # Reduce memory usage
        drop_last=True  # Drop last incomplete batch
    )
    
    return loader, dataset


if __name__ == "__main__":
    # Test
    print("\nTesting memory-efficient data loader...\n")
    loader, dataset = get_dataloader(data_dir="./data/brain_mri", batch_size=4)
    
    print(f"✓ Dataset size: {len(dataset)}")
    
    batch = next(iter(loader))
    print(f"✓ Batch image shape: {batch['image'].shape}")
    print(f"✓ Batch mask shape: {batch['mask'].shape}")
    print(f"✓ Memory-efficient loader works!\n")