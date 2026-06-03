import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import nibabel as nib
from pathlib import Path

class MedicalImageDataset(Dataset):
    """
    Load medical images from BraTS or ACDC dataset
    """
    def __init__(self, data_dir, modality="T1", transform=None, max_slices=5000):
        """
        Args:
            data_dir: Path to dataset folder
            modality: Type of MRI (T1, T2, FLAIR)
            transform: Image transforms
            max_slices: Maximum slices to load
        """
        self.data_dir = Path(data_dir)
        self.modality = modality
        self.transform = transform or transforms.ToTensor()
        self.max_slices = max_slices
        
        # Collect all image files
        self.images = []
        self.masks = []
        
        print(f"Loading images from {data_dir}...")
        self._load_data()
        
        print(f"Loaded {len(self.images)} images")
    
    def _load_data(self):
        """Load images and corresponding masks"""
        
        # Look for .nii.gz files (standard medical imaging format)
        for img_file in self.data_dir.rglob("*T1.nii.gz"):
            if len(self.images) >= self.max_slices:
                break
            
            try:
                # Load NIfTI image
                img_path = str(img_file)
                mask_path = str(img_file).replace("T1.nii.gz", "seg.nii.gz")
                
                if not os.path.exists(mask_path):
                    continue
                
                img_nifti = nib.load(img_path)
                mask_nifti = nib.load(mask_path)
                
                img_data = img_nifti.get_fdata()  # 3D volume
                mask_data = mask_nifti.get_fdata()
                
                # Extract 2D slices from 3D volume
                for slice_idx in range(img_data.shape[2]):
                    img_slice = img_data[:, :, slice_idx]
                    mask_slice = mask_data[:, :, slice_idx]
                    
                    # Skip empty slices
                    if mask_slice.sum() > 50:
                        self.images.append(img_slice)
                        self.masks.append(mask_slice)
                        
                        if len(self.images) >= self.max_slices:
                            break
                            
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """Get image and mask"""
        img = self.images[idx].astype(np.float32)
        mask = self.masks[idx].astype(np.float32)
        
        # Normalize image
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        
        # Resize to 256x256
        img = torch.from_numpy(img).unsqueeze(0)  # Add channel dim
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        # Simple resize
        from torch.nn.functional import interpolate
        img = interpolate(img.unsqueeze(0), size=(256, 256), mode='bilinear').squeeze(0)
        mask = interpolate(mask.unsqueeze(0), size=(256, 256), mode='nearest').squeeze(0)
        
        return {
            'image': img,
            'mask': mask,
            'index': idx
        }


def get_dataloader(data_dir, batch_size=16, num_workers=4):
    """Create DataLoader"""
    dataset = MedicalImageDataset(data_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return loader, dataset


if __name__ == "__main__":
    # Test
    loader, dataset = get_dataloader("./data/BraTS", batch_size=4)
    batch = next(iter(loader))
    print(f"Image shape: {batch['image'].shape}")
    print(f"Mask shape: {batch['mask'].shape}")