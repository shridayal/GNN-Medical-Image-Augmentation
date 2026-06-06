"""
Download brain data directly to ./data/brain_mri/
No temporary folders, clean organization
"""

import kagglehub
from pathlib import Path
import shutil
import os
from PIL import Image
import numpy as np

print("\n" + "="*70)
print("DOWNLOADING BRAIN DATA DIRECTLY TO PROJECT")
print("="*70 + "\n")

# Step 1: Download to Kaggle's cache (temporary)
print("Step 1: Downloading from Kaggle...")
try:
    path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    print(f"✓ Downloaded to: {path}\n")
except Exception as e:
    print(f"✗ Download failed: {e}")
    print("Make sure you have kaggle.json set up!")
    exit()

# Step 2: Create target directories
print("Step 2: Creating project directories...")
data_dir = Path("./data/brain_mri")
img_dir = data_dir / "images"
mask_dir = data_dir / "masks"

img_dir.mkdir(parents=True, exist_ok=True)
mask_dir.mkdir(parents=True, exist_ok=True)
print(f"✓ Created: {data_dir}\n")

# Step 3: Copy and organize files
print("Step 3: Copying and organizing files...\n")

count = 0
source_path = Path(path)

# Find all image files
for root, dirs, files in os.walk(source_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            src_file = Path(root) / file
            
            try:
                # Open image
                img = Image.open(src_file).convert('L')  # Convert to grayscale
                
                # Resize to 256x256
                img = img.resize((256, 256), Image.BILINEAR)
                
                # Save to our folder
                dst_path = img_dir / f"brain_{count:05d}.png"
                img.save(dst_path)
                
                # Create mask (threshold)
                img_array = np.array(img)
                mask = (img_array > 30).astype(np.uint8)
                mask_img = Image.fromarray((mask * 100).astype(np.uint8))
                
                mask_dst = mask_dir / f"brain_{count:05d}_mask.png"
                mask_img.save(mask_dst)
                
                count += 1
                
                # Progress
                if count % 50 == 0:
                    print(f"  ✓ Processed {count} images")
                
            except Exception as e:
                print(f"  ✗ Error processing {file}: {e}")
                continue

print(f"\n{'='*70}")
print(f"✓ DOWNLOAD COMPLETE!")
print(f"{'='*70}\n")

print(f"Total images processed: {count}")
print(f"Saved to: {data_dir}\n")

print(f"Folder structure:")
print(f"  ./data/brain_mri/")
print(f"  ├── images/ ({len(list(img_dir.glob('*.png')))} files)")
print(f"  └── masks/ ({len(list(mask_dir.glob('*.png')))} files)\n")

# Verify
if count > 0:
    print("="*70)
    print("✓ READY FOR TRAINING!")
    print("="*70)
    print("\nNext command:")
    print("  python train_pipeline.py\n")
else:
    print("✗ No images found!")
    print("Check the dataset structure")