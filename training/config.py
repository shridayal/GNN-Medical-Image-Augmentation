"""
Configuration for GNN-guided Diffusion Training with REAL Brain MRI Data
"""

# Model Architecture
MODEL_CONFIG = {
    "gat": {
        "input_dim": 4,              # [y, x, area, node_type]
        "hidden_dim": 64,
        "output_dim": 64,
        "num_heads": 8,
        "num_layers": 3,
    },
    "vae": {
        "latent_dim": 64,
        "struct_code_dim": 64,
        "init_features": 32,
        "in_channels": 1,
        "out_channels": 1,
    },
    "diffusion": {
        "in_channels": 1,
        "gnn_dim": 64,
        "timesteps": 1000,
        "init_features": 32,
    },
    "unet": {
        "in_channels": 1,
        "out_channels": 1,
        "init_features": 32,
    }
}

# Training Hyperparameters
TRAINING_CONFIG = {
    "batch_size": 4,                    # Increased for better stability
    "learning_rate_gat": 1e-3,
    "learning_rate_diffusion": 1e-4,
    "learning_rate_vae": 1e-3,
    "epochs": 20,                       # Increased epochs
    "device": "cuda",                   # Auto-detects if available
    "num_workers": 0,                   # Set to 0 for stability in Codespaces
    "early_stopping_patience": 10,
    "gradient_clip": 1.0,
    "warmup_epochs": 2,
}

# Data Configuration - REAL KAGGLE BRAIN MRI DATA
DATA_CONFIG = {
    "data_dir": "./data/brain_mri/Training",  # ✅ REAL DATA PATH
    "tumor_types": ["glioma", "meningioma", "pituitary"],
    "image_size": 256,                        # 256x256 images
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "normalize_method": "zscore",             # z-score normalization
    "augmentation_enabled": True,
    "max_images": None,                       # Use all available images
}

# Graph Construction Config
GRAPH_CONFIG = {
    "method": "region_based",                 # or "skeleton_based"
    "min_component_size": 50,                 # pixels
    "max_component_size": 50000,              # pixels
    "adjacency_threshold": 0.5,               # normalized distance
    "connectivity": 8,                        # 4 or 8 connectivity
}

# Diffusion Schedule Config
DIFFUSION_CONFIG = {
    "schedule_type": "linear",                # or "cosine"
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "timesteps": 1000,
    "clip_denoised": True,
}

# Artifact Injection Config (for robustness)
ARTIFACT_CONFIG = {
    "gaussian_noise": {"enabled": True, "std": [0.01, 0.05]},
    "rician_noise": {"enabled": True, "std": [0.01, 0.05]},
    "motion_blur": {"enabled": True, "kernel_size": [3, 5, 7]},
    "intensity_shift": {"enabled": True, "range": [-0.1, 0.1]},
}

# Paths
PATHS = {
    "models_dir": "./models",
    "results_dir": "./results",
    "logs_dir": "./logs",
    "checkpoints_dir": "./checkpoints",
    "synthetic_images_dir": "./results/synthetic_images",
    "synthetic_masks_dir": "./results/synthetic_masks",
    "visualizations_dir": "./results/visualizations",
}

# Evaluation Config
EVALUATION_CONFIG = {
    "fid_batch_size": 64,
    "num_samples_for_fid": 1000,
    "segmentation_model": "unet",             # For clinical utility test
    "metric_save_dir": "./results/metrics",
}

# Logging Config
LOGGING_CONFIG = {
    "log_interval": 10,                       # Log every N batches
    "save_checkpoint_interval": 5,            # Save every N epochs
    "save_visualizations": True,
    "visualization_interval": 2,              # Visualize every N epochs
}

# Seed for reproducibility
RANDOM_SEED = 42

print("✅ Config loaded: Using REAL Brain MRI data from Kaggle")
