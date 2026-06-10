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
    "batch_size": 4,                    # ✅ Real data needs smaller batches
    "learning_rate": 1e-3,              # ✅ Single learning rate (will be adapted per model)
    "epochs": 20,
    "device": "cuda",
    "num_workers": 0,
    "early_stopping_patience": 10,
    "gradient_clip": 1.0,
    "warmup_epochs": 2,
}

# Data Configuration - REAL KAGGLE BRAIN MRI DATA
DATA_CONFIG = {
    "data_dir": "./data/brain_mri/Training",
    "tumor_types": ["glioma", "meningioma", "pituitary"],
    "image_size": 256,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "normalize_method": "zscore",
    "augmentation_enabled": True,
    "max_images": None,
}

# Graph Construction Config
GRAPH_CONFIG = {
    "method": "region_based",
    "min_component_size": 50,
    "max_component_size": 50000,
    "adjacency_threshold": 0.5,
    "connectivity": 8,
}

# Diffusion Schedule Config
DIFFUSION_CONFIG = {
    "schedule_type": "linear",
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
    "segmentation_model": "unet",
    "metric_save_dir": "./results/metrics",
}

# Logging Config
LOGGING_CONFIG = {
    "log_interval": 10,
    "save_checkpoint_interval": 5,
    "save_visualizations": True,
    "visualization_interval": 2,
}

# Seed for reproducibility
RANDOM_SEED = 42

print("✅ Config loaded: Using REAL Brain MRI data from Kaggle")
