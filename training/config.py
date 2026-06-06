"""
Configuration for training
"""

# Model Architecture
MODEL_CONFIG = {
    "gat": {
        "input_dim": 5,
        "hidden_dim": 64,
        "output_dim": 32,
        "num_heads": 4,
    },
    "vae": {
        "latent_dim": 32,
        "struct_code_dim": 32,
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
    "batch_size": 4,
    "learning_rate": 1e-3,
    "epochs": 20,
    "device": "cuda",  # or "cpu"
    "num_workers": 4,
    "early_stopping_patience": 5,
}

# Data
DATA_CONFIG = {
    "data_dir": "./data",
    "max_slices": 1000,
    "image_size": (256, 256),
    "train_split": 0.8,
}

# Paths
PATHS = {
    "models_dir": "./models",
    "results_dir": "./results",
    "logs_dir": "./logs",
}