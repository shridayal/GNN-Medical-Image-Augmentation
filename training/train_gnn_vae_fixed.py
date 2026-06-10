"""
Train GNN-VAE model with REAL Brain MRI data from Kaggle
Graph Neural Network + Variational Autoencoder
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from pathlib import Path
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

# ✅ FIX: Import only what's needed, avoid circular imports
from graph.gat_model import GraphAttentionNetwork
from models.simple_vae import GNN_VAE, vae_loss
from torch_geometric.data import Data, Batch
from scipy import ndimage
import numpy as np

# Try to import config, but don't fail if circular
try:
    from training.config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG
except ImportError as e:
    print(f"Warning: Could not import config: {e}")
    MODEL_CONFIG = {'gat': {'input_dim': 4, 'output_dim': 64}, 
                    'vae': {'latent_dim': 64, 'struct_code_dim': 64}}
    TRAINING_CONFIG = {'learning_rate': 1e-3}
    DATA_CONFIG = {}

