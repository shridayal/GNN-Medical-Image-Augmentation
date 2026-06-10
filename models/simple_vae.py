"""
GNN-guided Variational Autoencoder (VAE)
Generates synthetic medical images conditioned on anatomical structure
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import MODEL_CONFIG


class VAEEncoder(nn.Module):
    """
    VAE Encoder: Image → Latent distribution
    Compresses 256×256 medical image to latent vector
    """
    def __init__(self, in_channels=1, latent_dim=32, init_features=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        
        print(f"\n   📐 VAE Encoder:")
        print(f"      Input: [{in_channels}, 256, 256]")
        print(f"      Latent dim: {latent_dim}")
        
        # Progressive downsampling: 256→128→64→32
        self.encoder = nn.Sequential(
            # Layer 1: 256 → 128
            nn.Conv2d(in_channels, init_features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_features),
            nn.ReLU(inplace=True),
            
            # Layer 2: 128 → 64
            nn.Conv2d(init_features, init_features*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_features*2),
            nn.ReLU(inplace=True),
            
            # Layer 3: 64 → 32
            nn.Conv2d(init_features*2, init_features*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_features*4),
            nn.ReLU(inplace=True),
            
            # Layer 4: 32 → 16
            nn.Conv2d(init_features*4, init_features*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_features*8),
            nn.ReLU(inplace=True),
        )
        
        # Calculate flattened size: 256 * 16 * 16
        self.flat_size = init_features * 8 * 16 * 16
        
        print(f"      Flattened: {self.flat_size}")
        
        # Latent layers
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, 1, 256, 256]
        Returns:
            mu, logvar: [batch_size, latent_dim]
        """
        # Encode
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        
        # Latent distribution
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    VAE Decoder: Latent vector → Image
    Reconstructs 256×256 medical image from latent vector
    """
    def __init__(self, latent_dim=32, init_features=32, out_channels=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        
        print(f"\n   📐 VAE Decoder:")
        print(f"      Input (latent): {latent_dim}")
        print(f"      Output: [{out_channels}, 256, 256]")
        
        # Project latent to spatial features
        self.flat_size = init_features * 8 * 16 * 16
        self.fc = nn.Linear(latent_dim, self.flat_size)
        
        # Progressive upsampling: 16→32→64→128→256
        self.decoder = nn.Sequential(
            # Layer 1: 16 → 32
            nn.ConvTranspose2d(init_features*8, init_features*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_features*4),
            nn.ReLU(inplace=True),
            
            # Layer 2: 32 → 64
            nn.ConvTranspose2d(init_features*4, init_features*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_features*2),
            nn.ReLU(inplace=True),
            
            # Layer 3: 64 → 128
            nn.ConvTranspose2d(init_features*2, init_features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_features),
            nn.ReLU(inplace=True),
            
            # Layer 4: 128 → 256
            nn.ConvTranspose2d(init_features, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
    
    def forward(self, z):
        """
        Args:
            z: [batch_size, latent_dim]
        Returns:
            recon: [batch_size, 1, 256, 256]
        """
        # Project to spatial features
        h = self.fc(z)
        h = h.view(h.size(0), -1, 16, 16)
        
        # Decode
        recon = self.decoder(h)
        
        return recon


class GNN_VAE(nn.Module):
    """
    Graph Neural Network guided Variational Autoencoder
    
    Combines:
    - VAE for image generation
    - GNN structural guidance for anatomical constraints
    - Reparameterization trick for efficient sampling
    """
    
    def __init__(self, latent_dim=32, struct_code_dim=32, init_features=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.struct_code_dim = struct_code_dim
        
        print(f"\n{'='*60}")
        print("🧠 Initializing GNN-VAE")
        print(f"{'='*60}")
        
        # VAE components
        self.encoder = VAEEncoder(
            in_channels=1,
            latent_dim=latent_dim,
            init_features=init_features
        )
        
        self.decoder = VAEDecoder(
            latent_dim=latent_dim,
            init_features=init_features,
            out_channels=1
        )
        
        # Structural guidance: Project graph features to latent space
        self.struct_guidance = nn.Sequential(
            nn.Linear(struct_code_dim, struct_code_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(struct_code_dim * 2, latent_dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        print(f"\n   📐 Structural Guidance:")
        print(f"      Input (graph): {struct_code_dim}")
        print(f"      Output (guidance): {latent_dim}")
        
        # Loss weights
        self.recon_weight = 1.0
        self.kl_weight = 0.001
        self.guidance_weight = 0.1
        
        print(f"\n✅ GNN-VAE initialized\n")
    
    def reparameterize(self, mu, logvar, temperature=1.0):
        """
        Reparameterization trick for efficient sampling
        
        Args:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log-variance of latent distribution [batch_size, latent_dim]
            temperature: Temperature for controlling stochasticity
        
        Returns:
            z: Sampled latent vector [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar * temperature)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x, struct_code=None):
        """
        Forward pass: Encode → Sample → Apply guidance → Decode
        
        Args:
            x: Input images [batch_size, 1, 256, 256]
            struct_code: Structural guidance from GNN [batch_size, struct_code_dim]
        
        Returns:
            recon: Reconstructed images [batch_size, 1, 256, 256]
            mu: Latent mean [batch_size, latent_dim]
            logvar: Latent log-variance [batch_size, latent_dim]
        """
        # ===== Encode =====
        mu, logvar = self.encoder(x)
        
        # ===== Reparameterize =====
        z = self.reparameterize(mu, logvar)
        
        # ===== Apply structural guidance =====
        if struct_code is not None:
            guidance = self.struct_guidance(struct_code)
            
            # Blend guidance with latent representation
            # Guidance acts as a "push" towards anatomically plausible region
            z = z + guidance * self.guidance_weight
        
        # ===== Decode =====
        recon = self.decoder(z)
        
        return recon, mu, logvar
    
    def generate(self, num_samples=4, struct_code=None, device='cpu', temperature=1.0):
        """
        Generate new synthetic images
        
        Args:
            num_samples: Number of images to generate
            struct_code: Optional structural guidance [num_samples, struct_code_dim]
            device: Device to generate on
            temperature: Sampling temperature (higher = more diverse)
        
        Returns:
            images: Generated images [num_samples, 1, 256, 256]
        """
        with torch.no_grad():
            # Sample from standard normal
            z = torch.randn(num_samples, self.latent_dim, device=device) * temperature
            
            # Apply structural guidance
            if struct_code is not None:
                guidance = self.struct_guidance(struct_code)
                z = z + guidance * self.guidance_weight
            
            # Decode
            images = self.decoder(z)
        
        return images
    
    def encode(self, x):
        """
        Encode image to latent distribution
        
        Args:
            x: Images [batch_size, 1, 256, 256]
        
        Returns:
            mu, logvar: Latent distribution parameters
        """
        return self.encoder(x)
    
    def decode(self, z):
        """
        Decode latent vector to image
        
        Args:
            z: Latent vectors [batch_size, latent_dim]
        
        Returns:
            recon: Reconstructed images [batch_size, 1, 256, 256]
        """
        return self.decoder(z)


def vae_loss(recon_x, x, mu, logvar, recon_weight=1.0, kl_weight=0.001):
    """
    VAE loss function
    
    Args:
        recon_x: Reconstructed images [batch_size, 1, 256, 256]
        x: Original images [batch_size, 1, 256, 256]
        mu: Latent mean [batch_size, latent_dim]
        logvar: Latent log-variance [batch_size, latent_dim]
        recon_weight: Weight for reconstruction loss
        kl_weight: Weight for KL divergence
    
    Returns:
        loss: Combined VAE loss
    """
    # ===== Reconstruction Loss =====
    # Use MSE for pixel-level reconstruction
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # ===== KL Divergence =====
    # Measure divergence from standard normal distribution
    # KL(N(mu, sigma) || N(0, I)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # ===== Total Loss =====
    total_loss = recon_weight * recon_loss + kl_weight * kl_loss
    
    return total_loss


def vae_loss_detailed(recon_x, x, mu, logvar, recon_weight=1.0, kl_weight=0.001):
    """
    VAE loss function with detailed component return
    
    Returns:
        (total_loss, recon_loss, kl_loss)
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_weight * recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing GNN-VAE")
    print("="*60 + "\n")
    
    # ===== Test Parameters =====
    batch_size = 4
    latent_dim = 32
    struct_code_dim = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}\n")
    
    # ===== Initialize Model =====
    model = GNN_VAE(
        latent_dim=latent_dim,
        struct_code_dim=struct_code_dim,
        init_features=32
    ).to(device)
    
    # ===== Test Forward Pass =====
    print("\n" + "="*60)
    print("Testing Forward Pass")
    print("="*60 + "\n")
    
    x = torch.randn(batch_size, 1, 256, 256).to(device)
    struct_code = torch.randn(batch_size, struct_code_dim).to(device)
    
    print(f"Input shape: {x.shape}")
    print(f"Struct code shape: {struct_code.shape}")
    
    recon, mu, logvar = model(x, struct_code)
    
    print(f"\nOutput shape: {recon.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    
    # ===== Test Loss =====
    print("\n" + "="*60)
    print("Testing Loss Computation")
    print("="*60 + "\n")
    
    loss = vae_loss(recon, x, mu, logvar)
    loss_detailed, recon_loss, kl_loss = vae_loss_detailed(recon, x, mu, logvar)
    
    print(f"Total Loss:       {loss.item():.6f}")
    print(f"Recon Loss:       {recon_loss.item():.6f}")
    print(f"KL Loss:          {kl_loss.item():.6f}")
    
    # ===== Test Generation =====
    print("\n" + "="*60)
    print("Testing Generation")
    print("="*60 + "\n")
    
    generated = model.generate(num_samples=4, struct_code=struct_code, device=device)
    
    print(f"Generated shape: {generated.shape}")
    print(f"Generated range: [{generated.min():.3f}, {generated.max():.3f}]")
    
    # ===== Count Parameters =====
    print("\n" + "="*60)
    print("Model Parameters")
    print("="*60 + "\n")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters:    {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n✅ All tests passed!\n")
