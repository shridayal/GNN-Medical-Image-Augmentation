import torch
import torch.nn as nn
from torch.nn import functional as F

class VAEEncoder(nn.Module):
    """Encoder for VAE"""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Flatten: 128 * 32 * 32
        self.fc_mu = nn.Linear(128 * 32 * 32, latent_dim)
        self.fc_logvar = nn.Linear(128 * 32 * 32, latent_dim)
    
    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    """Decoder for VAE"""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.fc = nn.Linear(latent_dim, 128 * 32 * 32)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 128, 32, 32)
        return self.decoder(h)


class GNN_VAE(nn.Module):
    """
    VAE guided by structural prior from GNN
    """
    def __init__(self, latent_dim=32, struct_code_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.struct_code_dim = struct_code_dim
        
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = VAEDecoder(latent_dim)
        
        # Structural guidance network
        self.struct_guidance = nn.Sequential(
            nn.Linear(struct_code_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh()
        )
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x, struct_code=None):
        # Encode
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        
        # Apply structural guidance
        if struct_code is not None:
            guidance = self.struct_guidance(struct_code)
            z = z + guidance * 0.5  # Blend guidance with latent
        
        # Decode
        recon = self.decoder(z)
        
        return recon, mu, logvar
    
    def generate(self, num_samples=4, struct_code=None, device='cpu'):
        """Generate new images"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            
            if struct_code is not None:
                guidance = self.struct_guidance(struct_code)
                z = z + guidance * 0.5
            
            images = self.decoder(z)
        
        return images


def vae_loss(recon_x, x, mu, logvar):
    """VAE loss function"""
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss


if __name__ == "__main__":
    model = GNN_VAE(latent_dim=32, struct_code_dim=32)
    x = torch.randn(4, 1, 256, 256)
    struct_code = torch.randn(4, 32)
    
    recon, mu, logvar = model(x, struct_code)
    loss = vae_loss(recon, x, mu, logvar)
    
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Loss: {loss.item():.4f}")