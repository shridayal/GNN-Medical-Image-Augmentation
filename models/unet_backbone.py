import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Double convolution block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    Simple U-Net for medical image generation
    """
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super().__init__()
        
        features = init_features
        
        # Encoder (downsampling)
        self.enc1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(features * 4, features * 8)
        
        # Decoder (upsampling)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, 2, 2)
        self.dec3 = DoubleConv(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, 2, 2)
        self.dec2 = DoubleConv(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, 2, 2)
        self.dec1 = DoubleConv(features * 2, features)
        
        # Final output
        self.final_conv = nn.Conv2d(features, out_channels, 1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(pool3)
        
        # Decoder
        upconv3 = self.upconv3(bottleneck)
        dec3 = self.dec3(torch.cat([upconv3, enc3], 1))
        
        upconv2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat([upconv2, enc2], 1))
        
        upconv1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat([upconv1, enc1], 1))
        
        # Final output
        output = self.final_conv(dec1)
        return output


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention to apply structural guidance from GNN
    """
    def __init__(self, feature_dim, struct_code_dim=32):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Project structural code to attention weights
        self.struct_to_attention = nn.Sequential(
            nn.Linear(struct_code_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, features, struct_code):
        """
        Apply structural guidance via attention
        
        Args:
            features: Image features (B, C, H, W)
            struct_code: Structural latent code (B, struct_code_dim)
            
        Returns:
            Attended features (B, C, H, W)
        """
        batch_size, channels, height, width = features.shape
        
        # Flatten spatial dimensions
        features_flat = features.view(batch_size, channels, -1)
        
        # Generate attention weights from structural code
        attention_weights = self.struct_to_attention(struct_code)  # (B, C)
        
        # Apply attention
        attention_weights = attention_weights.unsqueeze(2)  # (B, C, 1)
        attended_features = features_flat * attention_weights
        
        # Reshape back
        attended_features = attended_features.view(batch_size, channels, height, width)
        
        return attended_features


if __name__ == "__main__":
    model = UNet(in_channels=1, out_channels=1)
    x = torch.randn(4, 1, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")