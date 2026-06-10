"""
U-Net Architecture with GNN-guided Cross-Attention
For medical image synthesis with structural constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class DoubleConv(nn.Module):
    """
    Double convolution block with BatchNorm
    Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        if dropout > 0:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = None
    
    def forward(self, x):
        x = self.double_conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class AttentionGate(nn.Module):
    """
    Attention mechanism for skip connections
    Highlights relevant features from encoder
    """
    def __init__(self, channels_g, channels_x, channels_int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(channels_g, channels_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(channels_x, channels_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(channels_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        Args:
            g: Gating signal from decoder
            x: Skip connection from encoder
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention: Apply structural guidance from GNN to image features
    Modulates feature channels based on anatomical structure
    """
    def __init__(self, feature_channels, struct_code_dim=32, dropout=0.2):
        super().__init__()
        self.feature_channels = feature_channels
        self.struct_code_dim = struct_code_dim
        
        # Project structural code to attention weights
        self.struct_to_attention = nn.Sequential(
            nn.Linear(struct_code_dim, struct_code_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(struct_code_dim * 2, feature_channels),
            nn.ReLU(inplace=True),
            
            nn.Linear(feature_channels, feature_channels),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, features, struct_code):
        """
        Apply structural guidance via cross-attention
        
        Args:
            features: Image features [B, C, H, W]
            struct_code: Structural latent code [B, struct_code_dim]
        
        Returns:
            Attended features [B, C, H, W]
        """
        batch_size, channels, height, width = features.shape
        
        # ===== Channel Attention =====
        # Generate channel-wise attention from struct code
        channel_attention = self.struct_to_attention(struct_code)  # [B, C]
        channel_attention = channel_attention.view(batch_size, channels, 1, 1)
        
        # Apply channel attention
        features_ca = features * channel_attention
        
        # ===== Spatial Attention =====
        # Generate spatial attention map
        spatial_attention = self.spatial_attention(features_ca)  # [B, 1, H, W]
        
        # Apply spatial attention
        features_sa = features_ca * spatial_attention
        
        # ===== Residual Connection =====
        attended_features = features + features_sa
        
        return attended_features


class UNet(nn.Module):
    """
    U-Net for medical image synthesis
    
    Architecture:
    - Encoder: Progressive downsampling with skip connections
    - Bottleneck: Deep feature processing
    - Decoder: Progressive upsampling with attention gates
    - Output: Generated medical image
    
    Args:
        in_channels: Input channels (1 for grayscale)
        out_channels: Output channels (1 for grayscale)
        init_features: Base number of features (32)
        struct_code_dim: Dimension of structural guidance (32)
        use_attention: Whether to use attention gates (True)
        dropout: Dropout rate (0.0)
    """
    
    def __init__(self, in_channels=1, out_channels=1, init_features=32, 
                 struct_code_dim=32, use_attention=True, dropout=0.0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features
        self.struct_code_dim = struct_code_dim
        self.use_attention = use_attention
        
        features = init_features
        
        print(f"\n{'='*60}")
        print("🏗️  Initializing U-Net")
        print(f"{'='*60}")
        print(f"\n   Input:  [{in_channels}, 256, 256]")
        print(f"   Output: [{out_channels}, 256, 256]")
        print(f"   Init features: {init_features}")
        print(f"   Use attention: {use_attention}")
        
        # ===== ENCODER (Downsampling) =====
        # Level 1: 256 → 128
        self.enc1 = DoubleConv(in_channels, features, dropout=dropout)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Level 2: 128 → 64
        self.enc2 = DoubleConv(features, features * 2, dropout=dropout)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Level 3: 64 → 32
        self.enc3 = DoubleConv(features * 2, features * 4, dropout=dropout)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Level 4: 32 → 16
        self.enc4 = DoubleConv(features * 4, features * 8, dropout=dropout)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ===== BOTTLENECK =====
        # Level 5: 16 (deepest)
        self.bottleneck = DoubleConv(features * 8, features * 16, dropout=dropout)
        
        # ===== DECODER (Upsampling) =====
        # Level 4: 16 → 32
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, 
                                         kernel_size=2, stride=2)
        self.dec4 = DoubleConv(features * 16, features * 8, dropout=dropout)
        
        # Level 3: 32 → 64
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, 
                                         kernel_size=2, stride=2)
        self.dec3 = DoubleConv(features * 8, features * 4, dropout=dropout)
        
        # Level 2: 64 → 128
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, 
                                         kernel_size=2, stride=2)
        self.dec2 = DoubleConv(features * 4, features * 2, dropout=dropout)
        
        # Level 1: 128 → 256
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, 
                                         kernel_size=2, stride=2)
        self.dec1 = DoubleConv(features * 2, features, dropout=dropout)
        
        # ===== ATTENTION GATES =====
        if self.use_attention:
            self.att4 = AttentionGate(features * 8, features * 8, features * 4)
            self.att3 = AttentionGate(features * 4, features * 4, features * 2)
            self.att2 = AttentionGate(features * 2, features * 2, features)
            self.att1 = AttentionGate(features, features, features // 2)
        
        # ===== CROSS-ATTENTION (GNN Guidance) =====
        self.cross_attention = CrossAttentionBlock(
            feature_channels=features,
            struct_code_dim=struct_code_dim,
            dropout=dropout
        )
        
        # ===== FINAL OUTPUT =====
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
        self.output_activation = nn.Sigmoid()  # Output in [0, 1]
        
        # Initialize weights
        self._init_weights()
        
        print(f"\n✅ U-Net initialized\n")
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, struct_code=None):
        """
        Forward pass: Encode → Bottleneck → Decode
        
        Args:
            x: Input image [B, 1, 256, 256]
            struct_code: Optional structural guidance [B, struct_code_dim]
        
        Returns:
            output: Generated image [B, 1, 256, 256]
        """
        
        # ===== ENCODER =====
        # Level 1
        enc1 = self.enc1(x)           # [B, 32, 256, 256]
        pool1 = self.pool1(enc1)      # [B, 32, 128, 128]
        
        # Level 2
        enc2 = self.enc2(pool1)       # [B, 64, 128, 128]
        pool2 = self.pool2(enc2)      # [B, 64, 64, 64]
        
        # Level 3
        enc3 = self.enc3(pool2)       # [B, 128, 64, 64]
        pool3 = self.pool3(enc3)      # [B, 128, 32, 32]
        
        # Level 4
        enc4 = self.enc4(pool3)       # [B, 256, 32, 32]
        pool4 = self.pool4(enc4)      # [B, 256, 16, 16]
        
        # ===== BOTTLENECK =====
        bottleneck = self.bottleneck(pool4)  # [B, 512, 16, 16]
        
        # ===== DECODER =====
        # Level 4
        upconv4 = self.upconv4(bottleneck)  # [B, 256, 32, 32]
        
        if self.use_attention:
            enc4 = self.att4(upconv4, enc4)
        
        dec4 = self.dec4(torch.cat([upconv4, enc4], 1))  # [B, 256, 32, 32]
        
        # Level 3
        upconv3 = self.upconv3(dec4)  # [B, 128, 64, 64]
        
        if self.use_attention:
            enc3 = self.att3(upconv3, enc3)
        
        dec3 = self.dec3(torch.cat([upconv3, enc3], 1))  # [B, 128, 64, 64]
        
        # Level 2
        upconv2 = self.upconv2(dec3)  # [B, 64, 128, 128]
        
        if self.use_attention:
            enc2 = self.att2(upconv2, enc2)
        
        dec2 = self.dec2(torch.cat([upconv2, enc2], 1))  # [B, 64, 128, 128]
        
        # Level 1
        upconv1 = self.upconv1(dec2)  # [B, 32, 256, 256]
        
        if self.use_attention:
            enc1 = self.att1(upconv1, enc1)
        
        dec1 = self.dec1(torch.cat([upconv1, enc1], 1))  # [B, 32, 256, 256]
        
        # ===== CROSS-ATTENTION (GNN Guidance) =====
        if struct_code is not None:
            dec1 = self.cross_attention(dec1, struct_code)
        
        # ===== FINAL OUTPUT =====
        output = self.final_conv(dec1)      # [B, 1, 256, 256]
        output = self.output_activation(output)
        
        return output


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing U-Net with Cross-Attention")
    print("="*60 + "\n")
    
    # Parameters
    batch_size = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}\n")
    
    # ===== Initialize U-Net =====
    model = UNet(
        in_channels=1,
        out_channels=1,
        init_features=32,
        struct_code_dim=32,
        use_attention=True,
        dropout=0.1
    ).to(device)
    
    # ===== Test Forward Pass (without struct_code) =====
    print("="*60)
    print("Test 1: Forward pass (no structural guidance)")
    print("="*60 + "\n")
    
    x = torch.randn(batch_size, 1, 256, 256).to(device)
    print(f"Input shape: {x.shape}")
    
    y = model(x)
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
    print("✅ Test passed\n")
    
    # ===== Test Forward Pass (with struct_code) =====
    print("="*60)
    print("Test 2: Forward pass (with structural guidance)")
    print("="*60 + "\n")
    
    struct_code = torch.randn(batch_size, 32).to(device)
    print(f"Struct code shape: {struct_code.shape}")
    
    y_guided = model(x, struct_code)
    print(f"Output shape: {y_guided.shape}")
    print(f"Output range: [{y_guided.min():.3f}, {y_guided.max():.3f}]")
    print("✅ Test passed\n")
    
    # ===== Test with different batch size =====
    print("="*60)
    print("Test 3: Different batch size")
    print("="*60 + "\n")
    
    x_small = torch.randn(2, 1, 256, 256).to(device)
    struct_code_small = torch.randn(2, 32).to(device)
    
    y_small = model(x_small, struct_code_small)
    print(f"Input: {x_small.shape}, Output: {y_small.shape}")
    print("✅ Test passed\n")
    
    # ===== Count Parameters =====
    print("="*60)
    print("Model Statistics")
    print("="*60 + "\n")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters:    {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size:          {total_params * 4 / 1e6:.2f} MB")
    
    print("\n✅ All tests passed!\n")
