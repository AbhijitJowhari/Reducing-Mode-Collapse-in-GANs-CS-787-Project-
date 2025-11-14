"""
DCGAN models: Generator and Discriminator for CIFAR-10
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """DCGAN Generator with optional Stage 2 layers"""
    
    def __init__(self, latent_dim=100, nc=3, ngf=64, image_size=32):
        """
        Args:
            latent_dim: Dimension of latent vector
            nc: Number of channels (3 for RGB)
            ngf: Number of generator feature maps
            image_size: Output image size
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.nc = nc
        self.ngf = ngf
        self.image_size = image_size
        
        # Fully connected layer to project latent to initial feature map
        self.fc = nn.Linear(latent_dim, ngf * 8 * 4 * 4)
        
        # Main convolutional layers (transposes for upsampling)
        self.main = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # 32x32 -> 32x32 (1x1 conv to reduce channels to RGB)
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )
        
        # Stage 2 optional layers
        self.stage2_input_layer = None    # Linear layer before generator
        self.stage2_ff_layer = None       # Flatten + FC after generator
        self.stage2_conv_layer = None     # Conv up/downsample after generator
        
    def add_stage2_layers(self, config):
        """Add Stage 2 layers based on config"""
        
        # Input layer (before frozen generator)
        if config.add_stage2_input:
            self.stage2_input_layer = nn.Linear(
                config.latent_dim,
                config.latent_dim
            )
            print(f"  ✓ Input layer: {config.latent_dim} -> {config.latent_dim}")
        
        # Output feedforward layer
        if config.add_stage2_ff:
            output_size = self.nc * self.image_size * self.image_size
            self.stage2_ff_layer = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(output_size, config.sft_ff_dim),
                nn.ReLU(),
                nn.Linear(config.sft_ff_dim, output_size),
            )
            print(f"  ✓ Output FF layer: {output_size} -> {config.sft_ff_dim} -> {output_size}")
        
        # Output convolutional layer (up/downsample)
        if config.add_stage2_conv:
            self.stage2_conv_layer = nn.Sequential(
                # 32x32 -> 128x128 (upsample 4x)
                nn.ConvTranspose2d(self.nc, config.sft_conv_hidden, 4, stride=4, padding=0, bias=False),
                nn.BatchNorm2d(config.sft_conv_hidden),
                nn.ReLU(inplace=True),
                
                # 128x128 -> 256x256 (upsample 2x)
                nn.ConvTranspose2d(config.sft_conv_hidden, config.sft_conv_hidden, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(config.sft_conv_hidden),
                nn.ReLU(inplace=True),
                
                # 256x256 -> 128x128 (downsample 2x)
                nn.Conv2d(config.sft_conv_hidden, config.sft_conv_hidden // 2, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(config.sft_conv_hidden // 2),
                nn.ReLU(inplace=True),
                
                # 128x128 -> 32x32 (downsample 4x)
                nn.Conv2d(config.sft_conv_hidden // 2, self.nc, 4, stride=4, padding=0, bias=False),
                nn.Tanh()
            )
            print(f"  ✓ Conv layer: 32->128->256->128->32 (channels: {self.nc}->{config.sft_conv_hidden}->{self.nc})")
    
    def forward(self, z):
        """
        Generate image from latent vector
        
        Args:
            z: Latent vector, shape (batch_size, latent_dim)
        
        Returns:
            Generated image, shape (batch_size, nc, image_size, image_size)
        """
        # Apply input layer if it exists (Stage 2)
        if self.stage2_input_layer is not None:
            z = self.stage2_input_layer(z)
        
        # Project to feature map size
        x = self.fc(z)
        x = x.view(x.size(0), self.ngf * 8, 4, 4)
        
        # Generate image through main convolutional layers
        x = self.main(x)
        
        # Apply output layers if they exist (Stage 2)
        batch_size = x.size(0)
        
        # Priority: Conv layer first if it exists
        if self.stage2_conv_layer is not None:
            x = self.stage2_conv_layer(x)
        
        # Otherwise apply FF layer if it exists
        elif self.stage2_ff_layer is not None:
            x_flat = x.view(batch_size, -1)
            x_flat = self.stage2_ff_layer(x_flat)
            x = x_flat.view(batch_size, self.nc, self.image_size, self.image_size)
            x = torch.tanh(x)
        
        return x


class Discriminator(nn.Module):
    """DCGAN Discriminator"""
    
    def __init__(self, nc=3, ndf=64, image_size=32):
        """
        Args:
            nc: Number of input channels (3 for RGB)
            ndf: Number of discriminator feature maps
            image_size: Input image size
        """
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.image_size = image_size
        
        # Main convolutional layers (stride 2 for downsampling)
        self.main = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4x4 -> 2x2
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 2x2 -> 1x1 (final layer)
            nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False)
            # No sigmoid here - use BCEWithLogitsLoss
        )
    
    def forward(self, x):
        """
        Discriminate real vs fake images
        
        Args:
            x: Input image, shape (batch_size, nc, image_size, image_size)
        
        Returns:
            Discriminator logit, shape (batch_size, 1)
        """
        output = self.main(x)
        return output.view(-1, 1)  # Flatten to (batch_size, 1)
