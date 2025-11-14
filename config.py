"""
Configuration module for GAN training with MoG prior
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class GANConfig:
    """Configuration for class-conditional GAN training with MoG prior"""
    
    # ========== Model Architecture ==========
    latent_dim: int = 100
    nc: int = 3                     # Number of channels (RGB)
    ngf: int = 64                   # Generator feature maps
    ndf: int = 64                   # Discriminator feature maps
    image_size: int = 32            # CIFAR-10 image size
    num_classes: int = 10
    
    # ========== Stage 2 Layers ==========
    add_stage2_input: bool = True   # Add linear layer before generator
    add_stage2_ff: bool = True      # Add flatten+linear after generator
    add_stage2_conv: bool = False   # Add conv up/downsample (mutually exclusive with FF)
    
    sft_input_dim: int = 100        # Input layer: latent_dim -> latent_dim
    sft_ff_dim: int = 128           # Output layer hidden dimension
    sft_conv_hidden: int = 256      # Conv layer hidden channels
    
    # ========== Training Configuration ==========
    stage1_epochs: int = 50         # Stage 1 adversarial training
    stage2_epochs: int = 30         # Stage 2 class-conditional
    batch_size: int = 128
    
    # ========== Optimization ==========
    lr_generator: float = 0.0002
    lr_discriminator: float = 0.0002
    beta1: float = 0.5              # Adam beta1
    beta2: float = 0.999            # Adam beta2
    
    # ========== Training Dynamics ==========
    n_critic: int = 1               # Discriminator steps per generator step
    n_gen: int = 20                 # Generator steps per discriminator step
    ce_loss_weight: float = 0.5     # Weight for cross-entropy loss
    
    # ========== Checkpointing & Logging ==========
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    checkpoint_dir: str = "./outputs/checkpoints"
    save_interval: int = 5
    stage1_checkpoint: Optional[str] = None  # Path to pre-trained Stage 1
    
    # ========== MoG Configuration ==========
    mog_latent_dim: int = 100
    mog_seed: int = 42
    mog_separation_factor: float = 8.0
    mog_variance: float = 0.5
    
    def __post_init__(self):
        """Validate configuration"""
        if self.add_stage2_ff and self.add_stage2_conv:
            print("[Warning] Both FF and Conv layers enabled; Conv takes priority")
        
        if self.latent_dim != self.mog_latent_dim:
            print(f"[Warning] latent_dim ({self.latent_dim}) != mog_latent_dim ({self.mog_latent_dim})")
