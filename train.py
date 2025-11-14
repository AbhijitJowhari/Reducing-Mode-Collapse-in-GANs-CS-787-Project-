#!/usr/bin/env python3
"""
Main training script for class-conditional GAN with MoG prior and airbench classifier
"""

import argparse
import torch
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import GANConfig
from trainer import GANTrainer
from MoG import CIFAR10MixtureGaussian, MixtureGaussianConfig


def create_parser():
    """Create argument parser for training"""
    parser = argparse.ArgumentParser(
        description="Train class-conditional GAN with MoG prior",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Architecture D (recommended)
  python train.py --arch d --stage1_epochs 50 --stage2_epochs 30

  # Train with custom hyperparameters
  python train.py --arch d --lr_g 0.0002 --lr_d 0.0002 --ce_loss_weight 0.5

  # Load pre-trained Stage 1 and continue to Stage 2
  python train.py --arch d --stage1_checkpoint ./outputs/checkpoints/stage1_final.pt

  # Evaluate only (no training)
  python train.py --checkpoint ./outputs/checkpoints/stage2_final.pt --evaluate_only
        """
    )
    
    # ========== Architecture Selection ==========
    parser.add_argument(
        '--arch', '--architecture',
        type=str,
        choices=['a', 'b', 'c', 'd'],
        default='d',
        help='Architecture variant to train (default: d = Input+Output layers)'
    )
    
    # ========== Model Architecture ==========
    parser.add_argument('--latent_dim', type=int, default=100, help='Latent vector dimension')
    parser.add_argument('--nc', type=int, default=3, help='Number of channels (RGB)')
    parser.add_argument('--ngf', type=int, default=64, help='Generator feature maps')
    parser.add_argument('--ndf', type=int, default=64, help='Discriminator feature maps')
    parser.add_argument('--image_size', type=int, default=32, help='Image size (CIFAR-10=32)')
    
    # ========== Stage 2 Layer Configuration ==========
    parser.add_argument('--add_stage2_input', action='store_true', default=True,
                       help='Add linear layer before generator (Architecture D)')
    parser.add_argument('--no_stage2_input', dest='add_stage2_input', action='store_false',
                       help='Do NOT add input layer')
    
    parser.add_argument('--add_stage2_ff', action='store_true', default=True,
                       help='Add feedforward layer after generator')
    parser.add_argument('--no_stage2_ff', dest='add_stage2_ff', action='store_false',
                       help='Do NOT add feedforward layer')
    
    parser.add_argument('--add_stage2_conv', action='store_true', default=False,
                       help='Add convolutional up/downsample layer')
    
    parser.add_argument('--sft_ff_dim', type=int, default=128,
                       help='Output layer hidden dimension')
    parser.add_argument('--sft_conv_hidden', type=int, default=256,
                       help='Conv layer hidden channels')
    
    # ========== Training Configuration ==========
    parser.add_argument('--stage1_epochs', type=int, default=50,
                       help='Number of Stage 1 (adversarial) training epochs')
    parser.add_argument('--stage2_epochs', type=int, default=30,
                       help='Number of Stage 2 (class-conditional) training epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=128,
                       help='Batch size for training')
    
    # ========== Optimization ==========
    parser.add_argument('--lr_g', '--lr_generator', type=float, default=0.0002,
                       dest='lr_generator', help='Generator learning rate')
    parser.add_argument('--lr_d', '--lr_discriminator', type=float, default=0.0002,
                       dest='lr_discriminator', help='Discriminator learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta1 (momentum)')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2 (RMSprop)')
    
    # ========== Training Dynamics ==========
    parser.add_argument('--n_gen', type=int, default=20,
                       help='Generator steps per discriminator step')
    parser.add_argument('--ce_loss_weight', type=float, default=0.5,
                       help='Weight for cross-entropy loss in Stage 2')
    
    # ========== Checkpointing ==========
    parser.add_argument('--output_dir', '-o', type=str, default='./outputs/experiment',
                       help='Output directory for checkpoints and results')
    parser.add_argument('--stage1_checkpoint', type=str, default=None,
                       help='Path to pre-trained Stage 1 checkpoint')
    parser.add_argument('--save_interval', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # ========== General ==========
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                       help='Verbose training output')
    parser.add_argument('--quiet', '-q', dest='verbose', action='store_false',
                       help='Suppress verbose output')
    
    # ========== Evaluation ==========
    parser.add_argument('--evaluate_only', action='store_true', default=False,
                       help='Only evaluate, do not train')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint to evaluate')
    
    return parser


def configure_architecture(args, config):
    """Configure layers based on architecture choice"""
    arch = args.arch.lower()
    
    if arch == 'a':
        # Architecture A: Direct two-stage (naive)
        config.add_stage2_input = False
        config.add_stage2_ff = False
        config.add_stage2_conv = False
        print("✓ Architecture A: Direct two-stage training (naive approach)")
        
    elif arch == 'b':
        # Architecture B: Frozen + Output FF layer
        config.add_stage2_input = False
        config.add_stage2_ff = True
        config.add_stage2_conv = False
        print("✓ Architecture B: Frozen generator + output FF layer")
        
    elif arch == 'c':
        # Architecture C: Frozen + Conv layer
        config.add_stage2_input = False
        config.add_stage2_ff = False
        config.add_stage2_conv = True
        print("✓ Architecture C: Frozen generator + convolutional layer")
        
    elif arch == 'd':
        # Architecture D: Input + Output layers (proposed)
        config.add_stage2_input = True
        config.add_stage2_ff = True
        config.add_stage2_conv = False
        print("✓ Architecture D: Input + output learnable layers (proposed)")
    
    return config


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # ========== Create config from arguments ==========
    config = GANConfig(
        # Architecture
        latent_dim=args.latent_dim,
        nc=args.nc,
        ngf=args.ngf,
        ndf=args.ndf,
        image_size=args.image_size,
        
        # Stage 2 layers
        add_stage2_input=args.add_stage2_input,
        add_stage2_ff=args.add_stage2_ff,
        add_stage2_conv=args.add_stage2_conv,
        sft_ff_dim=args.sft_ff_dim,
        sft_conv_hidden=args.sft_conv_hidden,
        
        # Training
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size,
        
        # Optimization
        lr_generator=args.lr_generator,
        lr_discriminator=args.lr_discriminator,
        beta1=args.beta1,
        beta2=args.beta2,
        
        # Dynamics
        n_gen=args.n_gen,
        ce_loss_weight=args.ce_loss_weight,
        
        # Checkpointing
        checkpoint_dir=os.path.join(args.output_dir, 'checkpoints'),
        save_interval=args.save_interval,
        stage1_checkpoint=args.stage1_checkpoint,
        
        # General
        device=args.device,
        seed=args.seed,
    )
    
    # ========== Configure architecture ==========
    config = configure_architecture(args, config)
    
    # ========== Create output directories ==========
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    # ========== Set random seeds ==========
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # ========== Print configuration ==========
    print("\n" + "="*70)
    print("CLASS-CONDITIONAL GAN WITH MIXTURE OF GAUSSIANS PRIOR")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Latent Dim: {config.latent_dim}")
    print(f"  Image Size: {config.image_size}x{config.image_size}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Device: {config.device}")
    print(f"  Seed: {config.seed}")
    print(f"\nStage Configuration:")
    print(f"  Stage 1 Epochs: {config.stage1_epochs} (adversarial)")
    print(f"  Stage 2 Epochs: {config.stage2_epochs} (class-conditional)")
    print(f"  CE Loss Weight: {config.ce_loss_weight}")
    print(f"  Generator Updates per D Step: {config.n_gen}")
    print(f"\nOutput Directory: {args.output_dir}")
    print("="*70 + "\n")
    
    # ========== Initialize components ==========
    print("Initializing components...")
    
    # MoG
    mog_config = MixtureGaussianConfig(
        latent_dim=config.latent_dim,
        seed=config.seed
    )
    mog = CIFAR10MixtureGaussian(mog_config)
    print(" MoG initialized")
    
    # Classifier
    try:
        from airbench import train96
        print("Loading AIRBench96 classifier...")
        classifier = train96()
        print(" AIRBench96 classifier loaded")
    except ImportError as e:
        print(f" AIRBench not available: {e}")
        print("Make sure airbench is installed: pip install airbench")
        sys.exit(1)
    
    # ========== Training or Evaluation ==========
    if not args.evaluate_only:
        print("\n" + "="*70)
        print("Starting training...")
        print("="*70 + "\n")
        
        # Initialize trainer
        trainer = GANTrainer(config, mog, classifier)
        
        # Train
        trainer.train()
        
        print("\n✓ Training completed successfully!")
        
    else:
        print("\nEvaluation-only mode")
        if args.checkpoint is None:
            print(" Please provide --checkpoint path for evaluation")
            sys.exit(1)
        print(f"Checkpoint: {args.checkpoint}")
        # TODO: Implement evaluation
        print("Evaluation implementation coming soon")


if __name__ == '__main__':
    main()
