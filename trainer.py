"""
GAN Trainer with two-stage training and class-conditional loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from typing import Tuple
from tqdm import tqdm

from config import GANConfig
from models import Generator, Discriminator


class GANTrainer:
    """GAN trainer with two-stage training and class-conditional loss"""

    def __init__(
        self,
        config: GANConfig,
        mog,
        classifier_net,
    ):
        """
        Initialize GAN trainer.
        
        Args:
            config: GANConfig instance with all hyperparameters
            mog: Mixture of Gaussians sampler
            classifier_net: Pre-trained AIRBench96 classifier
        """
        self.config = config
        self.mog = mog

        # Classifier is already trained, just move to device
        self.classifier = classifier_net.to(config.device).half().eval()

        # Freeze classifier parameters
        for param in self.classifier.parameters():
            param.requires_grad = False

        # Initialize models
        self.generator = Generator(
            latent_dim=config.latent_dim,
            nc=config.nc,
            ngf=config.ngf,
            image_size=config.image_size
        ).to(config.device)
        
        self.discriminator = Discriminator(
            nc=config.nc,
            ndf=config.ndf,
            image_size=config.image_size
        ).to(config.device)

        # Optimizers
        self.opt_g = optim.Adam(
            self.generator.parameters(),
            lr=config.lr_generator,
            betas=(config.beta1, config.beta2)
        )
        self.opt_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr_discriminator,
            betas=(config.beta1, config.beta2)
        )

        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

        # Training tracking
        self.history = {
            'stage1_g_loss': [],
            'stage1_d_loss': [],
            'stage2_g_loss': [],
            'stage2_d_loss': [],
            'stage2_ce_loss': [],
        }

        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        self._print_initialization_info()

    def _print_initialization_info(self):
        """Print initialization information"""
        print("="*70)
        print("GAN Trainer Initialized")
        print("="*70)
        print(f"Device: {self.config.device}")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        print(f"Classifier parameters: {sum(p.numel() for p in self.classifier.parameters()):,} (frozen)")
        print("="*70)

    # ========== Loss Functions ==========

    def discriminator_loss(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Discriminator loss: classify real as 1, fake as 0

        L_D = E[BCE(D(real), 1)] + E[BCE(D(fake), 0)]
        """
        batch_size = real_images.size(0)
        device = self.config.device

        # Real images
        real_logits = self.discriminator(real_images)
        real_labels = torch.ones(batch_size, 1, device=device)
        loss_real = self.bce_loss(real_logits, real_labels)

        # Fake images (detach to prevent gradient flow to generator)
        fake_logits = self.discriminator(fake_images.detach())
        fake_labels = torch.zeros(batch_size, 1, device=device)
        loss_fake = self.bce_loss(fake_logits, fake_labels)

        return loss_real + loss_fake

    def generator_loss_stage1(self, fake_images: torch.Tensor) -> torch.Tensor:
        """
        Stage 1 Generator loss: fool the discriminator

        L_G = E[BCE(D(fake), 1)]
        """
        batch_size = fake_images.size(0)
        device = self.config.device

        fake_logits = self.discriminator(fake_images)
        real_labels = torch.ones(batch_size, 1, device=device)

        return self.bce_loss(fake_logits, real_labels)

    def generator_loss_stage2(
        self,
        fake_images: torch.Tensor,
        class_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 2 Generator loss: fool discriminator + match class

        L_G = E[BCE(D(fake), 1)] + λ_CE · E[CE(C(fake), k)]
        """
        batch_size = fake_images.size(0)
        device = self.config.device
        
        # Adversarial loss component
        fake_logits_disc = self.discriminator(fake_images)
        real_labels = torch.ones(batch_size, 1, device=device)
        loss_adv = self.bce_loss(fake_logits_disc, real_labels)
        
        # Class-conditional loss component
        # Convert generator output [-1,1] to [0,1] for CIFAR-10 format
        fake_images_normalized = (fake_images + 1) / 2.0  # [-1,1] → [0,1]
        fake_images_normalized = torch.clamp(fake_images_normalized, 0, 1)
        
        # Apply CIFAR-10 normalization
        CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
        CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)
        fake_images_normalized = (fake_images_normalized - CIFAR_MEAN) / CIFAR_STD
        
        # Convert to half precision to match classifier weights
        fake_images_normalized = fake_images_normalized.half()
        
        # Get classifier predictions
        with torch.no_grad():
            classifier_logits = self.classifier(fake_images_normalized)
        
        # Convert back to float32 for loss computation
        classifier_logits = classifier_logits.float()
        
        # Cross-entropy loss
        loss_ce = self.ce_loss(classifier_logits, class_labels)
        
        # Combined loss
        total_loss = loss_adv + self.config.ce_loss_weight * loss_ce
        
        return total_loss, loss_ce

    # ========== Checkpointing ==========

    def load_stage1_checkpoint(self, checkpoint_path: str) -> bool:
        """Load Stage 1 checkpoint if it exists"""
        if not os.path.exists(checkpoint_path):
            return False
        
        try:
            print(f"\nLoading Stage 1 checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
            
            # Load generator and discriminator states
            self.generator.load_state_dict(checkpoint['generator_state'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state'])
            
            # Load history if available
            if 'history' in checkpoint:
                self.history['stage1_g_loss'] = checkpoint['history'].get('stage1_g_loss', [])
                self.history['stage1_d_loss'] = checkpoint['history'].get('stage1_d_loss', [])
            
            print(f"✓ Stage 1 checkpoint loaded successfully")
            print(f"  - Epoch: {checkpoint.get('epoch', 'unknown')}")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}")
            return False

    def _save_checkpoint(self, epoch: int, stage: int):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'stage': stage,
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
        }
        path = os.path.join(self.config.checkpoint_dir, f"checkpoint_stage{stage}_epoch{epoch}.pt")
        torch.save(checkpoint, path)
        print(f"  ✓ Checkpoint saved: {path}")

    def save_stage1_final(self):
        """Save final Stage 1 checkpoint"""
        checkpoint = {
            'epoch': self.config.stage1_epochs,
            'stage': 1,
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'history': {
                'stage1_g_loss': self.history['stage1_g_loss'],
                'stage1_d_loss': self.history['stage1_d_loss']
            }
        }
        path = os.path.join(self.config.checkpoint_dir, "stage1_final.pt")
        torch.save(checkpoint, path)
        print(f"\n✓ Stage 1 final checkpoint saved: {path}")
        return path

    def save_final_model(self):
        """Save final trained generator and discriminator"""
        path = os.path.join(self.config.checkpoint_dir, "generator_final.pt")
        torch.save(self.generator.state_dict(), path)
        print(f"\n✓ Final generator saved: {path}")

        path = os.path.join(self.config.checkpoint_dir, "discriminator_final.pt")
        torch.save(self.discriminator.state_dict(), path)
        print(f"✓ Final discriminator saved: {path}")

    # ========== Training ==========

    def prepare_stage2(self):
        """Freeze generator and add Stage 2 layers"""
        print("\n" + "="*70)
        print("Preparing for Stage 2:")
        print("="*70)
        
        # Freeze ALL existing generator parameters
        for param in self.generator.parameters():
            param.requires_grad = False
        
        print("✓ Froze all base generator layers")
        
        # Add Stage 2 layers based on config
        self.generator.add_stage2_layers(self.config)
        
        # Collect trainable Stage 2 parameters
        stage2_params = []
        if self.generator.stage2_input_layer is not None:
            stage2_params.extend(self.generator.stage2_input_layer.parameters())
        if self.generator.stage2_ff_layer is not None:
            stage2_params.extend(self.generator.stage2_ff_layer.parameters())
        if self.generator.stage2_conv_layer is not None:
            stage2_params.extend(self.generator.stage2_conv_layer.parameters())
        
        # Create new optimizer for only Stage 2 layers
        if stage2_params:
            self.opt_g = optim.Adam(
                stage2_params,
                lr=self.config.lr_generator,
                betas=(self.config.beta1, self.config.beta2)
            )
            
            trainable_params = sum(p.numel() for p in stage2_params)
            print(f"✓ New optimizer created ({trainable_params:,} trainable params)")
        else:
            print("⚠ No Stage 2 layers added (check config flags)")
        
        print("="*70)

    def train_stage1(self):
        """Stage 1 Training: 1 discriminator step : N generator steps"""
        print("\n" + "="*70)
        print(f"STAGE 1: Adversarial Training ({self.config.stage1_epochs} epochs)")
        print(f"Training ratio: 1 D-step : {self.config.n_gen} G-steps")
        print("="*70)
        
        self.generator.train()
        self.discriminator.train()
        
        for epoch in range(self.config.stage1_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            num_batches = 0
            
            with tqdm(range(100), desc=f"Stage 1 | Epoch {epoch+1}/{self.config.stage1_epochs}",
                      ncols=100, leave=True) as pbar:
                
                for _ in pbar:
                    # Sample from MoG
                    z_batch, _ = self.mog.sample_batch(self.config.batch_size)
                    z_batch = torch.from_numpy(z_batch).float().to(self.config.device)
                    
                    # Random real images (dummy data for now)
                    real_images = torch.randn(
                        self.config.batch_size, self.config.nc, 
                        self.config.image_size, self.config.image_size,
                        device=self.config.device
                    )
                    
                    # ====== DISCRIMINATOR STEP (1x) ======
                    fake_images = self.generator(z_batch)
                    
                    self.opt_d.zero_grad()
                    d_loss = self.discriminator_loss(real_images, fake_images)
                    d_loss.backward()
                    self.opt_d.step()
                    
                    # ====== GENERATOR STEPS (N_gen times) ======
                    g_loss_accum = 0.0
                    for _ in range(self.config.n_gen):
                        z_batch, _ = self.mog.sample_batch(self.config.batch_size)
                        z_batch = torch.from_numpy(z_batch).float().to(self.config.device)
                        
                        fake_images = self.generator(z_batch)
                        
                        self.opt_g.zero_grad()
                        g_loss = self.generator_loss_stage1(fake_images)
                        g_loss.backward()
                        self.opt_g.step()
                        
                        g_loss_accum += g_loss.item()
                    
                    g_loss_avg = g_loss_accum / self.config.n_gen
                    
                    epoch_g_loss += g_loss_avg
                    epoch_d_loss += d_loss.item()
                    num_batches += 1
                    
                    pbar.set_postfix({
                        'D_loss': f'{d_loss.item():.4f}',
                        'G_loss': f'{g_loss_avg:.4f}'
                    })
            
            # Epoch statistics
            epoch_g_loss /= num_batches
            epoch_d_loss /= num_batches
            
            self.history['stage1_g_loss'].append(epoch_g_loss)
            self.history['stage1_d_loss'].append(epoch_d_loss)
            
            print(f"Stage 1 | Epoch [{epoch+1}/{self.config.stage1_epochs}] "
                  f"=> D_loss={epoch_d_loss:.4f}, G_loss={epoch_g_loss:.4f}")
            
            if (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint(epoch+1, stage=1)

    def train_stage2(self):
        """Stage 2: Class-conditional training with frozen base generator"""
        print("\n" + "="*70)
        print(f"STAGE 2: Class-Conditional Training ({self.config.stage2_epochs} epochs)")
        print(f"Training ratio: 1 D-step : {self.config.n_gen} G-steps")
        print("="*70)
        
        self.generator.train()
        self.discriminator.train()
        
        for epoch in range(self.config.stage2_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_ce_loss = 0.0
            num_batches = 0
            
            with tqdm(range(100), desc=f"Stage 2 | Epoch {epoch+1}/{self.config.stage2_epochs}",
                      ncols=100, leave=True) as pbar:
                
                for _ in pbar:
                    # Sample from MoG with class labels
                    z_batch, k_batch = self.mog.sample_batch(self.config.batch_size)
                    z_batch = torch.from_numpy(z_batch).float().to(self.config.device)
                    k_batch = torch.from_numpy(k_batch).long().to(self.config.device)
                    
                    # Random real images (dummy data)
                    real_images = torch.randn(
                        self.config.batch_size, self.config.nc,
                        self.config.image_size, self.config.image_size,
                        device=self.config.device
                    )
                    
                    # ====== DISCRIMINATOR STEP (1x) ======
                    fake_images = self.generator(z_batch)
                    
                    self.opt_d.zero_grad()
                    d_loss = self.discriminator_loss(real_images, fake_images)
                    d_loss.backward()
                    self.opt_d.step()
                    
                    # ====== GENERATOR STEPS (N_gen times) ======
                    g_loss_accum = 0.0
                    ce_loss_accum = 0.0
                    for _ in range(self.config.n_gen):
                        z_batch, k_batch = self.mog.sample_batch(self.config.batch_size)
                        z_batch = torch.from_numpy(z_batch).float().to(self.config.device)
                        k_batch = torch.from_numpy(k_batch).long().to(self.config.device)
                        
                        fake_images = self.generator(z_batch)
                        
                        self.opt_g.zero_grad()
                        g_loss, ce_loss = self.generator_loss_stage2(fake_images, k_batch)
                        g_loss.backward()
                        self.opt_g.step()
                        
                        g_loss_accum += g_loss.item()
                        ce_loss_accum += ce_loss.item()
                    
                    g_loss_avg = g_loss_accum / self.config.n_gen
                    ce_loss_avg = ce_loss_accum / self.config.n_gen
                    
                    epoch_g_loss += g_loss_avg
                    epoch_d_loss += d_loss.item()
                    epoch_ce_loss += ce_loss_avg
                    num_batches += 1
                    
                    pbar.set_postfix({
                        'D_loss': f'{d_loss.item():.4f}',
                        'G_loss': f'{g_loss_avg:.4f}',
                        'CE_loss': f'{ce_loss_avg:.4f}'
                    })
            
            # Epoch statistics
            epoch_g_loss /= num_batches
            epoch_d_loss /= num_batches
            epoch_ce_loss /= num_batches
            
            self.history['stage2_g_loss'].append(epoch_g_loss)
            self.history['stage2_d_loss'].append(epoch_d_loss)
            self.history['stage2_ce_loss'].append(epoch_ce_loss)
            
            print(f"Stage 2 | Epoch [{epoch+1}/{self.config.stage2_epochs}] "
                  f"=> D_loss={epoch_d_loss:.4f}, G_loss={epoch_g_loss:.4f}, CE_loss={epoch_ce_loss:.4f}")
            
            if (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint(epoch+1, stage=2)

    def train(self):
        """Execute full two-stage training with checkpoint loading"""
        start_time = time.time()
        
        # Try to load Stage 1 checkpoint if path provided
        stage1_loaded = False
        if self.config.stage1_checkpoint is not None:
            stage1_loaded = self.load_stage1_checkpoint(self.config.stage1_checkpoint)
        
        # Stage 1: Train only if not loaded from checkpoint
        if not stage1_loaded:
            print("\n" + "="*70)
            print("No Stage 1 checkpoint found. Training from scratch...")
            print("="*70)
            self.train_stage1()
            
            # Save Stage 1 checkpoint after training
            stage1_path = self.save_stage1_final()
            print(f"\nStage 1 training complete. You can resume from: {stage1_path}")
        else:
            print("\n" + "="*70)
            print("Stage 1 loaded from checkpoint. Skipping Stage 1 training.")
            print("="*70)
        
        # Prepare for Stage 2
        self.prepare_stage2()
        
        # Stage 2: Class-conditional training
        self.train_stage2()
        
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("Training Completed!")
        print("="*70)
        print(f"Total time: {total_time/60:.2f} minutes")
        
        # Save final model
        self.save_final_model()

    def generate_samples(self, n_samples: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate samples for visualization.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            (images, labels): Generated images and their MoG class labels
        """
        self.generator.eval()
        
        with torch.no_grad():
            z_batch, k_batch = self.mog.sample_batch(n_samples)
            z_batch = torch.from_numpy(z_batch).float().to(self.config.device)
            
            fake_images = self.generator(z_batch)
            fake_images = (fake_images + 1) / 2  # Rescale from [-1, 1] to [0, 1]
            fake_images = fake_images.cpu().numpy()
        
        return fake_images, k_batch
