# Class-Conditional GAN with Mixture of Gaussians Prior

## Overview

This repository implements a comprehensive exploration of **mode coverage in Generative Adversarial Networks (GANs)** using a **Mixture of Gaussians (MoG) prior** and **class-conditional supervision** from a pre-trained CIFAR-10 classifier.

### Problem Statement

Standard GANs suffer from **mode collapse**—the tendency to generate samples from only a small subset of the true data distribution, resulting in poor diversity across semantic classes. This work investigates whether mode coverage can be improved through two complementary modifications:

1. **Mixture of Gaussians (MoG) Prior:** Replace the standard isotropic Gaussian latent prior with a mixture where each component corresponds to a CIFAR-10 class
2. **Class-Conditional Supervision:** Use AIRBench96 (96.5% CIFAR-10 accuracy) to guide generation toward class-specific objectives

### Key Findings

Despite systematic exploration of four architectural variants, **no tested approach produced quantitatively superior results** compared to vanilla DCGAN on CIFAR-10. However, we identify a promising future direction: **unsupervised clustering-based mode enforcement**, which leverages unsupervised feature-space clustering to enforce mode diversity.

---

## Repository Structure

```
GAN-MoG-Mode-Coverage/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE
├── .gitignore
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── generator.py              # DCGAN Generator architecture
│   │   └── discriminator.py          # DCGAN Discriminator architecture
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                # GANTrainer class with two-stage training
│   │   ├── config.py                 # GANConfig dataclass
│   │   └── losses.py                 # Loss functions (adversarial, CE, mode-based)
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── mog_sampler.py            # MoG prior sampler
│   │   └── cifar10_loader.py         # CIFAR-10 data loading utilities
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py                # IS, FID, mode coverage evaluation
│   │   ├── visualization.py          # Image grid visualization, generation plots
│   │   └── checkpointing.py          # Model checkpoint save/load utilities
│   │
│   └── classifiers/
│       ├── __init__.py
│       └── airbench96.py             # AIRBench96 classifier interface
│
├── scripts/
│   ├── train.py                      # Main training script
│   ├── evaluate.py                   # Evaluation script (IS, FID, mode coverage)
│   ├── generate_samples.py           # Generate and visualize samples
│   └── train_classifier.py           # (Optional) Fine-tune classifier
│
├── notebooks/
│   ├── experiment_analysis.ipynb     # Results analysis and comparison
│   ├── architecture_comparison.ipynb # Visual comparison of architectures
│   └── mode_coverage_analysis.ipynb  # Mode coverage statistics
│
├── docs/
│   ├── ARCHITECTURE.md               # Detailed architecture descriptions
│   ├── RESULTS.md                    # Experimental results and analysis
│   ├── FUTURE_WORK.md                # Clustering-based approach details
│   └── report.pdf                    # Full LaTeX report
│
├── configs/
│   ├── arch_a.yaml                   # Configuration for Architecture A
│   ├── arch_b.yaml                   # Configuration for Architecture B
│   ├── arch_c.yaml                   # Configuration for Architecture C
│   └── arch_d.yaml                   # Configuration for Architecture D (best)
│
└── outputs/
    ├── checkpoints/                  # Model weights
    ├── results/                      # Experimental results (metrics, logs)
    ├── samples/                      # Generated image samples
    └── logs/                         # Training logs
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration, optional but recommended)
- PyTorch 1.9+

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/GAN-MoG-Mode-Coverage.git
   cd GAN-MoG-Mode-Coverage
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download CIFAR-10 dataset:**
   ```bash
   python scripts/download_cifar10.py
   ```

---

## Quick Start

### 1. Train a GAN Model

**Train Architecture D (recommended) with default settings:**
```bash
python scripts/train.py --config configs/arch_d.yaml --output_dir outputs/arch_d
```

**Train with custom hyperparameters:**
```bash
python scripts/train.py \
    --config configs/arch_d.yaml \
    --stage1_epochs 50 \
    --stage2_epochs 30 \
    --batch_size 128 \
    --ce_loss_weight 0.5 \
    --output_dir outputs/custom_run
```

**Training options:**
```
--config              Path to config file (required)
--output_dir          Directory to save results (default: outputs/)
--checkpoint          Path to pre-trained Stage 1 checkpoint
--device              Device: cuda or cpu (default: cuda)
--seed                Random seed (default: 42)
--verbose             Print training progress (default: True)
```

### 2. Evaluate Results

**Compute Inception Score and FID:**
```bash
python scripts/evaluate.py \
    --checkpoint outputs/arch_d/checkpoints/stage2_final.pt \
    --n_samples 5000 \
    --metrics IS FID mode_coverage
```

**Output:**
```
Inception Score: 6.06 ± 0.24
FID Score: 50.4
Mode Coverage: 10/10 classes
```

### 3. Generate Samples

**Generate 64 sample images:**
```bash
python scripts/generate_samples.py \
    --checkpoint outputs/arch_d/checkpoints/stage2_final.pt \
    --n_samples 64 \
    --output samples.png
```

**Generate class-conditional samples:**
```bash
python scripts/generate_samples.py \
    --checkpoint outputs/arch_d/checkpoints/stage2_final.pt \
    --n_samples_per_class 5 \
    --class_conditional true \
    --output class_conditional_samples.png
```

---

## Architecture Details

### Four Proposed Variants

#### Architecture A: Direct Two-Stage Training
- **Stage 1:** Standard GAN training with MoG-sampled noise
- **Stage 2:** Add cross-entropy loss alongside adversarial loss
- **Issue:** Abrupt loss function change causes gradient instability

```
Loss₁ = BCE(D(G(z)), 1)
Loss₂ = BCE(D(G(z)), 1) + λ_CE · CE(C(G(z)), k)  ← Different objective!
```

#### Architecture B: Frozen Generator + Output Layer
- **Stage 1:** Same as A
- **Stage 2:** Freeze generator, add trainable Flatten → Linear(3072, 128) → ReLU → Linear(128, 3072)
- **Limitation:** Output layer receives mode-collapsed input from frozen generator

#### Architecture C: Frozen Generator + Convolutional Layer
- **Stage 2:** Convolutional upsampling-downsampling pathway: 32→128→256→128→32
- **Limitation:** Same constraint as B—convolutional layer cannot overcome mode collapse from frozen input

#### Architecture D: Input + Output Learnable Layers (Proposed)
- **Stage 2:** 
  - **Input Layer:** Linear(100, 100) provides mode selection control
  - **Output Layer:** Refines class-specific features
- **Innovation:** Input layer modulates which modes frozen generator activates
- **Best Results:** IS ≈ 6.06, FID ≈ 50.4

```
z → L_input → z' → G_frozen → x → L_output → x' → [Discriminator, Classifier]
```

---

## Mathematical Formulation

### Mixture of Gaussians Prior

Instead of standard Gaussian \(z \sim \mathcal{N}(0, I)\), we sample:

```
p(z, k) = Σ_{k=1}^{10} π_k · N(z; μ_k, σ²I)

where:
  K = 10 (CIFAR-10 classes)
  π_k = 0.1 (equal mixing ratios)
  μ_k = class-specific arbitrary means
  σ² = 1 (shared variance)
```

Sampling returns both latent code **z** and class label **k**.

### Training Losses

**Stage 1 - Adversarial Loss:**
```
L_D = E_x[BCE(D(x), 1)] + E_z[BCE(D(G(z)), 0)]
L_G = E_z[BCE(D(G(z)), 1)]
```

**Stage 2 - Combined Loss (Architecture D):**
```
L_G = E_z,k[BCE(D(x'), 1)] + λ_CE · CE(C(x'), k)

where:
  x' = L_output(G(L_input(z)))
  λ_CE = 0.5 (balanced weighting)
```

---

## Experimental Results

All four architectures yielded similar quantitative performance:

| Architecture | Config | IS | FID |
|---|---|---|---|
| DCGAN Baseline | - | 6.08 | 50.2 |
| Arch A | λ=0.5, n=20 | 6.05 | 50.6 |
| Arch B | λ=0.5, FF | 6.03 | 50.7 |
| Arch C | λ=0.5, Conv | 6.04 | 50.5 |
| **Arch D** | **λ=0.5, I+FF** | **6.06** | **50.4** |

**Key Finding:** No architecture substantially outperformed vanilla DCGAN. However, all maintained stable training dynamics.

---

## Future Work: Clustering-Based Mode Enforcement

We propose an unsupervised approach to improve mode coverage:

### Concept

1. **Extract Features:** Get Inception-v3 embeddings for real/generated images
2. **Cluster:** Apply K-Means (k=10) in feature space
3. **Compare:** Measure cluster distribution discrepancy (KL divergence)
4. **Enforce:** Penalize generator if cluster distributions don't match

### Mathematical Formulation

```
φ_real = Inception(x_real)
φ_gen = Inception(G(z))

Clustering: c_i = argmin_k ||φ_i - μ_k||²

Mode Enforcement Loss:
L_mode = KL(P_real || P_gen) + α · ||Σ_real - Σ_gen||_F

Total Loss:
L_G = L_adversarial + λ_mode · L_mode
```

### Expected Improvements

- Discover actual data modes (unsupervised)
- Achieve IS > 7.0 on CIFAR-10
- Ensure all 10 classes represented
- Generalize to other datasets

---

## Configuration Files

### Example: Architecture D Configuration

```yaml
# configs/arch_d.yaml
model:
  latent_dim: 100
  nc: 3
  ngf: 64
  ndf: 64
  image_size: 32
  num_classes: 10

stage2:
  add_input_layer: true
  add_output_ff_layer: true
  add_output_conv_layer: false
  sft_input_dim: 100
  sft_ff_dim: 128
  sft_conv_hidden: 256

training:
  stage1_epochs: 50
  stage2_epochs: 30
  batch_size: 128
  
optimization:
  lr_generator: 0.0002
  lr_discriminator: 0.0002
  beta1: 0.5
  beta2: 0.999
  
dynamics:
  n_critic: 1
  n_gen: 20
  ce_loss_weight: 0.5
```

---

## Evaluation Metrics

### Inception Score (IS)
Measures image quality and diversity:
```
IS = exp(E_x[KL(p(y|x) || p(y))])
```
Higher is better. CIFAR-10 real data: ~10.5, Good GANs: ~8-9

### Fréchet Inception Distance (FID)
Measures distance between real and generated feature distributions:
```
FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2(Σ_real·Σ_gen)^{1/2})
```
Lower is better. CIFAR-10 baseline: ~40-50

### Mode Coverage
Percentage of CIFAR-10 classes represented in generated samples:
```
Coverage = (# unique classes) / 10 × 100%
```

---

## Code Examples

### Training a Custom Architecture

```python
from src.training.trainer import GANTrainer
from src.training.config import GANConfig
from src.data.mog_sampler import CIFAR10MixtureGaussian, MixtureGaussianConfig
from airbench import train96

# Setup
config = GANConfig(
    latent_dim=100,
    stage1_epochs=50,
    stage2_epochs=30,
    add_stage2_input=True,
    add_stage2_ff=True,
    ce_loss_weight=0.5
)

mog_config = MixtureGaussianConfig(latent_dim=100, seed=42)
mog = CIFAR10MixtureGaussian(mog_config)
classifier = train96()

# Train
trainer = GANTrainer(config, mog, classifier)
trainer.train()

# Evaluate
trainer.evaluate(n_samples=5000)
```

### Generating Samples

```python
import torch
from src.utils.visualization import display_samples

trainer.generator.eval()
with torch.no_grad():
    z = torch.randn(64, 100, 1, 1, device='cuda')
    samples = trainer.generator(z)
    samples = (samples + 1) / 2  # Normalize to [0,1]

display_samples(samples, nrow=8, title='Generated CIFAR-10 Images')
```

---

## Dependencies

See `requirements.txt` for all dependencies. Key packages:

- `torch>=1.9.0` - Deep learning framework
- `torchvision` - Image utilities
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `opencv-python` - Image processing
- `pillow` - Image library
- `matplotlib` - Visualization
- `tqdm` - Progress bars
- `pyyaml` - Configuration parsing

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{jowhari2025mog_gan,
  title={Exploring Mode Coverage in GANs Using Mixture of Gaussians Prior and Class-Conditional Loss},
  author={Jowhari, Abhijit Singh},
  year={2025},
  institution={IIT Kanpur}
}
```

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Contact & Support

**Author:** Abhijit Singh Jowhari  
**Roll Number:** 220031  
**Institute:** IIT Kanpur  
**Email:** abhijitsj22@iitk.ac.in

For issues, questions, or suggestions, please open an GitHub issue or contact the authors.

---

## Acknowledgments

- DCGAN architecture (Radford et al., 2016)
- AIRBench96 classifier
- CIFAR-10 dataset
- PyTorch and community contributions

---

## References

1. Radford, A., Metz, L., & Chintala, S. (2016). "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
2. Salimans, T., et al. (2016). "Improved Techniques for Training GANs"
3. Lucic, M., et al. (2017). "Are GANs Created Equal? A Large-Scale Empirical Study"
4. Heusel, M., et al. (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"