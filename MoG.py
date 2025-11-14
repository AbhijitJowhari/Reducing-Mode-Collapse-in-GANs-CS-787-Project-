"""
CIFAR-10 Mixture of Gaussians Sampler
======================================

This module creates a Mixture of Gaussians (MoG) model where:
- Each CIFAR-10 class (0-9) has its own Gaussian distribution
- Gaussians are non-overlapping (well-separated in latent space)
- Mixing coefficients π_k are proportional to class frequencies
- Sampling process: Select component k ~ Categorical(π), then sample z ~ N(μ_k, Σ_k)
- Returns (z, k): latent variable and its source class

Author: Abhijit Singh Jowhari
Date: 2025-11-11
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict
from scipy.stats import multivariate_normal


@dataclass
class MixtureGaussianConfig:
    """Configuration for Mixture of Gaussians for CIFAR-10 classes"""
    latent_dim: int = 16  # Dimension of latent space
    seed: int = 42
    separation_factor: float = 8.0  # Factor to ensure non-overlapping Gaussians
    variance_per_class: float = 0.5  # Variance within each class
    data_dir: str = "./cifar10_data"


class CIFAR10MixtureGaussian:
    """
    Mixture of Gaussian distributions, one for each CIFAR-10 class.

    Key properties:
    - 10 Gaussian components (one per CIFAR-10 class)
    - Non-overlapping: min_distance / max_sigma > 4.0
    - Class-balanced mixing coefficients: π_k = 1/10 for all k
    - Isotropic Gaussians: Σ_k = σ_k^2 * I

    Mathematical formulation:
    ========================
    p(z, k) = π_k * N(z | μ_k, Σ_k)

    Sampling:
    1. k ~ Categorical(π₁, π₂, ..., π₁₀)
    2. z | k ~ N(μ_k, Σ_k)
    3. Return (z, k)
    """

    CIFAR10_CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    def __init__(self, config: MixtureGaussianConfig = None):
        """
        Initialize mixture of Gaussians for CIFAR-10.

        Args:
            config: Configuration object (uses defaults if None)
        """
        if config is None:
            config = MixtureGaussianConfig()

        self.config = config
        self.latent_dim = config.latent_dim
        self.num_classes = 10

        # Set random seed
        np.random.seed(config.seed)

        # Initialize components
        self._compute_class_proportions()
        self._initialize_gaussians()

    def _compute_class_proportions(self):
        """
        Compute π_k (mixing coefficients) proportional to class frequency.

        CIFAR-10 is perfectly balanced: 5,000 samples per class out of 50,000 total.
        Therefore: π_k = 1/10 for all k ∈ {0, 1, ..., 9}
        """
        self.pi = np.ones(self.num_classes) / self.num_classes

        print("="*60)
        print("CIFAR-10 Class Proportions (π_k):")
        print("="*60)
        for k in range(self.num_classes):
            print(f"  Class {k:2d} ({self.CIFAR10_CLASSES[k]:12s}): π_{k} = {self.pi[k]:.4f}")

    def _initialize_gaussians(self):
        """
        Initialize non-overlapping Gaussians for each class.

        Strategy:
        - Place class means on a high-dimensional hypersphere
        - Use small isotropic covariances
        - Ensure separation ratio > 4.0 (< 0.1% tail overlap)
        """
        sep = self.config.separation_factor
        variance = self.config.variance_per_class

        self.means = np.zeros((self.num_classes, self.latent_dim))
        self.covariances = np.zeros((self.num_classes, self.latent_dim, self.latent_dim))

        # Place means on hypersphere (for high-dimensional separation)
        if self.latent_dim >= 10:
            for k in range(self.num_classes):
                # Random direction on unit sphere
                direction = np.random.randn(self.latent_dim)
                direction = direction / np.linalg.norm(direction)
                # Scale by separation factor
                self.means[k] = direction * sep * np.sqrt(self.latent_dim)
        else:
            # For low dimensions: grid placement
            for k in range(self.num_classes):
                for d in range(self.latent_dim):
                    if d < self.num_classes:
                        self.means[k, d] = (k - self.num_classes//2) * sep / 2.0
                    else:
                        self.means[k, d] = np.random.randn() * (sep / 4.0)

        # Isotropic covariances
        for k in range(self.num_classes):
            self.covariances[k] = np.eye(self.latent_dim) * variance

        print("\n" + "="*60)
        print("Gaussian Parameters:")
        print("="*60)
        print(f"Latent dimension: {self.latent_dim}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Variance per class: {variance}")
        print(f"Separation factor: {sep}")

        print(f"\nClass means (first 5 components shown):")
        for k in range(self.num_classes):
            mean_str = ", ".join([f"{self.means[k, d]:7.3f}" for d in range(min(5, self.latent_dim))])
            suffix = "..." if self.latent_dim > 5 else ""
            print(f"  Class {k:2d}: [{mean_str}{suffix}]")

    def _check_gaussians_non_overlapping(self) -> Tuple[bool, Dict]:
        """
        Verify non-overlapping property.

        Separation metric: min_distance / max_sigma
        - ratio > 4.0: < 0.1% tail overlap (well-separated)
        - ratio > 6.0: < 0.01% tail overlap (very well-separated)

        Returns:
            (is_non_overlapping, statistics_dict)
        """
        # Pairwise distances between means
        distances = np.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            for j in range(i+1, self.num_classes):
                dist = np.linalg.norm(self.means[i] - self.means[j])
                distances[i, j] = dist
                distances[j, i] = dist

        min_distance = np.min(distances[distances > 0])
        max_sigma = np.max(np.sqrt(np.diagonal(self.covariances[:, range(self.latent_dim), range(self.latent_dim)])))
        separation_ratio = min_distance / max_sigma
        is_non_overlapping = separation_ratio > 4.0

        return is_non_overlapping, {
            'min_distance': min_distance,
            'max_sigma': max_sigma,
            'separation_ratio': separation_ratio,
            'is_non_overlapping': is_non_overlapping
        }

    def sample(self, n_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample from the Mixture of Gaussians.

        Two-stage hierarchical sampling:
        1. Select component k with probability π_k: k ~ Categorical(π)
        2. Sample latent from component: z ~ N(μ_k, Σ_k)

        Args:
            n_samples: Number of samples to generate

        Returns:
            z: Latent variables, shape (n_samples, latent_dim), dtype float64
            k: Class indices (source components), shape (n_samples,), dtype int64
        """
        z = np.zeros((n_samples, self.latent_dim), dtype=np.float64)
        k = np.zeros(n_samples, dtype=np.int64)

        for i in range(n_samples):
            # Step 1: Select component with probability π_k
            k[i] = np.random.choice(self.num_classes, p=self.pi)

            # Step 2: Sample from selected Gaussian
            z[i] = np.random.multivariate_normal(
                self.means[k[i]],
                self.covariances[k[i]]
            )

        return z, k

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience wrapper for sampling batches.

        Args:
            batch_size: Number of samples

        Returns:
            (z_batch, k_batch): Latent variables and class labels
        """
        return self.sample(batch_size)

    def log_prob(self, z: np.ndarray) -> np.ndarray:
        """
        Compute log probability under the mixture model.

        log p(z) = log(Σ_k π_k * N(z | μ_k, Σ_k))

        Args:
            z: Latent samples, shape (n_samples, latent_dim)

        Returns:
            log_probs: Log probabilities, shape (n_samples,)
        """
        n_samples = z.shape[0]
        log_probs = np.zeros(n_samples)

        for i in range(n_samples):
            # Weighted sum of Gaussians
            probs_k = np.zeros(self.num_classes)
            for k in range(self.num_classes):
                probs_k[k] = self.pi[k] * multivariate_normal.pdf(
                    z[i],
                    mean=self.means[k],
                    cov=self.covariances[k]
                )
            log_probs[i] = np.log(np.sum(probs_k) + 1e-10)

        return log_probs

    def posterior_prob(self, z: np.ndarray) -> np.ndarray:
        """
        Compute posterior probability of each component given latent z.

        γ_k(z) = P(k | z) = π_k * N(z | μ_k, Σ_k) / p(z)

        This is the "responsibility" in EM algorithm.

        Args:
            z: Latent samples, shape (n_samples, latent_dim)

        Returns:
            gamma: Responsibilities, shape (n_samples, num_classes)
                   gamma[i, k] = P(component k | sample z[i])
        """
        n_samples = z.shape[0]
        gamma = np.zeros((n_samples, self.num_classes))

        for i in range(n_samples):
            # Compute likelihood for each component
            likelihoods = np.zeros(self.num_classes)
            for k in range(self.num_classes):
                likelihoods[k] = self.pi[k] * multivariate_normal.pdf(
                    z[i],
                    mean=self.means[k],
                    cov=self.covariances[k]
                )

            # Normalize to get posterior
            gamma[i] = likelihoods / (np.sum(likelihoods) + 1e-10)

        return gamma

    def get_statistics(self) -> Dict:
        """Return comprehensive statistics about the model"""
        is_non_overlapping, stats = self._check_gaussians_non_overlapping()

        return {
            'num_classes': self.num_classes,
            'latent_dim': self.latent_dim,
            'is_non_overlapping': is_non_overlapping,
            'separation_ratio': stats['separation_ratio'],
            'min_distance': stats['min_distance'],
            'max_sigma': stats['max_sigma'],
            'class_proportions': self.pi.copy(),
            'means': self.means.copy(),
            'covariances': self.covariances.copy()
        }


# Configuration
config = MixtureGaussianConfig(
    latent_dim=16,
    seed=42,
    separation_factor=8.0,
    variance_per_class=0.5
)

# Initialize MoG
print("Initializing CIFAR-10 Mixture of Gaussians...\n")
mog = CIFAR10MixtureGaussian(config)

# Check non-overlapping property
print("\n" + "="*60)
print("Non-overlapping Verification:")
print("="*60)
stats = mog.get_statistics()
print(f"✓ Non-overlapping: {stats['is_non_overlapping']}")
print(f"  Separation ratio: {stats['separation_ratio']:.3f}")
print(f"  (ratio > 4.0 means < 0.1% Gaussian tail overlap)")

# Sample from MoG
print("\n" + "="*60)
print("Sampling from Mixture of Gaussians:")
print("="*60)
n_samples = 1000
z_samples, k_samples = mog.sample(n_samples)

print(f"\nSampled {n_samples} examples:")
print(f"  Latent z shape: {z_samples.shape}")
print(f"  Class k shape: {k_samples.shape}")

# Show class distribution
print(f"\nClass distribution:")
unique, counts = np.unique(k_samples, return_counts=True)
for cls, count in zip(unique, counts):
    pct = 100 * count / n_samples
    print(f"  Class {cls} ({mog.CIFAR10_CLASSES[cls]:12s}): {count:3d} samples ({pct:5.1f}%)")

# Show sample examples
print(f"\nFirst 5 samples:")
for i in range(5):
    z_str = ", ".join([f"{z_samples[i, d]:7.3f}" for d in range(min(4, config.latent_dim))])
    cls_name = mog.CIFAR10_CLASSES[k_samples[i]]
    print(f"  {i}: z=[{z_str}...], class={k_samples[i]} ({cls_name})")

# Compute log probabilities
print("\n" + "="*60)
print("Log Probability Examples:")
print("="*60)
log_probs = mog.log_prob(z_samples[:5])
for i in range(5):
    print(f"  log p(z[{i}]) = {log_probs[i]:8.4f}")

# Compute posteriors
print("\n" + "="*60)
print("Posterior Probabilities:")
print("="*60)
posterior = mog.posterior_prob(z_samples[:3])
for i in range(3):
    max_prob_class = np.argmax(posterior[i])
    max_prob = posterior[i, max_prob_class]
    true_class = k_samples[i]
    match = "✓" if max_prob_class == true_class else "✗"
    print(f"\n  Sample {i} (true class={true_class}):")
    print(f"    Top 3 posterior probabilities:")
    top_k = np.argsort(posterior[i])[-3:][::-1]
    for rank, k in enumerate(top_k, 1):
        mark = " <- True class" if k == true_class else ""
        print(f"      {rank}. Class {k} ({mog.CIFAR10_CLASSES[k]:12s}): {posterior[i, k]:.4f}{mark}")

print("\n" + "="*60)
print("✓ Example completed successfully!")
print("="*60)
