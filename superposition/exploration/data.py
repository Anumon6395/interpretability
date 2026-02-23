"""Synthetic data for toy superposition: two binary features (A, B) -> high-dim input with noise."""

import torch
from torch.utils.data import Dataset


def make_input_from_features(
    a: torch.Tensor,
    b: torch.Tensor,
    input_dim: int,
    noise_std: float = 0.000001,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Map binary (A, B) to input_dim-dimensional vectors with distinct means per (A,B) and noise.

    Uses two random template vectors (one for A, one for B) so each of the 4 configs
    has a distinct mean; adds Gaussian noise so the same (A,B) gives many different x.
    """
    if seed is not None:
        torch.manual_seed(seed)
    device = a.device
    # Template for feature A: when A=1 we add t_A, when A=0 we add 0 (or -t_A for symmetry)
    t_A = torch.randn(input_dim, device=device)
    t_B = torch.randn(input_dim, device=device)
    # Ensure templates are not too aligned so (A,B) configs are separable
    t_A = t_A / t_A.norm()
    t_B = t_B / t_B.norm()

    # x = a * t_A + b * t_B + noise  (a, b in {0,1})
    a_ = a.float().unsqueeze(1)  # (N, 1)
    b_ = b.float().unsqueeze(1)
    x = a_ * t_A.unsqueeze(0) + b_ * t_B.unsqueeze(0)
    x = x + noise_std * torch.randn_like(x, device=device)
    return x


class TwoFeatureDataset(Dataset):
    """Dataset of (x, y_A, y_B) with x in R^input_dim, y_A, y_B in {0, 1}."""

    def __init__(
        self,
        num_samples: int,
        input_dim: int = 32,
        noise_std: float = 0.1,
        seed: int | None = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.noise_std = noise_std
        # Sample (A, B) uniformly from {0,1}^2
        a = torch.randint(0, 2, (num_samples,))
        b = torch.randint(0, 2, (num_samples,))
        x = make_input_from_features(a, b, input_dim, noise_std, seed=None)
        self.x = x
        self.y_a = a
        self.y_b = b

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y_a[idx], self.y_b[idx]
