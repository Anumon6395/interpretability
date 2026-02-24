"""
Toy superposition: train encoder + readout on two binary features in 2D embedding space,
then visualize embeddings and feature directions (expect ~90° between them).
"""

import math
from pathlib import Path
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from data import TwoFeatureDataset
from model import ToySuperpositionModel


def prediction_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Mean entropy of predicted distribution over the batch (rewarding certainty)."""
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs.clamp(min=1e-8))
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.mean()


def train(
    model: ToySuperpositionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    certainty_weight: float | None = 0.1,
    verbose: bool = True,
    early_stop_loss: float | None = None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
) -> list[float]:
    """
    certainty_weight: If None, loss is CE only (prediction-only). Else add entropy penalty.
    early_stop_loss: If set, stop when epoch average loss falls below this (saves time).
    scheduler: If set, step with epoch loss (e.g. ReduceLROnPlateau).
    """
    model.train()
    losses: list[float] = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n = 0
        for x, y_a, y_b in loader:
            x, y_a, y_b = x.to(device), y_a.to(device), y_b.to(device)
            optimizer.zero_grad()
            _, logits_a, logits_b = model(x)
            loss_a = F.cross_entropy(logits_a, y_a)
            loss_b = F.cross_entropy(logits_b, y_b)
            loss = loss_a + loss_b
            if certainty_weight is not None and certainty_weight != 0:
                entropy_a = prediction_entropy(logits_a)
                entropy_b = prediction_entropy(logits_b)
                loss = loss + certainty_weight * (entropy_a + entropy_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        avg = epoch_loss / max(n, 1)
        losses.append(avg)
        if scheduler is not None:
            scheduler.step(avg)
        if verbose and ((epoch + 1) % 100 == 0 or epoch == 0):
            print(f"Epoch {epoch + 1}/{num_epochs}  loss = {avg:.4f}")
        if early_stop_loss is not None and avg < early_stop_loss:
            if verbose:
                print(f"Early stop at epoch {epoch + 1} (loss {avg:.4f} < {early_stop_loss})")
            break
    return losses


def angle_between_degrees(w_a: torch.Tensor, w_b: torch.Tensor) -> float:
    """Angle in degrees between two unit vectors (0 to 180)."""
    cos_angle = (w_a * w_b).sum().item()
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def visualize(
    embeddings: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    w_a: np.ndarray,
    w_b: np.ndarray,
    angle_deg: float,
    save_path: str | None = None,
) -> None:
    """
    Two panels: scatter colored by A, scatter colored by B; overlay feature direction arrows.
    """
    scale = max(
        np.abs(embeddings).max() * 1.2,
        np.abs(w_a).max() * 2,
        np.abs(w_b).max() * 2,
        0.5,
    )
    arrow_scale = scale * 0.8

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left: color by feature A (0 vs 1)
    ax = axes[0]
    mask0 = y_a == 0
    mask1 = y_a == 1
    ax.scatter(embeddings[mask0, 0], embeddings[mask0, 1], c="tab:blue", alpha=0.6, label="A=0")
    ax.scatter(embeddings[mask1, 0], embeddings[mask1, 1], c="tab:red", alpha=0.6, label="A=1")
    ax.arrow(0, 0, w_a[0] * arrow_scale, w_a[1] * arrow_scale, head_width=0.08 * scale, head_length=0.05 * scale, fc="black", ec="black")
    ax.arrow(0, 0, w_b[0] * arrow_scale, w_b[1] * arrow_scale, head_width=0.08 * scale, head_length=0.05 * scale, fc="gray", ec="gray")
    ax.text(w_a[0] * arrow_scale * 1.1, w_a[1] * arrow_scale * 1.1, "A", fontsize=12)
    ax.text(w_b[0] * arrow_scale * 1.1, w_b[1] * arrow_scale * 1.1, "B", fontsize=12)
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Color by feature A")
    ax.set_xlabel("Embedding dim 0")
    ax.set_ylabel("Embedding dim 1")

    # Right: color by feature B (0 vs 1)
    ax = axes[1]
    mask0 = y_b == 0
    mask1 = y_b == 1
    ax.scatter(embeddings[mask0, 0], embeddings[mask0, 1], c="tab:green", alpha=0.6, label="B=0")
    ax.scatter(embeddings[mask1, 0], embeddings[mask1, 1], c="tab:orange", alpha=0.6, label="B=1")
    ax.arrow(0, 0, w_a[0] * arrow_scale, w_a[1] * arrow_scale, head_width=0.08 * scale, head_length=0.05 * scale, fc="black", ec="black")
    ax.arrow(0, 0, w_b[0] * arrow_scale, w_b[1] * arrow_scale, head_width=0.08 * scale, head_length=0.05 * scale, fc="gray", ec="gray")
    ax.text(w_a[0] * arrow_scale * 1.1, w_a[1] * arrow_scale * 1.1, "A", fontsize=12)
    ax.text(w_b[0] * arrow_scale * 1.1, w_b[1] * arrow_scale * 1.1, "B", fontsize=12)
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Color by feature B")
    ax.set_xlabel("Embedding dim 0")
    ax.set_ylabel("Embedding dim 1")

    fig.suptitle(f"Angle between feature directions: {angle_deg:.1f}°")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved figure to {save_path}")
    plt.show()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 32
    hidden_dim = 64
    embed_dim = 2
    num_train = 50000
    batch_size = 128
    num_epochs = 5000
    lr = 1e-2
    certainty_weight = 0.1  # penalize prediction uncertainty (entropy) to push toward orthogonal directions
    seed = random.randint(0, 1000000)

    torch.manual_seed(seed)
    train_dataset = TwoFeatureDataset(num_train, input_dim=input_dim, noise_std=0.15, seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = ToySuperpositionModel(input_dim=input_dim, hidden_dim=hidden_dim, embed_dim=embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train(model, train_loader, optimizer, device, num_epochs, certainty_weight=certainty_weight)

    # Fixed set for visualization (500 samples)
    viz_dataset = TwoFeatureDataset(500, input_dim=input_dim, noise_std=0.15, seed=seed + 1)
    viz_loader = DataLoader(viz_dataset, batch_size=500, shuffle=False)
    model.eval()
    with torch.no_grad():
        x_viz, y_a_viz, y_b_viz = next(iter(viz_loader))
        x_viz = x_viz.to(device)
        embed_viz, _, _ = model(x_viz)
        embed_np = embed_viz.cpu().numpy()
        y_a_np = y_a_viz.numpy()
        y_b_np = y_b_viz.numpy()

        w_a, w_b = model.get_feature_directions()
        w_a_np = w_a.cpu().numpy()
        w_b_np = w_b.cpu().numpy()
        angle_deg = angle_between_degrees(w_a, w_b)

    print(f"Angle between feature directions: {angle_deg:.1f}°")
    out_path = Path(__file__).resolve().parent / "superposition_embedding_viz.png"
    visualize(embed_np, y_a_np, y_b_np, w_a_np, w_b_np, angle_deg, save_path=str(out_path))


if __name__ == "__main__":
    main()
