"""
Sweep noise_std with CE-only loss; record angle and loss; save CSV and plots with 95% CIs.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

from data import TwoFeatureDataset
from model import ToySuperpositionModel
from train import angle_between_degrees, train


def next_run_version(out_dir: Path, base: str = "sweep_noise_results") -> int:
    """Return next version number (1-based) so we don't overwrite existing runs."""
    max_n = 0
    for p in out_dir.glob(f"{base}_*.csv"):
        try:
            n = int(p.stem.split("_")[-1])
            max_n = max(max_n, n)
        except (ValueError, IndexError):
            continue
    return max_n + 1


def ci_95_half(samples: list[float]) -> float:
    """Half-width of 95% CI for the mean (t-distribution)."""
    n = len(samples)
    if n < 2:
        return 0.0
    std = np.std(samples, ddof=1)
    t = stats.t.ppf(0.975, n - 1)
    return t * std / np.sqrt(n)


def run_one(
    noise_std: float,
    seed: int,
    device: torch.device,
    num_train: int,
    batch_size: int,
    num_epochs: int,
    input_dim: int,
    hidden_dim: int,
    early_stop_loss: float | None,
    lr: float,
    scheduler_patience: int,
    scheduler_factor: float,
) -> tuple[float, float, float]:
    """Train once (CE only); return (angle_deg, abs_angle_error, final_loss)."""
    torch.manual_seed(seed)
    train_dataset = TwoFeatureDataset(
        num_train, input_dim=input_dim, noise_std=noise_std, seed=seed
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = ToySuperpositionModel(
        input_dim=input_dim, hidden_dim=hidden_dim, embed_dim=2
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=1e-5,
    )
    losses = train(
        model,
        train_loader,
        optimizer,
        device,
        num_epochs,
        certainty_weight=None,
        verbose=False,
        early_stop_loss=early_stop_loss,
        scheduler=scheduler,
    )
    final_loss = losses[-1] if losses else float("nan")
    model.eval()
    with torch.no_grad():
        w_a, w_b = model.get_feature_directions()
        angle_deg = angle_between_degrees(w_a, w_b)
    abs_angle_error = abs(angle_deg - 90.0)
    return angle_deg, abs_angle_error, final_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep noise_std (CE only), plot with 95% CIs")
    parser.add_argument(
        "--noise-stds",
        type=str,
        default="0.01,0.02,0.03,0.05,0.08,0.1,0.12,0.15,0.2,0.25,0.3,0.4,0.5,0.6",
        help="Comma-separated noise_std values",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=20,
        help="Number of trials per noise_std (for confidence intervals)",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=5000,
        help="Training set size per run",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=150,
        help="Max epochs per run (high noise needs more epochs to converge)",
    )
    parser.add_argument(
        "--early-stop-loss",
        type=float,
        default=1e-3,
        help="Stop when epoch loss drops below this; use 0 to disable",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=15,
        help="ReduceLROnPlateau patience (epochs before reducing lr); higher helps when loss plateaus high",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau factor (new_lr = lr * factor)",
    )
    parser.add_argument(
        "--max-mean-loss",
        type=float,
        default=0.15,
        help="Stop sweep when mean loss for a noise_std exceeds this; then save CSV and chart",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for CSV and plots (default: same as script)",
    )
    args = parser.parse_args()

    noise_stds = [float(x.strip()) for x in args.noise_stds.split(",")]
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 32
    hidden_dim = 64
    early_stop_loss = args.early_stop_loss if args.early_stop_loss > 0 else None

    rows: list[dict] = []
    for noise_std in noise_stds:
        for seed in range(args.seeds):
            angle_deg, abs_angle_error, final_loss = run_one(
                noise_std=noise_std,
                seed=seed,
                device=device,
                num_train=args.num_train,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                early_stop_loss=early_stop_loss,
                lr=args.lr,
                scheduler_patience=args.scheduler_patience,
                scheduler_factor=args.scheduler_factor,
            )
            rows.append({
                "noise_std": noise_std,
                "seed": seed,
                "angle_deg": round(angle_deg, 4),
                "abs_angle_error": round(abs_angle_error, 4),
                "final_loss": round(final_loss, 6),
            })
            print(f"noise_std={noise_std} seed={seed} angle={angle_deg:.1f}° |angle-90|={abs_angle_error:.1f}° loss={final_loss:.4f}")
        # Stop sweep if mean loss for this noise_std exceeds threshold; flush chart with data so far
        current_losses = [r["final_loss"] for r in rows if r["noise_std"] == noise_std]
        if np.mean(current_losses) > args.max_mean_loss:
            print(f"\nStopping sweep: mean loss at noise_std={noise_std} is {np.mean(current_losses):.4f} > {args.max_mean_loss}")
            break

    # Version outputs so we don't overwrite previous runs
    version = next_run_version(out_dir)
    csv_path = out_dir / f"sweep_noise_results_{version}.csv"
    plot_path = out_dir / f"sweep_noise_plots_{version}.png"

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["noise_std", "seed", "angle_deg", "abs_angle_error", "final_loss"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}")

    # Aggregate per noise_std: mean and 95% CI half-width
    agg: dict[float, list[dict]] = {}
    for r in rows:
        agg.setdefault(r["noise_std"], []).append(r)
    noise_stds_sorted = sorted(agg.keys())

    angle_error_mean = {}
    angle_error_ci = {}
    loss_mean = {}
    loss_ci = {}
    for ns in noise_stds_sorted:
        group = agg[ns]
        errors = [x["abs_angle_error"] for x in group]
        losses = [x["final_loss"] for x in group]
        angle_error_mean[ns] = np.mean(errors)
        angle_error_ci[ns] = ci_95_half(errors)
        loss_mean[ns] = np.mean(losses)
        loss_ci[ns] = ci_95_half(losses)

    # Summary table
    print("\nMean |angle - 90°| (95% CI) by noise_std (lower is better):")
    for ns in noise_stds_sorted:
        m, h = angle_error_mean[ns], angle_error_ci[ns]
        print(f"  noise_std={ns}: {m:.1f}° ± {h:.1f}°  [{m - h:.1f}, {m + h:.1f}]")

    # Plots with 95% confidence intervals
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: distance from 90° vs noise_std with 95% CI
    ax = axes[0]
    xs = noise_stds_sorted
    ys = [angle_error_mean[ns] for ns in xs]
    ci_half = [angle_error_ci[ns] for ns in xs]
    ax.errorbar(xs, ys, yerr=ci_half, capsize=3, marker="o", label="Mean (95% CI)")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7, label="0° (orthogonal)")
    ax.set_xlabel("noise_std")
    ax.set_ylabel("|angle - 90°| (°)")
    ax.set_title("Distance from orthogonal vs input noise (95% CI, lower is better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: final loss vs noise_std with 95% CI
    ax = axes[1]
    ys = [loss_mean[ns] for ns in xs]
    ci_half = [loss_ci[ns] for ns in xs]
    ax.errorbar(xs, ys, yerr=ci_half, capsize=3, marker="o", label="Mean (95% CI)")
    ax.set_xlabel("noise_std")
    ax.set_ylabel("Final loss")
    ax.set_title("Final loss vs input noise (95% CI)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    print(f"Saved {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
