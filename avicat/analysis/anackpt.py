import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from omegaconf import DictConfig

# Use a non-interactive backend for plotting
matplotlib.use("Agg")

from crypto_dataset import MarketRegimeAnalyzer, create_crypto_dataloaders
from model import DeepStateSpaceModel


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    """Load a model and its configuration from a checkpoint."""
    print(f"Loading checkpoint from: {ckpt_path}")
    # --- THIS IS THE CORRECTED LINE ---
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    # ----------------------------------

    cfg = checkpoint["cfg"]
    print("Configuration loaded from checkpoint.")

    model = DeepStateSpaceModel(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print("Model loaded and in evaluation mode.")

    return model, cfg


def plot_transition_matrix(matrix: np.ndarray, save_path: Path):
    """Plot and save the state transition matrix as a heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar=True,
        linewidths=0.5,
    )
    plt.title("Market Regime Transition Matrix", fontsize=16)
    plt.xlabel("To State", fontsize=12)
    plt.ylabel("From State", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Transition matrix plot saved to: {save_path}")


def plot_feature_importance(stats: dict, feature_names: list, save_path: Path):
    """Plot and save the top N mean features for each state."""
    n_states = len(stats)
    top_n = 5  # Number of features to show

    fig, axes = plt.subplots(
        n_states,
        1,
        figsize=(12, n_states * 4),
        sharex=True,
    )
    if n_states == 1:
        axes = [axes]

    for i, (regime_name, data) in enumerate(stats.items()):
        mean_features = data["mean_features"]
        df = (
            pd.Series(mean_features)
            .sort_values(ascending=False)
            .head(top_n)
            .sort_values(ascending=True)
        )

        ax = axes[i]
        ax.barh(df.index, df.values, color=sns.color_palette("viridis", top_n))
        ax.set_title(f"{regime_name} - Top {top_n} Mean Feature Values", fontsize=14)
        ax.tick_params(axis="x", rotation=0)

    plt.xlabel("Mean Feature Value (Standardized)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Feature importance plot saved to: {save_path}")


def main(ckpt_path: str):
    """Main analysis function."""
    checkpoint_path = Path(ckpt_path)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found at {ckpt_path}")
        return

    # Create output directory
    results_dir = Path("analysis_results") / checkpoint_path.stem
    results_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving analysis results to: {results_dir}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model from Checkpoint
    model, cfg = load_model_from_checkpoint(str(checkpoint_path), device)

    # 2. Load Data
    # The analyzer needs the validation set to calculate statistics
    _, val_loader, data_config = create_crypto_dataloaders(cfg)
    feature_names = data_config["feature_names"]

    # 3. Initialize Analyzer
    analyzer = MarketRegimeAnalyzer(model, val_loader, feature_names)

    # 4. Perform Analysis
    print("\nExtracting market regime assignments...")
    states, features = analyzer.extract_states()

    print("Analyzing regime characteristics...")
    regime_stats = analyzer.analyze_regimes(states, features)

    print("Calculating transition matrix...")
    transition_matrix = analyzer.get_state_transitions(states)

    # 5. Save Results
    # Save regime statistics to JSON
    stats_path = results_dir / "regime_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(regime_stats, f, indent=4)
    print(f"Regime statistics saved to: {stats_path}")

    # Plot and save transition matrix
    plot_transition_matrix(transition_matrix, results_dir / "transition_matrix.png")

    # Plot and save feature importance
    plot_feature_importance(
        regime_stats,
        feature_names,
        results_dir / "feature_importance.png",
    )

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze a trained DSSM checkpoint for crypto market regimes.",
    )
    parser.add_argument(
        "ckpt_path",
        type=str,
        help="Path to the model checkpoint file (e.g., checkpoints/best_model.pt).",
    )
    args = parser.parse_args()
    main(args.ckpt_path)
