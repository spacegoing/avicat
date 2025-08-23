import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# Use a non-interactive backend for plotting
matplotlib.use("Agg")

from avicat.data.crypto_dataset import (
    MarketRegimeAnalyzer,
    create_crypto_dataloaders,
)
from avicat.models.model import DeepStateSpaceModel


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    """Load a model and its configuration from a checkpoint."""
    print(f"Loading checkpoint from: {ckpt_path}")
    # Load with weights_only=False to allow for unpickling of the config object
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

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
        # Ensure 'mean_features' exists and is a dictionary
        if "mean_features" in data and isinstance(data["mean_features"], dict):
            mean_features = data["mean_features"]
            # Create a sorted series for plotting
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
        else:
            # Handle cases where a state might have no data
            ax = axes[i]
            ax.set_title(f"{regime_name} - No data available", fontsize=14)
            ax.text(0.5, 0.5, "No samples found for this state.", ha='center', va='center')


    plt.xlabel("Mean Feature Value (Standardized)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Feature importance plot saved to: {save_path}")


def analyze_single_checkpoint(checkpoint_path: Path, device: torch.device):
    """Main analysis function for a single checkpoint."""
    print("\n" + "=" * 80)
    print(f"Analyzing: {checkpoint_path.name}")
    print("=" * 80)

    # Create output directory for this specific checkpoint
    results_dir = Path("analysis_results") / checkpoint_path.stem
    results_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving analysis results to: {results_dir}")

    # 1. Load Model from Checkpoint
    try:
        model, cfg = load_model_from_checkpoint(str(checkpoint_path), device)
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path.name}: {e}")
        return

    # 2. Load Data
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
    stats_path = results_dir / "regime_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(regime_stats, f, indent=4)
    print(f"Regime statistics saved to: {stats_path}")

    plot_transition_matrix(transition_matrix, results_dir / "transition_matrix.png")
    plot_feature_importance(
        regime_stats,
        feature_names,
        results_dir / "feature_importance.png",
    )
    print(f"✅ Analysis complete for {checkpoint_path.name}")


def main():
    """Finds all checkpoints and analyzes them."""
    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.exists():
        print(f"Error: Checkpoints directory not found at '{ckpt_dir}'")
        return

    # Find all .pt files in the directory
    checkpoint_files = sorted(list(ckpt_dir.glob("*.pt")))

    if not checkpoint_files:
        print(f"No checkpoint files (.pt) found in '{ckpt_dir}'")
        return

    print(f"Found {len(checkpoint_files)} checkpoints to analyze.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for ckpt_file in checkpoint_files:
        analyze_single_checkpoint(ckpt_file, device)

    print("\n" + "="*80)
    print("All analyses complete!")
    print("="*80)

if __name__ == "__main__":
    main()
