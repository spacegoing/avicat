import json
import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

warnings.filterwarnings("ignore")

from crypto_trainer import CryptoTrainer

from crypto_dataset import (
    MarketRegimeAnalyzer,
    create_crypto_dataloaders,
)
from datapipe import create_dataloaders as create_synthetic_dataloaders
from model import DeepStateSpaceModel


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="crypto_config",
)
def main(cfg: DictConfig):
    """Main training function with support for crypto data"""

    # Print config
    print("\n" + "=" * 80)
    print("Configuration")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))

    # Set seeds for reproducibility
    torch.manual_seed(cfg.data.seed)
    np.random.seed(cfg.data.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.data.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Device selection
    device = torch.device("cpu")
    if cfg.training.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif cfg.training.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        print("Using CPU")

    print("=" * 80)
    print("Deep State-Space Model for Crypto Market Regime Discovery")
    print("=" * 80)

    # Determine data source
    use_crypto_data = cfg.data.get("processed_data_dir", None) is not None

    if use_crypto_data and Path(cfg.data.processed_data_dir).exists():
        print("\n📊 Loading Crypto Market Data...")
        print("=" * 80)

        # Create crypto dataloaders
        train_loader, val_loader, data_config = create_crypto_dataloaders(cfg)

        # Update model config with actual data dimensions
        cfg.model.obs_dim = data_config["obs_dim"]
        cfg.data.seq_length = data_config["seq_length"]

        print("\n✓ Loaded crypto dataset:")
        print(f"  - Features: {data_config['obs_dim']} dimensions")
        print(f"  - Sequence length: {data_config['seq_length']} hours")
        print(f"  - Training sequences: {data_config['num_train_samples']}")
        print(f"  - Validation sequences: {data_config['num_val_samples']}")

        # Print feature groups
        print("\n📈 Feature Categories:")
        feature_names = data_config["feature_names"]

        # Categorize features
        price_features = [
            f for f in feature_names if "return" in f or "momentum" in f
        ]
        technical_features = [
            f
            for f in feature_names
            if any(x in f for x in ["rsi", "macd", "volatility", "atr"])
        ]
        sentiment_features = [
            f for f in feature_names if "fear" in f or "dominance" in f
        ]
        structure_features = [
            f
            for f in feature_names
            if "ratio" in f or "streak" in f or "direction" in f
        ]

        print(f"  - Price/Returns: {len(price_features)} features")
        print(f"  - Technical Indicators: {len(technical_features)} features")
        print(f"  - Sentiment: {len(sentiment_features)} features")
        print(f"  - Market Structure: {len(structure_features)} features")

    else:
        print(
            "\n⚠️ Crypto data not found. Using synthetic HMM data for testing...",
        )
        print("Run preprocess.py first to prepare crypto data.")
        print("=" * 80)

        # Fallback to synthetic data
        train_loader, val_loader = create_synthetic_dataloaders(cfg)
        data_config = {
            "obs_dim": cfg.data.obs_dim,
            "seq_length": cfg.data.seq_length,
            "feature_names": [f"feature_{i}" for i in range(cfg.data.obs_dim)],
        }

    print("\n🧠 Model Configuration:")
    print("=" * 80)
    print(f"  - Number of market regimes: {cfg.model.num_states}")
    print(f"  - State embedding dimension: {cfg.model.state_dim}")
    print(f"  - Observation dimension: {cfg.model.obs_dim}")
    print(f"  - Sequence length: {cfg.data.seq_length}")
    print(f"  - Batch size: {cfg.training.batch_size}")
    print(f"  - Learning rate: {cfg.training.learning_rate}")
    print(f"  - Number of epochs: {cfg.training.num_epochs}")
    print(f"  - Device: {device}")
    print(f"  - Output directory: {Path.cwd()}")

    # Initialize model
    print("\n🔧 Initializing model...")
    model = DeepStateSpaceModel(cfg)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"  ✓ Total parameters: {total_params:,}")
    print(f"  ✓ Trainable parameters: {trainable_params:,}")

    print("\n  📐 Model Architecture:")
    print(f"    - Transition network: MLP{cfg.model.transition.hidden_dims}")
    print(f"    - Emission network: MLP{cfg.model.emission.hidden_dims}")
    print(
        f"    - Inference RNN: GRU(hidden={cfg.model.inference.rnn_hidden_dim}, layers={cfg.model.inference.rnn_num_layers})",
    )
    print(
        f"    - Gumbel temperature: {cfg.model.gumbel.temperature_init} → {cfg.model.gumbel.temperature_min}",
    )
    print(
        f"    - KL weight: {cfg.model.regularization.kl_weight_init} → {cfg.model.regularization.kl_weight_max}",
    )

    # Initialize trainer
    print("\n🏋️ Initializing trainer...")
    trainer = CryptoTrainer(
        model,
        train_loader,
        val_loader,
        cfg,
        feature_names=data_config.get("feature_names", []),
    )

    print(f"  ✓ Optimizer: AdamW (lr={cfg.training.learning_rate})")
    print(f"  ✓ Gradient clipping: {cfg.training.grad_clip}")
    if cfg.training.scheduler.enabled:
        print(
            f"  ✓ LR scheduler: ReduceLROnPlateau (patience={cfg.training.scheduler.patience})",
        )
    if cfg.training.get("early_stopping", {}).get("enabled", False):
        print(
            f"  ✓ Early stopping: patience={cfg.training.early_stopping.patience}",
        )
    if cfg.training.logging.use_wandb:
        print(f"  ✓ W&B logging: {cfg.training.logging.project_name}")

    # Start training
    print("\n" + "=" * 80)
    print("🚀 Starting Training")
    print("=" * 80)

    try:
        trainer.train()

        print("\n" + "=" * 80)
        print("✅ Training Completed Successfully!")
        print("=" * 80)
        print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
        print(f"Best epoch: {trainer.best_epoch}")
        print(f"Final temperature: {model.temperature:.4f}")
        print(f"Final KL weight: {model.kl_weight:.6f}")

        # Post-training analysis for crypto data
        if use_crypto_data and cfg.get("analysis", {}).get(
            "compute_regime_statistics",
            False,
        ):
            print("\n" + "=" * 80)
            print("📊 Analyzing Discovered Market Regimes")
            print("=" * 80)

            analyzer = MarketRegimeAnalyzer(
                model,
                val_loader,
                data_config["feature_names"],
            )

            # Extract states
            print("\nExtracting market regime assignments...")
            states, features = analyzer.extract_states()

            # Analyze regimes
            regime_stats = analyzer.analyze_regimes(states, features)

            print("\n🎯 Discovered Market Regimes:")
            for regime_name, stats in regime_stats.items():
                print(f"\n  {regime_name}:")
                print(
                    f"    - Frequency: {stats['percentage']:.1f}% ({stats['count']} timesteps)",
                )
                print("    - Key characteristics:")
                for feat_name, feat_val in list(
                    stats["mean_features"].items(),
                )[:3]:
                    print(f"      • {feat_name}: {feat_val:+.3f}")

            # Get transition matrix
            transition_matrix = analyzer.get_state_transitions(states)
            print("\n🔄 Regime Transition Matrix:")
            print("    (rows: from state, cols: to state)")
            print(np.round(transition_matrix, 3))

            # Save analysis results
            if cfg.analysis.get("export_predictions", False):
                results_dir = Path(
                    cfg.analysis.get("results_dir", "analysis_results"),
                )
                results_dir.mkdir(exist_ok=True)

                # Save regime statistics
                with open(results_dir / "regime_statistics.json", "w") as f:
                    json.dump(regime_stats, f, indent=2)

                # Save transition matrix
                np.save(
                    results_dir / "transition_matrix.npy",
                    transition_matrix,
                )

                # Save state assignments
                np.save(results_dir / "predicted_states.npy", states)

                print(f"\n💾 Analysis results saved to: {results_dir}")

        # Save final config
        config_path = Path.cwd() / "final_config.yaml"
        with open(config_path, "w") as f:
            OmegaConf.save(cfg, f)
        print(f"\n📝 Configuration saved to: {config_path}")

    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("⚠️ Training Interrupted!")
        print("=" * 80)
        print(f"Best validation accuracy so far: {trainer.best_val_acc:.4f}")
        print(f"Best epoch: {trainer.best_epoch}")

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ Training Failed!")
        print("=" * 80)
        print(f"Error: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
