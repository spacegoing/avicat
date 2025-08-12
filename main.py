import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

warnings.filterwarnings("ignore")

from datapipe import create_dataloaders
from model import DeepStateSpaceModel
from trainer import Trainer


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function"""

    # Print config
    print("\n" + "=" * 70)
    print("Configuration")
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))

    # Set seeds for reproducibility
    torch.manual_seed(cfg.data.seed)
    np.random.seed(cfg.data.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.data.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")

    # Verify configuration consistency
    assert (
        cfg.model.num_states == cfg.data.num_true_states
    ), f"Model states ({cfg.model.num_states}) must match data states ({cfg.data.num_true_states})"
    assert (
        cfg.model.obs_dim == cfg.data.obs_dim
    ), f"Model obs_dim ({cfg.model.obs_dim}) must match data obs_dim ({cfg.data.obs_dim})"

    print("=" * 70)
    print("Deep State-Space Model with Neural Categorical Latents")
    print("=" * 70)
    print("Configuration Summary:")
    print(f"  - Number of states: {cfg.model.num_states}")
    print(f"  - Observation dimension: {cfg.data.obs_dim}")
    print(f"  - Sequence length: {cfg.data.seq_length}")
    print(f"  - Train sequences: {cfg.data.num_sequences_train}")
    print(f"  - Val sequences: {cfg.data.num_sequences_val}")
    print(f"  - Batch size: {cfg.training.batch_size}")
    print(f"  - Learning rate: {cfg.training.learning_rate}")
    print(f"  - Number of epochs: {cfg.training.num_epochs}")
    print(f"  - Initial temperature: {cfg.model.gumbel.temperature_init}")
    print(f"  - Initial KL weight: {cfg.model.regularization.kl_weight_init}")
    print(f"  - Device: {cfg.training.device}")
    print(f"  - Output directory: {Path.cwd()}")
    print("=" * 70)

    # Create data loaders
    print("\nCreating synthetic HMM dataset...")
    train_loader, val_loader = create_dataloaders(cfg)
    print(
        f"  ✓ Train batches: {len(train_loader)} ({cfg.data.num_sequences_train} sequences)",
    )
    print(
        f"  ✓ Val batches: {len(val_loader)} ({cfg.data.num_sequences_val} sequences)",
    )

    # Sample data statistics
    sample_batch = next(iter(train_loader))
    sample_x, sample_states = sample_batch
    print(f"  ✓ Batch shape: {sample_x.shape}")
    print(
        f"  ✓ Observation range: [{sample_x.min():.2f}, {sample_x.max():.2f}]",
    )

    # Initialize model
    print("\nInitializing model...")
    model = DeepStateSpaceModel(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"  ✓ Total parameters: {total_params:,}")
    print(f"  ✓ Trainable parameters: {trainable_params:,}")
    print("  ✓ Model architecture:")
    print(f"    - State embedding dim: {cfg.model.state_dim}")
    print(f"    - Transition network: {cfg.model.transition.hidden_dims}")
    print(f"    - Emission network: {cfg.model.emission.hidden_dims}")
    print(
        f"    - Inference RNN: {cfg.model.inference.rnn_hidden_dim} (bidirectional)",
    )

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer(model, train_loader, val_loader, cfg)
    print(f"  ✓ Optimizer: AdamW (lr={cfg.training.learning_rate})")
    print(f"  ✓ Gradient clipping: {cfg.training.grad_clip}")
    if cfg.training.scheduler.enabled:
        print(
            f"  ✓ LR scheduler: ReduceLROnPlateau (patience={cfg.training.scheduler.patience})",
        )
    if cfg.training.logging.use_wandb:
        print(f"  ✓ W&B logging: {cfg.training.logging.project_name}")

    # Start training
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    try:
        trainer.train()

        print("\n" + "=" * 70)
        print("Training Completed Successfully!")
        print("=" * 70)
        print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
        print(f"Output saved to: {Path.cwd()}")

        # Save final config
        config_path = Path.cwd() / "final_config.yaml"
        with open(config_path, "w") as f:
            OmegaConf.save(cfg, f)
        print(f"Configuration saved to: {config_path}")

    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("Training Interrupted!")
        print("=" * 70)
        print(f"Best validation accuracy so far: {trainer.best_val_acc:.4f}")

    except Exception as e:
        print("\n" + "=" * 70)
        print("Training Failed!")
        print("=" * 70)
        print(f"Error: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
