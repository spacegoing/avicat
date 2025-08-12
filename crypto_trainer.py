import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.optim as optim
import wandb
from omegaconf import DictConfig
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


def compute_clustering_accuracy(pred_states, true_states):
    """Compute clustering accuracy using Hungarian algorithm"""
    pred_flat = pred_states.cpu().numpy().flatten()
    true_flat = true_states.cpu().numpy().flatten()

    pred_labels = np.unique(pred_flat)
    true_labels = np.unique(true_flat)

    max_label = (
        max(
            max(pred_labels) if len(pred_labels) > 0 else 0,
            max(true_labels) if len(true_labels) > 0 else 0,
        )
        + 1
    )

    confusion = np.zeros((max_label, max_label))

    for p_label in pred_labels:
        for t_label in true_labels:
            confusion[p_label, t_label] = np.sum(
                (pred_flat == p_label) & (true_flat == t_label),
            )

    row_ind, col_ind = linear_sum_assignment(-confusion)
    correct = confusion[row_ind, col_ind].sum()
    total = len(pred_flat)

    return correct / total if total > 0 else 0


class CryptoTrainer:
    """Enhanced DSSM Trainer for Crypto Market Data"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        cfg: DictConfig,
        feature_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.feature_names = feature_names or []

        # Device setup
        self.device = torch.device("cpu")
        if cfg.training.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif (
            cfg.training.device == "mps" and torch.backends.mps.is_available()
        ):
            self.device = torch.device("mps")

        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=1e-5,
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler
        self.scheduler = None
        if cfg.training.scheduler.enabled:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=cfg.training.scheduler.factor,
                patience=cfg.training.scheduler.patience,
                min_lr=cfg.training.scheduler.min_lr,
            )

        # Early stopping
        self.early_stopping_enabled = cfg.training.get(
            "early_stopping",
            {},
        ).get("enabled", False)
        if self.early_stopping_enabled:
            self.early_stopping_patience = cfg.training.early_stopping.patience
            self.early_stopping_counter = 0
            self.early_stopping_min_delta = cfg.training.early_stopping.get(
                "min_delta",
                0.0001,
            )

        # Weights & Biases logging
        self.use_wandb = cfg.training.logging.use_wandb
        if self.use_wandb:
            wandb.init(
                project=cfg.training.logging.project_name,
                name=cfg.training.logging.run_name,
                config=dict(cfg),
                tags=["crypto", "dssm", "market-regimes"],
            )

            # Log feature names
            if self.feature_names:
                wandb.config.update({"features": self.feature_names})

        # Training state
        self.global_step = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")
        self.best_epoch = 0

        # Metrics tracking
        self.train_history = {
            "loss": [],
            "accuracy": [],
            "recon_loss": [],
            "kl_loss": [],
        }
        self.val_history = {
            "loss": [],
            "accuracy": [],
            "recon_loss": [],
            "kl_loss": [],
        }

        # Checkpointing
        self.checkpoint_dir = Path.cwd() / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

    def train_epoch(self, epoch):
        """Train for one epoch with enhanced monitoring"""
        self.model.train()
        epoch_metrics = {
            "loss": [],
            "accuracy": [],
            "recon_loss": [],
            "kl_loss": [],
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (x, true_states) in enumerate(pbar):
            x = x.to(self.device)
            true_states = true_states.to(self.device)

            # Forward pass
            elbo, components, pred_states = self.model.compute_elbo(
                x,
                return_components=True,
                return_states=True,
            )
            loss = -elbo

            # Skip batch if NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(
                    f"Warning: NaN/Inf loss detected at batch {batch_idx}, skipping...",
                )
                continue

            # For crypto data, we don't have true states, so accuracy is based on consistency
            # We can compute pseudo-accuracy based on state stability
            if true_states.sum() == 0:  # Pseudo states (all zeros)
                # Compute state consistency as proxy for accuracy
                state_changes = (
                    (pred_states[:, 1:] != pred_states[:, :-1]).float().mean()
                )
                accuracy = (
                    1.0 - state_changes.item()
                )  # Higher consistency = higher "accuracy"
            else:
                accuracy = compute_clustering_accuracy(
                    pred_states,
                    true_states,
                )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.training.grad_clip,
            )

            self.optimizer.step()

            # Annealing
            self.model.anneal_temperature()
            self.model.anneal_kl_weight()

            # Track metrics
            epoch_metrics["loss"].append(loss.item())
            epoch_metrics["accuracy"].append(accuracy)
            epoch_metrics["recon_loss"].append(
                components["reconstruction_loss"].item(),
            )
            epoch_metrics["kl_loss"].append(components["kl_loss"].item())

            self.global_step += 1

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.3f}",
                    "acc": f"{accuracy:.3f}",
                    "temp": f"{self.model.temperature:.3f}",
                    "kl_w": f"{self.model.kl_weight:.5f}",
                    "grad": f"{grad_norm:.3f}",
                },
            )

            # Log to W&B
            if self.use_wandb and self.global_step % 10 == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/accuracy": accuracy,
                        "train/reconstruction_loss": components[
                            "reconstruction_loss"
                        ].item(),
                        "train/kl_loss": components["kl_loss"].item(),
                        "train/elbo": elbo.item(),
                        "train/temperature": self.model.temperature,
                        "train/kl_weight": self.model.kl_weight,
                        "train/grad_norm": grad_norm,
                        "train/learning_rate": self.optimizer.param_groups[0][
                            "lr"
                        ],
                        "step": self.global_step,
                    },
                )

        # Compute epoch averages
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}

        # Update history
        for k, v in avg_metrics.items():
            self.train_history[k].append(v)

        return avg_metrics

    def validate(self, epoch):
        """Validate with market regime analysis"""
        self.model.eval()
        val_metrics = {
            "loss": [],
            "accuracy": [],
            "recon_loss": [],
            "kl_loss": [],
        }

        # State distribution tracking
        state_counts = np.zeros(self.cfg.model.num_states)

        with torch.no_grad():
            for x, true_states in self.val_loader:
                x = x.to(self.device)
                true_states = true_states.to(self.device)

                elbo, components, pred_states = self.model.compute_elbo(
                    x,
                    return_components=True,
                    return_states=True,
                )
                loss = -elbo

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                # Compute accuracy
                if true_states.sum() == 0:  # Pseudo states
                    state_changes = (
                        (pred_states[:, 1:] != pred_states[:, :-1])
                        .float()
                        .mean()
                    )
                    accuracy = 1.0 - state_changes.item()
                else:
                    accuracy = compute_clustering_accuracy(
                        pred_states,
                        true_states,
                    )

                val_metrics["loss"].append(loss.item())
                val_metrics["accuracy"].append(accuracy)
                val_metrics["recon_loss"].append(
                    components["reconstruction_loss"].item(),
                )
                val_metrics["kl_loss"].append(components["kl_loss"].item())

                # Track state distribution
                for state in pred_states.cpu().numpy().flatten():
                    state_counts[state] += 1

        # Compute averages
        avg_metrics = {
            k: np.mean(v) if v else 0 for k, v in val_metrics.items()
        }

        # Update history
        for k, v in avg_metrics.items():
            self.val_history[k].append(v)

        # Compute state distribution entropy (diversity measure)
        state_probs = state_counts / (state_counts.sum() + 1e-10)
        state_entropy = -np.sum(state_probs * np.log(state_probs + 1e-10))
        max_entropy = np.log(self.cfg.model.num_states)
        state_diversity = state_entropy / max_entropy  # Normalized entropy

        # Learning rate scheduling
        if self.scheduler:
            self.scheduler.step(avg_metrics["accuracy"])

        # Check for best model
        is_best = False
        if avg_metrics["accuracy"] > self.best_val_acc:
            self.best_val_acc = avg_metrics["accuracy"]
            self.best_val_loss = avg_metrics["loss"]
            self.best_epoch = epoch
            is_best = True
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        # Save checkpoint
        if (
            is_best
            or epoch % self.cfg.training.logging.get("save_interval", 50) == 0
        ):
            self.save_checkpoint(epoch, is_best=is_best)

        # Log to W&B
        if self.use_wandb:
            wandb.log(
                {
                    "val/loss": avg_metrics["loss"],
                    "val/accuracy": avg_metrics["accuracy"],
                    "val/reconstruction_loss": avg_metrics["recon_loss"],
                    "val/kl_loss": avg_metrics["kl_loss"],
                    "val/best_accuracy": self.best_val_acc,
                    "val/state_diversity": state_diversity,
                    "val/state_distribution": wandb.Histogram(state_counts),
                    "epoch": epoch,
                },
            )

            # Log state usage
            for i, count in enumerate(state_counts):
                wandb.log(
                    {
                        f"val/state_{i}_usage": count / state_counts.sum(),
                        "epoch": epoch,
                    },
                )

        return avg_metrics, state_diversity

    def train(self):
        """Main training loop with enhanced monitoring"""
        print(f"\n🚀 Starting training on {self.device}")
        print(
            f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}",
        )
        print(f"   Training samples: {len(self.train_loader.dataset)}")
        print(f"   Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(1, self.cfg.training.num_epochs + 1):
            # Training
            train_metrics = self.train_epoch(epoch)

            # Validation
            if epoch % self.cfg.training.logging.val_interval == 0:
                val_metrics, state_diversity = self.validate(epoch)

                # Print epoch summary
                print(f"\nEpoch {epoch}/{self.cfg.training.num_epochs}:")
                print(
                    f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}",
                )
                print(
                    f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}",
                )
                print(f"  State Diversity: {state_diversity:.3f}")
                print(
                    f"  Best Val Acc: {self.best_val_acc:.4f} (epoch {self.best_epoch})",
                )
                print(
                    f"  Temperature: {self.model.temperature:.4f}, KL Weight: {self.model.kl_weight:.6f}",
                )

                # Early stopping check
                if self.early_stopping_enabled:
                    if (
                        self.early_stopping_counter
                        >= self.early_stopping_patience
                    ):
                        print(
                            f"\n⏹️ Early stopping triggered (patience={self.early_stopping_patience})",
                        )
                        break

        print("\n✅ Training completed!")
        print(f"   Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"   Best epoch: {self.best_epoch}")

        # Save training history
        self.save_training_history()

        if self.use_wandb:
            wandb.finish()

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint with enhanced metadata"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "temperature": self.model.temperature,
            "kl_weight": self.model.kl_weight,
            "cfg": self.cfg,
            "feature_names": self.feature_names,
            "train_history": self.train_history,
            "val_history": self.val_history,
        }

        filename = (
            "best_model.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
        )
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)

        print(
            f"💾 {'Best model' if is_best else 'Checkpoint'} saved: {filepath}",
        )

        if self.use_wandb and is_best:
            wandb.save(str(filepath))

    def save_training_history(self):
        """Save training history to JSON"""
        history = {
            "train": self.train_history,
            "val": self.val_history,
            "best_val_acc": float(self.best_val_acc),
            "best_val_loss": float(self.best_val_loss),
            "best_epoch": int(self.best_epoch),
        }

        history_path = Path.cwd() / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        print(f"📊 Training history saved: {history_path}")
