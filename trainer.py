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


class Trainer:
    """DSSM Trainer"""

    def __init__(self, model, train_loader, val_loader, cfg: DictConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg

        self.device = torch.device(
            cfg.training.device if torch.cuda.is_available() else "cpu",
        )
        self.model = self.model.to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=1e-5,
        )

        if cfg.training.scheduler.enabled:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=cfg.training.scheduler.factor,
                patience=cfg.training.scheduler.patience,
                min_lr=cfg.training.scheduler.min_lr,
            )

        if cfg.training.logging.use_wandb:
            wandb.init(
                project=cfg.training.logging.project_name,
                name=cfg.training.logging.run_name,
                config=dict(cfg),
            )

        self.global_step = 0
        self.best_val_acc = 0.0

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        epoch_accuracies = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for x, true_states in pbar:
            x = x.to(self.device)
            true_states = true_states.to(self.device)

            elbo, components, pred_states = self.model.compute_elbo(
                x,
                return_components=True,
                return_states=True,
            )
            loss = -elbo

            if torch.isnan(loss):
                continue

            accuracy = compute_clustering_accuracy(pred_states, true_states)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.training.grad_clip,
            )
            self.optimizer.step()

            self.model.anneal_temperature()
            self.model.anneal_kl_weight()

            epoch_losses.append(loss.item())
            epoch_accuracies.append(accuracy)
            self.global_step += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.2e}",
                    "acc": f"{accuracy:.3f}",
                    "temp": f"{self.model.temperature:.3f}",
                    "kl_w": f"{self.model.kl_weight:.4f}",
                },
            )

            if (
                self.cfg.training.logging.use_wandb
                and self.global_step % 10 == 0
            ):
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/accuracy": accuracy,
                        "train/reconstruction_loss": components[
                            "reconstruction_loss"
                        ].item(),
                        "train/kl_loss": components["kl_loss"].item(),
                        "train/temperature": self.model.temperature,
                        "train/kl_weight": self.model.kl_weight,
                        "step": self.global_step,
                    },
                )

        return np.mean(epoch_losses), np.mean(epoch_accuracies)

    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        val_losses = []
        val_accuracies = []

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

                if torch.isnan(loss):
                    continue

                accuracy = compute_clustering_accuracy(
                    pred_states,
                    true_states,
                )
                val_losses.append(loss.item())
                val_accuracies.append(accuracy)

        avg_val_loss = np.mean(val_losses) if val_losses else float("inf")
        avg_val_acc = np.mean(val_accuracies) if val_accuracies else 0

        if self.cfg.training.scheduler.enabled:
            self.scheduler.step(avg_val_acc)

        if avg_val_acc > self.best_val_acc:
            self.best_val_acc = avg_val_acc
            self.save_checkpoint(epoch, is_best=True)

        if self.cfg.training.logging.use_wandb:
            wandb.log(
                {
                    "val/loss": avg_val_loss,
                    "val/accuracy": avg_val_acc,
                    "val/best_accuracy": self.best_val_acc,
                    "epoch": epoch,
                },
            )

        return avg_val_loss, avg_val_acc

    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}",
        )

        for epoch in range(1, self.cfg.training.num_epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)

            if epoch % self.cfg.training.logging.val_interval == 0:
                val_loss, val_acc = self.validate(epoch)
                print(
                    f"Epoch {epoch}: Train Loss: {train_loss:.2e}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.2e}, Val Acc: {val_acc:.4f}, "
                    f"Best Acc: {self.best_val_acc:.4f}",
                )

        print(
            f"Training completed! Best validation accuracy: {self.best_val_acc:.4f}",
        )

        if self.cfg.training.logging.use_wandb:
            wandb.finish()

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "cfg": self.cfg,
        }

        filename = (
            "best_model.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
        )
        torch.save(checkpoint, filename)

        if self.cfg.training.logging.use_wandb and is_best:
            wandb.save(filename)
