"""
Deep State-Space Model with Neural Categorical Latent Variables
PyTorch implementation with GPU support, synthetic dataset, and W&B logging
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb

# ============================================================================
# HYPERPARAMETERS
# ============================================================================


@dataclass
class ModelHParams:
    """Model hyperparameters"""

    num_states: int = 10  # K: number of discrete states
    state_dim: int = 64  # dimension of state embeddings
    obs_dim: int = 20  # dimension of observations

    # Transition network
    trans_hidden_dims: Tuple[int, ...] = (128, 128)
    trans_activation: str = "relu"
    trans_dropout: float = 0.1

    # Emission network
    emission_hidden_dims: Tuple[int, ...] = (128, 128)
    emission_activation: str = "relu"
    emission_dropout: float = 0.1
    emission_std: float = 0.1  # std for Gaussian emission

    # Inference network
    rnn_hidden_dim: int = 128
    rnn_num_layers: int = 2
    inference_hidden_dims: Tuple[int, ...] = (128,)

    # Gumbel-Softmax
    temperature: float = 1.0
    min_temperature: float = 0.5
    temperature_anneal_rate: float = 0.99995


@dataclass
class DataHParams:
    """Synthetic dataset hyperparameters"""

    num_sequences: int = 1000
    seq_length: int = 100
    num_true_states: int = 5  # true number of states in synthetic data
    obs_dim: int = 20
    transition_prob_concentration: float = 10.0  # Dirichlet concentration
    emission_noise_std: float = 0.1
    seed: int = 42


@dataclass
class TrainingHParams:
    """Training hyperparameters"""

    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    grad_clip: float = 5.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 10
    val_interval: int = 5
    use_wandb: bool = True
    project_name: str = "dssm-categorical"
    run_name: Optional[str] = None


# ============================================================================
# SYNTHETIC DATASET
# ============================================================================


class SyntheticHMMDataset(Dataset):
    """Synthetic HMM dataset for testing the model"""

    def __init__(self, hparams: DataHParams, split="train"):
        self.hparams = hparams
        self.split = split

        # Set seed for reproducibility
        np.random.seed(hparams.seed if split == "train" else hparams.seed + 1)

        # Generate HMM parameters
        self.transition_matrix = self._generate_transition_matrix()
        self.emission_means = self._generate_emission_means()

        # Generate sequences
        n_seqs = (
            hparams.num_sequences
            if split == "train"
            else hparams.num_sequences // 5
        )
        self.sequences, self.states = self._generate_sequences(n_seqs)

    def _generate_transition_matrix(self):
        """Generate a transition matrix with strong diagonal (sticky states)"""
        K = self.hparams.num_true_states
        alpha = self.hparams.transition_prob_concentration

        # Create transition matrix with strong diagonal
        trans_mat = np.random.dirichlet(np.ones(K) * alpha / K, size=K)
        # Make states more sticky
        for i in range(K):
            trans_mat[i, i] += 2.0
            trans_mat[i] /= trans_mat[i].sum()

        return trans_mat

    def _generate_emission_means(self):
        """Generate distinct emission means for each state"""
        K = self.hparams.num_true_states
        D = self.hparams.obs_dim

        # Generate well-separated means
        means = np.random.randn(K, D) * 2.0
        # Make them more separated
        for i in range(K):
            means[i] += i * 0.5

        return means

    def _generate_sequences(self, num_sequences):
        """Generate sequences from the HMM"""
        sequences = []
        states = []

        for _ in range(num_sequences):
            seq, state_seq = self._generate_single_sequence()
            sequences.append(seq)
            states.append(state_seq)

        return sequences, states

    def _generate_single_sequence(self):
        """Generate a single sequence"""
        T = self.hparams.seq_length
        K = self.hparams.num_true_states

        # Initial state (uniform)
        state_seq = np.zeros(T, dtype=np.int64)
        state_seq[0] = np.random.choice(K)

        # Generate state sequence
        for t in range(1, T):
            prev_state = state_seq[t - 1]
            state_seq[t] = np.random.choice(
                K,
                p=self.transition_matrix[prev_state],
            )

        # Generate observations
        obs_seq = np.zeros((T, self.hparams.obs_dim))
        for t in range(T):
            mean = self.emission_means[state_seq[t]]
            obs_seq[t] = (
                mean
                + np.random.randn(self.hparams.obs_dim)
                * self.hparams.emission_noise_std
            )

        return obs_seq.astype(np.float32), state_seq

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(
            self.states[idx],
        )


# ============================================================================
# MODEL COMPONENTS
# ============================================================================


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture"""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims,
        activation="relu",
        dropout=0.1,
    ):
        super().__init__()

        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # Not the last layer
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "elu":
                    layers.append(nn.ELU())
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NeuralTransitionModel(nn.Module):
    """Neural network for modeling state transitions p(c_t | c_{t-1})"""

    def __init__(self, hparams: ModelHParams):
        super().__init__()
        self.hparams = hparams

        # State embeddings
        self.state_embeddings = nn.Embedding(
            hparams.num_states,
            hparams.state_dim,
        )

        # Transition network
        self.transition_net = MLP(
            hparams.state_dim,
            hparams.num_states,
            hparams.trans_hidden_dims,
            hparams.trans_activation,
            hparams.trans_dropout,
        )

    def forward(self, prev_state):
        """
        Args:
            prev_state: [batch, num_states] (soft state from Gumbel-Softmax)
        Returns:
            logits: [batch, num_states]
        """
        # Get weighted embedding
        state_emb = torch.matmul(prev_state, self.state_embeddings.weight)
        # Get transition logits
        logits = self.transition_net(state_emb)
        return logits


class EmissionModel(nn.Module):
    """Neural network for emission distribution p(x_t | c_t)"""

    def __init__(self, hparams: ModelHParams):
        super().__init__()
        self.hparams = hparams

        # State embeddings (shared with transition model in main model)
        self.state_embeddings = nn.Embedding(
            hparams.num_states,
            hparams.state_dim,
        )

        # Emission network for mean
        self.emission_net = MLP(
            hparams.state_dim,
            hparams.obs_dim,
            hparams.emission_hidden_dims,
            hparams.emission_activation,
            hparams.emission_dropout,
        )

        # Fixed or learnable std
        self.log_std = nn.Parameter(
            torch.log(torch.tensor(hparams.emission_std)),
        )

    def forward(self, state):
        """
        Args:
            state: [batch, num_states] (soft state from Gumbel-Softmax)
        Returns:
            mean: [batch, obs_dim]
            std: [batch, obs_dim]
        """
        # Get weighted embedding
        state_emb = torch.matmul(state, self.state_embeddings.weight)
        # Get emission mean
        mean = self.emission_net(state_emb)
        std = self.log_std.exp().expand_as(mean)
        return mean, std


class InferenceNetwork(nn.Module):
    """Structured inference network q(C|X) using RNNs"""

    def __init__(self, hparams: ModelHParams):
        super().__init__()
        self.hparams = hparams

        # Bidirectional RNN for context
        self.bi_rnn = nn.GRU(
            hparams.obs_dim,
            hparams.rnn_hidden_dim,
            hparams.rnn_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=hparams.trans_dropout if hparams.rnn_num_layers > 1 else 0,
        )

        # State embeddings (will be shared)
        self.state_embeddings = nn.Embedding(
            hparams.num_states,
            hparams.state_dim,
        )

        # Network for q(c_t | c_{t-1}, h_t)
        self.inference_net = MLP(
            hparams.state_dim
            + 2 * hparams.rnn_hidden_dim,  # prev_state + context
            hparams.num_states,
            hparams.inference_hidden_dims,
            hparams.trans_activation,
            hparams.trans_dropout,
        )

    def forward(self, x, prev_state=None):
        """
        Args:
            x: [batch, seq_len, obs_dim] or [batch, obs_dim] for single step
            prev_state: [batch, num_states] (soft state) or None for first timestep
        Returns:
            logits: [batch, num_states] or [batch, seq_len, num_states]
        """
        batch_size = x.size(0)

        # Get context from bidirectional RNN
        if x.dim() == 3:  # Full sequence
            contexts, _ = self.bi_rnn(x)  # [batch, seq_len, 2*hidden]
            return self._get_all_logits(contexts)
        else:  # Single timestep
            # This is used during sequential processing
            context = x  # Assume x is already the context vector
            return self._get_single_logits(context, prev_state)

    def get_contexts(self, x):
        """Get context vectors from BiRNN"""
        contexts, _ = self.bi_rnn(x)
        return contexts

    def _get_all_logits(self, contexts):
        """Get logits for all timesteps (used for initialization)"""
        # For simplicity, just use contexts without previous states
        # In full implementation, would process sequentially
        return self.inference_net(contexts)

    def _get_single_logits(self, context, prev_state):
        """Get logits for single timestep"""
        batch_size = context.size(0)

        if prev_state is None:
            # First timestep - use zero state
            prev_state = torch.zeros(batch_size, self.hparams.num_states).to(
                context.device,
            )
            prev_state[:, 0] = 1.0  # Start with first state

        # Get weighted embedding of previous state
        prev_emb = torch.matmul(prev_state, self.state_embeddings.weight)

        # Concatenate with context
        combined = torch.cat([prev_emb, context], dim=-1)

        # Get logits
        logits = self.inference_net(combined)
        return logits


# ============================================================================
# MAIN MODEL
# ============================================================================


class DeepStateSpaceModel(nn.Module):
    """Deep State-Space Model with Neural Categorical Latents"""

    def __init__(self, hparams: ModelHParams):
        super().__init__()
        self.hparams = hparams

        # Shared state embeddings
        self.state_embeddings = nn.Embedding(
            hparams.num_states,
            hparams.state_dim,
        )

        # Model components
        self.transition_model = NeuralTransitionModel(hparams)
        self.emission_model = EmissionModel(hparams)
        self.inference_network = InferenceNetwork(hparams)

        # Share embeddings
        self.transition_model.state_embeddings = self.state_embeddings
        self.emission_model.state_embeddings = self.state_embeddings
        self.inference_network.state_embeddings = self.state_embeddings

        # Temperature for Gumbel-Softmax
        self.temperature = hparams.temperature

    def gumbel_softmax_sample(self, logits, temperature=1.0, hard=False):
        """Sample from Gumbel-Softmax distribution"""
        gumbel_noise = -torch.log(
            -torch.log(torch.rand_like(logits) + 1e-20) + 1e-20,
        )
        y = F.softmax((logits + gumbel_noise) / temperature, dim=-1)

        if hard:
            # Straight-through estimator
            y_hard = torch.zeros_like(y)
            y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
            y = (y_hard - y).detach() + y

        return y

    def compute_elbo(self, x, return_components=False, return_states=False):
        """
        Compute ELBO for a batch of sequences
        Args:
            x: [batch, seq_len, obs_dim]
            return_components: whether to return individual loss components
            return_states: whether to return predicted states
        Returns:
            elbo: scalar
            components: dict (if return_components=True)
            predicted_states: [batch, seq_len] (if return_states=True)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Get context vectors from BiRNN
        contexts = self.inference_network.get_contexts(
            x,
        )  # [batch, seq_len, 2*hidden]

        # Initialize losses
        recon_loss = 0.0
        kl_loss = 0.0

        # Store predicted states if needed
        if return_states:
            predicted_states = torch.zeros(
                batch_size,
                seq_len,
                dtype=torch.long,
            ).to(device)

        # Initial state (uniform prior)
        prev_state = (
            torch.ones(batch_size, self.hparams.num_states).to(device)
            / self.hparams.num_states
        )

        # Process sequence
        for t in range(seq_len):
            # Get posterior logits q(c_t | c_{t-1}, h_t)
            if t == 0:
                q_logits = self.inference_network._get_single_logits(
                    contexts[:, t],
                    None,
                )
            else:
                q_logits = self.inference_network._get_single_logits(
                    contexts[:, t],
                    prev_state,
                )

            q_probs = F.softmax(q_logits, dim=-1)

            # Store predicted state (argmax of posterior)
            if return_states:
                predicted_states[:, t] = q_probs.argmax(dim=-1)

            # Get prior logits p(c_t | c_{t-1})
            if t == 0:
                # Uniform prior for first timestep
                p_probs = torch.ones_like(q_probs) / self.hparams.num_states
            else:
                p_logits = self.transition_model(prev_state)
                p_probs = F.softmax(p_logits, dim=-1)

            # Compute KL divergence (analytical)
            kl_t = torch.sum(
                q_probs
                * (torch.log(q_probs + 1e-10) - torch.log(p_probs + 1e-10)),
                dim=-1,
            )
            kl_loss += kl_t.mean()

            # Sample state using Gumbel-Softmax
            curr_state = self.gumbel_softmax_sample(q_logits, self.temperature)

            # Compute reconstruction loss
            x_mean, x_std = self.emission_model(curr_state)
            x_t = x[:, t]

            # Gaussian log-likelihood
            log_prob = -0.5 * (
                torch.sum(((x_t - x_mean) / x_std) ** 2, dim=-1)
                + torch.sum(torch.log(2 * np.pi * x_std**2), dim=-1)
            )
            recon_loss += log_prob.mean()

            # Update previous state
            prev_state = curr_state

        # ELBO = E[log p(x|z)] - KL[q(z|x) || p(z)]
        elbo = recon_loss - kl_loss

        results = [elbo]

        if return_components:
            components = {
                "reconstruction_loss": recon_loss,
                "kl_loss": kl_loss,
                "elbo": elbo,
            }
            results.append(components)

        if return_states:
            results.append(predicted_states)

        return tuple(results) if len(results) > 1 else results[0]

    def anneal_temperature(self):
        """Anneal the temperature for Gumbel-Softmax"""
        self.temperature = max(
            self.hparams.min_temperature,
            self.temperature * self.hparams.temperature_anneal_rate,
        )


# ============================================================================
# TRAINER
# ============================================================================


def compute_clustering_accuracy(pred_states, true_states):
    """
    Compute clustering accuracy using Hungarian algorithm for alignment
    Args:
        pred_states: [batch, seq_len] predicted state assignments
        true_states: [batch, seq_len] true state assignments
    Returns:
        accuracy: float
    """

    # Flatten
    pred_flat = pred_states.cpu().numpy().flatten()
    true_flat = true_states.cpu().numpy().flatten()

    # Get unique labels
    pred_labels = np.unique(pred_flat)
    true_labels = np.unique(true_flat)

    # Create confusion matrix
    n_pred = len(pred_labels)
    n_true = len(true_labels)
    confusion = np.zeros((n_pred, n_true))

    for i, p_label in enumerate(pred_labels):
        for j, t_label in enumerate(true_labels):
            confusion[i, j] = np.sum(
                (pred_flat == p_label) & (true_flat == t_label),
            )

    # Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(-confusion)

    # Compute accuracy
    correct = confusion[row_ind, col_ind].sum()
    total = len(pred_flat)
    accuracy = correct / total

    return accuracy


class Trainer:
    """Trainer for the Deep State-Space Model"""

    def __init__(
        self,
        model: DeepStateSpaceModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        hparams: TrainingHParams,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.hparams = hparams

        # Move model to device
        self.model = self.model.to(hparams.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=hparams.learning_rate,
        )

        # Initialize wandb
        if hparams.use_wandb:
            wandb.init(
                project=hparams.project_name,
                name=hparams.run_name,
                config={
                    "model": model.hparams.__dict__,
                    "training": hparams.__dict__,
                },
            )

        self.global_step = 0
        self.best_val_elbo = -float("inf")
        self.best_val_acc = 0.0

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        epoch_accuracies = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (x, true_states) in enumerate(pbar):
            x = x.to(self.hparams.device)
            true_states = true_states.to(self.hparams.device)

            # Forward pass
            elbo, components, pred_states = self.model.compute_elbo(
                x,
                return_components=True,
                return_states=True,
            )
            loss = -elbo  # Minimize negative ELBO

            # Compute accuracy
            accuracy = compute_clustering_accuracy(pred_states, true_states)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.hparams.grad_clip,
            )

            self.optimizer.step()

            # Anneal temperature
            self.model.anneal_temperature()

            # Logging
            epoch_losses.append(loss.item())
            epoch_accuracies.append(accuracy)
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "acc": accuracy,
                    "recon": components["reconstruction_loss"].item(),
                    "kl": components["kl_loss"].item(),
                    "temp": self.model.temperature,
                },
            )

            # Log to wandb
            if (
                self.hparams.use_wandb
                and self.global_step % self.hparams.log_interval == 0
            ):
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/elbo": -loss.item(),
                        "train/accuracy": accuracy,
                        "train/reconstruction_loss": components[
                            "reconstruction_loss"
                        ].item(),
                        "train/kl_loss": components["kl_loss"].item(),
                        "train/temperature": self.model.temperature,
                        "step": self.global_step,
                    },
                )

        avg_accuracy = np.mean(epoch_accuracies)
        return np.mean(epoch_losses), avg_accuracy

    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        val_losses = []
        val_recon_losses = []
        val_kl_losses = []
        val_accuracies = []

        with torch.no_grad():
            for x, true_states in self.val_loader:
                x = x.to(self.hparams.device)
                true_states = true_states.to(self.hparams.device)

                elbo, components, pred_states = self.model.compute_elbo(
                    x,
                    return_components=True,
                    return_states=True,
                )
                loss = -elbo

                # Compute accuracy
                accuracy = compute_clustering_accuracy(
                    pred_states,
                    true_states,
                )

                val_losses.append(loss.item())
                val_recon_losses.append(
                    components["reconstruction_loss"].item(),
                )
                val_kl_losses.append(components["kl_loss"].item())
                val_accuracies.append(accuracy)

        avg_val_loss = np.mean(val_losses)
        avg_val_elbo = -avg_val_loss
        avg_val_recon = np.mean(val_recon_losses)
        avg_val_kl = np.mean(val_kl_losses)
        avg_val_acc = np.mean(val_accuracies)

        # Check if best model (based on accuracy)
        if avg_val_acc > self.best_val_acc:
            self.best_val_acc = avg_val_acc
            self.best_val_elbo = avg_val_elbo
            self.save_checkpoint(epoch, is_best=True)

        # Log to wandb
        if self.hparams.use_wandb:
            wandb.log(
                {
                    "val/loss": avg_val_loss,
                    "val/elbo": avg_val_elbo,
                    "val/accuracy": avg_val_acc,
                    "val/reconstruction_loss": avg_val_recon,
                    "val/kl_loss": avg_val_kl,
                    "val/best_accuracy": self.best_val_acc,
                    "val/best_elbo": self.best_val_elbo,
                    "epoch": epoch,
                },
            )

        return avg_val_loss, avg_val_elbo, avg_val_acc

    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.hparams.device}")
        print(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}",
        )

        for epoch in range(1, self.hparams.num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            if epoch % self.hparams.val_interval == 0:
                val_loss, val_elbo, val_acc = self.validate(epoch)
                print(
                    f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val ELBO: {val_elbo:.4f}, Val Acc: {val_acc:.4f}, "
                    f"Best Acc: {self.best_val_acc:.4f}",
                )

            # Save periodic checkpoint
            if epoch % 20 == 0:
                self.save_checkpoint(epoch, is_best=False)

        # Final validation
        val_loss, val_elbo, val_acc = self.validate(self.hparams.num_epochs)
        print(
            f"Final Validation - Loss: {val_loss:.4f}, ELBO: {val_elbo:.4f}, Accuracy: {val_acc:.4f}",
        )

        if self.hparams.use_wandb:
            wandb.finish()

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_elbo": self.best_val_elbo,
            "best_val_acc": self.best_val_acc,
            "temperature": self.model.temperature,
            "global_step": self.global_step,
        }

        filename = (
            "best_model.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
        )
        torch.save(checkpoint, filename)

        if self.hparams.use_wandb and is_best:
            wandb.save(filename)


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Initialize hyperparameters
    model_hparams = ModelHParams()
    data_hparams = DataHParams()
    train_hparams = TrainingHParams()

    print("Creating synthetic dataset...")
    # Create datasets
    train_dataset = SyntheticHMMDataset(data_hparams, split="train")
    val_dataset = SyntheticHMMDataset(data_hparams, split="val")

    print(
        f"Train sequences: {len(train_dataset)}, Val sequences: {len(val_dataset)}",
    )
    print(f"Sequence shape: {train_dataset[0][0].shape}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_hparams.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_hparams.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    print("Initializing model...")
    # Create model
    model = DeepStateSpaceModel(model_hparams)

    print("Starting training...")
    # Create trainer and train
    trainer = Trainer(model, train_loader, val_loader, train_hparams)
    trainer.train()

    print("Training completed!")


if __name__ == "__main__":
    main()
