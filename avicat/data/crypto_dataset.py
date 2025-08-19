import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CryptoTimeSeriesDataset(Dataset):
    """
    Dataset for crypto time series with automatic state discovery.
    The model will learn to cluster market regimes from the features.
    """

    def __init__(
        self,
        data_path: str,
        transform: Optional[callable] = None,
        augment: bool = False,
        noise_level: float = 0.01,
    ):
        """
        Args:
            data_path: Path to the .npz file containing sequences
            transform: Optional transform to apply to sequences
            augment: Whether to apply data augmentation
            noise_level: Noise level for augmentation
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.augment = augment
        self.noise_level = noise_level

        # Load data - allow pickle for datetime objects
        data = np.load(self.data_path, allow_pickle=True)
        self.sequences = data["sequences"].astype(np.float32)

        # Handle timestamps if present
        if "timestamps" in data:
            timestamps_raw = data["timestamps"]
            # Convert to string if needed for safety
            if timestamps_raw.dtype == object:
                self.timestamps = [str(ts) for ts in timestamps_raw]
            else:
                self.timestamps = timestamps_raw
        else:
            self.timestamps = None

        # Since we're doing unsupervised clustering, we don't have true labels
        # The model will discover states automatically
        self.n_samples = len(self.sequences)

        print(
            f"Loaded {self.n_samples} sequences of shape {self.sequences[0].shape}",
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            sequence: Time series features [seq_len, n_features]
            pseudo_states: Placeholder states (zeros) since we're unsupervised
        """
        sequence = self.sequences[idx].copy()

        # Data augmentation
        if self.augment and np.random.random() > 0.5:
            sequence = self._augment_sequence(sequence)

        # Apply transform if provided
        if self.transform:
            sequence = self.transform(sequence)

        # Convert to tensor
        sequence = torch.from_numpy(sequence).float()

        # Create pseudo states (will be ignored during training but needed for compatibility)
        # The model will learn its own state assignments
        seq_len = sequence.shape[0]
        pseudo_states = torch.zeros(seq_len, dtype=torch.long)

        return sequence, pseudo_states

    def _augment_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Apply data augmentation to sequence"""
        augmented = sequence.copy()

        # Add small Gaussian noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, self.noise_level, sequence.shape)
            augmented += noise

        # Random scaling (slight)
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.95, 1.05)
            augmented *= scale

        # Dropout some features randomly
        if np.random.random() > 0.7:
            n_features = sequence.shape[1]
            dropout_mask = np.random.binomial(1, 0.9, n_features)
            augmented *= dropout_mask[np.newaxis, :]

        return augmented


class CryptoDataModule:
    """
    Data module to handle all crypto data loading and preprocessing.
    Integrates with the existing training pipeline.
    """

    def __init__(
        self,
        processed_data_dir: str = "processed_data",
        batch_size: int = 64,
        num_workers: int = 0,
        augment_train: bool = True,
        noise_level: float = 0.01,
    ):
        self.processed_data_dir = Path(processed_data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment_train = augment_train
        self.noise_level = noise_level

        # Load metadata
        with open(self.processed_data_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        self.n_features = self.metadata["num_features"]
        self.seq_length = self.metadata["sequence_length"]
        self.feature_names = self.metadata["feature_names"]

    def setup(self):
        """Setup datasets"""
        self.train_dataset = CryptoTimeSeriesDataset(
            self.processed_data_dir / "train_data.npz",
            augment=self.augment_train,
            noise_level=self.noise_level,
        )

        self.val_dataset = CryptoTimeSeriesDataset(
            self.processed_data_dir / "val_data.npz",
            augment=False,
        )

        self.test_dataset = CryptoTimeSeriesDataset(
            self.processed_data_dir / "test_data.npz",
            augment=False,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_data_config(self) -> Dict[str, Any]:
        """Get configuration for model initialization"""
        return {
            "obs_dim": self.n_features,
            "seq_length": self.seq_length,
            "num_train_samples": len(self.train_dataset),
            "num_val_samples": len(self.val_dataset),
            "feature_names": self.feature_names,
        }


def create_crypto_dataloaders(cfg):
    """
    Factory function to create crypto dataloaders.
    Compatible with existing training pipeline.
    """
    # Initialize data module
    data_module = CryptoDataModule(
        processed_data_dir=cfg.data.get(
            "processed_data_dir",
            "processed_data",
        ),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.get("num_workers", 0),
        augment_train=cfg.data.get("augment_train", True),
        noise_level=cfg.data.get("noise_level", 0.01),
    )

    # Setup datasets
    data_module.setup()

    # Get dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Update config with data dimensions
    data_config = data_module.get_data_config()

    # Print data statistics
    print("=" * 60)
    print("Crypto Dataset Statistics")
    print("=" * 60)
    print(f"Features: {data_config['obs_dim']}")
    print(f"Sequence Length: {data_config['seq_length']}")
    print(f"Train Samples: {data_config['num_train_samples']}")
    print(f"Val Samples: {data_config['num_val_samples']}")
    print(f"Batch Size: {cfg.training.batch_size}")
    print("=" * 60)

    return train_loader, val_loader, data_config


class MarketRegimeAnalyzer:
    """
    Post-training analysis tool to interpret discovered market regimes.
    """

    def __init__(self, model, data_loader, feature_names):
        self.model = model
        self.data_loader = data_loader
        self.feature_names = feature_names
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def extract_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract predicted states for all sequences"""
        self.model.eval()

        all_states = []
        all_features = []

        for batch_x, _ in self.data_loader:
            batch_x = batch_x.to(self.device)

            # Get predicted states
            _, _, pred_states = self.model.compute_elbo(
                batch_x,
                return_components=True,
                return_states=True,
            )

            all_states.append(pred_states.cpu().numpy())
            all_features.append(batch_x.cpu().numpy())

        all_states = np.concatenate(all_states, axis=0)
        all_features = np.concatenate(all_features, axis=0)

        return all_states, all_features

    def analyze_regimes(
        self,
        states: np.ndarray,
        features: np.ndarray,
    ) -> Dict:
        """Analyze characteristics of discovered regimes"""
        n_states = self.model.cfg.model.num_states
        regime_stats = {}

        # Flatten states for analysis
        states_flat = states.flatten()

        # Reshape features to (total_timesteps, n_features)
        batch_size, seq_len, n_features = features.shape
        features_flat = features.reshape(-1, n_features)

        for state_id in range(n_states):
            # Find all time points assigned to this state
            mask = states_flat == state_id

            if mask.sum() == 0:
                continue

            # Extract features for this state
            state_features = features_flat[
                mask
            ]  # Now 2D: (n_samples, n_features)

            # Calculate statistics
            regime_stats[f"State_{state_id}"] = {
                "count": int(mask.sum()),
                "percentage": float(mask.sum() / len(states_flat) * 100),
                "mean_features": {
                    name: float(state_features[:, i].mean())
                    for i, name in enumerate(
                        self.feature_names[:5],
                    )  # Top 5 features
                },
                "std_features": {
                    name: float(state_features[:, i].std())
                    for i, name in enumerate(self.feature_names[:5])
                },
            }

        return regime_stats

    def get_state_transitions(self, states: np.ndarray) -> np.ndarray:
        """Calculate state transition matrix"""
        n_states = self.model.cfg.model.num_states
        transition_matrix = np.zeros((n_states, n_states))

        for seq in states:
            for t in range(len(seq) - 1):
                transition_matrix[seq[t], seq[t + 1]] += 1

        # Normalize rows
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums

        return transition_matrix


if __name__ == "__main__":
    # Example usage
    from omegaconf import OmegaConf

    # Create a sample config
    cfg = OmegaConf.create(
        {
            "data": {
                "processed_data_dir": "processed_data",
                "num_workers": 0,
                "augment_train": True,
                "noise_level": 0.01,
            },
            "training": {"batch_size": 64},
        },
    )

    # Create dataloaders
    train_loader, val_loader, data_config = create_crypto_dataloaders(cfg)

    # Sample batch
    sample_batch = next(iter(train_loader))
    x, pseudo_states = sample_batch
    print(f"\nSample batch shape: {x.shape}")
    print(f"Features range: [{x.min():.3f}, {x.max():.3f}]")
