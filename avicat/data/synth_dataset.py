import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticHMNDataset(Dataset):
    """
    Generates a synthetic dataset based on a Hidden Markov Model (HMM)
    with discrete states. Configurable difficulty through various parameters.
    """

    def __init__(
        self,
        n_samples,
        n_features,
        n_states,
        sequence_length,
        transition_bias=0.9,
        emission_overlap=0.7,  # NEW: Control state overlap (0=no overlap, 1=complete overlap)
        emission_noise_scale=1.5,  # NEW: Increase observation noise
        state_persistence_prob=0.15,  # NEW: Probability of random state switches
        rare_state_prob=0.1,  # NEW: Make some states rare
    ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_states = n_states
        self.sequence_length = sequence_length
        self.transition_bias = transition_bias
        self.emission_overlap = emission_overlap
        self.emission_noise_scale = emission_noise_scale
        self.state_persistence_prob = state_persistence_prob
        self.rare_state_prob = rare_state_prob

        # Generate both the observed data and the true latent states
        self.data, self.states = self._generate_data()

    def _generate_data(self):
        """Generates sequences and their corresponding true state labels with configurable difficulty."""

        # Create emission means with controlled overlap
        # Instead of random means, create them in a structured way with overlap
        emission_means = torch.zeros(self.n_states, self.n_features)

        # Create base patterns for each state
        for i in range(self.n_states):
            # Each state has a base pattern
            base_pattern = torch.randn(self.n_features) * (
                2.0 - self.emission_overlap
            )

            # Add shared components for overlap
            shared_pattern = (
                torch.randn(self.n_features) * self.emission_overlap
            )

            emission_means[i] = base_pattern + shared_pattern

            # Add some structured correlation between adjacent states
            if i > 0:
                emission_means[i] += (
                    emission_means[i - 1] * self.emission_overlap * 0.5
                )

        # Normalize to control magnitude
        emission_means = emission_means / (emission_means.std() + 1e-8) * 2.0

        # Create transition matrix with more complex dynamics
        transition_matrix = torch.full(
            (self.n_states, self.n_states),
            (1 - self.transition_bias) / (self.n_states - 1),
        )

        # Add self-transition bias
        for i in range(self.n_states):
            transition_matrix[i, i] = self.transition_bias

        # Make some states harder to reach (rare states)
        if self.n_states > 2:
            rare_states = torch.randperm(self.n_states)[
                : max(1, self.n_states // 3)
            ]
            for rare_state in rare_states:
                transition_matrix[:, rare_state] *= self.rare_state_prob
                # Renormalize
                transition_matrix = transition_matrix / transition_matrix.sum(
                    dim=1,
                    keepdim=True,
                )

        # Add some noise to transition matrix for irregularity
        transition_noise = torch.rand_like(transition_matrix) * 0.1
        transition_matrix = transition_matrix + transition_noise
        transition_matrix = transition_matrix / transition_matrix.sum(
            dim=1,
            keepdim=True,
        )

        all_sequences = []
        all_states = []

        for _ in range(self.n_samples):
            states = torch.zeros(self.sequence_length, dtype=torch.long)
            sequence = torch.zeros(self.sequence_length, self.n_features)

            # Initial state - bias towards common states
            if torch.rand(1) < 0.7:
                states[0] = torch.randint(0, min(2, self.n_states), (1,))
            else:
                states[0] = torch.randint(0, self.n_states, (1,))

            # Generate observation with variable noise
            noise_std = self.emission_noise_scale * (
                0.5 + torch.rand(1).item()
            )
            sequence[0] = torch.normal(
                mean=emission_means[states[0]],
                std=noise_std,
            )

            for t in range(1, self.sequence_length):
                # Add random state switches to make it harder
                if torch.rand(1) < self.state_persistence_prob:
                    # Random jump to any state
                    states[t] = torch.randint(0, self.n_states, (1,))
                else:
                    # Follow transition matrix
                    states[t] = torch.multinomial(
                        transition_matrix[states[t - 1]],
                        1,
                    ).squeeze()

                # Emit observation with time-varying noise
                noise_std = self.emission_noise_scale * (
                    0.5 + 0.5 * torch.sin(torch.tensor(t * 0.1)).abs()
                )

                # Add observation
                obs_mean = emission_means[states[t]]

                # Add some temporal correlation in observations
                if t > 0:
                    temporal_correlation = 0.2 * self.emission_overlap
                    obs_mean = (
                        1 - temporal_correlation
                    ) * obs_mean + temporal_correlation * sequence[t - 1]

                sequence[t] = torch.normal(mean=obs_mean, std=noise_std)

                # Add occasional outliers
                if torch.rand(1) < 0.05:  # 5% outlier probability
                    sequence[t] += torch.randn_like(sequence[t]) * 3.0

            all_sequences.append(sequence)
            all_states.append(states)

        return torch.stack(all_sequences), torch.stack(all_states)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """Returns a tuple of (data, true_states) for a given index."""
        return self.data[idx], self.states[idx]


def create_dataloaders(cfg):
    """Creates training and validation dataloaders with harder synthetic data."""

    # Extract difficulty parameters from config or use challenging defaults
    difficulty_params = {
        "emission_overlap": cfg.data.get(
            "emission_overlap",
            0.7,
        ),  # High overlap
        "emission_noise_scale": cfg.data.get(
            "emission_noise_scale",
            1.5,
        ),  # High noise
        "state_persistence_prob": cfg.data.get(
            "state_persistence_prob",
            0.15,
        ),  # Random jumps
        "rare_state_prob": cfg.data.get(
            "rare_state_prob",
            0.1,
        ),  # Unbalanced states
    }

    dataset = SyntheticHMNDataset(
        n_samples=cfg.data.num_sequences_train + cfg.data.num_sequences_val,
        n_features=cfg.data.obs_dim,
        n_states=cfg.data.num_true_states,
        sequence_length=cfg.data.seq_length,
        transition_bias=cfg.data.get(
            "transition_bias",
            0.7,
        ),  # Lower bias for more transitions
        **difficulty_params,
    )

    # Split dataset into training and validation
    train_size = cfg.data.num_sequences_train
    val_size = cfg.data.num_sequences_val

    # Use fixed split instead of random for reproducibility
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(
        dataset,
        range(train_size, train_size + val_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,  # Drop last incomplete batch for consistency
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    return train_loader, val_loader
