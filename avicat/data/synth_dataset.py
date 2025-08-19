import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticHMNDataset(Dataset):
    """
    Generates a synthetic dataset based on a Hidden Markov Model (HMM)
    with discrete states. Now returns both observations and true states.
    """

    def __init__(
        self,
        n_samples,
        n_features,
        n_states,
        sequence_length,
        transition_bias=0.9,
    ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_states = n_states
        self.sequence_length = sequence_length
        self.transition_bias = transition_bias
        # Generate both the observed data and the true latent states
        self.data, self.states = self._generate_data()

    def _generate_data(self):
        """Generates sequences and their corresponding true state labels."""
        # Emission means for each state
        emission_means = torch.randn(self.n_states, self.n_features)

        # Transition matrix with a bias towards self-transition
        transition_matrix = torch.full(
            (self.n_states, self.n_states),
            (1 - self.transition_bias) / (self.n_states - 1),
        )
        for i in range(self.n_states):
            transition_matrix[i, i] = self.transition_bias

        all_sequences = []
        all_states = []

        for _ in range(self.n_samples):
            states = torch.zeros(self.sequence_length, dtype=torch.long)
            sequence = torch.zeros(self.sequence_length, self.n_features)

            # Initial state
            states[0] = torch.randint(0, self.n_states, (1,))
            sequence[0] = torch.normal(mean=emission_means[states[0]], std=1.0)

            for t in range(1, self.sequence_length):
                # Transition to the next state
                states[t] = torch.multinomial(
                    transition_matrix[states[t - 1]],
                    1,
                ).squeeze()
                # Emit observation
                sequence[t] = torch.normal(
                    mean=emission_means[states[t]],
                    std=1.0,
                )

            all_sequences.append(sequence)
            all_states.append(states)

        return torch.stack(all_sequences), torch.stack(all_states)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """Returns a tuple of (data, true_states) for a given index."""
        return self.data[idx], self.states[idx]


def create_dataloaders(cfg):
    """Creates training and validation dataloaders."""
    dataset = SyntheticHMNDataset(
        n_samples=cfg.data.num_sequences_train + 200,
        n_features=cfg.data.obs_dim,
        n_states=cfg.data.num_true_states,
        sequence_length=cfg.data.seq_length,
        transition_bias=0.9,
    )

    # Split dataset into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    return train_loader, val_loader
