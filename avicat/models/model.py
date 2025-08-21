import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class MLP(nn.Module):
    """Multi-layer perceptron"""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims,
        activation="relu",
        dropout=0.0,
    ):
        super().__init__()

        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []

        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i + 1])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)

            if i < len(dims) - 2:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "elu":
                    layers.append(nn.ELU())

                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NeuralTransitionModel(nn.Module):
    """Neural transition model p(c_t | c_{t-1})"""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg.model

        self.state_embeddings = nn.Embedding(
            self.cfg.num_states,
            self.cfg.state_dim,
        )
        nn.init.normal_(self.state_embeddings.weight, 0, 0.1)

        self.transition_net = MLP(
            self.cfg.state_dim,
            self.cfg.num_states,
            self.cfg.transition.hidden_dims,
            self.cfg.transition.activation,
            self.cfg.transition.dropout,
        )

        # Initialize for self-transitions
        with torch.no_grad():
            self.transition_net.net[-1].bias.data = (
                torch.eye(self.cfg.num_states).sum(0) * 0.5
            )

    def forward(self, prev_state):
        state_emb = torch.matmul(prev_state, self.state_embeddings.weight)
        return self.transition_net(state_emb)


class EmissionModel(nn.Module):
    """Emission model p(x_t | c_t)"""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg.model

        self.state_embeddings = nn.Embedding(
            self.cfg.num_states,
            self.cfg.state_dim,
        )

        self.emission_net = MLP(
            self.cfg.state_dim,
            self.cfg.obs_dim,
            self.cfg.emission.hidden_dims,
            self.cfg.emission.activation,
            self.cfg.emission.dropout,
        )

        self.log_std = nn.Parameter(
            torch.log(torch.tensor(self.cfg.emission.std_init)),
        )

    def forward(self, state):
        state_emb = torch.matmul(state, self.state_embeddings.weight)
        mean = self.emission_net(state_emb)
        std = torch.clamp(
            self.log_std.exp(),
            min=self.cfg.emission.std_min,
            max=self.cfg.emission.std_max,
        ).expand_as(mean)
        return mean, std


class InferenceNetwork(nn.Module):
    """Improved Inference network q(C|X) with GRU for temporal modeling"""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg.model

        # Bidirectional RNN for context extraction
        self.bi_rnn = nn.GRU(
            self.cfg.obs_dim,
            self.cfg.inference.rnn_hidden_dim,
            self.cfg.inference.rnn_num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0,
        )

        self.context_norm = nn.LayerNorm(self.cfg.inference.rnn_hidden_dim)

        # State embeddings
        self.state_embeddings = nn.Embedding(
            self.cfg.num_states,
            self.cfg.state_dim,
        )
        nn.init.normal_(self.state_embeddings.weight, 0, 0.1)

        # NEW: GRU for inference (unidirectional)
        # Input: concatenation of context and previous state embedding
        inference_input_dim = (
            self.cfg.inference.rnn_hidden_dim + self.cfg.state_dim
        )
        self.inference_gru = nn.GRU(
            inference_input_dim,
            self.cfg.inference.rnn_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # Output projection from GRU hidden to logits
        self.output_projection = nn.Sequential(
            nn.Linear(
                self.cfg.inference.rnn_hidden_dim,
                self.cfg.inference.hidden_dims[0],
            ),
            nn.Tanh(),
            nn.Linear(self.cfg.inference.hidden_dims[0], self.cfg.num_states),
        )

        # Initialize output projection
        for m in self.output_projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def get_contexts(self, x):
        """Extract bidirectional context from observations"""
        contexts, _ = self.bi_rnn(x)
        return self.context_norm(contexts)

    def forward_sequence(self, contexts, initial_state=None):
        """
        Process full sequence with GRU
        Args:
            contexts: [batch, seq_len, 2*rnn_hidden_dim] from BiRNN
            initial_state: [batch, num_states] initial state distribution
        Returns:
            logits: [batch, seq_len, num_states]
        """
        batch_size, seq_len, _ = contexts.shape
        device = contexts.device

        # Initialize hidden state for inference GRU
        h_inference = torch.zeros(
            1,
            batch_size,
            self.cfg.inference.rnn_hidden_dim,
        ).to(device)

        # Initialize previous state
        if initial_state is None:
            prev_state = torch.ones(batch_size, self.cfg.num_states).to(device)
            prev_state = prev_state / self.cfg.num_states
        else:
            prev_state = initial_state

        all_logits = []

        for t in range(seq_len):
            # Get state embedding for previous state
            prev_state_emb = torch.matmul(
                prev_state,
                self.state_embeddings.weight,
            )

            # Concatenate context and previous state embedding
            inference_input = torch.cat(
                [
                    contexts[:, t, :],  # [batch, 2*rnn_hidden]
                    prev_state_emb,  # [batch, state_dim]
                ],
                dim=-1,
            ).unsqueeze(1)  # [batch, 1, input_dim]

            # Pass through inference GRU
            gru_out, h_inference = self.inference_gru(
                inference_input,
                h_inference,
            )
            # gru_out: [batch, 1, rnn_hidden_dim]

            # Project to logits
            logits = self.output_projection(
                gru_out.squeeze(1),
            )  # [batch, num_states]
            all_logits.append(logits)

            # Update previous state (using softmax of current logits)
            prev_state = F.softmax(
                torch.clamp(logits, min=-10, max=10),
                dim=-1,
            )

        return torch.stack(all_logits, dim=1)  # [batch, seq_len, num_states]


class DeepStateSpaceModel(nn.Module):
    """Deep State-Space Model with improved inference network"""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # Shared state embeddings
        self.state_embeddings = nn.Embedding(
            cfg.model.num_states,
            cfg.model.state_dim,
        )
        nn.init.normal_(self.state_embeddings.weight, 0, 0.1)

        # Model components
        self.transition_model = NeuralTransitionModel(cfg)
        self.emission_model = EmissionModel(cfg)
        self.inference_network = InferenceNetwork(cfg)

        # Share embeddings
        self.transition_model.state_embeddings = self.state_embeddings
        self.emission_model.state_embeddings = self.state_embeddings
        self.inference_network.state_embeddings = self.state_embeddings

        # Annealing parameters
        self.temperature = cfg.model.gumbel.temperature_init
        self.kl_weight = cfg.model.regularization.kl_weight_init

    def gumbel_softmax_sample(self, logits, temperature=1.0):
        eps = 1e-10
        U = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
        y = (logits + gumbel_noise) / temperature
        return F.softmax(y, dim=-1)

    def compute_elbo(self, x, return_components=False, return_states=False):
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Get bidirectional contexts
        contexts = self.inference_network.get_contexts(
            x,
        )  # [batch, seq_len, 2*hidden]

        # Initialize losses
        recon_loss = 0.0
        kl_loss = 0.0

        if return_states:
            predicted_states = torch.zeros(
                batch_size,
                seq_len,
                dtype=torch.long,
            ).to(device)

        # Initialize for sequential processing
        prev_state_soft = (
            torch.ones(batch_size, self.cfg.model.num_states).to(device)
            / self.cfg.model.num_states
        )
        h_inference = torch.zeros(
            1,
            batch_size,
            self.cfg.model.inference.rnn_hidden_dim,
        ).to(device)

        for t in range(seq_len):
            # Get posterior using improved inference network with GRU
            prev_state_emb = torch.matmul(
                prev_state_soft,
                self.state_embeddings.weight,
            )

            # Prepare input for inference GRU
            inference_input = torch.cat(
                [
                    contexts[:, t, :],  # BiRNN context
                    prev_state_emb,  # Previous state embedding
                ],
                dim=-1,
            ).unsqueeze(1)

            # Pass through inference GRU
            gru_out, h_inference = self.inference_network.inference_gru(
                inference_input,
                h_inference,
            )
            q_logits = self.inference_network.output_projection(
                gru_out.squeeze(1),
            )

            # Clip and normalize posterior
            q_logits = torch.clamp(q_logits, min=-10, max=10)
            q_probs = F.softmax(q_logits, dim=-1)
            q_probs = q_probs + 1e-10
            q_probs = q_probs / q_probs.sum(dim=-1, keepdim=True)

            if return_states:
                predicted_states[:, t] = q_probs.argmax(dim=-1)

            # Get prior p(c_t | c_{t-1})
            if t == 0:
                p_probs = torch.ones_like(q_probs) / self.cfg.model.num_states
            else:
                p_logits = self.transition_model(prev_state_soft)
                p_logits = torch.clamp(p_logits, min=-10, max=10)
                p_probs = F.softmax(p_logits, dim=-1)
                p_probs = p_probs + 1e-10
                p_probs = p_probs / p_probs.sum(dim=-1, keepdim=True)

            # KL divergence with annealing
            kl_t = torch.sum(
                q_probs * (torch.log(q_probs) - torch.log(p_probs)),
                dim=-1,
            )
            kl_loss += self.kl_weight * kl_t.mean()

            # Sample state using Gumbel-Softmax
            curr_state_soft = self.gumbel_softmax_sample(
                q_logits,
                self.temperature,
            )

            # Compute reconstruction loss
            x_mean, x_std = self.emission_model(curr_state_soft)
            x_t = x[:, t]

            diff = (x_t - x_mean) / (x_std + 1e-8)
            log_prob = -0.5 * (
                torch.sum(diff**2, dim=-1)
                + torch.sum(torch.log(2 * np.pi * (x_std**2 + 1e-8)), dim=-1)
            )
            recon_loss += log_prob.mean()

            # Update previous state
            prev_state_soft = curr_state_soft

        # Normalize by sequence length
        recon_loss = recon_loss / seq_len
        kl_loss = kl_loss / seq_len

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
        self.temperature = max(
            self.cfg.model.gumbel.temperature_min,
            self.temperature * self.cfg.model.gumbel.temperature_anneal_rate,
        )

    def anneal_kl_weight(self):
        self.kl_weight = min(
            self.cfg.model.regularization.kl_weight_max,
            self.kl_weight * self.cfg.model.regularization.kl_anneal_rate,
        )
