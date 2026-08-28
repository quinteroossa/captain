"""Actor-Critic network for PPO on CAPTAIN's cell-selection problem.

Architecture
------------
Shared trunk  : MLP — takes (n_features, n_cells), outputs per-cell embeddings
Policy head   : linear → per-cell scores → Plackett-Luce distribution (K cells sampled without replacement)
Value head    : attention-pooled per-cell embeddings → linear → scalar V(s)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCriticCellNN(nn.Module):
    """Shared-trunk actor-critic for CAPTAIN's cell-selection action space.

    Args:
        input_dim:  Number of features per cell (n_features from FeatureExtractor).
        hidden_dim: Hidden layer size(s) for the shared trunk.
        activation: Activation function ('relu', 'tanh', 'gelu').
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | list[int] = 64,
        activation: str = "relu",
    ):
        super().__init__()

        activation_classes = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
        act_cls = activation_classes[activation]

        hidden_dims = [hidden_dim] if isinstance(hidden_dim, int) else list(hidden_dim)

        trunk_layers = []
        current_dim = input_dim
        for h in hidden_dims:
            trunk_layers += [nn.Linear(current_dim, h), act_cls()]
            current_dim = h
        self.trunk = nn.Sequential(*trunk_layers)

        self.policy_head = nn.Linear(current_dim, 1)

        # Attention pooling: learned cell-importance weights for V(s) estimation.
        # Mean pooling weights all 58K cells equally — high-risk cells (EN/CR) get
        # diluted by the majority LC cells, making V(s) insensitive to risk state.
        self.attn_head  = nn.Linear(current_dim, 1)
        self.value_head = nn.Linear(current_dim, 1)

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        # Larger init: breaks gradient cancellation sooner.
        # Xavier gives σ≈0.18 for 64→1 — scores too similar across 58K cells at
        # init, causing selected/non-selected gradient terms to cancel.
        nn.init.normal_(self.policy_head.weight, 0, 1.0)
        nn.init.zeros_(self.policy_head.bias)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Shared trunk: (n_features, n_cells) → (n_cells, hidden_dim)."""
        return self.trunk(x.t())

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """Per-cell policy scores (logits): (n_features, n_cells) → (n_cells,)."""
        return self.policy_head(self._embed(x)).squeeze(-1)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """State value V(s) via attention pooling: (n_features, n_cells) → scalar."""
        emb    = self._embed(x)                              # (n_cells, hidden_dim)
        attn   = torch.softmax(self.attn_head(emb), dim=0)  # (n_cells, 1)
        pooled = (attn * emb).sum(dim=0)                    # (hidden_dim,)
        return self.value_head(pooled).squeeze(-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Scores and value in one shared-trunk pass.

        Returns:
            scores: (n_cells,)
            value:  scalar
        """
        emb        = self._embed(x)
        cell_scores = self.policy_head(emb).squeeze(-1)
        attn        = torch.softmax(self.attn_head(emb), dim=0)
        pooled      = (attn * emb).sum(dim=0)
        state_value = self.value_head(pooled).squeeze(-1)
        return cell_scores, state_value


def quantile_huber_loss(
    q_values: torch.Tensor,
    target: torch.Tensor,
    taus: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Quantile Huber loss ρ_τ for IQN value training.

    Args:
        q_values: (N,) — Z_τ(s) estimates for each quantile level
        target:   scalar — GAE return target
        taus:     (N,) — the quantile levels used
        kappa:    Huber threshold (1.0 as in IQN paper)

    Returns:
        scalar loss
    """
    u = target.detach() - q_values                                    # (N,)
    huber = torch.where(u.abs() <= kappa,
                        0.5 * u ** 2,
                        kappa * (u.abs() - 0.5 * kappa))              # (N,)
    rho = (taus - (u.detach() < 0).float()).abs() * huber / kappa     # (N,)
    return rho.mean()


class IQNActorCriticCellNN(nn.Module):
    """IQN distributional actor-critic for CAPTAIN.

    Identical trunk and policy head to ActorCriticCellNN.
    Replaces the scalar value head with an IQN quantile head:
        Z_τ(s) = value_head(pooled ⊙ φ(τ))
    where φ(τ) = ReLU(Linear(cos(π·i·τ) for i=1..n_cos)).

    forward() returns (scores, mean_value) — same interface as ActorCriticCellNN
    so rollout code is unchanged. Use forward_train() during the PPO update
    to get per-quantile values for the quantile Huber loss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | list[int] = 64,
        activation: str = "relu",
        n_cos: int = 64,
    ):
        super().__init__()

        activation_classes = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
        act_cls = activation_classes[activation]

        hidden_dims = [hidden_dim] if isinstance(hidden_dim, int) else list(hidden_dim)

        trunk_layers = []
        current_dim = input_dim
        for h in hidden_dims:
            trunk_layers += [nn.Linear(current_dim, h), act_cls()]
            current_dim = h
        self.trunk = nn.Sequential(*trunk_layers)

        self.policy_head = nn.Linear(current_dim, 1)
        self.attn_head   = nn.Linear(current_dim, 1)

        # IQN quantile head
        self.n_cos      = n_cos
        self.phi_head   = nn.Linear(n_cos, current_dim)   # cosine embed → hidden
        self.value_head = nn.Linear(current_dim, 1)        # conditioned → scalar

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        nn.init.normal_(self.policy_head.weight, 0, 1.0)
        nn.init.zeros_(self.policy_head.bias)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x.t())

    def _pool(self, emb: torch.Tensor) -> torch.Tensor:
        attn = torch.softmax(self.attn_head(emb), dim=0)
        return (attn * emb).sum(dim=0)   # (hidden_size,)

    def quantile_values(self, pooled: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        """Z_τ(s) for a batch of τ levels.

        Args:
            pooled: (hidden_size,) — attention-pooled state embedding
            taus:   (N,) — quantile levels in [0, 1]

        Returns:
            (N,) — quantile return estimates
        """
        i        = torch.arange(1, self.n_cos + 1, device=taus.device, dtype=taus.dtype)
        cos_feat = torch.cos(taus.unsqueeze(1) * i.unsqueeze(0) * torch.pi)  # (N, n_cos)
        phi      = torch.relu(self.phi_head(cos_feat))                        # (N, hidden)
        h        = pooled.unsqueeze(0) * phi                                  # (N, hidden)
        return self.value_head(h).squeeze(-1)                                 # (N,)

    def forward(self, x: torch.Tensor, n_taus: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        """Scores and mean quantile value — same interface as ActorCriticCellNN.

        Returns:
            scores:     (n_cells,)
            mean_value: scalar — E_τ[Z_τ(s)], used for GAE during rollout
        """
        emb         = self._embed(x)
        cell_scores = self.policy_head(emb).squeeze(-1)
        pooled      = self._pool(emb)
        taus        = torch.rand(n_taus, device=x.device)
        mean_value  = self.quantile_values(pooled, taus).mean()
        return cell_scores, mean_value

    def forward_train(
        self,
        x: torch.Tensor,
        taus: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single trunk pass returning scores + quantile values for PPO update.

        Args:
            taus: (N,) — quantile levels, sampled fresh each call

        Returns:
            scores:     (n_cells,)
            mean_value: scalar
            q_values:   (N,) — Z_τ(s) for each τ (for quantile Huber loss)
        """
        emb         = self._embed(x)
        cell_scores = self.policy_head(emb).squeeze(-1)
        pooled      = self._pool(emb)
        q_values    = self.quantile_values(pooled, taus)
        return cell_scores, q_values.mean(), q_values


def plackett_luce_sample(
    scores: torch.Tensor,
    k: int,
    constraint_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample K cells without replacement via Plackett-Luce and return log-prob.

    Log-prob is divided by K so that the PPO importance ratio
    exp(new_logp - old_logp) stays well-behaved regardless of K.
    Without this, K=50 sequential log-probs compound to ~-548 nats,
    making tiny policy changes produce large ratios that defeat PPO clipping.

    Args:
        scores:           (n_cells,) raw logits from policy head
        k:                number of cells to select
        constraint_mask:  (n_cells,) bool — True = cell already protected / invalid

    Returns:
        selected:  (k,) indices of selected cells
        log_prob:  scalar — log P(selected | scores) / K
    """
    n_cells = scores.shape[0]
    device  = scores.device

    excl     = constraint_mask.clone() if constraint_mask is not None else torch.zeros(n_cells, dtype=torch.bool, device=device)
    selected = []
    log_prob = torch.tensor(0.0, device=device)

    for _ in range(k):
        masked_scores = scores.masked_fill(excl, float("-inf"))
        if torch.all(torch.isinf(masked_scores)):
            break
        dist     = Categorical(logits=masked_scores)
        idx      = dist.sample()
        log_prob = log_prob + dist.log_prob(idx)
        selected.append(idx)
        excl     = excl.clone()
        excl[idx] = True

    return torch.stack(selected), log_prob


def plackett_luce_log_prob(
    scores: torch.Tensor,
    selected: torch.Tensor,
    constraint_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Recompute log-prob of a previously sampled action (needed for PPO update).

    Returns log P / K (matching plackett_luce_sample) so the importance ratio
    exp(new_logp - old_logp) is K-independent.

    Args:
        scores:          (n_cells,) current policy logits
        selected:        (k,) indices that were chosen
        constraint_mask: (n_cells,) bool — True = invalid

    Returns:
        log_prob: scalar (divided by K)
    """
    n_cells = scores.shape[0]
    device  = scores.device

    excl     = constraint_mask.clone() if constraint_mask is not None else torch.zeros(n_cells, dtype=torch.bool, device=device)
    log_prob = torch.tensor(0.0, device=device)

    for idx in selected:
        masked_scores = scores.masked_fill(excl, float("-inf"))
        dist     = Categorical(logits=masked_scores)
        log_prob = log_prob + dist.log_prob(idx)
        excl     = excl.clone()
        excl[idx] = True

    return log_prob
