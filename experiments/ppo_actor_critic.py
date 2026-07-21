"""Actor-Critic network for PPO on CAPTAIN's cell-selection problem.

Architecture
------------
Shared trunk  : same MLP as CellNN — takes (n_features, n_cells), outputs per-cell embeddings
Policy head   : linear → per-cell scores → Plackett-Luce distribution (K cells sampled without replacement)
Value head    : mean-pools per-cell embeddings → linear → scalar V(s)

Why a shared trunk?
    The trunk learns spatial features useful for both "which cells matter" (policy)
    and "how good is this state overall" (value). Sharing reduces parameters and
    encourages the representations to stay grounded.

Why Plackett-Luce for K=50?
    Top-K selection with torch.topk is deterministic — no log-prob, no gradient.
    Plackett-Luce models the probability of an ordered K-subset as sequential
    categorical draws without replacement:
        P(i1, i2, ..., iK) = Π_{j=1}^{K} softmax(scores[remaining])[ij]
    log P is the sum of per-step log-softmax values, computable in O(K·N).
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

        # Shared trunk: maps each cell's features to a latent embedding
        trunk_layers = []
        current_dim = input_dim
        for h in hidden_dims:
            trunk_layers += [nn.Linear(current_dim, h), act_cls()]
            current_dim = h
        self.trunk = nn.Sequential(*trunk_layers)

        # Policy head: maps per-cell embedding to a scalar score
        self.policy_head = nn.Linear(current_dim, 1)

        # Value head: maps mean-pooled embedding to a scalar state value
        # Mean pooling over cells collapses (n_cells, h) → (h,) before the linear
        self.value_head = nn.Linear(current_dim, 1)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Shared trunk forward pass.

        Args:
            x: (n_features, n_cells)

        Returns:
            embeddings: (n_cells, hidden_dim)
        """
        return self.trunk(x.t())  # (n_cells, n_features) → (n_cells, hidden_dim)

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """Per-cell policy scores (logits).

        Args:
            x: (n_features, n_cells)

        Returns:
            scores: (n_cells,)
        """
        emb = self._embed(x)
        return self.policy_head(emb).squeeze(-1)  # (n_cells,)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """State value estimate V(s).

        Args:
            x: (n_features, n_cells)

        Returns:
            value: scalar tensor
        """
        emb = self._embed(x)
        pooled = emb.mean(dim=0)  # (hidden_dim,) — average over all cells
        return self.value_head(pooled).squeeze(-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Scores and value in one pass (shares trunk computation).

        Args:
            x: (n_features, n_cells)

        Returns:
            scores: (n_cells,)
            value:  scalar
        """
        emb = self._embed(x)
        cell_scores = self.policy_head(emb).squeeze(-1)
        state_value = self.value_head(emb.mean(dim=0)).squeeze(-1)
        return cell_scores, state_value


def plackett_luce_sample(
    scores: torch.Tensor,
    k: int,
    constraint_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample K cells without replacement via Plackett-Luce and return log-prob.

    At each of K steps:
        1. Apply softmax over remaining (unselected, unconstrained) cells
        2. Sample one cell from that distribution
        3. Accumulate log-prob
        4. Mask the selected cell out

    Args:
        scores:           (n_cells,) raw logits from policy head
        k:                number of cells to select
        constraint_mask:  (n_cells,) bool — True = cell already protected / invalid

    Returns:
        selected:  (k,) indices of selected cells
        log_prob:  scalar — log P(selected | scores) under Plackett-Luce
    """
    n_cells = scores.shape[0]
    device = scores.device

    excl = constraint_mask.clone() if constraint_mask is not None else torch.zeros(n_cells, dtype=torch.bool, device=device)
    selected = []
    log_prob = torch.tensor(0.0, device=device)

    for _ in range(k):
        masked_scores = scores.masked_fill(excl, float("-inf"))

        if torch.all(torch.isinf(masked_scores)):
            break

        dist = Categorical(logits=masked_scores)
        idx = dist.sample()

        log_prob = log_prob + dist.log_prob(idx)
        selected.append(idx)
        excl = excl.clone()
        excl[idx] = True

    return torch.stack(selected), log_prob


def plackett_luce_log_prob(
    scores: torch.Tensor,
    selected: torch.Tensor,
    constraint_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Recompute log-prob of a previously sampled action (needed for PPO update).

    PPO compares the log-prob under the current policy vs the old policy.
    This function recomputes log P(selected | current scores).

    Args:
        scores:          (n_cells,) current policy logits
        selected:        (k,) indices that were chosen
        constraint_mask: (n_cells,) bool — True = invalid

    Returns:
        log_prob: scalar
    """
    n_cells = scores.shape[0]
    device = scores.device

    excl = constraint_mask.clone() if constraint_mask is not None else torch.zeros(n_cells, dtype=torch.bool, device=device)
    log_prob = torch.tensor(0.0, device=device)

    for idx in selected:
        masked_scores = scores.masked_fill(excl, float("-inf"))
        dist = Categorical(logits=masked_scores)
        log_prob = log_prob + dist.log_prob(idx)
        # Clone before in-place write: masked_fill saves the mask for backward,
        # so modifying it in-place after the fact triggers a version mismatch.
        excl = excl.clone()
        excl[idx] = True

    return log_prob
