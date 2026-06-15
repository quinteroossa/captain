"""Policy network for cell selection in conservation planning.

This module provides the neural network architecture and wrapper class for
scoring cells and selecting optimal locations for protection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from captain.agents.spatial_cnn import SpatialCNN

logger = logging.getLogger(__name__)


class CellNN(nn.Module):
    """Neural network for scoring individual cells.

    An MLP that takes cell features and outputs a scalar score
    indicating the priority for protection. Supports dynamic depth and layer sizes.

    Architecture:
        Input: (n_features, n_cells)
        -> Transpose to (n_cells, n_features)
        -> [Linear -> Activation] x Number of hidden layers
        -> Linear(last_hidden_dim, 1)
        -> Squeeze to (n_cells,)

    Examples:
        >>> # Single hidden layer
        >>> model = CellNN(input_dim=13, hidden_dim=32)

        >>> # Deep network: 3 hidden layers (64 -> 32 -> 16 nodes)
        >>> model = CellNN(input_dim=13, hidden_dim=[64, 32, 16])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | list[int] = 32,
        activation: str = "relu",
    ):
        """Initialize cell scoring network.

        Args:
            input_dim: Number of input features per cell.
            hidden_dim: Hidden layer dimension(s). Can be an int or a list of ints.
            activation: Activation function ('relu', 'tanh', 'gelu').
        """
        super().__init__()

        # 1. Map to activation CLASSES instead of instances.
        # This avoids sharing a single instance across multiple layers,
        # which can break model tracing/exporting (ONNX/TorchScript).
        activation_classes = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
        }
        if activation not in activation_classes:
            raise ValueError(
                f"Unknown activation: {activation}. Use one of {list(activation_classes)}"
            )
        act_cls = activation_classes[activation]

        # 2. Normalize input to a list of integers
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = list(hidden_dim)

        # 3. Construct layers dynamically
        layers = []
        current_in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_in_dim, h_dim))
            layers.append(act_cls())
            current_in_dim = h_dim

        # Final projection layer to map down to a scalar score
        layers.append(nn.Linear(current_in_dim, 1))

        # Pack it all into the sequential block
        self.net = nn.Sequential(*layers)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # Keeps tracking intact regardless of type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Features tensor of shape (n_features, n_cells).

        Returns:
            Scores tensor of shape (n_cells,).
        """
        # Transpose: (f, n) -> (n, f)
        x = x.t()
        # Forward: (n, f) -> (n, 1)
        out = self.net(x)
        # Squeeze: (n, 1) -> (n,)
        return out.squeeze(-1)


class CellCNNPolicy(nn.Module):
    """Composite module registering SpatialCNN and CellNN as submodules.

    This allows Evolution Strategies to flatten/unflatten all parameters
    (CNN + CellNN) as a single vector. The feature extractor holds a reference
    to the same CNN object, so weight updates propagate automatically.

    The forward pass delegates to CellNN only; the CNN is called separately
    by FeatureExtractorCNN during observation.

    Args:
        cnn: SpatialCNN module for spatial feature extraction.
        cell_nn: CellNN module for cell scoring.

    Example:
        >>> from captain.agents.spatial_cnn import SpatialCNN
        >>> cnn = SpatialCNN(12, 8, 8)
        >>> cell_nn = CellNN(8, 16)
        >>> composite = CellCNNPolicy(cnn, cell_nn)
        >>> policy = PolicyNetwork(composite, device="cpu")
        >>> len(policy.get_flat_weights())  # CNN + CellNN params
        2041
    """

    def __init__(self, cnn: SpatialCNN, cell_nn: CellNN):
        super().__init__()
        self.cnn = cnn
        self.cell_nn = cell_nn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CellNN only.

        Args:
            x: Features tensor of shape (n_features, n_cells).

        Returns:
            Scores tensor of shape (n_cells,).
        """
        return self.cell_nn(x)


class PolicyNetwork:
    """Wrapper for cell-scoring neural network with weight management.

    Provides methods for weight extraction/injection (for evolution strategies)
    and action selection with constraint handling.

    Attributes:
        model: The underlying CellNN network.
        device: PyTorch device for computations.

    Example:
        >>> model = CellNN(input_dim=13, hidden_dim=32)
        >>> policy = PolicyNetwork(model, device="cuda")
        >>> actions = policy.get_actions(features, n_cells=10, constraint_mask=mask)
    """

    # Tie-breaking noise scale
    TIE_BREAK_NOISE: float = 1e-7

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ):
        """Initialize policy network wrapper.

        Args:
            model: CellNN model or compatible module.
            device: PyTorch device.
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(seed)
            self.seeded_init(model, seed)

    def seeded_init(self, model, seed):
        # Save the current random state
        current_state = torch.random.get_rng_state()

        # Set the seed for the init phase
        torch.manual_seed(seed)

        # Apply standard torch init methods
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Restore the previous random state
        torch.random.set_rng_state(current_state)

    def get_flat_weights(self) -> np.ndarray:
        """Extract all model parameters as a flat 1D NumPy array.

        Returns:
            1D array of all model parameters concatenated.
        """
        with torch.no_grad():
            parameters = nn.utils.parameters_to_vector(self.model.parameters())
            return parameters.cpu().numpy()

    def set_flat_weights(self, new_weights: np.ndarray | torch.Tensor) -> None:
        """Inject a flat weight vector into model parameters.

        Args:
            new_weights: 1D array of weights to inject.
        """
        if isinstance(new_weights, np.ndarray):
            new_weights = torch.from_numpy(new_weights).float()

        new_weights = new_weights.to(self.device)

        with torch.no_grad():
            nn.utils.vector_to_parameters(new_weights, self.model.parameters())

    def get_n_params(self) -> int:
        """Get total number of trainable parameters.

        Returns:
            Number of parameters.
        """
        return sum(p.numel() for p in self.model.parameters())

    def select_k_cells(self, scores: torch.Tensor, n_cells: int) -> torch.Tensor:
        """Select top-k cells by score with tie-breaking noise.

        Args:
            scores: Cell scores tensor of shape (n_cells,).
            n_cells: Number of cells to select.

        Returns:
            Indices of selected cells.
        """
        try:
            noise = torch.randn_like(scores, generator=self.gen) * self.TIE_BREAK_NOISE
        except TypeError:
            # Fallback for older PyTorch versions
            noise = (
                torch.randn(
                    scores.shape,
                    generator=self.gen,
                    dtype=scores.dtype,
                    device=scores.device,
                )
                * self.TIE_BREAK_NOISE
            )

        _, top_indices = torch.topk(scores + noise, n_cells, sorted=False)
        return top_indices

    def get_scores(self, observation: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Compute cell scores from observations.

        Args:
            observation: Feature tensor of shape (n_features, n_cells).

        Returns:
            Score tensor of shape (n_cells,).
        """
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()

        observation = observation.to(self.device)

        with torch.no_grad():
            scores = self.model(observation)

        return scores

    def get_actions(
        self,
        observation: torch.Tensor | np.ndarray,
        constraint_mask: torch.Tensor | np.ndarray | None = None,
        **kwargs,  # Captures any extra arguments
    ) -> torch.Tensor:
        """Get indices of cells to protect (Global Version).

        Args:
            observation: Feature tensor of shape (n_features, n_cells).
            constraint_mask: Boolean mask of cells to exclude (True = exclude).
            **kwargs: Must contain ``n_cells`` (number of cells to select).

        Returns:
            Indices of selected cells.
        """
        n_cells = kwargs.get("n_cells")
        if n_cells is None:
            raise ValueError(
                "Global PolicyNetwork requires 'n_cells' keyword argument."
            )

        scores = self.get_scores(observation)

        if constraint_mask is not None:
            if isinstance(constraint_mask, np.ndarray):
                constraint_mask = torch.from_numpy(constraint_mask)
            scores = scores.masked_fill(constraint_mask.to(self.device), float("-inf"))

        return self.select_k_cells(scores, n_cells)

    def to(self, device: torch.device | str) -> PolicyNetwork:
        """Move policy to specified device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.gen = torch.Generator(device=self.device)
        return self

    def save(self, path: str) -> None:
        """Save model weights to file.

        Args:
            path: Path to save weights.
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved policy weights to {path}")

    def load(self, path: str) -> None:
        """Load model weights from file.

        Args:
            path: Path to load weights from.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Loaded policy weights from {path}")


class RegionalPolicyNetwork(PolicyNetwork):
    """Extension of PolicyNetwork for handling stratified regional allocations."""

    def select_regional_k_cells(
        self,
        scores: torch.Tensor,
        region_masks: dict[str | int, torch.Tensor],
        region_k: dict[str | int, int],
    ) -> torch.Tensor:
        """Select top-k cells independently within non-overlapping regions.

        Args:
            scores: Global cell scores tensor of shape (n_cells fire-mapped).
            region_masks: Dict mapping region IDs to boolean tensors of shape (n_cells,).
            region_k: Dict mapping region IDs to their specific allocation target (K_i).

        Returns:
            Flat tensor containing all selected global cell indices across all regions.
        """
        # 1. Apply tie-breaking noise globally once
        # noise = torch.randn_like(scores, generator=self.gen) * self.TIE_BREAK_NOISE

        try:
            noise = torch.randn_like(scores, generator=self.gen) * self.TIE_BREAK_NOISE
        except TypeError:
            # Fallback for older PyTorch versions
            noise = (
                torch.randn(
                    scores.shape,
                    generator=self.gen,
                    dtype=scores.dtype,
                    device=scores.device,
                )
                * self.TIE_BREAK_NOISE
            )

        perturbed_scores = scores + noise

        all_selected_global_indices = []

        # 2. Iterate through regions (efficient since number of regions is small)
        for region_id, mask in region_masks.items():
            k_i = region_k.get(region_id, 0)
            if k_i <= 0:
                continue

            # Convert boolean mask to global indices
            global_indices = torch.nonzero(mask).squeeze(-1)
            if len(global_indices) == 0:
                continue

            # Extract local scores for this region
            local_scores = perturbed_scores[global_indices]

            # Edge case safety: ensure K_i doesn't exceed valid, unconstrained cells
            valid_cells = torch.sum(local_scores > float("-inf")).item()
            actual_k = min(k_i, valid_cells)

            if actual_k > 0:
                # Local Top-K selection
                _, top_local_indices = torch.topk(local_scores, actual_k, sorted=False)

                # Map local indices back to global positions
                top_global_indices = global_indices[top_local_indices]
                all_selected_global_indices.append(top_global_indices)

        # 3. Combine all regional picks into a single action vector
        if all_selected_global_indices:
            return torch.cat(all_selected_global_indices)

        return torch.empty(0, dtype=torch.long, device=self.device)

    def get_actions(
        self,
        observation: torch.Tensor | np.ndarray,
        constraint_mask: torch.Tensor | np.ndarray | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Get indices of cells to protect (Regional Version).

        Args:
            observation: Feature tensor of shape (n_features, n_cells).
            region_masks: Dict mapping region IDs to boolean masks.
            region_k: Dict mapping region IDs to their target integers.
            constraint_mask: Global boolean mask of cells to exclude entirely.

        Returns:
            Indices of all selected cells across the grid.

        """
        region_masks = kwargs.get("region_masks")
        region_k = kwargs.get("region_k")

        if region_masks is None or region_k is None:
            raise ValueError(
                "RegionalPolicyNetwork requires 'region_masks' and 'region_k'."
            )

        scores = self.get_scores(observation)

        if constraint_mask is not None:
            if isinstance(constraint_mask, np.ndarray):
                constraint_mask = torch.from_numpy(constraint_mask)
            scores = scores.masked_fill(constraint_mask.to(self.device), float("-inf"))

        # Align masks to device
        device_masks = {}
        for r_id, mask in region_masks.items():
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            device_masks[r_id] = mask.to(self.device)

        return self.select_regional_k_cells(scores, device_masks, region_k)
