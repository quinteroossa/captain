"""Custom environment components for dissertation experiments.

Kept separate from captain/ to avoid modifying Daniele's upstream code.
"""

import numpy as np
import torch
from pyperlin import FractalPerlin2D

from captain.data.spatial_data import SpatialData, StochasticSpatialData


class SampledIntensityDisturbance(StochasticSpatialData):
    """StochasticSpatialData with intensity sampled per episode.

    Instead of a fixed intensity value, samples from Uniform(intensity_min,
    intensity_max) at the start of each episode. This creates a distribution
    of episode returns driven by environmental uncertainty — the prerequisite
    for CVaR-based objectives in Scenario 4.

    BioEnv calls reset() once per episode and update() once per timestep.
    Intensity is resampled in reset(), kept fixed within the episode in update().
    """

    def __init__(
        self,
        *args,
        intensity_min: float = 0.05,
        intensity_max: float = 0.5,
        impact_factor: float = 0.5,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max
        self.impact_factor = impact_factor
        self._rng = np.random.default_rng(seed)
        self._current_intensity = float(
            self._rng.uniform(intensity_min, intensity_max)
        )

    def reset(self) -> None:
        """Episode reset: restore data AND resample intensity for the new episode."""
        SpatialData.reset(self)
        self._current_intensity = float(
            self._rng.uniform(self.intensity_min, self.intensity_max)
        )

    def update(self, time_step: int = 1, trigger_events: bool = True, **kwargs):
        """Timestep update: restore data only (keep this episode's intensity)."""
        SpatialData.reset(self)   # data reset only — does NOT resample intensity
        if trigger_events:
            self.apply_stochastic_events(
                intensity=self._current_intensity,
                impact_factor=self.impact_factor,
            )

    def apply_stochastic_events(self, intensity: float, impact_factor: float = 0.5):
        """Quantile-based threshold fix for FractalPerlin2D range mismatch.

        The upstream formula `noise < (1 - risk_map) * intensity` assumes noise
        in [0, 1], but FractalPerlin2D outputs ~[-0.4, 0.45]. This makes intensity
        a dead parameter (~97% of cells fire regardless).

        Fix: use torch.quantile so that `intensity` directly controls the fraction
        of zero-risk cells disturbed per step, independent of the noise distribution.
        """
        noise_2d = self.noise_generator()[
            :, : self.binary_mask_2d.shape[0], : self.binary_mask_2d.shape[1]
        ].squeeze(0)

        q_threshold = torch.quantile(noise_2d.flatten(), intensity)
        event_mask_2d = noise_2d < q_threshold * (1.0 - self.risk_map)

        flat_event_mask = event_mask_2d[self.binary_mask_2d]
        self._data[:, flat_event_mask] *= impact_factor

    @property
    def current_intensity(self) -> float:
        return self._current_intensity
