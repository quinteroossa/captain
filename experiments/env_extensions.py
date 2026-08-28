"""Custom environment components for dissertation experiments.

Kept separate from captain/ to avoid modifying Daniele's upstream code.
"""

import numpy as np
import torch

from captain.agents.rewards import CalcReward
from captain.data.spatial_data import SpatialData, StochasticSpatialData


class CalcRewardMarginalCost(CalcReward):
    """Cost of cells protected THIS step (marginal), not cumulative.

    CalcRewardPersistentCost computes dot(costs, all_protected) which grows
    every step — giving wrong per-step credit to PPO. This class computes only
    the cost of the K cells newly added this step.
    """

    def __init__(self, name: str = "marginal_cost", rescaler: float = 1.0):
        super().__init__(name, rescaler, positive=False)
        self._prev_protection = None

    def calc_reward(self, env) -> float:
        current = env.protection_matrix.data.flatten()
        if self._prev_protection is None:
            new_cells = current
        else:
            if self._prev_protection.device != current.device:
                self._prev_protection = self._prev_protection.to(current.device)
            new_cells = (current - self._prev_protection).clamp(min=0)
        self._prev_protection = current.clone()
        return torch.dot(env.costs.data.flatten(), new_cells).item() * self._rescaler

    def reset(self) -> None:
        self._prev_protection = None


class CalcRewardExtRiskLevel(CalcReward):
    """Dense per-step reward from weighted sum of current risk counts.

    CalcRewardExtRisk fires only on category shifts (rare, ~0 per step).
    This class rewards the current risk state every step:
        reward = (counts · threat_weights) / n_species
    Agent gets signal for maintaining/improving species status at each timestep.
    """

    def __init__(
        self,
        name: str = "ext_risk_level",
        rescaler: float = 1.0,
        threat_weights=None,
        device: str = "cpu",
    ):
        super().__init__(name, rescaler, positive=True)
        if threat_weights is None:
            raise ValueError("threat_weights required, e.g. [1, 0, -8, -16, -32]")
        self.device = torch.device(device)
        if isinstance(threat_weights, (list, np.ndarray)):
            threat_weights = torch.tensor(threat_weights, dtype=torch.float32)
        self._threat_weights = threat_weights.to(self.device)

    def calc_reward(self, env) -> float:
        if self._threat_weights.device != env.device:
            self._threat_weights = self._threat_weights.to(env.device)
        counts = env.ext_risk.species_per_class(env.current_ext_risk)
        weighted = (counts * self._threat_weights) / env.n_species
        return weighted.sum().item() * self._rescaler

    def to(self, device):
        self.device = torch.device(device)
        self._threat_weights = self._threat_weights.to(self.device)
        return self


class CalcRewardCellValue(CalcReward):
    """Regret-based per-step credit for biodiversity value of newly protected cells.

    Computes the counterfactual advantage of the actual cell selection over a
    random selection from the available pool at the same timestep:

        reward = avg_value(selected_K) − avg_value(all_available_before_step)

    This is a per-action control variate (Williams, 1992; Foerster et al., 2018
    COMA) that removes the state-dependent component from the reward signal.
    The critic no longer needs to explain away per-timestep expected value;
    advantages directly measure "did I pick above-average cells?"

    Without the baseline, the critic absorbs the timestep-varying expected cell
    value, leaving near-zero advantages and no policy gradient. With the baseline:
      - Random policy → reward ≈ 0 (by definition)
      - Good policy   → reward > 0 (selected above-average cells)
      - The critic learns V(s) ≈ 0, so advantages = regret directly

    priority_weights should be NON-NEGATIVE and increase with threat level,
    e.g. [1, 0, 8, 16, 32] — mirrors ES weights [1, 0, -8, -16, -32] semantically:
    LC gets small positive value (keep safe), NT=0, VU/EN/CR get increasing priority.
    """

    def __init__(
        self,
        priority_weights,
        name: str = "cell_value",
        rescaler: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__(name, rescaler, positive=True)
        self.device = torch.device(device)
        if isinstance(priority_weights, (list, np.ndarray)):
            priority_weights = torch.tensor(priority_weights, dtype=torch.float32)
        self._priority_weights = priority_weights.to(self.device)
        self._prev_protection = None

    def calc_reward(self, env) -> float:
        current = env.protection_matrix.data.flatten()
        if self._prev_protection is None:
            prev = torch.zeros_like(current)
            new_mask = (current > 0).float()
        else:
            if self._prev_protection.device != current.device:
                self._prev_protection = self._prev_protection.to(current.device)
            prev = self._prev_protection
            new_mask = (current - prev).clamp(min=0)
        self._prev_protection = current.clone()

        n_new = new_mask.sum().clamp(min=1)
        dev = env.sdms.data.device
        new_mask = new_mask.to(dev)
        prev = prev.to(dev)

        w = self._priority_weights.to(dev)[env.current_ext_risk.to(dev)]  # (n_species,)
        cell_value = (env.sdms.data_min_threshold * w.unsqueeze(1)).sum(dim=0)  # (n_cells,)

        selected_avg = (cell_value * new_mask).sum() / n_new

        # Baseline: mean value of cells that were available before this step's selection
        available = (prev == 0).float().to(dev)
        n_available = available.sum().clamp(min=1)
        baseline = (cell_value * available).sum() / n_available

        return (selected_avg - baseline).item() / env.n_species * self._rescaler

    def reset(self) -> None:
        self._prev_protection = None

    def to(self, device):
        self.device = torch.device(device)
        self._priority_weights = self._priority_weights.to(self.device)
        return self


class SampledIntensityDisturbance(StochasticSpatialData):
    """StochasticSpatialData with intensity sampled per episode.

    Instead of a fixed intensity value, samples from Uniform(intensity_min,
    intensity_max) at the start of each episode. This creates a distribution
    of episode returns driven by environmental uncertainty — the prerequisite
    for CVaR-based objectives in Scenario 4.

    BioEnv calls reset() once per episode and update() once per timestep.
    Intensity is resampled in reset(), kept fixed within the episode in update().

    Units: `data` must be a degradation-intensity layer — 0 = no disturbance,
    1 = fully degraded — matching what `BioEnv.update_carrying_capacity()`
    expects from `self.disturbance.data`. Baseline is 0 everywhere; cells hit
    by a stochastic event are SET (not multiplied) to `impact_factor`. Do not
    construct this with `data=mask` (all 1s) — that convention comes from the
    standalone visual demo (`examples/plot_input_data.py`), which was never
    wired into `BioEnv`. Plugged into carrying-capacity math it means
    "undisturbed" cells are treated as maximally disturbed while "event" cells
    end up comparatively better off — the opposite of the intended effect.
    See DEVELOPMENT_NOTES.md, "Disturbance units inverted" (2026-07-23).
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

        Cells hit by an event are SET to `impact_factor` (a degradation-intensity
        value, 0-1), not multiplied by it — baseline (unhit) cells stay at 0.
        """
        noise_2d = self.noise_generator()[
            :, : self.binary_mask_2d.shape[0], : self.binary_mask_2d.shape[1]
        ].squeeze(0)

        q_threshold = torch.quantile(noise_2d.flatten(), intensity)
        event_mask_2d = noise_2d < q_threshold * (1.0 - self.risk_map)

        flat_event_mask = event_mask_2d[self.binary_mask_2d]
        self._data[:, flat_event_mask] = impact_factor

    @property
    def current_intensity(self) -> float:
        return self._current_intensity
