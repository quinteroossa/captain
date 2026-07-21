"""CAPTAIN environment wrapper for PPO step-by-step interaction.

Why this is needed
------------------
CAPTAIN's EpisodeRunner.run_episode() runs a full episode internally — PPO
can't intercept individual steps. This wrapper exposes a standard
    reset() → obs
    step(action) → (obs, reward, done, info)
interface so PPO's rollout loop can drive the environment one step at a time,
storing (obs, action, log_prob, reward, value, done) in a rollout buffer.

Episode structure with K=50
---------------------------
Each call to step() corresponds to one protection+environment timestep:
    1. Apply action (protect K=50 cells)
    2. Advance the environment (BioEnv.step)
    3. Compute reward
    4. Return next observation

With budget=17,000 and K=50, protection is exhausted after 340 steps.
After that the agent still observes and the env still steps (for the remaining
n_time_steps), but no new cells are protected — matching CAPTAIN's original
episode logic.
"""

from __future__ import annotations

import numpy as np
import torch

import captain as cn
from captain.algorithms.budget_manager import GlobalBudgetManager


class CaptainPPOEnv:
    """Step-by-step CAPTAIN environment wrapper for PPO.

    Args:
        env:               BioEnv instance
        feature_extractor: FeatureExtractor instance
        rewards:           Rewards instance
        budget_manager:    GlobalBudgetManager instance
        n_steps:           Total timesteps per episode (e.g. 50)
        k:                 Cells to protect per step (K=50)
        device:            torch device
    """

    def __init__(
        self,
        env: cn.BioEnv,
        feature_extractor: cn.FeatureExtractor,
        rewards: cn.Rewards,
        budget_manager: GlobalBudgetManager,
        n_steps: int = 50,
        k: int = 50,
        device: str | torch.device = "cpu",
    ):
        self.env = env
        self.feature_extractor = feature_extractor
        self.rewards = rewards
        self.budget_manager = budget_manager
        self.n_steps = n_steps
        self.k = k
        self.device = torch.device(device)

        self._t = 0

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def reset(self) -> torch.Tensor:
        """Reset environment to start of episode.

        Returns:
            obs: (n_features, n_cells)
        """
        self.env.reset()
        self.rewards.reset()
        # GlobalBudgetManager is stateless (reads from env.protected_cells_mask) — no reset needed
        self._t = 0
        return self._observe()

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, dict]:
        """Apply action, advance environment, return transition.

        Args:
            action: (k,) indices of cells to protect

        Returns:
            obs:    (n_features, n_cells) — next observation
            reward: scalar float
            done:   True when episode ends (t == n_steps)
            info:   dict with episode statistics (only populated at done)
        """
        # 1. Protect selected cells (if budget remains)
        budget_kwargs = self.budget_manager.get_step_context(self.env)
        has_budget = not budget_kwargs["done"]
        cells_available = not self.env.no_action_mask.all()

        if has_budget and cells_available and len(action) > 0:
            # Filter to valid (unprotected) cells only
            valid = action[~self.env.no_action_mask[action]]
            if len(valid) > 0:
                self.env.update_protection_matrix(valid)

        # 2. Advance environment (dispersal, growth, carrying capacity update)
        self.env.step()

        # 3. Compute reward (per-step, not cumulative)
        # calc_reward() accumulates into episode_rewards and appends to episode_reward_history.
        # We derive the per-step scalar from the last history entry to avoid coupling to
        # the cumulative total (which grows over the episode).
        self.rewards.calc_reward(self.env)
        last = torch.tensor(self.rewards.episode_reward_history[-1], dtype=torch.float32)
        reward = float(
            (last * self.rewards._reward_weights * self.rewards._reward_calibration).sum().item()
        )

        # 4. Advance timestep
        self._t += 1
        done = self._t >= self.n_steps

        # 5. Next observation
        obs = self._observe()

        info = {}
        if done:
            info = {
                "protected_cells": int(self.env.protected_cells_mask.sum().item()),
                "total_reward": float(self.rewards.get_weighted_reward()),
            }

        return obs, reward, done, info

    @property
    def constraint_mask(self) -> torch.Tensor:
        """Boolean mask of cells that cannot be selected (already protected or invalid)."""
        return self.env.no_action_mask

    @property
    def n_cells(self) -> int:
        return self.env.n_cells

    @property
    def n_features(self) -> int:
        return self.feature_extractor.n_features

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _observe(self) -> torch.Tensor:
        """Extract features from current environment state."""
        return self.feature_extractor.observe(self.env)
