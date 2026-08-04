#!/usr/bin/env python
"""Train PPO with stochastic disturbance (RQ2).

Identical to train_ppo.py except create_env() replaces the static disturbance
layer with SampledIntensityDisturbance — Perlin-noise disturbance whose
intensity is resampled from Uniform(intensity_min, intensity_max) at the start
of each episode.  This produces a distribution of episode returns driven by
environmental uncertainty, which is the prerequisite for CVaR (RQ3).

Usage:
    uv run python experiments/train_ppo_stochastic.py \\
        --config experiments/configs/ppo_stochastic.yaml \\
        --data-dir /path/to/data \\
        --run-name ppo_stochastic_v1 \\
        --wandb
"""

import numpy as np
from pathlib import Path
from pyperlin import FractalPerlin2D

import captain as cn
from experiments.env_extensions import (
    CalcRewardCellValue, CalcRewardMarginalCost, SampledIntensityDisturbance,
)
import experiments.train_ppo as _base
from experiments.train_ppo import (
    GlobalBudgetManager, CaptainPPOEnv, torch, main,
)


def create_env(data_dir: Path, cfg: dict) -> CaptainPPOEnv:
    device = cfg["device"]

    if cfg["subset"]:
        present_dir = data_dir / "subset/present_sdms"
        future_dir  = data_dir / "subset/future_sdms"
        trait_file  = data_dir / "subset/species_tbl.csv"
    else:
        present_dir = data_dir / "present_sdms"
        future_dir  = data_dir / "future_sdms"
        trait_file  = data_dir / "species_tbl.csv"

    mask, _ = cn.data_loader.load_map(data_dir / "env_layers/area_mask.npy")

    sdm = cn.load_spatial_data_from_dir(
        dir=present_dir,
        future_dir=future_dir,
        mask=mask,
        lower_bound=0,
        upper_bound=1,
        n_time_steps=cfg["n_time_steps"],
        min_threshold=cfg["min_habitat_suitability"],
    )

    # --- Stochastic disturbance (RQ2) ---
    risk_map, _ = cn.data_loader.load_map(
        data_dir / "env_layers/area_swept_disturbance.tif"
    )
    coherence     = cfg["disturbance_coherence"]
    padded_height = (risk_map.shape[0] // coherence + 1) * coherence
    padded_width  = (risk_map.shape[1] // coherence + 1) * coherence
    noise_gen = FractalPerlin2D(
        shape=(1, padded_height, padded_width),
        resolutions=[(coherence, coherence), (coherence, coherence)],
        factors=cfg["disturbance_factors"],
    )
    disturbance = SampledIntensityDisturbance(
        data=np.zeros_like(mask),
        risk_map=risk_map,
        mask=mask,
        binary_mask_2d=np.nan_to_num(mask),
        noise_generator=noise_gen,
        delta_per_step=None,
        lower_bound=0,
        upper_bound=1,
        intensity_min=cfg["disturbance_intensity_min"],
        intensity_max=cfg["disturbance_intensity_max"],
        impact_factor=cfg["disturbance_impact_factor"],
        seed=cfg.get("seed", 42),
    )

    protection = cn.SpatialData(
        data=np.zeros((1,) + mask.shape), mask=mask, lower_bound=0, upper_bound=1,
    )
    costs = cn.load_spatial_data(
        file=data_dir / "env_layers/cost.tif",
        future_file=data_dir / "env_layers/future_cost.tif",
        mask=mask, lower_bound=0, upper_bound=1,
        n_time_steps=cfg["n_time_steps"],
    )

    traits              = cn.data_loader.load_trait_table(trait_file, sdm.names, ref_column="species", fill_gaps=True)
    sensitivity         = traits["sensitivity_disturbance"].to_numpy(copy=True)[:, np.newaxis]
    growth_rates        = traits["growth_rate"].to_numpy(copy=True) + 1.0
    carrying_capacity   = cfg["avg_carrying_capacity"] / traits["conservation_status"].to_numpy(copy=True)
    conservation_status = traits["conservation_status"].to_numpy(copy=True) - 1

    ext_risk = cn.ExtinctionRisk(init_status=conservation_status, n_classes=5, alpha=0.5)

    disp_rate   = cfg["dispersal_rate"]
    disp_window = cfg["dispersal_window"]
    disp_file   = data_dir / f"dispersal_d{disp_rate}_t{disp_window}_NEW.npz"
    if not disp_file.exists():
        cn.grid_utils.save_dispersal_distances(
            lambda_0=disp_rate, coords=sdm._coords, threshold=disp_window, filename=str(disp_file)
        )
    dispersal_matrix = cn.grid_utils.load_dispersal_distances(str(disp_file))

    env = cn.BioEnv(
        sdms=sdm, disturbance=disturbance, costs=costs,
        protection_matrix=protection, species_k=carrying_capacity,
        growth_rates=growth_rates, sensitivity_rates=sensitivity,
        cached_dispersal_matrix=dispersal_matrix, ext_risk=ext_risk, device=device,
    )

    feature_extractor = cn.FeatureExtractor(
        env, feature_set=None, time_rescale=cfg["n_time_steps"] / 2, device=device,
    )

    costs_rescaler = float(1.0 / costs.data.sum())
    rewards = cn.Rewards(
        reward_obj_list=[
            CalcRewardCellValue(
                priority_weights=np.array([1, 1, 8, 16, 32]), device=device
            ),
            CalcRewardMarginalCost(rescaler=costs_rescaler),
        ],
        reward_weights=np.array([
            cfg.get("reward_weight_cell_value", 1.0),
            cfg.get("reward_weight_cost", -0.5),
        ]),
    )
    rewards._reward_calibration = torch.ones(
        len(rewards._reward_obj_list), dtype=torch.float32
    )

    budget_manager = GlobalBudgetManager(
        total_target=cfg["target_protected_cells"],
        cells_per_time_step=cfg["cells_per_step"],
        feature_updates_per_time_step=1,
    )

    return CaptainPPOEnv(
        env=env, feature_extractor=feature_extractor, rewards=rewards,
        budget_manager=budget_manager, n_steps=cfg["n_time_steps"],
        k=cfg["k"], device=device,
    )


# Patch base module so main() uses this create_env
_base.create_env = create_env


if __name__ == "__main__":
    main()
