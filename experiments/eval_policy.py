#!/usr/bin/env python
"""Evaluate a trained ES policy: protected vs no-protection counterfactual.

Mirrors Daniele's run_inference.py pattern:
  1. Run N episodes with the trained policy → final extinction-risk outcomes
  2. Run N episodes with NoBudgetManager → counterfactual (no protection)
  3. Report transition matrix, threat counts, total cost for both

No reward calibration is applied during eval (matches Daniele's NoRewards usage).
The comparison between (1) and (2) is the key dissertation metric.

For deterministic runs (es_baseline), n_episodes=1 suffices — same result every
time. For stochastic runs (es_stochastic_v3), n_episodes>1 samples the return
distribution across different disturbance intensities.

Usage:
    uv run python experiments/eval_policy.py \\
        --run-name es_baseline \\
        --data-dir /home/quinteroossa/captain_data/captain3data

    uv run python experiments/eval_policy.py \\
        --run-name es_stochastic_v3 \\
        --data-dir /home/quinteroossa/captain_data/captain3data \\
        --n-episodes 10
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from pyperlin import FractalPerlin2D

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

import captain as cn
from experiments.env_extensions import SampledIntensityDisturbance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)

CLASS_NAMES = ["LC", "NT", "VU", "EN", "CR"]


# =============================================================================
# Args
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained ES policy")
    parser.add_argument("--run-name", type=str, required=True,
                        help="Name of the training run (reads results/<run-name>/)")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Path to captain3data directory")
    parser.add_argument("--n-episodes", type=int, default=1,
                        help="Episodes to run (>1 useful for stochastic runs; deterministic is the same every time)")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cuda/mps/cpu)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for evaluation")
    return parser.parse_args()


# =============================================================================
# Environment — mirrors train_es.py / train_es_stochastic.py create_episode_runner
# =============================================================================

def create_episode_runner(data_dir: Path, cfg: dict) -> cn.EpisodeRunner:
    device = cfg["device"]

    if cfg.get("subset", False):
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

    # Disturbance — deterministic or stochastic depending on config
    if cfg.get("disturbance_mode") in ("fixed", "distribution"):
        risk_map, _ = cn.data_loader.load_map(
            data_dir / "env_layers/area_swept_disturbance.tif"
        )
        coherence     = cfg["disturbance_coherence"]
        padded_height = (risk_map.shape[0] // coherence + 1) * coherence
        padded_width  = (risk_map.shape[1] // coherence + 1) * coherence
        noise_gen     = FractalPerlin2D(
            shape=(1, padded_height, padded_width),
            resolutions=[(coherence, coherence), (coherence, coherence)],
            factors=cfg["disturbance_factors"],
        )
        binary_mask_2d = np.nan_to_num(mask)
        if cfg["disturbance_mode"] == "fixed":
            intensity_min = intensity_max = cfg["disturbance_intensity"]
        else:
            intensity_min = cfg["disturbance_intensity_min"]
            intensity_max = cfg["disturbance_intensity_max"]
        disturbance = SampledIntensityDisturbance(
            data=np.zeros_like(mask),
            risk_map=risk_map,
            mask=mask,
            binary_mask_2d=binary_mask_2d,
            noise_generator=noise_gen,
            delta_per_step=None,
            lower_bound=0,
            upper_bound=1,
            intensity_min=intensity_min,
            intensity_max=intensity_max,
            impact_factor=cfg["disturbance_impact_factor"],
            seed=cfg.get("seed", 42),
        )
    else:
        disturbance = cn.load_spatial_data(
            file=data_dir / "env_layers/area_swept_disturbance.tif",
            future_file=data_dir / "env_layers/future_area_swept_disturbance.tif",
            mask=mask,
            lower_bound=0,
            upper_bound=1,
            n_time_steps=cfg["n_time_steps"],
        )

    protection = cn.SpatialData(
        data=np.zeros((1,) + mask.shape),
        mask=mask,
        lower_bound=0,
        upper_bound=1,
    )

    costs = cn.load_spatial_data(
        file=data_dir / "env_layers/cost.tif",
        future_file=data_dir / "env_layers/future_cost.tif",
        mask=mask,
        lower_bound=0,
        upper_bound=1,
        n_time_steps=cfg["n_time_steps"],
    )

    traits = cn.data_loader.load_trait_table(
        trait_file, sdm.names, ref_column="species", fill_gaps=True
    )

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
            lambda_0=disp_rate,
            coords=sdm._coords,
            threshold=disp_window,
            filename=str(disp_file),
        )
    dispersal_matrix = cn.grid_utils.load_dispersal_distances(str(disp_file))

    env = cn.BioEnv(
        sdms=sdm,
        disturbance=disturbance,
        costs=costs,
        protection_matrix=protection,
        species_k=carrying_capacity,
        growth_rates=growth_rates,
        sensitivity_rates=sensitivity,
        cached_dispersal_matrix=dispersal_matrix,
        ext_risk=ext_risk,
        device=device,
    )

    feature_extractor = cn.FeatureExtractor(
        env,
        feature_set=None,
        time_rescale=cfg["n_time_steps"] / 2,
        device=device,
    )

    model  = cn.CellNN(input_dim=feature_extractor.n_features, hidden_dim=cfg["hidden_dim"])
    policy = cn.PolicyNetwork(model, seed=cfg.get("seed", 42), device=device)

    rewards = cn.Rewards(
        reward_obj_list=[
            cn.CalcRewardExtRisk(
                threat_weights=np.array([1, 0, -8, -16, -32]), device=device
            ),
            cn.CalcRewardPersistentCost(rescaler=float(1.0 / costs.data.sum())),
        ],
        reward_weights=np.array([1.0, 1.0]),
    )

    budget_manager = cn.GlobalBudgetManager(
        total_target=cfg["target_protected_cells"],
        cells_per_time_step=cfg["cells_per_step"],
        feature_updates_per_time_step=1,
    )

    return cn.EpisodeRunner(
        env=env,
        feature_extractor=feature_extractor,
        policy_network=policy,
        rewards=rewards,
        n_steps=cfg["n_time_steps"],
        budget_manager=budget_manager,
    )


# =============================================================================
# Metrics
# =============================================================================

def collect_metrics(env: cn.BioEnv) -> dict:
    """Mirrors Daniele's run_inference.py: reports outcomes only, no reward calibration."""
    n_classes  = env.ext_risk._n_classes
    final_risk = env.current_ext_risk
    init_risk  = env.ext_risk._init_status
    counts     = env.ext_risk.species_per_class(final_risk)

    transition = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for i in range(len(init_risk)):
        transition[init_risk[i].item(), final_risk[i].item()] += 1

    prot_flat  = env.protection_matrix.data.flatten()
    cost_flat  = env.costs.data.flatten()
    total_cost = torch.dot(cost_flat, prot_flat).item()
    n_protected = int(env.protected_cells_mask.sum().item())

    return {
        "threat_counts":     {CLASS_NAMES[i]: int(counts[i].item()) for i in range(n_classes)},
        "transition_matrix": transition.tolist(),
        "total_cost":        total_cost,
        "n_protected":       n_protected,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    results_dir  = Path("results") / args.run_name
    config_file  = results_dir / "config.yaml"
    weights_file = results_dir / "trained_weights.npy"

    if not config_file.exists():
        raise FileNotFoundError(f"No config at {config_file} — check --run-name")
    if not weights_file.exists():
        raise FileNotFoundError(f"No weights at {weights_file} — has training completed?")

    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    if args.device is not None:
        cfg["device"] = args.device
    elif torch.cuda.is_available():
        cfg["device"] = "cuda"
    elif torch.backends.mps.is_available():
        cfg["device"] = "mps"
    else:
        cfg["device"] = "cpu"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    weights = np.load(weights_file)

    print("=" * 60)
    print(f"Evaluating: {args.run_name}")
    print(f"  Episodes : {args.n_episodes}")
    print(f"  Device   : {cfg['device']}")
    print(f"  Weights  : {weights_file}")
    print("=" * 60)

    episode = create_episode_runner(args.data_dir, cfg)
    print(f"  Grid     : {episode.env.n_cells} cells, {episode.env.n_species} species")

    # ------------------------------------------------------------------
    # WITH protection — trained policy (Daniele's pattern: NoRewards,
    # focus on ecological outcomes not calibrated reward signal)
    # ------------------------------------------------------------------
    protected_metrics = []
    for i in range(args.n_episodes):
        episode.run_episode(params=weights)
        m = collect_metrics(episode.env)
        protected_metrics.append(m)
        counts_str = "  ".join(f"{k}:{v}" for k, v in m["threat_counts"].items())
        print(f"  [protected] ep {i+1:2d}: cost={m['total_cost']:.4f}  "
              f"protected={m['n_protected']}  [{counts_str}]")

    # ------------------------------------------------------------------
    # WITHOUT protection — counterfactual baseline (Daniele's pattern:
    # NoBudgetManager, same env and policy but no cells can be protected)
    # Answers: how much does protection actually change outcomes?
    # ------------------------------------------------------------------
    print()
    no_prot_runner = cn.EpisodeRunner(
        env=episode.env,
        feature_extractor=episode.feature_extractor,
        policy_network=episode.policy,
        rewards=cn.NoRewards(),
        n_steps=cfg["n_time_steps"],
        budget_manager=cn.NoBudgetManager(),
    )

    unprotected_metrics = []
    for i in range(args.n_episodes):
        no_prot_runner.run_episode(params=weights)
        m = collect_metrics(episode.env)
        unprotected_metrics.append(m)
        counts_str = "  ".join(f"{k}:{v}" for k, v in m["threat_counts"].items())
        print(f"  [no prot.] ep {i+1:2d}:                             [{counts_str}]")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON: with vs without protection")
    print("=" * 60)

    print("\n  WITH protection (trained policy):")
    for cls in CLASS_NAMES:
        vals = [m["threat_counts"][cls] for m in protected_metrics]
        print(f"    {cls:2s} : {np.mean(vals):6.1f} ± {np.std(vals):.1f}")
    costs_p = [m["total_cost"] for m in protected_metrics]
    print(f"    Cost : {np.mean(costs_p):.4f} ± {np.std(costs_p):.4f}")

    print("\n  WITHOUT protection (counterfactual):")
    for cls in CLASS_NAMES:
        vals = [m["threat_counts"][cls] for m in unprotected_metrics]
        print(f"    {cls:2s} : {np.mean(vals):6.1f} ± {np.std(vals):.1f}")

    tm = np.mean([m["transition_matrix"] for m in protected_metrics], axis=0)
    print("\n  Transition matrix — WITH protection (rows=initial, cols=final):")
    header = "        " + "  ".join(f"{c:>5}" for c in CLASS_NAMES)
    print(header)
    for i, row_name in enumerate(CLASS_NAMES):
        row = "  ".join(f"{tm[i][j]:5.1f}" for j in range(len(CLASS_NAMES)))
        print(f"    {row_name:2s}  [ {row} ]")

    # Save results
    out_file = results_dir / "eval_results.json"
    with open(out_file, "w") as f:
        json.dump({
            "run_name":   args.run_name,
            "n_episodes": args.n_episodes,
            "seed":       args.seed,
            "protected": {
                "mean_threat_counts": {
                    cls: float(np.mean([m["threat_counts"][cls] for m in protected_metrics]))
                    for cls in CLASS_NAMES
                },
                "mean_cost":              float(np.mean(costs_p)),
                "std_cost":               float(np.std(costs_p)),
                "mean_transition_matrix": tm.tolist(),
                "episodes":               protected_metrics,
            },
            "unprotected": {
                "mean_threat_counts": {
                    cls: float(np.mean([m["threat_counts"][cls] for m in unprotected_metrics]))
                    for cls in CLASS_NAMES
                },
                "episodes": unprotected_metrics,
            },
        }, f, indent=2)

    print(f"\n  Results → {out_file}")


if __name__ == "__main__":
    main()
