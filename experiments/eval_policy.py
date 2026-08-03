#!/usr/bin/env python
"""Evaluate a trained ES policy: protected vs no-protection counterfactual.

Mirrors Daniele's run_inference.py pattern:
  1. Run N episodes with the trained policy → final extinction-risk outcomes
  2. Run N episodes with NoBudgetManager → counterfactual (no protection)
  3. Report transition matrix, threat counts, cost breakdown, spatial map

Enhanced for dissertation comparison:
  - Saves protection grid as .npy for cross-model Jaccard comparison
  - Cost breakdown by initial threat category
  - Species recovery and decline rates per category
  - Spatial protection map via cn.plots.plot_grid

Usage:
    uv run python experiments/eval_policy.py \\
        --run-name es_baseline_full \\
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


# =============================================================================
# Environment
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
            mask=mask, lower_bound=0, upper_bound=1,
            n_time_steps=cfg["n_time_steps"],
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
    ext_risk            = cn.ExtinctionRisk(init_status=conservation_status, n_classes=5, alpha=0.5)

    disp_rate   = cfg["dispersal_rate"]
    disp_window = cfg["dispersal_window"]
    disp_file   = data_dir / f"dispersal_d{disp_rate}_t{disp_window}_NEW.npz"
    if not disp_file.exists():
        cn.grid_utils.save_dispersal_distances(
            lambda_0=disp_rate, coords=sdm._coords,
            threshold=disp_window, filename=str(disp_file),
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
    model  = cn.CellNN(input_dim=feature_extractor.n_features, hidden_dim=cfg["hidden_dim"])
    policy = cn.PolicyNetwork(model, seed=cfg.get("seed", 42), device=device)
    rewards = cn.Rewards(
        reward_obj_list=[
            cn.CalcRewardExtRisk(threat_weights=np.array([1, 0, -8, -16, -32]), device=device),
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
        env=env, feature_extractor=feature_extractor, policy_network=policy,
        rewards=rewards, n_steps=cfg["n_time_steps"], budget_manager=budget_manager,
    )


# =============================================================================
# Metrics
# =============================================================================

def collect_metrics(env: cn.BioEnv) -> dict:
    n_classes  = env.ext_risk._n_classes
    final_risk = env.current_ext_risk
    init_risk  = env.ext_risk._init_status
    counts     = env.ext_risk.species_per_class(final_risk)

    # Transition matrix
    transition = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for i in range(len(init_risk)):
        transition[init_risk[i].item(), final_risk[i].item()] += 1

    # Cost
    prot_flat   = env.protection_matrix.data.flatten()
    cost_flat   = env.costs.data.flatten()
    total_cost  = torch.dot(cost_flat, prot_flat).item()
    n_protected = int(env.protected_cells_mask.sum().item())

    # Recovery / decline rates per initial category
    # recovery = improved to safer category; decline = moved to more threatened
    recovery_rate = {}
    decline_rate  = {}
    for c in range(n_classes):
        mask_c = (init_risk == c)
        n_c    = mask_c.sum().item()
        if n_c == 0:
            recovery_rate[CLASS_NAMES[c]] = None
            decline_rate[CLASS_NAMES[c]]  = None
            continue
        improved = ((final_risk < init_risk) & mask_c).sum().item()
        declined = ((final_risk > init_risk) & mask_c).sum().item()
        recovery_rate[CLASS_NAMES[c]] = round(improved / n_c, 4)
        decline_rate[CLASS_NAMES[c]]  = round(declined / n_c, 4)

    # Cost breakdown by initial threat category:
    # For each category c, sum costs of cells where ≥1 species with init_risk==c has habitat
    sdm_data  = env.sdms.data_min_threshold        # (n_species, n_cells)
    cost_flat_dev = cost_flat.to(sdm_data.device)
    cost_by_category = {}
    for c in range(n_classes):
        species_in_c = (init_risk == c).to(sdm_data.device)
        if species_in_c.sum() == 0:
            cost_by_category[CLASS_NAMES[c]] = 0.0
            continue
        habitat_mask = (sdm_data[species_in_c] > 0).any(dim=0)   # cells with habitat for cat c
        prot_c       = prot_flat.to(sdm_data.device) * habitat_mask.float()
        cost_by_category[CLASS_NAMES[c]] = float(torch.dot(cost_flat_dev, prot_c).item())

    # Protection grid for spatial comparison (2D numpy array, NaN outside study area)
    prot_grid = env.protection_matrix.reconstruct_grid[0].cpu().numpy()

    return {
        "threat_counts":      {CLASS_NAMES[i]: int(counts[i].item()) for i in range(n_classes)},
        "transition_matrix":  transition.tolist(),
        "total_cost":         total_cost,
        "n_protected":        n_protected,
        "recovery_rate":      recovery_rate,
        "decline_rate":       decline_rate,
        "cost_by_category":   cost_by_category,
        "protection_grid":    prot_grid,   # excluded from JSON, saved separately
    }


def print_transition_matrix(tm: np.ndarray):
    header = "         " + "  ".join(f"{c:>5}" for c in CLASS_NAMES)
    print(header)
    for i, row_name in enumerate(CLASS_NAMES):
        row = "  ".join(f"{tm[i][j]:5.1f}" for j in range(len(CLASS_NAMES)))
        print(f"    {row_name:2s}   [ {row} ]")


def save_spatial_plot(grid: np.ndarray, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, cmap="YlOrRd", interpolation="nearest")
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="Protected (1=yes)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    results_dir  = Path("results") / args.run_name
    config_file  = results_dir / "config.yaml"
    weights_file = results_dir / "trained_weights.npy"

    if not config_file.exists():
        raise FileNotFoundError(f"No config at {config_file}")
    if not weights_file.exists():
        raise FileNotFoundError(f"No weights at {weights_file}")

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
    print("=" * 60)

    episode = create_episode_runner(args.data_dir, cfg)
    print(f"  Grid     : {episode.env.n_cells} cells, {episode.env.n_species} species\n")

    # ------------------------------------------------------------------ WITH protection
    protected_metrics = []
    for i in range(args.n_episodes):
        episode.run_episode(params=weights)
        m = collect_metrics(episode.env)
        protected_metrics.append(m)
        counts_str = "  ".join(f"{k}:{v}" for k, v in m["threat_counts"].items())
        print(f"  [protected] ep {i+1:2d}: cost={m['total_cost']:.4f}  "
              f"protected={m['n_protected']}  [{counts_str}]")

    # ------------------------------------------------------------------ WITHOUT protection
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

    # ------------------------------------------------------------------ Summary
    print("\n" + "=" * 60)
    print("RESULTS: with vs without protection")
    print("=" * 60)

    print("\n  Final species counts — WITH protection:")
    for cls in CLASS_NAMES:
        vals = [m["threat_counts"][cls] for m in protected_metrics]
        print(f"    {cls:2s} : {np.mean(vals):6.1f} ± {np.std(vals):.1f}")

    print("\n  Final species counts — WITHOUT protection:")
    for cls in CLASS_NAMES:
        vals = [m["threat_counts"][cls] for m in unprotected_metrics]
        print(f"    {cls:2s} : {np.mean(vals):6.1f} ± {np.std(vals):.1f}")

    print("\n  Recovery rates (fraction improved to safer category) — WITH protection:")
    for cls in CLASS_NAMES:
        vals = [m["recovery_rate"][cls] for m in protected_metrics if m["recovery_rate"][cls] is not None]
        if vals:
            print(f"    {cls:2s} : {np.mean(vals):.3f}")

    print("\n  Decline rates (fraction moved to more threatened) — WITH protection:")
    for cls in CLASS_NAMES:
        vals = [m["decline_rate"][cls] for m in protected_metrics if m["decline_rate"][cls] is not None]
        if vals:
            print(f"    {cls:2s} : {np.mean(vals):.3f}")

    print("\n  Cost breakdown by initial threat category (budget spent on habitat for each):")
    for cls in CLASS_NAMES:
        vals = [m["cost_by_category"][cls] for m in protected_metrics]
        print(f"    {cls:2s} : {np.mean(vals):.4f}")
    costs_p = [m["total_cost"] for m in protected_metrics]
    print(f"    Total : {np.mean(costs_p):.4f} ± {np.std(costs_p):.4f}")

    tm = np.mean([m["transition_matrix"] for m in protected_metrics], axis=0)
    print("\n  Transition matrix — WITH protection (rows=initial, cols=final):")
    print_transition_matrix(tm)

    tm_np = np.mean([m["transition_matrix"] for m in unprotected_metrics], axis=0)
    print("\n  Transition matrix — WITHOUT protection:")
    print_transition_matrix(tm_np)

    # ------------------------------------------------------------------ Save
    # Spatial grids — saved separately for Jaccard comparison
    grids_protected = np.stack([m["protection_grid"] for m in protected_metrics])
    mean_grid = np.nanmean(grids_protected, axis=0)

    np.save(results_dir / "eval_protection_grid.npy", mean_grid)
    save_spatial_plot(
        mean_grid,
        title=f"Protection map — {args.run_name}",
        out_path=results_dir / "eval_protection_map.png",
    )

    # JSON results (exclude protection_grid — too large)
    def strip_grid(m):
        return {k: v for k, v in m.items() if k != "protection_grid"}

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
                "mean_recovery_rate":     {
                    cls: float(np.mean([m["recovery_rate"][cls] for m in protected_metrics
                                        if m["recovery_rate"][cls] is not None]))
                    for cls in CLASS_NAMES
                    if any(m["recovery_rate"][cls] is not None for m in protected_metrics)
                },
                "mean_decline_rate": {
                    cls: float(np.mean([m["decline_rate"][cls] for m in protected_metrics
                                        if m["decline_rate"][cls] is not None]))
                    for cls in CLASS_NAMES
                    if any(m["decline_rate"][cls] is not None for m in protected_metrics)
                },
                "mean_cost_by_category": {
                    cls: float(np.mean([m["cost_by_category"][cls] for m in protected_metrics]))
                    for cls in CLASS_NAMES
                },
                "mean_transition_matrix": tm.tolist(),
                "episodes": [strip_grid(m) for m in protected_metrics],
            },
            "unprotected": {
                "mean_threat_counts": {
                    cls: float(np.mean([m["threat_counts"][cls] for m in unprotected_metrics]))
                    for cls in CLASS_NAMES
                },
                "mean_transition_matrix": tm_np.tolist(),
                "episodes": [strip_grid(m) for m in unprotected_metrics],
            },
        }, f, indent=2)

    print(f"\n  Results  → {out_file}")
    print(f"  Grid     → {results_dir}/eval_protection_grid.npy")
    print(f"  Map      → {results_dir}/eval_protection_map.png")


if __name__ == "__main__":
    main()
