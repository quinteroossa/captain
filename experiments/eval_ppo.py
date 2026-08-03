#!/usr/bin/env python
"""Evaluate a trained PPO policy: protected vs no-protection counterfactual.

Mirrors experiments/eval_policy.py (ES) but loads an ActorCriticCellNN and
runs episodes via CaptainPPOEnv.  Inference is greedy (top-K by score) rather
than Plackett-Luce sampling so results are deterministic per episode.

Outputs eval_results.json and eval_protection_grid.npy in the same format as
the ES eval, so compare_policies.py works across all three models unchanged.

Usage:
    uv run python experiments/eval_ppo.py \\
        --run-name ppo_v11 \\
        --data-dir /home/quinteroossa/captain_data/captain3data \\
        --n-episodes 5
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

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

import captain as cn
from captain.algorithms.budget_manager import GlobalBudgetManager
from experiments.env_extensions import CalcRewardCellValue, CalcRewardMarginalCost
from experiments.ppo_actor_critic import ActorCriticCellNN
from experiments.ppo_env_wrapper import CaptainPPOEnv

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
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


# =============================================================================
# Environment  (mirrors create_env() in train_ppo.py exactly)
# =============================================================================

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

    disturbance = cn.load_spatial_data(
        file=data_dir / "env_layers/area_swept_disturbance.tif",
        mask=mask,
        lower_bound=0,
        upper_bound=1,
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

    traits             = cn.data_loader.load_trait_table(trait_file, sdm.names, ref_column="species", fill_gaps=True)
    sensitivity        = traits["sensitivity_disturbance"].to_numpy(copy=True)[:, np.newaxis]
    growth_rates       = traits["growth_rate"].to_numpy(copy=True) + 1.0
    carrying_capacity  = cfg["avg_carrying_capacity"] / traits["conservation_status"].to_numpy(copy=True)
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

    costs_rescaler = float(1.0 / costs.data.sum())
    rewards = cn.Rewards(
        reward_obj_list=[
            CalcRewardCellValue(
                priority_weights=np.array([1, 0, 8, 16, 32]), device=device
            ),
            CalcRewardMarginalCost(rescaler=costs_rescaler),
        ],
        reward_weights=np.array([
            cfg.get("reward_weight_cell_value", 1.0),
            cfg.get("reward_weight_cost", -0.5),
        ]),
    )
    # Disable calibration (set to ones as in training)
    rewards._reward_calibration = torch.ones(
        len(rewards._reward_obj_list), dtype=torch.float32
    )

    budget_manager = GlobalBudgetManager(
        total_target=cfg["target_protected_cells"],
        cells_per_time_step=cfg["cells_per_step"],
        feature_updates_per_time_step=1,
    )

    return CaptainPPOEnv(
        env=env,
        feature_extractor=feature_extractor,
        rewards=rewards,
        budget_manager=budget_manager,
        n_steps=cfg["n_time_steps"],
        k=cfg["k"],
        device=device,
    )


# =============================================================================
# Metrics  (identical to ES eval — both work on cn.BioEnv directly)
# =============================================================================

def collect_metrics(env: cn.BioEnv) -> dict:
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

    sdm_data      = env.sdms.data_min_threshold
    cost_flat_dev = cost_flat.to(sdm_data.device)
    cost_by_category = {}
    for c in range(n_classes):
        species_in_c = (init_risk == c).to(sdm_data.device)
        if species_in_c.sum() == 0:
            cost_by_category[CLASS_NAMES[c]] = 0.0
            continue
        habitat_mask = (sdm_data[species_in_c] > 0).any(dim=0)
        prot_c       = prot_flat.to(sdm_data.device) * habitat_mask.float()
        cost_by_category[CLASS_NAMES[c]] = float(torch.dot(cost_flat_dev, prot_c).item())

    prot_grid = np.array(env.protection_matrix.reconstruct_grid[0])

    return {
        "threat_counts":     {CLASS_NAMES[i]: int(counts[i].item()) for i in range(n_classes)},
        "transition_matrix": transition.tolist(),
        "total_cost":        total_cost,
        "n_protected":       n_protected,
        "recovery_rate":     recovery_rate,
        "decline_rate":      decline_rate,
        "cost_by_category":  cost_by_category,
        "protection_grid":   prot_grid,  # excluded from JSON, saved separately
    }


def normalize_obs(obs: torch.Tensor) -> torch.Tensor:
    mean = obs.mean(dim=-1, keepdim=True)
    std  = obs.std(dim=-1, keepdim=True).clamp(min=1e-8)
    return (obs - mean) / std


def save_spatial_plot(grid: np.ndarray, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, cmap="YlOrRd", interpolation="nearest")
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="Protected (1=yes)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def print_transition_matrix(tm: np.ndarray):
    header = "         " + "  ".join(f"{c:>5}" for c in CLASS_NAMES)
    print(header)
    for i, row_name in enumerate(CLASS_NAMES):
        row = "  ".join(f"{tm[i][j]:5.1f}" for j in range(len(CLASS_NAMES)))
        print(f"    {row_name:2s}   [ {row} ]")


# =============================================================================
# Episode runners
# =============================================================================

def run_protected_episode(
    captain_env: CaptainPPOEnv,
    model: ActorCriticCellNN,
    device: torch.device,
) -> dict:
    """Greedy top-K episode. Returns collect_metrics() output."""
    obs  = captain_env.reset()
    done = False

    model.eval()
    with torch.no_grad():
        while not done:
            obs_t  = normalize_obs(obs.to(device))
            scores = model.scores(obs_t)            # (n_cells,)

            constraint = captain_env.constraint_mask
            # Mask out already-protected / unavailable cells
            masked = scores.clone()
            masked[~constraint] = float("-inf")
            k      = captain_env.k
            action = torch.topk(masked, k).indices

            obs, _reward, done, _info = captain_env.step(action)

    return collect_metrics(captain_env.env)


def run_unprotected_episode(captain_env: CaptainPPOEnv) -> dict:
    """No-protection counterfactual: reset the underlying BioEnv and step
    through n_steps without protecting any cells.

    BioEnv.reset() already resets ext_risk via set_init_ext_risk(), so
    extinction tracking is clean.  We step the simulator but never call
    update_protection_matrix(), so the protection grid stays all-zeros.
    """
    bio_env = captain_env.env
    bio_env.reset()

    for _ in range(captain_env.n_steps):
        bio_env.step()

    return collect_metrics(bio_env)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    results_dir  = Path("results") / args.run_name
    config_file  = results_dir / "config.yaml"
    weights_file = results_dir / "trained_weights.pt"

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

    print("=" * 60)
    print(f"Evaluating PPO: {args.run_name}")
    print(f"  Episodes : {args.n_episodes}")
    print(f"  Device   : {cfg['device']}")
    print("=" * 60)

    device = torch.device(cfg["device"])
    captain_env = create_env(args.data_dir, cfg)
    print(f"  Grid     : {captain_env.n_cells} cells, {captain_env.env.n_species} species")
    print(f"  Features : {captain_env.n_features}\n")

    model = ActorCriticCellNN(
        input_dim=captain_env.n_features,
        hidden_dim=cfg["hidden_dim"],
        activation=cfg["activation"],
    ).to(device)
    model.load_state_dict(torch.load(weights_file, map_location=device))

    # ------------------------------------------------------------------ protected
    protected_metrics = []
    for i in range(args.n_episodes):
        m = run_protected_episode(captain_env, model, device)
        protected_metrics.append(m)
        counts_str = "  ".join(f"{k}:{v}" for k, v in m["threat_counts"].items())
        print(f"  [protected] ep {i+1:2d}: cost={m['total_cost']:.4f}  "
              f"protected={m['n_protected']}  [{counts_str}]")

    # ------------------------------------------------------------------ unprotected
    print()
    unprotected_metrics = []
    for i in range(args.n_episodes):
        m = run_unprotected_episode(captain_env)
        unprotected_metrics.append(m)
        counts_str = "  ".join(f"{k}:{v}" for k, v in m["threat_counts"].items())
        print(f"  [no prot.] ep {i+1:2d}:                             [{counts_str}]")

    # ------------------------------------------------------------------ summary
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

    print("\n  Recovery rates — WITH protection:")
    for cls in CLASS_NAMES:
        vals = [m["recovery_rate"][cls] for m in protected_metrics if m["recovery_rate"][cls] is not None]
        if vals:
            print(f"    {cls:2s} : {np.mean(vals):.3f}")

    print("\n  Decline rates — WITH protection:")
    for cls in CLASS_NAMES:
        vals = [m["decline_rate"][cls] for m in protected_metrics if m["decline_rate"][cls] is not None]
        if vals:
            print(f"    {cls:2s} : {np.mean(vals):.3f}")

    print("\n  Cost breakdown by initial threat category:")
    for cls in CLASS_NAMES:
        vals = [m["cost_by_category"][cls] for m in protected_metrics]
        print(f"    {cls:2s} : {np.mean(vals):.4f}")
    costs_p = [m["total_cost"] for m in protected_metrics]
    print(f"    Total : {np.mean(costs_p):.4f} ± {np.std(costs_p):.4f}")

    print("\n  Transition matrix (mean over episodes) — WITH protection:")
    mean_tm = np.mean([m["transition_matrix"] for m in protected_metrics], axis=0)
    print_transition_matrix(mean_tm)

    # ------------------------------------------------------------------ save
    def _make_serialisable(m: dict) -> dict:
        out = {k: v for k, v in m.items() if k != "protection_grid"}
        return out

    def _aggregate(metrics: list[dict]) -> dict:
        keys = ["threat_counts", "transition_matrix", "total_cost", "n_protected",
                "recovery_rate", "decline_rate", "cost_by_category"]
        agg = {}
        for key in keys:
            vals = [m[key] for m in metrics]
            if key in ("threat_counts",):
                classes = list(vals[0].keys())
                agg[f"mean_{key}"] = {c: float(np.mean([v[c] for v in vals])) for c in classes}
            elif key == "transition_matrix":
                agg[f"mean_{key}"] = np.mean(vals, axis=0).tolist()
            elif key in ("recovery_rate", "decline_rate", "cost_by_category"):
                classes = list(vals[0].keys())
                agg[f"mean_{key}"] = {
                    c: (float(np.mean([v[c] for v in vals if v[c] is not None]))
                        if any(v[c] is not None for v in vals) else None)
                    for c in classes
                }
            elif key == "total_cost":
                agg["mean_cost"] = float(np.mean(vals))
                agg["std_cost"]  = float(np.std(vals))
            else:
                agg[f"mean_{key}"] = float(np.mean(vals))
        agg["episodes"] = [_make_serialisable(m) for m in metrics]
        return agg

    output = {
        "run_name":  args.run_name,
        "model":     "ppo",
        "n_episodes": args.n_episodes,
        "protected":   _aggregate(protected_metrics),
        "unprotected": _aggregate(unprotected_metrics),
    }

    out_dir = results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "eval_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results → {json_path}")

    # Protection grid from last protected episode (deterministic greedy, all eps same)
    last_grid = protected_metrics[-1]["protection_grid"]
    np.save(out_dir / "eval_protection_grid.npy", last_grid)
    print(f"  Grid    → {out_dir / 'eval_protection_grid.npy'}")

    save_spatial_plot(
        last_grid,
        title=f"Protection map — {args.run_name}",
        out_path=out_dir / "eval_protection_map.png",
    )
    print(f"  Map     → {out_dir / 'eval_protection_map.png'}")


if __name__ == "__main__":
    main()
