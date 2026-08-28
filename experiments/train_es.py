#!/usr/bin/env python
"""Train a conservation policy with Evolution Strategies (ES) baseline.

Usage:
    uv run python experiments/train_es.py \\
        --data-dir /path/to/captain3data \\
        --config experiments/configs/es_base.yaml \\
        --run-name es_baseline

Cluster (SLURM):
    Set --data-dir to the cluster data path.
    All outputs are written to results/<run-name>/.
"""

import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

import captain as cn
from experiments.utils.wandb_logger import WandbLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)


# =============================================================================
# Config
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train CAPTAIN with Evolution Strategies")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Path to captain3data directory")
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/es_base.yaml"),
                        help="Path to YAML config file")
    parser.add_argument("--run-name", type=str, default="es_run",
                        help="Name for this run — outputs go to results/<run-name>/")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="Override n_epochs from config")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cuda/cpu/mps)")
    parser.add_argument("--recalibrate", action="store_true", default=False,
                        help="Force recompute reward calibration even if file already exists")
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="captain-dissertation",
                        help="W&B project name")
    return parser.parse_args()


def load_config(config_path: Path, args) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # CLI overrides
    if args.n_epochs is not None:
        cfg["n_epochs"] = args.n_epochs
    if args.device is not None:
        cfg["device"] = args.device
    else:
        if torch.cuda.is_available():
            cfg["device"] = "cuda"
        elif torch.backends.mps.is_available():
            cfg["device"] = "mps"
        else:
            cfg["device"] = "cpu"
    return cfg


# =============================================================================
# Environment setup
# =============================================================================

def create_episode_runner(data_dir: Path, cfg: dict, results_dir: Path) -> cn.EpisodeRunner:
    device = cfg["device"]

    if cfg["subset"]:
        present_sdms_dir = data_dir / "subset/present_sdms"
        future_sdms_dir  = data_dir / "subset/future_sdms"
        trait_file       = data_dir / "subset/species_tbl.csv"
    else:
        present_sdms_dir = data_dir / "present_sdms"
        future_sdms_dir  = data_dir / "future_sdms"
        trait_file       = data_dir / "species_tbl.csv"

    mask, _ = cn.data_loader.load_map(data_dir / "env_layers/area_mask.npy")

    sdm = cn.load_spatial_data_from_dir(
        dir=present_sdms_dir,
        future_dir=future_sdms_dir,
        mask=mask,
        lower_bound=0,
        upper_bound=1,
        n_time_steps=cfg["n_time_steps"],
        min_threshold=cfg["min_habitat_suitability"],
    )

    disturbance = cn.load_spatial_data(
        file=data_dir / "env_layers/area_swept_disturbance.tif",
        future_file=data_dir / "env_layers/future_area_swept_disturbance.tif",
        mask=mask,
        lower_bound=0,
        upper_bound=1,
        n_time_steps=cfg["n_time_steps"],
    )

    protection = cn.SpatialData(
        data=np.zeros(disturbance.shape),
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

    sensitivity        = traits["sensitivity_disturbance"].to_numpy(copy=True)[:, np.newaxis]
    growth_rates       = traits["growth_rate"].to_numpy(copy=True) + 1.0
    carrying_capacity  = cfg["avg_carrying_capacity"] / traits["conservation_status"].to_numpy(copy=True)
    conservation_status = traits["conservation_status"].to_numpy(copy=True) - 1

    ext_risk = cn.ExtinctionRisk(init_status=conservation_status, n_classes=5, alpha=0.5)

    disp_rate   = cfg["dispersal_rate"]
    disp_window = cfg["dispersal_window"]
    disp_file   = data_dir / f"dispersal_d{disp_rate}_t{disp_window}_NEW.npz"
    if not disp_file.exists():
        logging.info(f"Computing dispersal matrix: {disp_file}")
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
    policy = cn.PolicyNetwork(model, seed=cfg["seed"], device=device)

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
# Main
# =============================================================================

def main():
    args = parse_args()
    cfg  = load_config(args.config, args)

    # Global seeds for reproducibility
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    results_dir    = Path("results") / args.run_name
    calibration_file = results_dir / "reward_calibration.json"
    os.makedirs(results_dir, exist_ok=True)

    # Save config snapshot alongside results for reproducibility
    with open(results_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    print("=" * 60)
    print(f"CAPTAIN — ES Training  |  run: {args.run_name}")
    print("=" * 60)
    print(f"  Device     : {cfg['device']}")
    print(f"  Data       : {args.data_dir}")
    print(f"  Results    : {results_dir}")
    print(f"  Subset     : {cfg['subset']}")
    print(f"  Epochs     : {cfg['n_epochs']}")

    if cfg["device"] in ("cuda", "mps"):
        episode_runners = [create_episode_runner(args.data_dir, cfg, results_dir)]
    else:
        episode_runners = [
            create_episode_runner(args.data_dir, cfg, results_dir)
            for _ in range(cfg["n_parallel_workers"])
        ]

    episode = episode_runners[0]
    print(f"  Grid       : {episode.env.n_cells} cells, {episode.env.n_species} species")
    print(f"  Features   : {episode.feature_extractor.n_features}")
    print(f"  Parameters : {len(episode.policy.get_flat_weights())}")

    trainer = cn.EvolStrategiesTrainer(
        episode_runners,
        initial_coeffs=episode.policy.get_flat_weights(),
        scheduler=cn.LearningScheduler(
            initial_alpha=cfg["initial_alpha"],
            initial_sigma=cfg["initial_sigma"],
        ),
        n_perturbations=cfg["n_perturbations"],
        seed=cfg["seed"],
    )

    wb = WandbLogger(
        enabled=args.wandb,
        project=args.wandb_project,
        name=args.run_name,
        config=cfg,
        group="es_baseline",
    )

    if cfg["calibrate_rewards"]:
        if not calibration_file.exists() or args.recalibrate:
            print(f"\nCalibrating rewards with {cfg['n_probes']} probes...")
            multipliers = trainer.get_reward_calibrated_weights(
                n_probes=cfg["n_probes"], verbose=True
            )
            trainer.save_reward_calibration(multipliers, calibration_file)
        else:
            print(f"\nUsing existing reward calibration: {calibration_file}")

    trainer.load_reward_calibration(calibration_file, verbose=True)

    with open(calibration_file) as f:
        calib_dict = json.load(f)
    print("\nReward calibration multipliers:")
    for name, val in calib_dict.items():
        triggered = val != 1.0
        flag = "" if triggered else "  ← NEVER TRIGGERED (check probe episodes)"
        print(f"  {name}: {val:.4f}{flag}")
    wb.log_raw({"calibration/" + k: v for k, v in calib_dict.items()})

    logger = cn.algorithms.TrainingLogger(
        trainer=trainer,
        episode=episode,
        results_dir=results_dir,
        log_file="training_log.tsv",
        weights_file="trained_weights.npy",
        plot_freq=cfg["plot_train_freq"],
    )

    print(f"\nTraining for {cfg['n_epochs']} epochs...")
    print("-" * 60)
    t_start = time.time()

    for epoch in range(cfg["n_epochs"]):
        t0 = time.time()
        avg_reward, summary = trainer.train_epoch()
        logger.log_epoch(epoch, avg_reward, summary, time.time() - t0)
        wb.log(epoch, avg_reward, summary, trainer)

    print("-" * 60)
    print(f"Done in {time.time() - t_start:.1f}s")
    print(f"Log     : {logger.log_path}")
    print(f"Weights : {logger.weights_path}")

    wb.finish()
    trainer.close()


if __name__ == "__main__":
    main()
