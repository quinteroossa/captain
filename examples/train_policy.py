#!/usr/bin/env python
"""Example: Train a conservation policy with Evolution Strategies.

This script demonstrates how to:
1. Load real spatial data (GeoTIFF habitat suitability maps)
2. Set up multiple parallel episode runners
3. Train a policy using Evolution Strategies
4. Log training progress

Requirements:
- Example data in DATA_DIR (see below)
- Species trait CSV file

To run with synthetic data instead, see run_episode.py.
"""

import warnings

# Filter out the specific PyTorch Sparse CSR beta warning
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

import captain as cn

# Configure logging to print INFO messages to your console/Slurm log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],  # Sends output to the terminal/stderr
)

# Device configuration: run on GPU is available (needs CUDA)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
# =============================================================================
# Configuration
# =============================================================================

# Data paths - UPDATE THESE to point to your data
DATA_DIR = Path("/Users/quinteroossa/Documents/ucl_dissertation/captain_data/captain3data")  # <-- Change this!

# Here using a subset of 30 species for testing
PRESENT_SDMS_DIR = "subset/present_sdms"
FUTURE_SDMS_DIR = "subset/future_sdms"
SPECIES_TRAIT_FILE = "subset/species_tbl.csv"
DISTURBANCE_FILE = "env_layers/area_swept_disturbance.tif"
FUTURE_DISTURBANCE_FILE = "env_layers/future_area_swept_disturbance.tif"
COST_FILE = "env_layers/cost.tif"
FUTURE_COST_FILE = "env_layers/future_cost.tif"
DATA_MASK = "env_layers/area_mask.npy"

# Training and policy parameters
N_EPOCHS = 3
N_PERTURBATIONS = 2  # Number of parallel episode evaluations (sequential on GPU)
N_PROBES = 5  # Number of perturbations for reward calibration
N_PARALLEL_WORKERS = 2  # Number of CPUs (if not CUDA)
N_TIME_STEPS = 50  # Time duration of each episode
TARGET_PROTECTED_CELLS = 17000  # Total number of cells to be protected
CELLS_PER_STEP = 1000  # number of new cells protected at each time step
CALIBRATE_REWARDS = True

# Species parameters
AVG_CARRYING_CAPACITY = 100  # 'individuals' per cells (* empirical relative abundance)
DISPERSAL_RATE = 0.5  # can be an array (per-species values)
DISPERSAL_WINDOW = 3
MIN_HABITAT_SUITABILITY = 0.05  # can be an array (per-species values)

# Output
RESULTS_DIR = DATA_DIR / "training_results"
LOG_FILE = "training_log.tsv"
MODEL_FILE = "trained_weights.npy"
CALIBRATION_FILE = DATA_DIR / "reward_calibration.json"
PLOT_FEATURES = False
PLOT_TRAIN_FREQ = 1  # plot intermediate protection results during training
PLOT_DATA = False

os.makedirs(RESULTS_DIR, exist_ok=True)
if PLOT_FEATURES:
    # RESULTS_DIR_PLOTS = RESULTS_DIR / "data_plots"
    # os.makedirs(RESULTS_DIR_PLOTS, exist_ok=True)
    RESULTS_DIR_FEATURE_PLOTS = RESULTS_DIR / "feature_plots"
    os.makedirs(RESULTS_DIR_FEATURE_PLOTS, exist_ok=True)


# =============================================================================
# Episode Setup Function
# =============================================================================


def create_episode_runner() -> cn.EpisodeRunner:
    """Create an episode runner with real data.

    This function loads spatial data and creates all components
    needed for one episode runner. Called once per parallel worker.
    """
    # Load present and future species distribution maps
    global PLOT_DATA, PLOT_FEATURES
    mask, _ = cn.data_loader.load_map(DATA_DIR / DATA_MASK)

    sdm = cn.load_spatial_data_from_dir(
        dir=DATA_DIR / PRESENT_SDMS_DIR,
        future_dir=DATA_DIR / FUTURE_SDMS_DIR,
        mask=mask,
        lower_bound=0,
        upper_bound=1,
        n_time_steps=N_TIME_STEPS,
        min_threshold=MIN_HABITAT_SUITABILITY,
    )

    # Load disturbance layer with predicted future change
    disturbance = cn.load_spatial_data(
        file=DATA_DIR / DISTURBANCE_FILE,
        future_file=DATA_DIR / FUTURE_DISTURBANCE_FILE,
        mask=mask,
        lower_bound=0,
        upper_bound=1,
        n_time_steps=N_TIME_STEPS,
    )

    # Protection matrix (starts empty)
    protection = cn.SpatialData(
        data=np.zeros(disturbance.shape),
        mask=mask,
        lower_bound=0,
        upper_bound=1,
    )

    # Load costs with predicted future change
    costs = cn.load_spatial_data(
        file=DATA_DIR / COST_FILE,
        future_file=DATA_DIR / FUTURE_COST_FILE,
        mask=mask,
        lower_bound=0,
        upper_bound=1,
        n_time_steps=N_TIME_STEPS,
    )

    # Load species traits
    # simple imputation of missing data (could be replaced e.g. RF imputation)
    traits = cn.data_loader.load_trait_table(
        DATA_DIR / SPECIES_TRAIT_FILE, sdm.names, ref_column="species", fill_gaps=True
    )

    # extract parameters for simulation
    sensitivity = traits["sensitivity_disturbance"].to_numpy(copy=True)[:, np.newaxis]
    growth_rates = traits["growth_rate"].to_numpy(copy=True) + 1.0
    carrying_capacity = AVG_CARRYING_CAPACITY / traits["conservation_status"].to_numpy(
        copy=True
    )
    conservation_status = traits["conservation_status"].to_numpy(copy=True) - 1

    # Initial extinction risk from conservation status
    ext_risk = cn.ExtinctionRisk(
        init_status=conservation_status,
        n_classes=5,
        alpha=0.5,
    )

    # Load or create dispersal matrix (cached for efficiency)
    disp_file = DATA_DIR / f"dispersal_d{DISPERSAL_RATE}_t{DISPERSAL_WINDOW}_NEW.npz"
    if not disp_file.exists():
        print(f"Creating dispersal matrix: {disp_file}")
        cn.grid_utils.save_dispersal_distances(
            lambda_0=DISPERSAL_RATE,
            coords=sdm._coords,
            threshold=DISPERSAL_WINDOW,
            filename=str(disp_file),
        )
    dispersal_matrix = cn.grid_utils.load_dispersal_distances(str(disp_file))

    # dispersal_matrix = None
    # dispersal_rates = np.random.random(sdm.shape[0])

    # Create environment
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
        device=DEVICE,
    )

    # Create agent components
    feature_extractor = cn.FeatureExtractor(
        env,
        feature_set=None,  # Use default feature set (can be customized)
        time_rescale=N_TIME_STEPS / 2,
        device=DEVICE,
    )

    if PLOT_FEATURES:
        feature_extractor.plot_features(
            env, rescale=False, outdir=RESULTS_DIR_FEATURE_PLOTS
        )
        PLOT_FEATURES = False

    env.ext_risk.species_per_class(env.current_ext_risk)

    model = cn.CellNN(input_dim=feature_extractor.n_features, hidden_dim=16)
    policy = cn.PolicyNetwork(model, seed=SEED, device=DEVICE)

    rewards = cn.Rewards(
        reward_obj_list=[
            cn.CalcRewardExtRisk(
                threat_weights=np.array([1, 0, -8, -16, -32]), device=DEVICE
            ),
            cn.CalcRewardPersistentCost(rescaler=float(1.0 / costs.data.sum())),
        ],
        reward_weights=np.array([1.0, 1.0]),
    )

    if PLOT_DATA:
        cn.plots.plot_grid(
            np.sum(env.reconstruct_h_grid > 1, axis=0),
            title="species richness",
            outfile=RESULTS_DIR_FEATURE_PLOTS / "species_richness",
        )
        PLOT_DATA = False

    # Create episode runner
    budget_manager = cn.GlobalBudgetManager(
        total_target=TARGET_PROTECTED_CELLS,
        cells_per_time_step=CELLS_PER_STEP,
        feature_updates_per_time_step=1,
    )

    ep = cn.EpisodeRunner(
        env=env,
        feature_extractor=feature_extractor,
        policy_network=policy,
        rewards=rewards,
        n_steps=N_TIME_STEPS,
        budget_manager=budget_manager,
    )

    # plot end features
    if PLOT_FEATURES:
        ep.run_episode()
        feature_extractor.plot_features(
            env, rescale=False, outdir=RESULTS_DIR_FEATURE_PLOTS
        )

    return ep


# =============================================================================
# Main Training Loop
# =============================================================================


def main():
    print("=" * 60)
    print("CAPTAIN-3 Training")
    print("=" * 60)

    # Check data directory
    if not DATA_DIR.exists() or str(DATA_DIR) == "/path/to/your/data":
        print("\nERROR: Please update DATA_DIR in this script to point to your data.")
        print("       See the example data repository for the expected format.")
        raise FileNotFoundError

    print(f"\nDevice: {DEVICE}")

    if DEVICE == "cuda":
        # GPU mode: single runner, sequential perturbations (avoids pickling CUDA tensors)
        print(
            f"Creating 1 episode runner on {DEVICE} ({N_PERTURBATIONS} perturbations)..."
        )
        episode_runners = [create_episode_runner()]
    else:
        # CPU mode: multiple runners with multiprocessing pool
        print(f"Creating {N_PARALLEL_WORKERS} parallel episode runners on CPU...")
        episode_runners = [create_episode_runner() for _ in range(N_PARALLEL_WORKERS)]

    # Get reference to first runner for logging
    episode = episode_runners[0]
    print(f"  Grid: {episode.env.n_cells} cells, {episode.env.n_species} species")
    print(f"  Features: {episode.feature_extractor.n_features}")
    print(f"  Policy parameters: {len(episode.policy.get_flat_weights())}")

    # Create trainer
    trainer = cn.EvolStrategiesTrainer(
        episode_runners,
        initial_coeffs=episode.policy.get_flat_weights(),
        scheduler=cn.LearningScheduler(initial_alpha=0.2, initial_sigma=0.3),
        epsilon_reward=0.5,
        n_perturbations=N_PERTURBATIONS,
        seed=SEED,
    )

    # Heuristic Reward Calibration
    if CALIBRATE_REWARDS:
        multipliers = trainer.get_reward_calibrated_weights(
            n_probes=N_PROBES, verbose=True
        )
        trainer.save_reward_calibration(multipliers, CALIBRATION_FILE)

    # Initialize Logger
    trainer.load_reward_calibration(CALIBRATION_FILE, verbose=True)

    logger = cn.algorithms.TrainingLogger(
        trainer=trainer,
        episode=episode,
        results_dir=RESULTS_DIR,
        log_file=LOG_FILE,
        weights_file=MODEL_FILE,
        plot_freq=PLOT_TRAIN_FREQ,
    )

    # Training loop
    print(f"\nTraining for {N_EPOCHS} epochs...")
    print("-" * 60)

    t_start = time.time()

    for epoch in range(N_EPOCHS):
        t0 = time.time()
        avg_reward, summary = trainer.train_epoch()
        # Log progress
        logger.log_epoch(epoch, avg_reward, summary, time.time() - t0)

    # Summary
    print("-" * 60)
    print(f"Training complete in {time.time() - t_start:.1f}s")
    print(f"Log saved to: {logger.log_path}")
    print(f"Weights saved to: {logger.weights_path}")
    # TODO save scheduler checkpoint to restart

    # Cleanup
    trainer.close()


if __name__ == "__main__":
    main()

"""
# Check hex code (memory address)
print(hex(id(trainer)))
print(hex(id(logger.trainer)))
"""
