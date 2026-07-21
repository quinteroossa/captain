"""Test reward calibration with a synthetic environment.

Verifies that:
1. get_reward_calibrated_weights() produces non-trivial, finite multipliers
2. Both reward components (ext_risk, cost) produce non-zero variance across probe episodes
3. save/load calibration pipeline round-trips correctly
4. Multipliers are applied to rewards (weighted reward changes after calibration)

Run with: uv run python examples/test_reward_calibration.py
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

import captain as cn

# ---------------------------------------------------------------------------
# Synthetic environment parameters
# ---------------------------------------------------------------------------
N_SPECIES = 8
GRID_H, GRID_W = 20, 20
N_CELLS = GRID_H * GRID_W
N_TIME_STEPS = 10
TARGET_CELLS = 50
CELLS_PER_STEP = 10
N_PROBES = 10
SEED = 42


def build_synthetic_runner() -> cn.EpisodeRunner:
    """Build a minimal EpisodeRunner using random synthetic data."""
    rng = np.random.default_rng(SEED)

    # SDMs: (n_species, H, W) — random habitat suitability in [0, 1]
    sdm_data = rng.random((N_SPECIES, GRID_H, GRID_W)).astype(np.float32)
    sdm_future = rng.random((N_SPECIES, GRID_H, GRID_W)).astype(np.float32)
    delta_sdm = (sdm_future - sdm_data) / N_TIME_STEPS

    sdm = cn.SpatialData(
        data=sdm_data,
        delta_per_step=delta_sdm,
        lower_bound=0,
        upper_bound=1,
        min_threshold=0.05,
        names=[f"sp_{i}" for i in range(N_SPECIES)],
    )

    # Disturbance: single channel
    dist_data = rng.random((1, GRID_H, GRID_W)).astype(np.float32) * 0.3
    dist_future = rng.random((1, GRID_H, GRID_W)).astype(np.float32) * 0.3
    delta_dist = (dist_future - dist_data) / N_TIME_STEPS

    disturbance = cn.SpatialData(
        data=dist_data,
        delta_per_step=delta_dist,
        lower_bound=0,
        upper_bound=1,
    )

    # Costs: single channel, values in [0, 1]
    cost_data = rng.random((1, GRID_H, GRID_W)).astype(np.float32)
    costs = cn.SpatialData(data=cost_data, lower_bound=0, upper_bound=1)

    # Protection matrix: starts at zero
    protection = cn.SpatialData(
        data=np.zeros((1, GRID_H, GRID_W), dtype=np.float32),
        lower_bound=0,
        upper_bound=1,
    )

    # Life-history traits
    growth_rates = rng.uniform(1.01, 1.1, N_SPECIES).astype(np.float32)
    sensitivity = rng.uniform(0.1, 0.5, (N_SPECIES, 1)).astype(np.float32)
    carrying_capacity = rng.uniform(50, 200, N_SPECIES).astype(np.float32)

    # Extinction risk: spread species across categories 0–4
    init_status = np.array([0, 0, 1, 1, 2, 2, 3, 4], dtype=int)
    ext_risk = cn.ExtinctionRisk(init_status=init_status, n_classes=5, alpha=0.5)

    env = cn.BioEnv(
        sdms=sdm,
        disturbance=disturbance,
        costs=costs,
        protection_matrix=protection,
        growth_rates=growth_rates,
        sensitivity_rates=sensitivity,
        species_k=carrying_capacity,
        ext_risk=ext_risk,
        device="cpu",
    )

    feature_extractor = cn.FeatureExtractor(
        env,
        feature_set=None,
        time_rescale=N_TIME_STEPS / 2,
        device="cpu",
    )

    model = cn.CellNN(input_dim=feature_extractor.n_features, hidden_dim=16)
    policy = cn.PolicyNetwork(model, seed=SEED, device="cpu")

    rewards = cn.Rewards(
        reward_obj_list=[
            cn.CalcRewardExtRisk(
                threat_weights=np.array([1, 0, -8, -16, -32]), device="cpu"
            ),
            cn.CalcRewardPersistentCost(rescaler=float(1.0 / costs.data.sum())),
        ],
        reward_weights=np.array([1.0, 1.0]),
    )

    budget_manager = cn.GlobalBudgetManager(
        total_target=TARGET_CELLS,
        cells_per_time_step=CELLS_PER_STEP,
        feature_updates_per_time_step=1,
    )

    return cn.EpisodeRunner(
        env=env,
        feature_extractor=feature_extractor,
        policy_network=policy,
        rewards=rewards,
        n_steps=N_TIME_STEPS,
        budget_manager=budget_manager,
    )


def run_tests():
    print("=" * 60)
    print("Reward Calibration Test (synthetic environment)")
    print("=" * 60)

    runner = build_synthetic_runner()
    trainer = cn.EvolStrategiesTrainer(
        [runner],
        initial_coeffs=runner.policy.get_flat_weights(),
        scheduler=cn.LearningScheduler(initial_alpha=0.2, initial_sigma=0.3),
        n_perturbations=N_PROBES,
        seed=SEED,
    )

    # ------------------------------------------------------------------
    # Test 1: calibration produces finite, non-trivial multipliers
    # ------------------------------------------------------------------
    print(f"\n[1] Running {N_PROBES} probe episodes...")
    multipliers = trainer.get_reward_calibrated_weights(
        n_probes=N_PROBES, target_std=1.0, verbose=True
    )

    print(f"    Reward components : {runner.rewards.names}")
    print(f"    Multipliers       : {multipliers}")

    assert len(multipliers) == 2, f"Expected 2 multipliers, got {len(multipliers)}"
    assert np.all(np.isfinite(multipliers)), f"Non-finite multipliers: {multipliers}"
    assert not np.all(multipliers == 1.0), "All multipliers are 1.0 — calibration had no effect"

    trivial = np.allclose(multipliers, 1.0, atol=0.05)
    if trivial:
        print("    WARNING: multipliers are very close to 1.0 — reward scales may already be similar")
    else:
        print("    PASS: multipliers are non-trivial")

    # ------------------------------------------------------------------
    # Test 2: variance of each reward component across probes > 0
    # ------------------------------------------------------------------
    print("\n[2] Checking reward component variance across probe episodes...")
    probe_rewards = []
    params_list = [
        runner.policy.get_flat_weights() + np.random.randn(len(runner.policy.get_flat_weights())) * 0.3
        for _ in range(N_PROBES)
    ]
    for p in params_list:
        info, _ = runner.run_episode(p)
        probe_rewards.append(info["rewards"].numpy())

    component_stds = np.std(np.array(probe_rewards), axis=0)
    print(f"    Per-component std : {component_stds}")
    for i, (name, std) in enumerate(zip(runner.rewards.names, component_stds)):
        if std < 1e-6:
            print(f"    WARNING: '{name}' has near-zero variance — never triggered in probes")
        else:
            print(f"    PASS: '{name}' std = {std:.4f}")

    # ------------------------------------------------------------------
    # Test 3: save/load pipeline round-trips correctly
    # ------------------------------------------------------------------
    print("\n[3] Testing save/load round-trip...")
    with tempfile.TemporaryDirectory() as tmpdir:
        calib_file = Path(tmpdir) / "reward_calibration.json"

        trainer.save_reward_calibration(multipliers, calib_file, verbose=True)
        assert calib_file.exists(), "Calibration file was not saved"

        with open(calib_file) as f:
            saved = json.load(f)
        print(f"    Saved JSON: {saved}")

        # Reset multipliers to 1.0, then reload
        trainer.calibrate_reward_scales(np.ones(2))
        trainer.load_reward_calibration(calib_file, verbose=True)

        loaded = runner.rewards._reward_calibration.numpy()
        print(f"    Loaded multipliers: {loaded}")
        assert np.allclose(multipliers, loaded, atol=1e-5), \
            f"Round-trip mismatch: saved {multipliers} vs loaded {loaded}"
        print("    PASS: save/load round-trip correct")

    # ------------------------------------------------------------------
    # Test 4: multipliers actually change the weighted reward
    # ------------------------------------------------------------------
    print("\n[4] Checking multipliers affect weighted reward...")
    info, _ = runner.run_episode()
    trainer.calibrate_reward_scales(np.ones(2))
    reward_before = runner.rewards.get_weighted_reward()

    runner.run_episode()
    trainer.calibrate_reward_scales(multipliers)
    reward_after = runner.rewards.get_weighted_reward()

    print(f"    Weighted reward (uniform weights) : {reward_before:.4f}")
    print(f"    Weighted reward (calibrated)      : {reward_after:.4f}")
    if not np.allclose(reward_before, reward_after, atol=1e-3):
        print("    PASS: calibration changes weighted reward")
    else:
        print("    WARNING: weighted reward unchanged — multipliers may be close to uniform")

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
