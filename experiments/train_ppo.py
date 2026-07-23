#!/usr/bin/env python
"""Train a conservation policy with PPO (Scenario 3 — deterministic disturbance).

Architecture decisions
----------------------
- K=50 cells protected per step via Plackett-Luce sampling (log-prob available)
- Shared-trunk ActorCriticCellNN: policy head + value head
- CleanRL-style PPO loop: collect T=128 steps → GAE → 4 update epochs

Why PPO over ES
---------------
ES estimates gradients via reward-weighted noise — it is blind to which specific
timestep decisions caused the reward. PPO uses a value function baseline (GAE)
to assign credit to individual timestep actions, enabling learning from the
dense per-step reward signal in CAPTAIN.

Usage:
    uv run python experiments/train_ppo.py \\
        --data-dir /path/to/captain3data \\
        --run-name ppo_baseline \\
        --wandb
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
import torch.nn as nn
import yaml
from torch.distributions import Categorical

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

import captain as cn
from captain.algorithms.budget_manager import GlobalBudgetManager
from experiments.env_extensions import CalcRewardExtRiskLevel, CalcRewardMarginalCost
from experiments.ppo_actor_critic import ActorCriticCellNN, plackett_luce_log_prob, plackett_luce_sample
from experiments.ppo_env_wrapper import CaptainPPOEnv
from experiments.utils.wandb_logger import WandbLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)


# =============================================================================
# Args and config
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train CAPTAIN with PPO")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/ppo_base.yaml"))
    parser.add_argument("--run-name", type=str, default="ppo_run")
    parser.add_argument("--n-updates", type=int, default=None, help="Override n_updates from config")
    parser.add_argument("--rollout-steps", type=int, default=None, help="Override rollout_steps from config")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb-project", type=str, default="captain-dissertation")
    return parser.parse_args()


def load_config(path: Path, args) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if args.n_updates is not None:
        cfg["n_updates"] = args.n_updates
    if args.rollout_steps is not None:
        cfg["rollout_steps"] = args.rollout_steps
    if args.device is not None:
        cfg["device"] = args.device
    else:
        cfg["device"] = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return cfg


# =============================================================================
# Environment setup
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

    rewards = cn.Rewards(
        reward_obj_list=[
            CalcRewardExtRiskLevel(
                threat_weights=np.array([1, 0, -8, -16, -32]), device=device
            ),
            CalcRewardMarginalCost(),
        ],
        reward_weights=np.array([
            cfg.get("reward_weight_ext_risk", 1.0),
            cfg.get("reward_weight_cost", 1.0),
        ]),
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
# Reward calibration
# =============================================================================

def calibrate_rewards(captain_env, model, cfg, device) -> np.ndarray:
    """Run probe episodes with the untrained policy to calibrate reward scales.

    Mirrors ES trainer's get_reward_calibrated_weights(): normalises each
    reward component to std=1 across probes, preventing one component from
    dominating the signal purely due to scale differences.
    """
    n_probes = cfg.get("n_calibration_probes", 20)
    k = cfg["k"]
    all_totals = []

    model.eval()
    print(f"\nCalibrating rewards with {n_probes} probe episodes...")
    for i in range(n_probes):
        obs = captain_env.reset()
        done = False
        with torch.no_grad():
            while not done:
                scores, _ = model(obs.to(device))
                constraint = captain_env.constraint_mask
                action, _ = plackett_luce_sample(scores, k, constraint)
                obs, _, done, _ = captain_env.step(action)
        # Raw per-component sum over the episode (before weights/calibration)
        history = torch.tensor(captain_env.rewards.episode_reward_history, dtype=torch.float32)
        all_totals.append(history.sum(dim=0).numpy())
        if (i + 1) % 5 == 0:
            print(f"  probe {i+1}/{n_probes}")

    all_totals = np.array(all_totals)  # (n_probes, n_components)
    stds = np.std(all_totals, axis=0)
    valid = stds > 1e-6
    multipliers = np.ones_like(stds)
    multipliers[valid] = 1.0 / stds[valid]
    multipliers = np.clip(multipliers, 0.001, 1000.0)

    names = [r._name for r in captain_env.rewards._reward_obj_list]
    print("Calibration multipliers:")
    for name, val, std in zip(names, multipliers, stds):
        flag = "" if std > 1e-6 else "  ← NEVER TRIGGERED"
        print(f"  {name}: {val:.4f}  (probe std={std:.4f}){flag}")

    return multipliers


# =============================================================================
# Rollout buffer
# =============================================================================

class RolloutBuffer:
    """Stores PPO rollout transitions in CPU tensors.

    Why CPU? The observation (n_features, n_cells) ≈ 13×58000 per step.
    Storing T=128 of these on GPU would exhaust VRAM. We collect on CPU
    and move minibatches to GPU during the update.
    """

    def __init__(self, rollout_steps: int, n_features: int, n_cells: int, k: int):
        self.T = rollout_steps
        self.k = k
        self.obs      = torch.zeros(rollout_steps, n_features, n_cells)
        self.actions  = torch.zeros(rollout_steps, k, dtype=torch.long)
        self.logprobs = torch.zeros(rollout_steps)
        self.rewards  = torch.zeros(rollout_steps)
        self.values   = torch.zeros(rollout_steps)
        self.dones    = torch.zeros(rollout_steps)
        self.masks    = torch.zeros(rollout_steps, n_cells, dtype=torch.bool)  # constraint mask per step
        self._ptr = 0

    def add(self, obs, action, logprob, reward, value, done, mask):
        self.obs[self._ptr]      = obs.cpu()
        self.actions[self._ptr]  = action.cpu()
        self.logprobs[self._ptr] = logprob.cpu().detach()
        self.rewards[self._ptr]  = reward
        self.values[self._ptr]   = value.cpu().detach()
        self.dones[self._ptr]    = float(done)
        self.masks[self._ptr]    = mask.cpu()
        self._ptr += 1

    def full(self) -> bool:
        return self._ptr >= self.T

    def reset(self):
        self._ptr = 0


# =============================================================================
# GAE advantage computation
# =============================================================================

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalised Advantage Estimates.

    Why GAE?
    --------
    A plain Monte Carlo return has high variance (every reward in the episode
    affects every advantage estimate). GAE trades a little bias for much
    lower variance via the lambda parameter:
        δt = rt + γ·V(st+1) - V(st)          (TD error)
        At = δt + (γλ)·δt+1 + (γλ)²·δt+2 + ...

    λ=0 → pure TD (low variance, high bias)
    λ=1 → pure MC (high variance, no bias)
    λ=0.95 is the CleanRL default — a good empirical starting point.
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0

    for t in reversed(range(T)):
        next_val = next_value if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    cfg  = load_config(args.config, args)
    device = torch.device(cfg["device"])

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    results_dir = Path("results") / args.run_name
    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    print("=" * 60)
    print(f"CAPTAIN — PPO  |  run: {args.run_name}")
    print("=" * 60)
    print(f"  Device  : {cfg['device']}")
    print(f"  K       : {cfg['k']} cells/step")
    print(f"  Updates : {cfg['n_updates']}")

    # Environment
    captain_env = create_env(args.data_dir, cfg)
    n_features  = captain_env.n_features
    n_cells     = captain_env.n_cells
    print(f"  Grid    : {n_cells} cells, {captain_env.env.n_species} species")
    print(f"  Features: {n_features}")

    # Network and optimiser
    model = ActorCriticCellNN(
        input_dim=n_features,
        hidden_dim=cfg["hidden_dim"],
        activation=cfg["activation"],
    ).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"], eps=1e-5)

    # Rollout buffer (CPU)
    buffer = RolloutBuffer(
        rollout_steps=cfg["rollout_steps"],
        n_features=n_features,
        n_cells=n_cells,
        k=cfg["k"],
    )

    wb = WandbLogger(
        enabled=args.wandb,
        project=args.wandb_project,
        name=args.run_name,
        config=cfg,
        group="ppo_baseline",
    )

    # Logging
    log_path = results_dir / "training_log.tsv"
    with open(log_path, "w") as f:
        f.write("update\tpolicy_loss\tvalue_loss\tentropy\ttotal_loss\treward_mean\ttime\n")

    # Reward calibration
    calibration_file = results_dir / "reward_calibration.json"
    if not calibration_file.exists():
        multipliers = calibrate_rewards(captain_env, model, cfg, device)
        calib_dict = {r._name: float(m) for r, m in zip(captain_env.rewards._reward_obj_list, multipliers)}
        with open(calibration_file, "w") as f:
            json.dump(calib_dict, f, indent=4)
    else:
        print(f"\nUsing existing reward calibration: {calibration_file}")
        with open(calibration_file) as f:
            calib_dict = json.load(f)
        multipliers = np.array(list(calib_dict.values()))

    captain_env.rewards._reward_calibration = torch.tensor(multipliers, dtype=torch.float32)
    wb.log_raw({"calibration/" + k: v for k, v in calib_dict.items()})

    print(f"\nTraining for {cfg['n_updates']} updates...")
    print("-" * 60)

    obs = captain_env.reset()
    done = False
    t_start = time.time()

    for update in range(cfg["n_updates"]):
        t0 = time.time()

        # Learning rate annealing
        if cfg.get("lr_anneal"):
            frac = 1.0 - update / cfg["n_updates"]
            for pg in optimiser.param_groups:
                pg["lr"] = cfg["lr"] * frac

        # ------------------------------------------------------------------
        # Phase 1: Collect rollout
        # ------------------------------------------------------------------
        buffer.reset()
        episode_rewards = []
        ep_reward = 0.0
        episode_ext_risks = []
        episode_transitions = []

        model.eval()
        with torch.no_grad():
            while not buffer.full():
                obs_t = obs.to(device)
                scores, value = model(obs_t)

                constraint = captain_env.constraint_mask
                action, logprob = plackett_luce_sample(scores, cfg["k"], constraint)

                obs_next, reward, done, info = captain_env.step(action)
                ep_reward += reward

                buffer.add(obs, action, logprob, reward, value, done, constraint)
                obs = obs_next

                if done:
                    episode_rewards.append(ep_reward)
                    ep_reward = 0.0
                    if "extinction_risk" in info:
                        episode_ext_risks.append(info["extinction_risk"])
                    if "transition_matrix" in info:
                        episode_transitions.append(info["transition_matrix"])
                    obs = captain_env.reset()
                    done = False

            # Bootstrap value at end of rollout for GAE
            obs_t = obs.to(device)
            _, next_value = model(obs_t)

        # ------------------------------------------------------------------
        # Phase 2: Compute advantages
        # ------------------------------------------------------------------
        advantages, returns = compute_gae(
            buffer.rewards, buffer.values, buffer.dones,
            next_value.detach().cpu(),
            cfg["gamma"], cfg["gae_lambda"],
        )
        # Normalise advantages — reduces variance across minibatches
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ------------------------------------------------------------------
        # Phase 3: PPO update
        # ------------------------------------------------------------------
        model.train()
        indices = np.arange(cfg["rollout_steps"])
        pl_losses, vf_losses, ent_losses = [], [], []

        for _ in range(cfg["n_epochs_per_update"]):
            np.random.shuffle(indices)
            for start in range(0, cfg["rollout_steps"], cfg["minibatch_size"]):
                mb_idx = indices[start : start + cfg["minibatch_size"]]

                mb_obs   = buffer.obs[mb_idx].to(device)
                mb_acts  = buffer.actions[mb_idx].to(device)
                mb_logp  = buffer.logprobs[mb_idx].to(device)
                mb_adv   = advantages[mb_idx].to(device)
                mb_ret   = returns[mb_idx].to(device)
                mb_masks = buffer.masks[mb_idx].to(device)

                # Recompute log-probs and values under current policy
                new_logprobs = []
                new_values   = []
                entropies    = []

                for i in range(len(mb_idx)):
                    obs_i    = mb_obs[i]
                    acts_i   = mb_acts[i]
                    mask_i   = mb_masks[i]
                    scores_i, val_i = model(obs_i)

                    lp = plackett_luce_log_prob(scores_i, acts_i, mask_i)
                    new_logprobs.append(lp)
                    new_values.append(val_i)

                    # Entropy: average entropy of the per-step categoricals
                    ent = Categorical(logits=scores_i).entropy()
                    entropies.append(ent)

                new_logprobs = torch.stack(new_logprobs)
                new_values   = torch.stack(new_values)
                entropy      = torch.stack(entropies).mean()

                # PPO clipped policy loss
                # Why clip? Prevents the policy from taking too large an update step,
                # which can collapse performance. The ratio r = exp(new_logp - old_logp)
                # measures how much the policy changed; clipping to [1-ε, 1+ε] limits this.
                log_ratio = new_logprobs - mb_logp
                ratio     = torch.exp(log_ratio)
                pg_loss1  = -mb_adv * ratio
                pg_loss2  = -mb_adv * torch.clamp(ratio, 1 - cfg["clip_coef"], 1 + cfg["clip_coef"])
                pg_loss   = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped to match policy clip)
                vf_loss = nn.functional.mse_loss(new_values, mb_ret)

                # Total loss
                loss = pg_loss + cfg["vf_coef"] * vf_loss - cfg["ent_coef"] * entropy

                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                optimiser.step()

                pl_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(entropy.item())

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        mean_reward   = np.mean(episode_rewards) if episode_rewards else float("nan")
        mean_pl_loss  = np.mean(pl_losses)
        mean_vf_loss  = np.mean(vf_losses)
        mean_entropy  = np.mean(ent_losses)
        total_loss    = mean_pl_loss + cfg["vf_coef"] * mean_vf_loss - cfg["ent_coef"] * mean_entropy
        elapsed       = time.time() - t0

        # Average extinction risk counts across completed episodes this rollout
        mean_ext_risk = {}
        if episode_ext_risks:
            for key in episode_ext_risks[0]:
                mean_ext_risk[key] = float(np.mean([e[key] for e in episode_ext_risks]))

        # Sum transition matrices across episodes (rows=initial, cols=final)
        mean_transition = None
        if episode_transitions:
            mean_transition = torch.stack(episode_transitions).float().mean(dim=0)

        print(
            f"Update {update:4d} | "
            f"reward: {mean_reward:7.3f} | "
            f"pl: {mean_pl_loss:6.4f} | "
            f"vf: {mean_vf_loss:6.4f} | "
            f"ent: {mean_entropy:5.3f} | "
            f"time: {elapsed:.1f}s"
        )
        if mean_ext_risk:
            risk_str = "  ".join(f"{k}:{v:.1f}" for k, v in mean_ext_risk.items())
            print(f"           | ext_risk: {risk_str}")

        with open(log_path, "a") as f:
            f.write(f"{update}\t{mean_pl_loss:.6f}\t{mean_vf_loss:.6f}\t"
                    f"{mean_entropy:.6f}\t{total_loss:.6f}\t{mean_reward:.4f}\t{elapsed:.1f}\n")

        if args.wandb:
            wandb_data = {
                "update": update,
                "reward/mean": mean_reward,
                "loss/policy": mean_pl_loss,
                "loss/value": mean_vf_loss,
                "loss/entropy": mean_entropy,
                "lr": optimiser.param_groups[0]["lr"],
                **{f"extinction_risk/{k}": v for k, v in mean_ext_risk.items()},
            }
            if mean_transition is not None:
                n = mean_transition.shape[0]
                class_names = ["LC", "NT", "VU", "EN", "CR"][:n]
                for r in range(n):
                    for c in range(n):
                        wandb_data[f"transition/{class_names[r]}_to_{class_names[c]}"] = mean_transition[r, c].item()
            wb.log_raw(wandb_data)

        if update % cfg["plot_train_freq"] == 0:
            torch.save(model.state_dict(), results_dir / f"weights_update_{update}.pt")

    torch.save(model.state_dict(), results_dir / "trained_weights.pt")
    print("-" * 60)
    print(f"Done in {time.time() - t_start:.1f}s")
    wb.finish()


if __name__ == "__main__":
    main()
