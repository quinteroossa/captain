#!/usr/bin/env python
"""Train CVaR-PPO with stochastic disturbance (RQ3).

Static CVaR-PPO after Tang et al. (2020) "Worst Cases Policy Gradients":
  1. Collect N complete episodes under stochastic disturbance.
  2. Rank by total return; keep worst ceil(cvar_alpha * N) episodes.
  3. Run a standard PPO update on only those worst-case trajectories.

This optimises CVaR_alpha(G) — the expected return over the worst alpha
fraction of disturbance draws — rather than E[G]. The episode is the
correct unit because disturbance intensity is resampled i.i.d. at each
reset(), making the return distribution aleatoric across episodes.

The naive CVaR Bellman operator (applying CVaR step-by-step) is NOT used
because CVaR is not decomposable via Bellman and is known to misconverge
(Tamar et al. 2015).

Usage:
    uv run python experiments/train_ppo_cvar.py \\
        --config experiments/configs/ppo_cvar.yaml \\
        --data-dir /path/to/captain3data \\
        --run-name ppo_cvar_v1 \\
        --wandb
"""

import os
import time
import yaml
import logging
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

from experiments.train_ppo import (
    parse_args,
    load_config,
    normalize_obs,
    compute_gae,
)
from experiments.train_ppo_stochastic import create_env
from experiments.ppo_actor_critic import (
    ActorCriticCellNN,
    plackett_luce_log_prob,
    plackett_luce_sample,
)
from experiments.utils.wandb_logger import WandbLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)


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

    n_episodes = cfg["n_episodes_per_update"]
    cvar_alpha = cfg["cvar_alpha"]
    n_keep     = max(1, int(np.ceil(cvar_alpha * n_episodes)))

    print("=" * 60)
    print(f"CAPTAIN — CVaR-PPO  |  run: {args.run_name}")
    print("=" * 60)
    print(f"  Device      : {cfg['device']}")
    print(f"  K           : {cfg['k']} cells/step")
    print(f"  Updates     : {cfg['n_updates']}")
    print(f"  Episodes/upd: {n_episodes}  →  keep worst {n_keep} (alpha={cvar_alpha})")

    captain_env = create_env(args.data_dir, cfg)
    n_features  = captain_env.n_features
    n_cells     = captain_env.n_cells
    print(f"  Grid        : {n_cells} cells, {captain_env.env.n_species} species")

    model = ActorCriticCellNN(
        input_dim=n_features,
        hidden_dim=cfg["hidden_dim"],
        activation=cfg["activation"],
    ).to(device)

    if args.resume_from is not None:
        model.load_state_dict(torch.load(args.resume_from, map_location=device))
        print(f"  Resumed     : {args.resume_from}  (start_update={args.start_update})")

    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"], eps=1e-5)

    captain_env.rewards._reward_calibration = torch.ones(
        len(captain_env.rewards._reward_obj_list), dtype=torch.float32
    )

    wb = WandbLogger(
        enabled=args.wandb,
        project=args.wandb_project,
        name=args.run_name,
        config=cfg,
        group="ppo_cvar",
    )

    log_path = results_dir / "training_log.tsv"
    with open(log_path, "w") as f:
        f.write("update\tpolicy_loss\tvalue_loss\tentropy\tmean_return\tcvar_value\ttime\n")

    print(f"\nTraining for {cfg['n_updates']} updates...")
    print("-" * 60)

    t_start = time.time()

    for update in range(cfg["n_updates"]):
        t0 = time.time()

        # LR annealing — continues correctly when resuming via --start-update
        if cfg.get("lr_anneal"):
            total_updates = cfg["n_updates"] + args.start_update
            frac = 1.0 - (args.start_update + update) / total_updates
            for pg in optimiser.param_groups:
                pg["lr"] = cfg["lr"] * frac

        # ------------------------------------------------------------------
        # Phase 1: Collect N complete episodes
        # ------------------------------------------------------------------
        episodes = []  # list of (total_return, steps, info)

        model.eval()
        with torch.no_grad():
            for _ in range(n_episodes):
                steps = []
                ep_return = 0.0
                obs  = captain_env.reset()
                done = False

                while not done:
                    obs_t  = normalize_obs(obs.to(device))
                    scores, value = model(obs_t)
                    constraint    = captain_env.constraint_mask
                    action, logprob = plackett_luce_sample(scores, cfg["k"], constraint)

                    obs_next, reward, done, info = captain_env.step(action)
                    ep_return += reward

                    steps.append((
                        obs.cpu(),
                        action.cpu(),
                        logprob.cpu().detach(),
                        reward,
                        value.squeeze().cpu().detach(),
                        done,
                        constraint.cpu(),
                    ))
                    obs = obs_next

                episodes.append((ep_return, steps, info))

        # ------------------------------------------------------------------
        # Phase 2: Filter to worst alpha fraction (CVaR filter)
        # ------------------------------------------------------------------
        episodes.sort(key=lambda x: x[0])  # ascending: lowest return first
        worst    = episodes[:n_keep]
        all_rets = [ep[0] for ep in episodes]
        cvar_value  = float(np.mean([ep[0] for ep in worst]))
        mean_return = float(np.mean(all_rets))

        # ------------------------------------------------------------------
        # Phase 3: Build update tensors from worst episodes
        # ------------------------------------------------------------------
        all_obs, all_actions, all_logprobs = [], [], []
        all_values, all_masks = [], []
        all_advantages, all_returns_gae = [], []
        worst_ext_risks   = []
        worst_transitions = []

        for _, steps, info in worst:
            rewards_ep = torch.tensor([s[3] for s in steps], dtype=torch.float32)
            values_ep  = torch.tensor([s[4].item() for s in steps], dtype=torch.float32)
            dones_ep   = torch.tensor([float(s[5]) for s in steps], dtype=torch.float32)

            # Normalize per-episode rewards (same as standard PPO)
            r_std = rewards_ep.std()
            if r_std > 1e-8:
                rewards_ep = (rewards_ep - rewards_ep.mean()) / r_std

            # GAE with next_value=0: episodes always terminate cleanly
            adv_ep, ret_ep = compute_gae(
                rewards_ep, values_ep, dones_ep,
                torch.tensor(0.0), cfg["gamma"], cfg["gae_lambda"],
            )

            all_obs.extend([s[0] for s in steps])
            all_actions.extend([s[1] for s in steps])
            all_logprobs.extend([s[2] for s in steps])
            all_values.extend([s[4] for s in steps])
            all_masks.extend([s[6] for s in steps])
            all_advantages.append(adv_ep)
            all_returns_gae.append(ret_ep)

            if "extinction_risk" in info:
                worst_ext_risks.append(info["extinction_risk"])
            if "transition_matrix" in info:
                worst_transitions.append(info["transition_matrix"])

        obs_tensor   = torch.stack(all_obs)
        acts_tensor  = torch.stack(all_actions)
        logp_tensor  = torch.stack(all_logprobs)
        vals_tensor  = torch.stack(all_values)
        masks_tensor = torch.stack(all_masks)
        adv_tensor   = torch.cat(all_advantages)
        ret_tensor   = torch.cat(all_returns_gae)

        # Normalize advantages across all filtered transitions
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        T_filtered = len(obs_tensor)

        # ------------------------------------------------------------------
        # Phase 4: PPO update on filtered transitions
        # ------------------------------------------------------------------
        model.train()
        indices = np.arange(T_filtered)
        pl_losses, vf_losses, ent_losses = [], [], []

        for _ in range(cfg["n_epochs_per_update"]):
            np.random.shuffle(indices)
            for start in range(0, T_filtered, cfg["minibatch_size"]):
                mb_idx = indices[start : start + cfg["minibatch_size"]]

                mb_obs   = obs_tensor[mb_idx]
                mb_acts  = acts_tensor[mb_idx].to(device)
                mb_logp  = logp_tensor[mb_idx].to(device)
                mb_adv   = adv_tensor[mb_idx].to(device)
                mb_ret   = ret_tensor[mb_idx].to(device)
                mb_masks = masks_tensor[mb_idx].to(device)
                mb_vals  = vals_tensor[mb_idx].to(device)

                new_logprobs, new_values, entropies = [], [], []
                for i in range(len(mb_idx)):
                    obs_i   = normalize_obs(mb_obs[i].to(device))
                    acts_i  = mb_acts[i]
                    mask_i  = mb_masks[i]
                    scores_i, val_i = model(obs_i)

                    lp  = plackett_luce_log_prob(scores_i, acts_i, mask_i)
                    ent = Categorical(logits=scores_i).entropy()
                    new_logprobs.append(lp)
                    new_values.append(val_i)
                    entropies.append(ent)

                new_logprobs = torch.stack(new_logprobs)
                new_values   = torch.stack(new_values)
                entropy      = torch.stack(entropies).mean()

                log_ratio = new_logprobs - mb_logp
                ratio     = torch.exp(log_ratio)
                pg_loss   = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - cfg["clip_coef"], 1 + cfg["clip_coef"]),
                ).mean()

                v_clipped = mb_vals + torch.clamp(
                    new_values - mb_vals, -cfg["clip_coef"], cfg["clip_coef"]
                )
                vf_loss = 0.5 * torch.max(
                    (new_values - mb_ret) ** 2,
                    (v_clipped  - mb_ret) ** 2,
                ).mean()

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
        mean_pl  = float(np.mean(pl_losses))
        mean_vf  = float(np.mean(vf_losses))
        mean_ent = float(np.mean(ent_losses))
        elapsed  = time.time() - t0

        mean_ext_risk = {}
        if worst_ext_risks:
            for key in worst_ext_risks[0]:
                mean_ext_risk[key] = float(np.mean([e[key] for e in worst_ext_risks]))

        mean_transition = None
        if worst_transitions:
            mean_transition = torch.stack(worst_transitions).float().mean(dim=0)

        print(
            f"Update {update:4d} | "
            f"ret_mean: {mean_return:7.3f}  cvar: {cvar_value:7.3f} | "
            f"pl: {mean_pl:6.4f}  vf: {mean_vf:6.4f}  ent: {mean_ent:5.3f} | "
            f"time: {elapsed:.1f}s"
        )
        if mean_ext_risk:
            risk_str = "  ".join(f"{k}:{v:.1f}" for k, v in mean_ext_risk.items())
            print(f"           | ext_risk (worst eps): {risk_str}")

        with open(log_path, "a") as f:
            f.write(
                f"{args.start_update + update}\t{mean_pl:.6f}\t{mean_vf:.6f}\t"
                f"{mean_ent:.6f}\t{mean_return:.4f}\t{cvar_value:.4f}\t{elapsed:.1f}\n"
            )

        if args.wandb:
            wandb_data = {
                "update":           args.start_update + update,
                "reward/mean":      mean_return,
                "reward/cvar":      cvar_value,
                "loss/policy":      mean_pl,
                "loss/value":       mean_vf,
                "loss/entropy":     mean_ent,
                "lr":               optimiser.param_groups[0]["lr"],
                **{f"extinction_risk/threat_{i}": v
                   for i, v in enumerate(mean_ext_risk.values())},
            }
            if mean_transition is not None:
                n = mean_transition.shape[0]
                class_names = ["LC", "NT", "VU", "EN", "CR"][:n]
                for r in range(n):
                    for c in range(n):
                        wandb_data[f"transition/{class_names[r]}_to_{class_names[c]}"] = (
                            mean_transition[r, c].item()
                        )
            wb.log_raw(wandb_data)

        if update % cfg.get("plot_train_freq", 50) == 0:
            torch.save(model.state_dict(), results_dir / f"weights_update_{args.start_update + update}.pt")

    torch.save(model.state_dict(), results_dir / "trained_weights.pt")
    print("-" * 60)
    print(f"Done in {time.time() - t_start:.1f}s")
    wb.finish()


if __name__ == "__main__":
    main()
