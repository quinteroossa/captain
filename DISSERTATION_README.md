# CAPTAIN — Dissertation Extension

**UCL MSc Dissertation 2026**  
Ana Maria Quintero Ossa — supervised by Maria Perez-Ortiz (UCL) and Daniele Silvestro (original CAPTAIN author)

This fork extends [CAPTAIN v3](https://github.com/captain-project/captain3preview) with three cascading contributions for risk-sensitive conservation planning under climate uncertainty.

---

## Research Questions

### RQ1 — PPO with Plackett-Luce action space
Replaces CAPTAIN's Evolution Strategies (ES) optimiser with Proximal Policy Optimization (PPO). The key challenge was that CAPTAIN's `CellNN` performs greedy top-K cell selection with no probability distribution, making standard PPO inapplicable. This was solved using a **Plackett-Luce sequential categorical** formulation: at each step, K cells are drawn one at a time from a softmax over scores, giving valid log-probabilities for the policy gradient.

Reward is `CalcRewardCellValue` (counterfactual regret: avg value of selected cells minus avg value of all available cells) plus `CalcRewardMarginalCost`.

**Key finding:** Most of the EN-recovery improvement over ES comes from action granularity (K=50 vs K=1000), not the algorithm itself. PPO's remaining contributions are sample efficiency (9.5h vs 26h wall time) and enabling the IQN distributional critic.

### RQ2 — Stochastic disturbance environment
Integrates parametrizable spatially-coherent Perlin-noise disturbance into the training loop, replacing CAPTAIN's deterministic per-episode climate trajectory. Implemented as `SampledIntensityDisturbance` in `experiments/env_extensions.py`, which samples a fresh disturbance intensity each episode from `[intensity_min, intensity_max]`.

Disturbance knobs: `intensity_min/max`, `coherence`, `impact_factor`, `factors` (fractal persistence). Configured identically across PPO and ES K=50 for fair comparison.

### RQ3 — CVaR risk-sensitive objective via IQN
Incorporates CVaR-based risk-sensitive optimization using an **Implicit Quantile Network (IQN)** distributional critic. The agent learns the full return distribution across disturbance episodes and is trained on the worst-α tail (CVaR), optimizing for worst-case ecological outcomes rather than expected value.

Implementation avoids naive Bellman-CVaR (known misconvergence): IQN learns quantile returns, CVaR filtering selects the worst-α episodes for the PPO update, and the policy gradient is computed only over that subset.

**Finding:** With a single climate scenario and mild disturbance, the return distribution is narrow, making the CVaR tail nearly identical to the mean. The null result is informative: meaningful risk-sensitive gains require multi-scenario climate inputs or stronger disturbance variance.

---

## Repository Structure

```
experiments/
  configs/          # YAML configs for all runs
  env_extensions.py # SampledIntensityDisturbance, CalcRewardCellValue, CalcRewardMarginalCost
  ppo_actor_critic.py   # ActorCriticCellNN with IQN distributional head
  ppo_env_wrapper.py    # CaptainPPOEnv: wraps BioEnv for PPO step interface
  train_es.py           # ES baseline (K=1000)
  train_es_stochastic.py  # ES + stochastic disturbance (K=1000 and K=50 ablation)
  train_ppo.py          # PPO + Plackett-Luce
  train_ppo_stochastic.py # PPO + stochastic disturbance
  train_ppo_iqn.py      # IQN distributional critic (α=1 and α=0.25)
  train_ppo_cvar.py     # CVaR episode filtering
  eval_policy.py        # ES evaluation → eval_results.json + eval_protection_grid.npy
  eval_ppo.py           # PPO evaluation
  eval_ppo_iqn.py       # IQN evaluation

models/               # Trained weights for all final models
  es_baseline_k1000.npy
  es_stochastic_k1000.npy
  es_stochastic_k50.npy
  ppo_stochastic.pt
  iqn_alpha1.pt
  iqn_alpha025.pt

results/
  final/            # eval_results.json for all final models
  plots/            # Spatial comparison figures (dissertation figures)
```

---

## Key Results

| Model | LC | NT | VU | EN | CR |
|---|---|---|---|---|---|
| ES K=1000 (stochastic) | 545 | 28 | 6 | 1 | 3 |
| ES K=50 (ablation) | 545 | 27 | 6 | 0 | 5 |
| PPO stochastic | 541 | 30 | 8 | 0 | 4 |
| IQN α=1 | 540 | 20 | 13 | 0 | 10 |
| IQN α=0.25 (CVaR) | 541 | 25 | 9 | 0 | 7 |

---

## Running Experiments

All commands use `uv run`. Data must be downloaded separately (not in repo).

```bash
# ES baseline
uv run python experiments/train_es_stochastic.py \
    --data-dir /path/to/captain3data \
    --config experiments/configs/es_stochastic_k50.yaml \
    --run-name es_stochastic_k50

# PPO stochastic
uv run python experiments/train_ppo_stochastic.py \
    --data-dir /path/to/captain3data \
    --config experiments/configs/ppo_stochastic.yaml \
    --run-name ppo_stochastic_v2

# IQN (α=1 ablation)
uv run python experiments/train_ppo_iqn.py \
    --data-dir /path/to/captain3data \
    --config experiments/configs/ppo_iqn_alpha1_reduced.yaml \
    --run-name ppo_iqn_alpha1_reduced

# Evaluate ES model
uv run python experiments/eval_policy.py \
    --run-name es_stochastic_k50 \
    --data-dir /path/to/captain3data

# Evaluate PPO/IQN model
uv run python experiments/eval_ppo_iqn.py \
    --run-name ppo_iqn_alpha1_reduced \
    --data-dir /path/to/captain3data
```

---

## Dependencies

Same as upstream CAPTAIN — see `README.md`. Additional packages: `cleanrl`-style PPO (inlined), `pyperlin` for Perlin noise disturbance.
