# CAPTAIN — Dissertation Extension

**UCL MSc Dissertation 2026**  
Ana Maria Quintero Ossa — supervised by Maria Perez-Ortiz (UCL) and Daniele Silvestro (original CAPTAIN author)

This fork extends [CAPTAIN v3](https://github.com/captain-project/captain3preview) for risk-sensitive conservation planning under climate uncertainty. Three contributions are layered on top of the original ES optimiser.

---

## Contributions

**PPO with Plackett-Luce action space**  
Replaces CAPTAIN's Evolution Strategies with PPO. CAPTAIN's greedy top-K cell selection has no probability distribution, so log-probabilities are derived via a sequential Plackett-Luce formulation: K cells are drawn one at a time from a softmax over scores. Reward is a counterfactual regret signal (`CalcRewardCellValue`) plus a marginal cost penalty. Most EN-recovery improvement over ES comes from action granularity (K=50 vs K=1000), confirmed by an ES K=50 ablation; PPO's remaining contributions are sample efficiency (9.5h vs 26h wall time) and enabling the distributional critic.

**Stochastic disturbance environment**  
Adds parametrizable spatially-coherent Perlin-noise disturbance to the training loop, replacing CAPTAIN's deterministic per-episode climate trajectory. Disturbance intensity is sampled fresh each episode from `[intensity_min, intensity_max]`. Knobs: `intensity_min/max`, `coherence`, `impact_factor`, `factors`.

**CVaR risk-sensitive objective via IQN**  
Adds an Implicit Quantile Network distributional critic. The agent learns the full return distribution across disturbance episodes and is trained on the worst-α tail (CVaR), targeting worst-case ecological outcomes. Naive Bellman-CVaR is avoided (known misconvergence); instead, IQN learns quantile returns and CVaR filtering selects the worst-α episodes for the PPO update. With a single climate scenario and mild disturbance the return distribution is narrow, so CVaR gains are modest — informative as a null result requiring multi-scenario inputs to activate.

---

## Results

| Model | LC | NT | VU | EN | CR |
|---|---|---|---|---|---|
| ES K=1000 (stochastic) | 545 | 28 | 6 | 1 | 3 |
| ES K=50 (ablation) | 545 | 27 | 6 | 0 | 5 |
| PPO stochastic | 541 | 30 | 8 | 0 | 4 |
| IQN α=1 | 540 | 20 | 13 | 0 | 10 |
| IQN α=0.25 (CVaR) | 541 | 25 | 9 | 0 | 7 |

---

## Repository Structure

```
experiments/
  configs/              # YAML configs for all runs
  env_extensions.py     # SampledIntensityDisturbance, CalcRewardCellValue, CalcRewardMarginalCost
  ppo_actor_critic.py   # ActorCriticCellNN with IQN distributional head
  ppo_env_wrapper.py    # CaptainPPOEnv: wraps BioEnv for PPO step interface
  train_es.py / train_es_stochastic.py
  train_ppo.py / train_ppo_stochastic.py
  train_ppo_iqn.py / train_ppo_cvar.py
  eval_policy.py        # ES evaluation
  eval_ppo.py / eval_ppo_iqn.py

models/                 # Trained weights for all final models
results/final/          # eval_results.json for all final models
results/plots/          # Spatial comparison figures
```

---

## Reproducing Runs

```bash
# ES K=50 ablation
uv run python experiments/train_es_stochastic.py \
    --data-dir /path/to/captain3data \
    --config experiments/configs/es_stochastic_k50.yaml \
    --run-name es_stochastic_k50

# PPO stochastic
uv run python experiments/train_ppo_stochastic.py \
    --data-dir /path/to/captain3data \
    --config experiments/configs/ppo_stochastic.yaml \
    --run-name ppo_stochastic_v2

# IQN α=1
uv run python experiments/train_ppo_iqn.py \
    --data-dir /path/to/captain3data \
    --config experiments/configs/ppo_iqn_alpha1_reduced.yaml \
    --run-name ppo_iqn_alpha1_reduced

# Evaluate
uv run python experiments/eval_policy.py --run-name es_stochastic_k50 --data-dir /path/to/captain3data
uv run python experiments/eval_ppo_iqn.py --run-name ppo_iqn_alpha1_reduced --data-dir /path/to/captain3data
```

See `README.md` for installation instructions.
