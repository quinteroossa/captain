#!/usr/bin/env python
"""Evaluate a trained PPO policy under stochastic disturbance (RQ2/RQ3).

Identical to eval_ppo.py except create_env() uses SampledIntensityDisturbance
so each episode draws a different disturbance intensity.  Run with --n-episodes 10
to sample the return distribution; the output JSON includes per-episode returns
so CVaR_0.2 can be computed in post-processing.

Usage:
    uv run python experiments/eval_ppo_stochastic.py \\
        --run-name ppo_stochastic_v1 \\
        --data-dir /path/to/captain3data \\
        --n-episodes 10
"""

import experiments.eval_ppo as _base
from experiments.train_ppo_stochastic import create_env

_base.create_env = create_env

if __name__ == "__main__":
    _base.main()
