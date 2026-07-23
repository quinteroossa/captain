# Development Notes
# Risk-Sensitive RL for Systematic Conservation Planning
# Ana Maria Quintero Ossa — UCL MSc 2026

This document tracks experimental design decisions and their rationale,
structured to support dissertation writing. Each scenario is a distinct
experimental condition that answers a specific research sub-question.

---

## Experimental Progression

```
Scenario 0 (ES, deterministic)
    ↓  add stochastic disturbance
Scenario 1 (ES, stochastic fixed intensity)
    ↓  make intensity a random variable
Scenario 2 (ES, stochastic sampled intensity)  ← genuine return distribution exists here
    ↓  replace ES with PPO
Scenario 3 (PPO, stochastic sampled intensity)
    ↓  add CVaR objective
Scenario 4 (PPO + CVaR, stochastic sampled intensity)  ← primary novel contribution
```

---

## Scenario 0 — ES Baseline (Deterministic Disturbance)

**Script:** `experiments/train_es.py`
**Config:** `experiments/configs/es_base.yaml`
**Status:** Implemented, running locally (200 epochs test), full run pending cluster.

### What it is
Original CAPTAIN training loop with no modifications. Evolution Strategies
optimiser, deterministic climate trajectory, deterministic disturbance layer
(`area_swept_disturbance.tif` stepped linearly each timestep).

### Why this scenario
Establishes the performance floor. All subsequent scenarios are compared
against this. Reproducing the upstream result validates that the local
setup (data, code, seeds) is correct before any modifications are made.

### What it tells us
- Baseline biodiversity outcome (extinction risk trajectory, CR/EN species counts)
- Baseline training stability (reward curve, Jaccard index)
- Wall-clock time per epoch on local machine and cluster (informs PPO feasibility)

### Key metrics to report
- Final reward vs epoch
- Jaccard index (policy consistency across perturbations)
- Species status table at episode end (LC/NT/VU/EN/CR counts)
- Epochs to convergence

---

## Scenario 1 — ES + Stochastic Disturbance (Fixed Intensity)

**Script:** `experiments/train_es_stochastic.py`
**Config:** `experiments/configs/es_stochastic.yaml`
**Status:** Implemented, runs successfully. Known limitation: `intensity`
and `impact_factor` are hardcoded in `StochasticSpatialData.update()` (0.3
and 0.5), config values not yet wired through `BioEnv`.

### What it is
Same as Scenario 0 but `StochasticSpatialData` replaces the deterministic
disturbance layer. At each timestep, spatially coherent Perlin-noise events
are sampled and applied — some cells receive reduced disturbance exposure.
The `risk_map` is `area_swept_disturbance.tif` (pending confirmation from
Daniele of what this layer represents physically). Intensity is fixed across
all episodes and timesteps.

### Why this scenario
First integration test of the stochastic disturbance layer into training.
Verifies the environment behaves differently from Scenario 0, and quantifies
how much training instability the disturbance adds.

**Decision rationale:** Use `data=mask` (binary 1s for valid cells) as the
base disturbance value, following the upstream demo in `plot_input_data.py`.
Stochastic events multiply affected cells by `impact_factor=0.5`.
This was chosen over using the raw `area_swept_disturbance.tif` values as
`data` because: (1) those values are very low (mean 0.021), making events
nearly undetectable; (2) the demo uses the mask and it is the design Daniele
demoed. Revisit once he confirms the intended semantics of `data`.

### Known issues / open questions
- ~29% of cells fire each step regardless of location (because `risk_map`
  mean ≈ 0.021 makes the threshold effectively `0.3 × 1.0 = 0.3` everywhere)
- Risk direction may be inverted in `apply_stochastic_events()` — awaiting
  Daniele's confirmation
- `area_swept_disturbance.tif` has no CRS — spatial alignment with SDMs unverified
- `intensity` and `impact_factor` not yet configurable via BioEnv

### What it tells us
- Whether ES can still learn under stochastic disturbance
- How much variance the disturbance adds to episode returns
- Comparison against Scenario 0: does protection strategy change when
  disturbance is stochastic?

---

## Scenario 2 — ES + Stochastic Disturbance (Sampled Intensity) *(Proposal)*

**Script:** `experiments/train_es_stochastic.py` (to be extended)
**Config:** `experiments/configs/es_stochastic_sampled.yaml` (to be created)
**Status:** Proposal — not yet implemented. Requires small upstream fix to
wire `intensity` through `BioEnv`, then sample it per episode at the trainer
level. To be discussed with James and Daniele before committing.

### What it is
Same as Scenario 1 but `intensity` is sampled from a distribution at the
start of each episode rather than fixed:

```python
intensity ~ Uniform(intensity_min, intensity_max)   # e.g. Uniform(0.05, 0.5)
```

**What `intensity` controls:** it is the probability threshold that determines
how many cells get disturbed each timestep. A noise value is generated for
every cell (via Perlin noise); if that noise value falls below `intensity`,
the cell is hit by a disturbance event and its value is reduced. So:
- `intensity = 0.05` → ~5% of cells disturbed per timestep (mild scenario)
- `intensity = 0.5`  → ~50% of cells disturbed per timestep (severe scenario)

By sampling a new `intensity` at the start of each episode, we are simulating
uncertainty in how bad the disturbance season will be — some years few marine
heatwaves occur, some years many. The agent does not know in advance which
intensity it will face; it must learn a policy that performs well across the
full range.

This creates a **distribution of episode returns** driven purely by
environmental randomness (aleatoric uncertainty). High-intensity episodes
→ worse outcomes → lower tail of the return distribution.

### Why this scenario
**This is the key step that makes CVaR meaningful.**

With fixed intensity (Scenario 1), all episodes face the same disturbance
severity — there is no distribution to take the worst tail of. With sampled
intensity, the return distribution has genuine spread:

- Low-intensity episodes → returns cluster near the deterministic baseline
- High-intensity episodes → returns shift down, more species at risk
- CVaR(α) then targets the worst-α% episodes, which are the high-intensity ones

**Decision rationale:** Intensity is the natural uncertainty parameter for
marine heatwaves — we don't know the future frequency or severity of events.
Sampling it per episode is the simplest, most ecologically motivated way to
create aleatoric uncertainty. Alternatives considered:
- Multiple climate scenario files (not available in current dataset)
- Noise on `delta_per_step` in SDMs (possible but less targeted)
- Both could be added later as additional uncertainty sources

### What it tells us
- Shape of the return distribution under environmental uncertainty
- How much the worst-case episodes differ from the mean
- Whether ES is sensitive to high-intensity disturbance regimes
- Sets the ground truth return distribution that CVaR will act on in Scenario 4

---

## Scenario 3 — PPO + Stochastic Disturbance (Sampled Intensity)

**Script:** `experiments/train_ppo.py`
**Config:** `experiments/configs/ppo_base.yaml`
**Status:** Implemented. Running on full dataset (583 species, 58,315 cells).

### What it is
Replace ES with PPO while keeping the stochastic environment from Scenario 2.

**Log-prob problem solved — Plackett-Luce (K=50):**
The original CAPTAIN CellNN uses greedy top-K selection, which is deterministic
(no probability distribution → no log-prob → no policy gradient). Solved by
switching to Plackett-Luce sampling: model K sequential categorical draws
without replacement. Log P = sum of K per-step log-softmax values.
K=50 chosen as a compromise between K=1 (too slow to cover 17K budget) and
K=1000 (too expensive per step for Plackett-Luce's O(K·N) loop).

### Architecture — `ActorCriticCellNN`

#### Overview

CAPTAIN's environment produces a state observation of shape `(13, 58315)` —
13 features per cell across 58,315 valid spatial cells. The network must
produce two outputs: a score per cell (for action selection) and a scalar
state value (for advantage estimation).

The architecture is a **shared-trunk actor-critic**: one MLP processes all
cells identically, and two separate heads read from the shared embeddings.

```
Input: (13 features, 58,315 cells)
  ↓ transpose → (58315, 13)          each cell treated as one data point

Shared trunk (applied independently to each cell):
  Linear(13 → 64) → ReLU
  Linear(64 → 64) → ReLU
  → embeddings: (58315, 64)

      ┌─────────────────────────────────────┐
      │                                     │
  Policy head                          Value head
  Linear(64 → 1)               attention_head(64 → 1) → softmax over cells
  → per-cell scores (58315,)   → weighted sum → pooled (64,)
  → Plackett-Luce(K=50)        → Linear(64 → 1) → scalar V(s)
  → action + log P
```

#### Feature breakdown (n_features = 13)

| Feature | Description |
|---|---|
| `time` | Normalised timestep (t / n_time_steps×0.5) |
| `disturbance` | Raw disturbance value per cell |
| `disturbance_conv` | Spatially smoothed disturbance (convolved) |
| `species_richness` | Number of species present in cell |
| `total_population` | Total population across all species in cell |
| `ext_risk_0…4` | Count of species in each IUCN category in cell (5 features) |
| `cost` | Protection cost per cell |
| `protection_matrix` | Whether cell is currently protected (0/1) |
| `protection_matrix_conv` | Spatially smoothed protection coverage |

Note: extinction risk is encoded as **per-category counts per cell** (5 values),
not per-species (which would be 583 values). This keeps the input manageable
while still informing the policy about local biodiversity risk.

#### Design decisions

**Why shared trunk?**
Policy and value learn from the same spatial features. Sharing the trunk
reduces parameters and encourages representations useful for both objectives.
Standard in CleanRL PPO. Tradeoff: gradient interference between policy and
value losses — separate trunks are an alternative if training is unstable.

**Why two trunk layers?**
A single `Linear(13→64)` is too shallow to capture non-linear interactions
between features (e.g. high cost + high extinction risk = high priority).
Two layers `[64, 64]` add capacity with negligible compute cost under backprop.

**Why Plackett-Luce instead of greedy top-K?**
Greedy top-K is deterministic — no probability distribution, no log-prob,
no policy gradient. Plackett-Luce models K sequential categorical draws
without replacement: `P(i1,…,iK) = Π softmax(scores[remaining])[ij]`.
Log P = sum of K log-softmax values, computable in O(K·N).

**Why K=50?**
With budget=17,000 and `cells_per_step=50`, one episode = 340 protection
steps. K=1 would require 17,000 steps per episode (too slow for rollout
collection). K=1000 makes the Plackett-Luce loop expensive (1000 sequential
categoricals). K=50 balances episode length and per-step cost.

**Why mean pooling for value head (baseline)?**
Collapses the spatial dimension cheaply. Treats all 58,315 cells equally when
estimating V(s). Sufficient for the baseline PPO run — keeping the architecture
simple so the only variable vs ES is the optimiser.

**Planned variant — attention pooling:**
Mean pooling weights all cells equally, but most cells have near-zero
population and don't drive episode quality. A learned attention pooling would
let the value head focus on high-risk, high-population cells:
```
attn = softmax(Linear(64→1)(emb))   # (58315, 1) — learned cell importance
pooled = Σ attn_i × emb_i           # (64,)      — importance-weighted sum
V(s) = Linear(64→1)(pooled)
```
To be run as a separate experiment after the baseline PPO is validated.

### Reward calibration
PPO uses probe-based calibration identical to ES: run `n_calibration_probes=20`
episodes with the untrained policy, compute std of each reward component,
set multiplier = 1/std. Saved to `results/<run>/reward_calibration.json`.
Both cost and extinction_risk are enabled (`reward_weights=[1.0, 1.0]`),
with calibration preventing cost from dominating the signal.

### Key hyperparameters
- K=50 cells/step, n_time_steps=200, rollout_steps=128
- γ=0.99, λ=0.95 (GAE), clip_coef=0.2, ent_coef=0.05
- lr=3e-4 with linear annealing

### What it tells us
- Whether PPO achieves better sample efficiency than ES (fewer epochs to same reward)
- Whether PPO produces qualitatively different protection strategies
- Credit assignment: does PPO learn which early-episode decisions matter most?

---

## Scenario 4 — PPO + CVaR + Stochastic Disturbance (Sampled Intensity)

**Script:** `experiments/train_ppo_cvar.py` (to be created)
**Config:** `experiments/configs/ppo_cvar.yaml` (to be created)
**Status:** Pending Scenarios 1–3. Primary novel contribution of dissertation.

### What it is
Add a CVaR-based risk-sensitive objective on top of Scenario 3. Rather than
maximising expected return E[G], the agent maximises CVaR_α[G] — the expected
return in the worst α% of episodes (defined by the sampled intensity
distribution):

```
CVaR_α[G] = E[G | G ≤ VaR_α[G]]
```

**Implementation approach (to finalise with James):**
Do NOT apply CVaR inside the Bellman operator — this is a known
misconvergence pitfall. Instead, use a distributional approach:
estimate the full return distribution across episodes in a rollout buffer,
then use CVaR as the policy gradient objective. IQN (Implicit Quantile
Networks) is the leading candidate over C51/QR-DQN because it directly
parameterises the quantile function and allows explicit CVaR computation.

### Why this scenario
Addresses RQ3: does optimising for worst-case outcomes (CVaR) produce
policies that are more robust to high-intensity disturbance episodes, at the
cost of lower average performance?

**The ecological motivation:** Species extinction is irreversible. A policy
that performs well on average but causes mass extinction in 10% of climate
scenarios is unacceptable for real-world conservation. CVaR explicitly
penalises those tail outcomes.

### Key design decisions still open
- CVaR confidence level α (e.g. 0.1 = worst 10% of episodes)
- Whether CVaR applies to total return or per-species extinction counts
- How to estimate the return distribution within a PPO rollout buffer
- Whether to use a separate risk level parameter or make α trainable

### What it tells us
- Whether CVaR-based training reduces worst-case extinction outcomes
- The tradeoff between mean performance and tail robustness
- Whether a risk-sensitive agent learns qualitatively different spatial priorities
  (e.g. preferentially protecting high-disturbance-risk areas)

---

## Implementation Decisions Log

### Reward structure mismatch: ES vs PPO per-step credit — fixed 2026-07-23

**Files:** `experiments/env_extensions.py`, `experiments/train_ppo.py`

**Symptom:** PPO policy stuck at maximum entropy (`loss/entropy ≈ ln(58315) ≈ 10.97`),
reward oscillating -100 to 0, value loss high (~140 at start). Confirmed via four
wandb runs: `ppo_baseline_full`, `ppo_extrisk_only`, `ppo_200steps`, `ppo_baseline`.

**Root cause 1 — Cumulative cost gives wrong per-step credit:**
`CalcRewardPersistentCost` computes `dot(costs, protection_matrix)` — the total cost
of ALL currently protected cells. This grows monotonically every episode step.
Step 1 gets penalised ~0, step 340 gets a large penalty regardless of which cells
were selected at that step. PPO's GAE cannot assign credit to individual decisions
because early and late steps receive structurally different penalties regardless of
policy quality. `ppo_baseline_full` (cost enabled) oscillated -100 to 0;
`ppo_extrisk_only` (cost disabled) confirmed cost was the source of oscillation.

**Root cause 2 — Delta-based extinction risk reward is near-zero:**
`CalcRewardExtRisk` rewards *changes* in species counts between IUCN categories.
Species rarely shift category in a single timestep — they must lose all their
population in a cell first, which takes many steps. Result: reward ≈ 0 nearly
every step. Confirmed by `ppo_extrisk_only`: reward flat at ~0, entropy stuck
at maximum. No gradient signal for the policy.

**Why the same reward design worked for ES:**
ES optimises total episode return (sum over all steps). Cumulative cost and
delta-based extinction risk are both correct in expectation over a full episode.
PPO needs meaningful per-step rewards for GAE to assign credit to individual
timestep decisions — the same reward functions don't transfer.

**Fix applied:**
- `CalcRewardMarginalCost` (`experiments/env_extensions.py`): cost of K cells
  newly protected THIS step only (delta of protection matrix). Dense, non-growing,
  correctly credits the current action.
- `CalcRewardExtRiskLevel` (`experiments/env_extensions.py`): weighted sum of
  current risk category counts per step: `(counts · [1,0,-8,-16,-32]) / n_species`.
  Dense per-step signal — agent is rewarded every step for the current state of
  species risk, not only on rare category transitions.

---

### Perlin noise range mismatch in `apply_stochastic_events()` — fixed in experiments

**Date:** 2026-07-14
**File:** `experiments/env_extensions.py` — `SampledIntensityDisturbance.apply_stochastic_events()`
**Upstream file (not modified):** `captain/data/spatial_data.py:306`

**Bug:** The upstream formula `noise < (1 - risk_map) * intensity` assumes Perlin
noise in [0, 1]. `FractalPerlin2D` outputs ~[-0.4, 0.45] centred around 0.
At `intensity=0.3` the threshold is ~0.3 but ~97% of noise values are already
below that, so nearly all cells fire regardless of `intensity` — it is a dead parameter.

**Fix applied (quantile threshold):**
```python
q_threshold = torch.quantile(noise_2d.flatten(), intensity)
event_mask_2d = noise_2d < q_threshold * (1.0 - self.risk_map)
```
`intensity` now directly equals the fraction of zero-risk cells disturbed per step,
independent of the noise distribution. Overridden only in `SampledIntensityDisturbance`
— upstream `StochasticSpatialData` is untouched.

**Options considered and rejected:**
- Min-max normalise: per-frame normalisation shifts the absolute threshold each step.
- Replace FractalPerlin2D: unnecessary — the problem is the assumption, not the generator.

**Open question (not fixed here):** Risk direction may be inverted. `(1 - risk_map)`
means high-risk cells have a lower threshold and fire *less* — contradicts the comment
"high risk areas require very little noise to trigger an event." Awaiting confirmation
from Daniele on intended `risk_map` semantics before touching the direction.

---

## Open Implementation Decisions

| Decision | Current status | To discuss with |
|---|---|---|
| Risk direction in `apply_stochastic_events()` | Potentially inverted — awaiting confirmation | Daniele |
| What `area_swept_disturbance.tif` represents | Unknown CRS, physical meaning unclear | Daniele |
| What `oil.tif` represents (depth?) | 333 valid cells, values 1600–19000 | Daniele |
| Whether to build composite `risk_map` from lat + disturbance | Proposed, not built | Daniele |
| Wire `intensity` through `BioEnv.env_step()` | Hardcoded at 0.3 currently | Implement |
| K=1 vs top-K for PPO action space | K=1 confirmed by Daniele | Implement |
| Shared trunk vs separate actor/critic | Shared recommended, to confirm | James |
| Discount factor γ for PPO | Currently 1.0 (unused) | James |
| CVaR implementation: IQN vs C51 vs QR-DQN | IQN preferred, not confirmed | James |
| CVaR confidence level α | Not decided | James / Maria |
| Memory in disturbance (temporal autocorrelation) | Memoryless currently — assess value | James (tomorrow) |


* Ask which randomness makes sense (cost, environment)



