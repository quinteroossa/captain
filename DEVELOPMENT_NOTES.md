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
**Status:** Implemented. Training ran (no crash) but produced a reward curve
that climbed smoothly for ~18 epochs then collapsed — traced to a units bug
in the disturbance layer, fixed 2026-07-23 (see Implementation Decisions Log).
Rerun pending to confirm the fix stabilises training.

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

**Decision rationale (superseded 2026-07-23):** Originally used `data=mask`
(binary 1s for valid cells) as the base disturbance value, following the
upstream demo in `plot_input_data.py`, with stochastic events multiplying
affected cells by `impact_factor=0.5`. This inverted the units `BioEnv`
expects — see "Disturbance units inverted" below. Fixed: base is now
`np.zeros_like(mask)` (0 = no disturbance), events SET affected cells to
`impact_factor` instead of multiplying down.

### Known issues / open questions
- **Fixed 2026-07-23:** disturbance units were inverted relative to what
  `update_carrying_capacity()` expects — see Implementation Decisions Log.
  Rerun needed to confirm training stabilises.
- Risk direction may be inverted in `apply_stochastic_events()` — awaiting
  Daniele's confirmation (separate, still-open question — see below)
- `area_swept_disturbance.tif` has no CRS — spatial alignment with SDMs unverified
- `intensity` and `impact_factor` now wired through `SampledIntensityDisturbance`
  (config-driven), but still hardcoded defaults in the base upstream
  `StochasticSpatialData.update()` if used directly outside `experiments/`

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

**Script:** `experiments/train_ppo_stochastic.py` (to be created)
**Config:** `experiments/configs/ppo_stochastic.yaml` (to be created)
**Status:** Planned. Log-prob blocker resolved — see architecture below.
Intermediate deterministic PPO implemented in `experiments/train_ppo.py`;
stochastic wiring pending Scenario 2 completion.

### What it is
Replace ES with PPO while keeping the Scenario 2 stochastic environment.

**Log-prob problem solved — Plackett-Luce (K=50):**
The original CAPTAIN CellNN uses greedy top-K selection — deterministic, no
probability distribution, no log-prob, no policy gradient. Solved by
Plackett-Luce sampling: K sequential categorical draws without replacement.
Log P = sum of K log-softmax values. K=50 balances episode length (340 steps)
vs per-step Plackett-Luce cost (O(K·N)).

### Architecture — `ActorCriticCellNN` (`experiments/ppo_actor_critic.py`)

```
Input: (13 features, 58,315 cells)
  ↓ transpose → (58315, 13)

Shared trunk (per-cell MLP):
  Linear(13 → 64) → ReLU
  Linear(64 → 64) → ReLU
  → embeddings: (58315, 64)

      ┌────────────────────────────┐
  Policy head                 Value head
  Linear(64 → 1)              mean-pool → Linear(64 → 1)
  → scores (58315,)           → scalar V(s)
  → Plackett-Luce(K=50)
  → action + log P
```

**Feature breakdown (n_features = 13):** time, disturbance, disturbance_conv,
species_richness, total_population, ext_risk_0…4 (count per IUCN class per
cell), cost, protection_matrix, protection_matrix_conv.

**Planned variant — attention pooling for value head:** mean pooling weights
all cells equally; a learned attention head would focus on high-risk cells.
Deferred until baseline PPO is validated.

**Reward fix required for PPO** (see Implementation Decisions Log):
ES rewards (cumulative cost, delta-based extinction risk) break PPO's per-step
credit assignment. Replaced with `CalcRewardMarginalCost` (cost of this step's
cells) and `CalcRewardExtRiskLevel` (dense per-step risk level signal).

### Key hyperparameters
K=50 cells/step, n_time_steps=200, rollout_steps=128, γ=0.99, λ=0.95,
clip_coef=0.2, ent_coef=0.05, lr=3e-4 with linear annealing.

### What it tells us
- Whether PPO achieves better sample efficiency than ES
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
were selected at that step. PPO's GAE cannot assign credit to individual decisions.
`ppo_baseline_full` (cost enabled) oscillated -100 to 0; `ppo_extrisk_only` (cost
disabled) confirmed cost was the source of oscillation.

**Root cause 2 — Delta-based extinction risk reward is near-zero:**
`CalcRewardExtRisk` rewards *changes* in species counts between IUCN categories.
Species rarely shift category in a single timestep. Confirmed by `ppo_extrisk_only`:
reward flat at ~0, entropy stuck at maximum — no gradient signal for the policy.

**Why the same reward design worked for ES:**
ES optimises total episode return (sum over all steps). Cumulative cost and
delta-based extinction risk are both correct in expectation over a full episode.
PPO needs meaningful per-step rewards for GAE to assign credit — the same reward
functions don't transfer.

**Fix applied:**
- `CalcRewardMarginalCost`: cost of K cells newly protected THIS step only (delta
  of protection matrix). Dense, non-growing, correctly credits the current action.
- `CalcRewardExtRiskLevel`: weighted sum of current risk counts per step:
  `(counts · [1, 0, -8, -16, -32]) / n_species`. Dense per-step signal every step.

---

### Disturbance units inverted — fixed 2026-07-23

**Date:** 2026-07-23
**Files:** `experiments/env_extensions.py` — `SampledIntensityDisturbance`;
`experiments/train_es_stochastic.py` — `create_episode_runner()`
**Upstream (not modified):** `captain/environment/bioenv.py:357-377` —
`update_carrying_capacity()`

**Symptom:** ES + stochastic disturbance training (Scenario 1) did not crash,
but the reward curve climbed smoothly for ~18 epochs then collapsed sharply,
with all five extinction-risk category counts (`threat_0`…`threat_4`)
reverting or overshooting past their starting values at the same epoch
(wandb screenshots, 2026-07-23). ES weight std (`w_std`) was also growing
rapidly across epochs in an earlier short test run.

**Root cause:** `update_carrying_capacity()` treats `self.disturbance.data`
as a degradation-intensity layer: `eff_dist = (1 - protection) * disturbance.data`,
then `k_dist = 1 - species_sensitivity @ eff_dist` — i.e. **higher
`disturbance.data` means lower carrying capacity**. The deterministic layer
(`area_swept_disturbance.tif`) matches this: mean ≈0.14 within the study
mask, mostly small.

`SampledIntensityDisturbance` was constructed with `data=mask` (all 1s) and
applied events via `self._data[:, flat_event_mask] *= impact_factor`
(multiplying *down* to 0.5). This convention was carried over from
`examples/plot_input_data.py`, a standalone visual demo that was never wired
into `BioEnv` and never exercised this code path. The effect once plugged
into `update_carrying_capacity()`:

| | `eff_dist` | `k_dist` (sensitivity≈0.88) |
|---|---|---|
| Deterministic baseline | ≈0.14 | ≈0.88 (12% capacity loss) |
| Stochastic, cell with **no event** | ≈1.0 | ≈0.12 (88% capacity loss — every step, every undisturbed cell) |
| Stochastic, cell **hit by an event** | ≈0.5 | ≈0.56 (44% loss — better off than an "undisturbed" cell) |

Undisturbed cells were being crushed to a fraction of carrying capacity
domain-wide, every timestep, while "event" cells were comparatively better
off — inverted relative to intent, and severe enough to explain slow
population attrition building over ~18 epochs before a threshold was crossed
and category counts spiked simultaneously across the grid.

**Fix applied:**
```python
# train_es_stochastic.py — base layer is now 0 (no disturbance), not 1
disturbance = SampledIntensityDisturbance(
    data=np.zeros_like(mask),
    ...
)

# env_extensions.py — events SET affected cells to impact_factor, not multiply
self._data[:, flat_event_mask] = impact_factor
```
Baseline is now 0 everywhere (no disturbance); an event raises a cell to
`impact_factor` (a degradation-intensity value, 0-1), consistent with how
the deterministic layer is scaled.

**Status:** Fix applied, not yet re-run to confirm the reward curve
stabilises. `plot_input_data.py` intentionally left unchanged (visual demo
only, not used for training).

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
| Disturbance units (0=none vs 1=none) in `SampledIntensityDisturbance` | Fixed 2026-07-23 — rerun pending to confirm | Implement (verify) |

* Ask which randomness makes sense (cost, environment)
