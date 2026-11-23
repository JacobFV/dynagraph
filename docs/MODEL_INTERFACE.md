# compact model interface (owner: agent MODEL)

File: `src/physmem/models/compact.py`. All tensors torch; mask convention everywhere: **True = valid**.
`d` = token width of the compact model (default 128). Model outputs are always **raw SI units**
(the model holds target-scaler buffers fit on TRAIN data; `model.set_target_scaler(mean, std)`).

## top level

```python
from physmem.models.compact import build_model, PhysMemModel
model = build_model(variant: str, d: int = 128, passes_train=(1, 2, 4), core: nn.Module | None = None)
#   variant in {"b00","b10","b01","b11"} (+ "b10u"/"b00u" untied compute-matched core control)
#   b?0 = no physics loss, b?1 = physics loss (loss flag only; architecture depends on first digit)
#   first digit 0 = GenericMemoryEncoder, 1 = StructuredMemoryEncoder
out: ModelOutput = model(events: EventTensors, query: QueryTensors, passes: int = 4)
```

## split API (insert an external core, e.g. the Inkling recurrent core, between encode and readout)

```python
tokens, mask, slots = model.encode(events: EventTensors, query: QueryTensors)
#   tokens (B, M, d) float ; mask (B, M) bool ; slots: dict (see below)
tokens = model.refine(tokens, mask, passes: int)          # == model.core(tokens, mask, passes)
out    = model.readout(tokens, mask, query: QueryTensors, slots) -> ModelOutput
```

`model.core` is any `nn.Module` with
```python
core.forward(tokens: (B,M,d) float, mask: (B,M) bool, passes: int) -> (B,M,d) float
```
Requirements on a replacement core: it must not reorder/drop tokens (readout uses token positions via
`slots["token_entity"]` and `slots["typed_index"]`), must leave masked tokens finite, and must not change `d`
(project in/out inside the adapter). Refinement passes do NOT advance physical time.
`build_model(..., core=my_core)` or `model.core = my_core` swaps it. `model.freeze(encoder=True, head=True)` freezes
the compact encoder/head (for adapter-only training).

## slots dict (returned by encode, consumed by readout / interventions)

| key | shape | meaning |
|---|---|---|
| `token_entity` | (B,M) long | local entity index each token refers to, -1 for none (latents / typed slots) |
| `token_kind` | (B,M) long | 0 latent, 1 (entity,prop) slot, 2 entity token, 3 typed FRICTION, 4 typed MASS |
| `typed_index` | dict prop->int | token position of typed slots (structured only), e.g. `{Prop.FRICTION: M-2, Prop.MASS: M-1}` |
| `typed` | dict prop->dict(mean (B,), logvar (B,), latent (B,L)) | explicit typed estimates (structured only); friction is dimensionless, mass in kg |
| `edge_gate` | (B,N,N) float | relational edge weights used by the encoder (structured only; lesion target for e11) |

Generic (b00/b01) slots carry only `token_entity` (all -1) and `token_kind` (all 0).

## interventions (structured only)

```python
tokens2, slots2 = model.intervene(tokens, mask, slots, prop=Prop.FRICTION, mean: (B,) tensor, logvar: (B,) tensor)
out2 = model.readout(model.refine(tokens2, mask, passes), mask, query, slots2)
```
Overwrites the typed slot's (mean, logvar), re-encodes that slot token (latent kept), and returns new
tokens/slots (inputs not mutated). `out2.friction_mean` equals the intervened mean.
Edge lesion for e11: `model.encode(events, query, edge_lesion=(B,N,N) bool)` (True = cut edge) re-encodes
with those relational edges removed. Token lesion: set `mask[:, j] = False` before `refine`.

## ModelOutput (schema.py)
mean/logvar (B,H,7) raw units in the canonical target frame; friction_mean/logvar (B,), mass_mean/logvar (B,)
(logvar of the raw quantity); contact_logits (B,H,4); slots = the dict above; cost = {"passes": k, "tokens": M}.

## analytic baselines (`src/physmem/models/analytic.py`)
`ConstantVelocity()`, `CoulombAnalytic()`, `CoulombOracle()` — same call `m(events, query, passes=None, labels=None)`
(only the oracle reads `labels`, and it is labelled `oracle` everywhere). All return ModelOutput.

## early data observations (MODEL, 05:35Z, on data/v1 val/train/test)
- friction is WEAKLY identifiable in w2 v1: post-push free slides are short (typically 0.1-1.5 cm, < 0.15 s,
  i.e. 0-3 pose samples at 20 Hz with 2 mm noise). Simulator finite difference (k=0, friction 0.35 -> 0.65) moves
  the final object position by only ~1-8 mm. The analytic history-based least-squares friction estimate
  (grid posterior, model-error 4 mm fit on train) is calibrated (mean z^2 ~0.95) but only slightly better than the
  prior (abs err 0.092 vs 0.094, corr 0.23 for k>0). Expect small friction-related effects (h7, e07, e16).
- velocities at horizons >= 1.0 s are ~0 (objects at rest); the target scaler floors per-horizon std at 10% of the
  max-horizon std per dim to keep normalized NLL meaningful.
- v2 check (05:25Z): ground-truth post-push slide distance (val, 100 roots x 4 pushes) quantiles 10/50/90/99% =
  0.00/0.02/1.5/6.2 cm (object speed at push end is often ~0); simulator FD friction 0.35->0.65 at k=0 moves the
  final position by median 0.27 cm, 90th pct 2.5 cm. The analytic history estimator is still ~prior-level on v2
  (abs err 0.093 vs prior 0.094, corr 0.12). Friction matters for a minority (~10-20%) of v2 samples.
