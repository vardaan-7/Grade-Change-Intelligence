# Grade Change Intelligence — Paper Machine Copilot

A decision-support copilot for the grade-change transition on a paper machine,
built for Honeywell's *"Grade Change Intelligence in Paper Making Process"*
hackathon problem statement. It watches Basis Weight during a recipe
transition, predicts a spec breach before it happens, and recommends
safe secondary-loop setpoints to shorten the stabilization window.

> **Note on data:** there's no public historian feed for this problem, so
> `simulator.py` generates physically-motivated synthetic data (first-order
> lag, cross-loop coupling, operator overrides, sensor faults) rather than
> pretending to load a real DCS export. Everything downstream — the
> early-warning model, the correlation engine, the optimizer — runs on top
> of that simulated stream exactly as it would on live tags.

---

## 1. The 2.5% threshold, mathematically

The problem statement defines "off-spec" as Basis Weight deviating more
than **2.5%** from the recipe's setpoint target:

```
pct_dev(t) = (BW_pv(t) - BW_target(t)) / BW_target(t) * 100
off_spec(t) = |pct_dev(t)| > 2.5
```

The early-warning model doesn't just threshold `pct_dev` — that would only
catch the breach *as it happens*, not before. Instead it's trained to
predict a forward-looking label:

```
label(t) = 1 if max(|pct_dev(t : t+H)|) > 2.5   else 0
```

where `H` is a 6-sample look-ahead horizon (~1 minute at 10s sampling).
Features are the rate of change of Basis Weight, the current deviation,
and the live offsets of steam pressure / machine speed / moisture from
their setpoints — the three variables the simulator's physics say actually
drive Basis Weight drift. A `LogisticRegression` is fit on the run's own
history so far (warm-start style); if a run never actually goes at-risk,
it falls back to a transparent rule (weighted proximity + momentum) rather
than silently failing.

## 2. Architecture

```
simulator.py   →   raw process tags (PV/SP for 4 loops + moisture/ash/caliper/BW)
                          │
                          ▼
engine.py      →   EarlyWarningModel   (risk %)
                    correlation_engine (rolling ρ matrix)
                    SetpointOptimizer  (safe setpoints + settling-time projection)
                    explain()          (why / source for every trigger)
                          │
                          ▼
app.py         →   Streamlit dashboard: KPIs, alert banner, trajectory chart,
                    correlation heatmap, setpoint table, Accept/Reject loop
                          │
                          ▼
audit_trail.csv →  append-only log of every operator decision
```

Each module is independently testable: `simulator.py` has no dependency on
`engine.py`, and `engine.py` has no dependency on Streamlit at all — you
could drive it from a notebook or a batch script just as easily.

## 3. Hidden correlations the simulator encodes

These are the relationships the correlation engine is meant to "discover"
live rather than have hard-coded into the dashboard:

- **Steam pressure → Moisture** (dominant, negative): more drying steam
  pulls moisture down, which the sheet's Basis Weight reading is
  sensitive to via a secondary drag term.
- **Machine speed → Moisture** (weaker, positive): higher speed shortens
  dryer-can dwell time, leaving the sheet slightly wetter.
- **Filler flow → Ash** (direct, near-linear): filler loading is the main
  ash driver, modeled almost 1:1 with noise.
- **Stock flow → Basis Weight** (direct): the primary lever, but lagged
  through an 8-sample first-order filter so it doesn't respond instantly.

## 4. Setpoint optimization

The optimizer treats "how tightly tuned" the four secondary loops are as
a single scalar gain and does a coordinate search over candidate gains,
projecting a simplified first-order settling trajectory for each. The
winning gain is translated back into concrete Stock Flow / Filler Flow /
Steam Pressure / Machine Speed setpoints by moving each loop from its
current value toward the recipe target in proportion to the gain
improvement purchased — then every recommendation is clamped to ±15% of
the recipe-nominal value as a stand-in for the mill's safe operating
envelope. This is deliberately simple (a real deployment would swap in a
proper MPC or Bayesian optimizer) but it is fully transparent: every
number in the "Recommended Safe Setpoints" table can be traced back to the
gain search in `SetpointOptimizer.optimize()`.

## 5. Explainability & audit trail

Every recommendation the UI shows is backed by an `explain()` lookup that
states *why* it fired and *what it's based on* — historical run data,
recipe boundaries, or a rolling correlation window. Operator Accept/Reject
decisions are appended (never overwritten) to `audit_trail.csv`, alongside
the exact risk score and recommended setpoints at that moment, so model
accuracy can be reviewed after the fact against what an operator actually
did with the recommendation.

## 6. Extras beyond the core spec

- **Financial impact counter** — cumulative and live ₹/min loss, computed
  from off-spec minutes × machine throughput × a configurable cost/tonne.
- **Sensor validation guard** — every tick is checked for physically
  implausible values (negative pressure, out-of-range speed, flagged
  dropouts); faults are surfaced as warnings and the pipeline forward-fills
  rather than crashing.
- **Async-style replay** — a session-state cursor plus an auto-stream
  toggle simulate a live historian feed advancing across the run.

## 7. Running locally

```bash
git clone <your-repo-url>
cd grade-change-intelligence
pip install -r requirements.txt
streamlit run app.py
```


## 8. File-by-file

| File | Responsibility |
|---|---|
| `simulator.py` | Synthetic grade-change time series with lag, coupling, operator overrides, sensor faults |
| `engine.py` | Early-warning classifier, correlation engine, setpoint optimizer, explainability library |
| `app.py` | Streamlit UI: KPIs, alerts, charts, Accept/Reject, audit trail |
| `requirements.txt` | Pinned dependencies |
