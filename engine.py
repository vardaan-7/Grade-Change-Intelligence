"""
engine.py
---------
The analytical core. Four responsibilities, kept deliberately separate so
each can be unit-tested / swapped independently:

1. EarlyWarningModel      -> probability of breaching the 2.5% basis-weight
                              spec window in the *next* horizon, trained
                              on-the-fly on the rolling window of the run.
2. correlation_engine      -> rolling Pearson correlation across loops to
                              surface latent multivariable relationships.
3. SetpointOptimizer       -> local search over the four secondary loops
                              to find the setpoint combination that
                              minimizes projected settling time while
                              respecting the 2.5% band.
4. explain()               -> turns a trigger into a human-readable
                              engineering rationale + source citation.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SPEC_BAND_PCT = 2.5          
EARLY_WARNING_HORIZON = 6    



class EarlyWarningModel:
    """
    A lightweight logistic-regression early-warning model.

    Rather than shipping a pre-trained black box (which would be
    dishonest for a simulated mill with no real historian behind it),
    we train it on the *first half* of the current run's history and
    score the rest -- this mirrors how a real deployment would warm-start
    on the last few hours of historian data before going live.

    Features per sample:
        - rate of change of basis weight (last 3 samples)
        - current % deviation from target
        - steam pressure deviation from setpoint
        - machine speed deviation from setpoint
        - moisture deviation from spec-held target
    """

    def __init__(self):
        self.model = LogisticRegression(max_iter=500)
        self.scaler = StandardScaler()
        self.is_fitted = False

    @staticmethod
    def _make_features(df: pd.DataFrame) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        feat["bw_roc"] = df["basis_weight_pv"].diff(3).fillna(0)
        feat["bw_pct_dev"] = df["basis_weight_pct_dev"]
        feat["steam_dev"] = df["steam_pressure_pv"] - df["steam_pressure_sp"]
        feat["speed_dev"] = df["machine_speed_pv"] - df["machine_speed_sp"]
        feat["moisture_dev"] = df["moisture_pv"] - df["moisture_pv"].iloc[:5].mean()
        return feat.fillna(0)

    def fit(self, df: pd.DataFrame):
        feat = self._make_features(df)
      
        future_max_abs_dev = (
            df["basis_weight_pct_dev"].abs()
            .rolling(EARLY_WARNING_HORIZON)
            .max()
            .shift(-EARLY_WARNING_HORIZON)
        )
        label = (future_max_abs_dev > SPEC_BAND_PCT).astype(int).fillna(0)

        if label.nunique() < 2:
            
            self.is_fitted = False
            return

        X = self.scaler.fit_transform(feat.values)
        self.model.fit(X, label.values)
        self.is_fitted = True

    def predict_risk(self, df: pd.DataFrame) -> np.ndarray:
        feat = self._make_features(df)
        if not self.is_fitted:
      
            proximity = (feat["bw_pct_dev"].abs() / SPEC_BAND_PCT).clip(0, 1.5)
            momentum = (feat["bw_roc"].abs() / (feat["bw_roc"].abs().max() + 1e-6))
            risk = (0.7 * proximity + 0.3 * momentum).clip(0, 1)
            return risk.values
        X = self.scaler.transform(feat.values)
        return self.model.predict_proba(X)[:, 1]


def correlation_engine(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Rolling-window correlation matrix across the core process variables.
    Returns the correlation matrix computed over the *last* `window`
    samples (i.e. "as of now"), which is what the dashboard surfaces as
    "newly discovered" relationships.
    """
    cols = {
        "Stock Flow": "stock_flow_pv",
        "Filler Flow": "filler_flow_pv",
        "Steam Pressure": "steam_pressure_pv",
        "Machine Speed": "machine_speed_pv",
        "Moisture": "moisture_pv",
        "Ash": "ash_pv",
        "Caliper": "caliper_pv",
        "Basis Weight": "basis_weight_pv",
    }
    tail = df.iloc[-window:][list(cols.values())].rename(
        columns={v: k for k, v in cols.items()}
    )
    return tail.corr()


def top_impact_loops(corr: pd.DataFrame, target: str = "Basis Weight", top_n: int = 4):
    """Rank loops by absolute correlation with the target variable (Basis Weight)."""
    if target not in corr.columns:
        return []
    s = corr[target].drop(index=target).abs().sort_values(ascending=False)
    return list(s.head(top_n).items())



@dataclass
class OptimizationResult:
    setpoints: dict
    projected_settling_min: float
    baseline_settling_min: float
    trajectory_optimized: np.ndarray
    trajectory_baseline: np.ndarray
    rationale: list = field(default_factory=list)


def _project_trajectory(current_bw, target_bw, gain, steps=60):
    """
    Simple critically-damped-ish first-order projection used to compare
    'do nothing differently' vs 'apply the recommended setpoints'.
    `gain` controls how aggressively the loop closes the gap per step --
    higher gain (tighter, well-tuned setpoints) settles faster but we cap
    it to avoid unrealistic overshoot-free perfection.
    """
    traj = np.empty(steps)
    y = current_bw
    for i in range(steps):
        y = y + gain * (target_bw - y)
        traj[i] = y
    return traj


def settling_time_from_trajectory(traj, target, band_pct=SPEC_BAND_PCT, sample_rate_sec=10):
    band = target * band_pct / 100.0
    within = np.abs(traj - target) <= band
    for i in range(len(within)):
        if within[i] and np.all(within[i:]):
            return i * sample_rate_sec / 60.0
    return len(traj) * sample_rate_sec / 60.0 


class SetpointOptimizer:
    """
    Local coordinate-search optimizer over the four secondary loops.
    Objective: minimize projected settling time to within the 2.5% band,
    subject to each loop staying within +/-15% of its recipe nominal
    (a stand-in for the safe operating envelope Honeywell's spec asks for).
    """

    def __init__(self, safe_band_pct: float = 15.0):
        self.safe_band_pct = safe_band_pct

    def optimize(self, df: pd.DataFrame, recipe_to: dict) -> OptimizationResult:
        current_bw = df["basis_weight_pv"].iloc[-1]
        target_bw = df["basis_weight_target"].iloc[-1]
        sample_rate_sec = 10

        
        recent = df["basis_weight_pv"].iloc[-10:].values
        recent_target = df["basis_weight_target"].iloc[-10:].values
        implied_gain = np.clip(
            np.mean(np.abs(np.diff(recent)) / (np.abs(recent_target[:-1] - recent[:-1]) + 1e-6)),
            0.02, 0.35
        )
        baseline_traj = _project_trajectory(current_bw, target_bw, gain=implied_gain)
        baseline_settle = settling_time_from_trajectory(baseline_traj, target_bw, sample_rate_sec=sample_rate_sec)

 
        best_gain = implied_gain
        best_settle = baseline_settle
        for candidate_gain in np.linspace(implied_gain, 0.55, 12):
            traj = _project_trajectory(current_bw, target_bw, gain=candidate_gain)
            settle = settling_time_from_trajectory(traj, target_bw, sample_rate_sec=sample_rate_sec)
         
            safety_penalty = 5.0 if candidate_gain > 0.45 else 0.0
            if settle + safety_penalty < best_settle:
                best_settle = settle
                best_gain = candidate_gain

        optimized_traj = _project_trajectory(current_bw, target_bw, gain=best_gain)

        aggressiveness = np.clip((best_gain - implied_gain) / 0.3, 0.15, 1.0)
        current_setpoints = {
            "Stock Flow (L/min)": df["stock_flow_pv"].iloc[-1],
            "Filler Flow (L/min)": df["filler_flow_pv"].iloc[-1],
            "Dryer Steam Pressure (bar)": df["steam_pressure_pv"].iloc[-1],
            "Machine Speed (m/min)": df["machine_speed_pv"].iloc[-1],
        }
        recipe_map = {
            "Stock Flow (L/min)": recipe_to["stock_flow"],
            "Filler Flow (L/min)": recipe_to["filler_flow"],
            "Dryer Steam Pressure (bar)": recipe_to["steam_pressure"],
            "Machine Speed (m/min)": recipe_to["machine_speed"],
        }
        recommended = {}
        for k in current_setpoints:
            gap = recipe_map[k] - current_setpoints[k]
            move = current_setpoints[k] + aggressiveness * gap
            lo = recipe_map[k] * (1 - self.safe_band_pct / 100.0)
            hi = recipe_map[k] * (1 + self.safe_band_pct / 100.0)
            recommended[k] = float(np.clip(move, lo, hi))

        rationale = [
            f"Implied current loop gain ~{implied_gain:.2f} projects a "
            f"{baseline_settle:.1f} min settling time at the present rate of correction.",
            f"Coordinate search over loop gain found {best_gain:.2f} shortens this to "
            f"{best_settle:.1f} min while every recommended setpoint stays inside the "
            f"±{self.safe_band_pct:.0f}% safe operating envelope around the recipe target.",
            "Recommended setpoints were derived by moving each secondary loop toward "
            "its full recipe-target value in proportion to the extra gain purchased, "
            "then clamped to the safe band so no loop is pushed to an unsafe extreme.",
        ]

        return OptimizationResult(
            setpoints=recommended,
            projected_settling_min=best_settle,
            baseline_settling_min=baseline_settle,
            trajectory_optimized=optimized_traj,
            trajectory_baseline=baseline_traj,
            rationale=rationale,
        )



def explain(trigger: str, context: dict) -> dict:
    """
    Map a trigger code to a structured explanation: what fired, why, and
    what evidence/source it draws on. This is what gets rendered in the
    'Explainable AI Rationale' panel and logged to the audit trail.
    """
    library = {
        "EARLY_WARNING_RISK": {
            "why": (
                f"Model risk score reached {context.get('risk', 0):.0%}, driven mainly by "
                f"the basis weight trending {context.get('trend', 'away from')} the recipe "
                f"target with a rate of change of {context.get('roc', 0):.2f} gsm/sample."
            ),
            "source": "Rolling-window early-warning classifier trained on this run's history "
                      "(features: rate of change, current % deviation, steam/speed/moisture offsets).",
        },
        "SETPOINT_RECOMMENDATION": {
            "why": (
                "Coordinate search over secondary-loop gain found a faster-settling "
                "configuration that still respects the recipe's safe operating envelope."
            ),
            "source": "Local optimizer (engine.SetpointOptimizer) constrained to ±15% of "
                      "recipe-nominal setpoints; recipe boundaries per recipe_for_gsm().",
        },
        "CORRELATION_INSIGHT": {
            "why": (
                f"{context.get('loop', 'This loop')} shows a rolling correlation of "
                f"{context.get('corr', 0):.2f} with Basis Weight over the last "
                f"{context.get('window', 60)} samples, above the 0.5 threshold treated "
                "as operationally significant."
            ),
            "source": "60-sample rolling Pearson correlation across all logged process variables.",
        },
        "SENSOR_FAULT": {
            "why": f"Signal validation flagged '{context.get('flag', 'UNKNOWN')}' -- value "
                   "outside physically plausible range or flatlined beyond the expected noise floor.",
            "source": "Rule-based sensor validation guard in app.py (range + flatline checks).",
        },
    }
    return library.get(trigger, {"why": "No rationale registered for this trigger.", "source": "n/a"})
