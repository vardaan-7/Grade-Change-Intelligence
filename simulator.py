import numpy as np
import pandas as pd



def recipe_for_gsm(gsm: float) -> dict:
    """Return the nominal secondary-loop setpoints for a given basis weight (gsm)."""
    return {
        "stock_flow": 180.0 + (gsm - 60.0) * 3.1,       # L/min
        "filler_flow": 22.0 + (gsm - 60.0) * 0.42,      # L/min
        "steam_pressure": 3.1 + (gsm - 60.0) * 0.018,   # bar
        "machine_speed": 950.0 - (gsm - 60.0) * 4.2,    # m/min
        "target_moisture": 6.8,                         # % (spec-held, not a free variable)
        "target_ash": 18.0,                              # %
        "target_caliper": 0.09 + (gsm - 60.0) * 0.0009,  # mm
    }


def _first_order_lag(target_series: np.ndarray, tau: float, initial: float) -> np.ndarray:
    """
    Simple discrete first-order lag filter: y[t] = y[t-1] + (dt/tau)*(target - y[t-1]).
    Models the fact that a physical loop can't jump instantly to a new setpoint.
    """
    out = np.empty_like(target_series, dtype=float)
    y = initial
    alpha = 1.0 / tau  
    for i, tgt in enumerate(target_series):
        y = y + alpha * (tgt - y)
        out[i] = y
    return out


def simulate_grade_change(
    from_gsm: float = 60.0,
    to_gsm: float = 100.0,
    duration_min: int = 90,
    sample_rate_sec: int = 10,
    transition_start_min: int = 10,
    ramp_duration_min: int = 6,
    seed: int = 42,
    inject_anomalies: bool = True,
    operator_overrides: bool = True,
) -> pd.DataFrame:
    """
    Build a full time-series DataFrame for one grade change event.

    Returns columns:
        timestamp, t_min, phase, target_basis_weight,
        stock_flow_sp/_pv, filler_flow_sp/_pv, steam_pressure_sp/_pv,
        machine_speed_sp/_pv, moisture_pv, ash_pv, caliper_pv,
        basis_weight_pv, basis_weight_pct_dev, sensor_flag
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration_min * 60 / sample_rate_sec)
    t_min = np.arange(n_samples) * (sample_rate_sec / 60.0)

    recipe_from = recipe_for_gsm(from_gsm)
    recipe_to = recipe_for_gsm(to_gsm)

    ramp_start_sample = int(transition_start_min * 60 / sample_rate_sec)
    ramp_end_sample = int((transition_start_min + ramp_duration_min) * 60 / sample_rate_sec)

    def ramp(key):
        sp = np.full(n_samples, recipe_from[key])
        if ramp_end_sample > ramp_start_sample:
            ramp_slice = np.linspace(recipe_from[key], recipe_to[key],
                                      ramp_end_sample - ramp_start_sample)
            sp[ramp_start_sample:ramp_end_sample] = ramp_slice
        sp[ramp_end_sample:] = recipe_to[key]
        return sp

    stock_sp = ramp("stock_flow")
    filler_sp = ramp("filler_flow")
    steam_sp = ramp("steam_pressure")
    speed_sp = ramp("machine_speed")
    bw_target = ramp("target_moisture") * 0 + np.where(
        np.arange(n_samples) < ramp_start_sample, from_gsm,
        np.where(np.arange(n_samples) >= ramp_end_sample, to_gsm,
                 np.linspace(from_gsm, to_gsm, n_samples)[
                     np.arange(n_samples)])
    )

    bw_target = np.full(n_samples, from_gsm)
    if ramp_end_sample > ramp_start_sample:
        bw_target[ramp_start_sample:ramp_end_sample] = np.linspace(
            from_gsm, to_gsm, ramp_end_sample - ramp_start_sample)
    bw_target[ramp_end_sample:] = to_gsm


    if operator_overrides:
        n_overrides = rng.integers(2, 5)
        for _ in range(n_overrides):
            idx = rng.integers(ramp_start_sample, min(n_samples - 1, ramp_end_sample + 100))
            nudge = rng.normal(0, 1) * rng.choice([1, 1, 1, -1])
            loop = rng.choice(["stock_flow", "steam_pressure", "machine_speed"])
            window = slice(idx, min(idx + 30, n_samples))
            if loop == "stock_flow":
                stock_sp[window] += nudge * 4.0
            elif loop == "steam_pressure":
                steam_sp[window] += nudge * 0.06
            else:
                speed_sp[window] += nudge * 6.0

    stock_pv = _first_order_lag(stock_sp, tau=8, initial=stock_sp[0]) + rng.normal(0, 0.6, n_samples)
    filler_pv = _first_order_lag(filler_sp, tau=10, initial=filler_sp[0]) + rng.normal(0, 0.15, n_samples)
    steam_pv = _first_order_lag(steam_sp, tau=12, initial=steam_sp[0]) + rng.normal(0, 0.02, n_samples)
    speed_pv = _first_order_lag(speed_sp, tau=6, initial=speed_sp[0]) + rng.normal(0, 1.2, n_samples)


    steam_dev = steam_pv - recipe_from["steam_pressure"]
    speed_dev = speed_pv - recipe_from["machine_speed"]
    moisture_pv = (
        recipe_from["target_moisture"]
        - 1.35 * steam_dev
        + 0.006 * speed_dev
        + rng.normal(0, 0.08, n_samples)
    )
    moisture_pv = _first_order_lag(moisture_pv, tau=5, initial=moisture_pv[0])

    ash_pv = 18.0 + 0.11 * (filler_pv - recipe_from["filler_flow"]) + rng.normal(0, 0.25, n_samples)

    
    bw_pv = (
        bw_target
        + 0.09 * (stock_pv - _first_order_lag(stock_sp, tau=1, initial=stock_sp[0]))
        - 0.012 * (speed_pv - speed_sp)
        + 0.6 * (moisture_pv - recipe_from["target_moisture"])
        + rng.normal(0, 0.35, n_samples)
    )
    bw_pv = _first_order_lag(bw_pv, tau=4, initial=bw_pv[0])

    caliper_pv = (
        0.09 + (bw_pv - 60.0) * 0.0009 + rng.normal(0, 0.0015, n_samples)
    )

    
    sensor_flag = np.array(["OK"] * n_samples, dtype=object)
    if inject_anomalies:
        n_anom = rng.integers(1, 4)
        for _ in range(n_anom):
            idx = rng.integers(0, n_samples)
            width = rng.integers(1, 6)
            end = min(idx + width, n_samples)
            kind = rng.choice(["dropout", "spike", "stuck", "negative_pressure"])
            if kind == "dropout":
                steam_pv[idx:end] = np.nan
                sensor_flag[idx:end] = "STEAM_SENSOR_DROPOUT"
            elif kind == "spike":
                bw_pv[idx:end] += rng.normal(8, 2)
                sensor_flag[idx:end] = "SCANNER_SPIKE"
            elif kind == "stuck":
                speed_pv[idx:end] = speed_pv[idx]
                sensor_flag[idx:end] = "SPEED_SENSOR_STUCK"
            else:
                steam_pv[idx:end] = -abs(rng.normal(0.3, 0.1))
                sensor_flag[idx:end] = "NEGATIVE_PRESSURE_FAULT"

    pct_dev = (bw_pv - bw_target) / bw_target * 100.0

    phase = np.where(
        np.arange(n_samples) < ramp_start_sample, "PRE_TRANSITION",
        np.where(np.arange(n_samples) < ramp_end_sample, "RAMPING", "STABILIZING")
    )

    df = pd.DataFrame({
        "t_min": t_min,
        "phase": phase,
        "basis_weight_target": bw_target,
        "stock_flow_sp": stock_sp, "stock_flow_pv": stock_pv,
        "filler_flow_sp": filler_sp, "filler_flow_pv": filler_pv,
        "steam_pressure_sp": steam_sp, "steam_pressure_pv": steam_pv,
        "machine_speed_sp": speed_sp, "machine_speed_pv": speed_pv,
        "moisture_pv": moisture_pv,
        "ash_pv": ash_pv,
        "caliper_pv": caliper_pv,
        "basis_weight_pv": bw_pv,
        "basis_weight_pct_dev": pct_dev,
        "sensor_flag": sensor_flag,
    })
    df.attrs["from_gsm"] = from_gsm
    df.attrs["to_gsm"] = to_gsm
    df.attrs["recipe_to"] = recipe_to
    df.attrs["ramp_start_sample"] = ramp_start_sample
    df.attrs["ramp_end_sample"] = ramp_end_sample
    df.attrs["sample_rate_sec"] = sample_rate_sec
    return df
