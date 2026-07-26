"""
app.py
------
Grade Change Intelligence -- Streamlit front-end.

Layout:
    Sidebar   : run configuration (from/to grade, run controls, seed)
    Top row   : KPI metric cards (current BW, deviation, financial loss, ETA to stable)
    Alert bar : early-warning banner + Accept/Reject operator loop
    Row 2     : live trajectory vs projected future-state trajectory
    Row 3     : correlation heatmap + top-impact loops
    Row 4     : recommended setpoints table + explainability panel
    Footer    : append-only audit trail viewer

Run locally:    streamlit run app.py
Deploy         : push this folder to a public GitHub repo, then
                 point Streamlit Community Cloud at app.py -- see README.
"""

import os
import time
import csv
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulator import simulate_grade_change, recipe_for_gsm
from engine import (
    EarlyWarningModel,
    correlation_engine,
    top_impact_loops,
    SetpointOptimizer,
    explain,
    SPEC_BAND_PCT,
)

st.set_page_config(
    page_title="Grade Change Intelligence | Paper Machine Copilot",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stMetric {
        background-color: #10161f;
        border: 1px solid #232c3a;
        border-radius: 10px;
        padding: 14px 16px 8px 16px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.65rem; }
    .risk-banner-high {
        background: linear-gradient(90deg, #4a1010, #2a0d0d);
        border-left: 5px solid #ff4b4b;
        padding: 14px 18px; border-radius: 8px; margin-bottom: 10px;
    }
    .risk-banner-med {
        background: linear-gradient(90deg, #4a3a10, #2a220d);
        border-left: 5px solid #f0a500;
        padding: 14px 18px; border-radius: 8px; margin-bottom: 10px;
    }
    .risk-banner-low {
        background: linear-gradient(90deg, #103a1c, #0d2a17);
        border-left: 5px solid #3ddc84;
        padding: 14px 18px; border-radius: 8px; margin-bottom: 10px;
    }
    .rationale-box {
        background-color: #10161f; border: 1px solid #232c3a;
        border-radius: 8px; padding: 10px 14px; margin-bottom: 6px; font-size: 0.92rem;
    }
    footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

AUDIT_LOG_PATH = "audit_trail.csv"
INR_PER_TONNE_WASTE = 42000 


with st.sidebar:
    st.title("🏭 Mill Configuration")
    st.caption("Configure the grade change event to simulate, then run.")

    from_gsm = st.slider("From grade (gsm)", 40, 140, 60, step=5)
    to_gsm = st.slider("To grade (gsm)", 40, 140, 100, step=5)
    duration_min = st.slider("Run duration (min)", 30, 180, 90, step=10)
    transition_start = st.slider("Transition start (min)", 2, 30, 10)
    ramp_duration = st.slider("Ramp duration (min)", 1, 20, 6)

    st.divider()
    seed = st.number_input("Random seed", value=42, step=1)
    inject_anomalies = st.checkbox("Inject sensor anomalies", value=True)
    operator_overrides = st.checkbox("Simulate operator overrides", value=True)

    st.divider()
    cost_per_tonne = st.number_input(
        "Off-spec cost (₹ / tonne)", value=INR_PER_TONNE_WASTE, step=1000,
        help="Stylized cost used to translate instability into a financial loss counter."
    )
    machine_tonnes_per_hr = st.number_input("Machine output (tonnes/hr)", value=18.0, step=0.5)

    st.divider()
    run_clicked = st.button("▶ Run Simulation", use_container_width=True, type="primary")
    st.caption("Human-in-the-loop: your Accept/Reject decisions below are logged to an "
               "append-only audit trail for model-accuracy review.")



if "df" not in st.session_state or run_clicked:
    st.session_state.df = simulate_grade_change(
        from_gsm=from_gsm, to_gsm=to_gsm, duration_min=duration_min,
        transition_start_min=transition_start, ramp_duration_min=ramp_duration,
        seed=int(seed), inject_anomalies=inject_anomalies,
        operator_overrides=operator_overrides,
    )
    st.session_state.cursor = max(20, st.session_state.df.attrs["ramp_start_sample"])
    st.session_state.last_decision = None

df_full = st.session_state.df
recipe_to = df_full.attrs["recipe_to"]


if "cursor" not in st.session_state:
    st.session_state.cursor = max(20, df_full.attrs["ramp_start_sample"])

st.title("Grade Change Intelligence")
st.caption(
    f"Live copilot for the {int(from_gsm)} → {int(to_gsm)} gsm grade change · "
    f"Basis weight spec band: ±{SPEC_BAND_PCT}% of recipe target"
)

col_step, col_auto, col_reset = st.columns([1, 1, 1])
with col_step:
    if st.button("⏭ Advance 2 min", use_container_width=True):
        st.session_state.cursor = min(len(df_full) - 1, st.session_state.cursor + 12)
with col_auto:
    play = st.toggle("▶ Auto-stream", value=False,
                      help="Replays the run like a live historian feed.")
with col_reset:
    if st.button("⏮ Jump to transition start", use_container_width=True):
        st.session_state.cursor = df_full.attrs["ramp_start_sample"]

cursor = st.session_state.cursor
df = df_full.iloc[: cursor + 1].copy()


def validate_sensors(row) -> list:
    """Rule-based physical-plausibility checks. Never raises -- always
    returns a (possibly empty) list of fault codes so the pipeline keeps
    running even when a sensor misbehaves."""
    faults = []
    if pd.isna(row["steam_pressure_pv"]) or row["steam_pressure_pv"] < 0:
        faults.append("NEGATIVE_OR_MISSING_STEAM_PRESSURE")
    if row["machine_speed_pv"] < 0 or row["machine_speed_pv"] > 2000:
        faults.append("MACHINE_SPEED_OUT_OF_RANGE")
    if row["sensor_flag"] != "OK":
        faults.append(row["sensor_flag"])
    return faults

latest = df.iloc[-1]
sensor_faults = validate_sensors(latest)


df_clean = df.copy()
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
df_clean[numeric_cols] = df_clean[numeric_cols].ffill().bfill()

ew_model = EarlyWarningModel()
if len(df_clean) >= 20:
    ew_model.fit(df_clean)
    risk_scores = ew_model.predict_risk(df_clean)
    current_risk = float(risk_scores[-1])
else:
    current_risk = 0.0

pct_dev_now = float(latest["basis_weight_pct_dev"])
bw_roc = float(df_clean["basis_weight_pv"].diff(3).iloc[-1]) if len(df_clean) > 3 else 0.0

off_spec_mask = df_clean["basis_weight_pct_dev"].abs() > SPEC_BAND_PCT
off_spec_minutes = off_spec_mask.sum() * (df_full.attrs["sample_rate_sec"] / 60.0)
waste_tonnes = (off_spec_minutes / 60.0) * machine_tonnes_per_hr
financial_loss = waste_tonnes * cost_per_tonne
loss_rate_per_min = (machine_tonnes_per_hr / 60.0) * cost_per_tonne if off_spec_mask.iloc[-1] else 0.0


k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Basis Weight (PV)", f"{latest['basis_weight_pv']:.1f} gsm",
          f"{pct_dev_now:+.2f}% vs target")
k2.metric("Target (recipe)", f"{latest['basis_weight_target']:.1f} gsm")
k3.metric("Early-Warning Risk", f"{current_risk:.0%}",
          "Rising" if bw_roc > 0.05 else ("Falling" if bw_roc < -0.05 else "Stable"))
k4.metric("Off-Spec Time So Far", f"{off_spec_minutes:.1f} min")
k5.metric("Financial Loss (cumulative)", f"₹{financial_loss:,.0f}",
          f"₹{loss_rate_per_min:,.0f}/min live" if loss_rate_per_min > 0 else "Currently on-spec")


optimizer = SetpointOptimizer()
opt_result = optimizer.optimize(df_clean, recipe_to)

if abs(pct_dev_now) > SPEC_BAND_PCT or current_risk > 0.6:
    banner_class, banner_label = "risk-banner-high", "🔴 HIGH RISK -- corrective action recommended now"
elif current_risk > 0.3:
    banner_class, banner_label = "risk-banner-med", "🟠 ELEVATED RISK -- monitor closely, setpoints below"
else:
    banner_class, banner_label = "risk-banner-low", "🟢 STABLE -- within safe operating margin"

ew_context = {
    "risk": current_risk,
    "trend": "away from" if bw_roc > 0 and pct_dev_now > 0 else "toward",
    "roc": bw_roc,
}
ew_explanation = explain("EARLY_WARNING_RISK", ew_context)

st.markdown(f"""
<div class="{banner_class}">
<b>{banner_label}</b><br>
{ew_explanation['why']}
</div>
""", unsafe_allow_html=True)

if sensor_faults:
    for f in sensor_faults:
        sf_expl = explain("SENSOR_FAULT", {"flag": f})
        st.warning(f"⚠️ Sensor validation: **{f}** -- {sf_expl['why']} Pipeline continues on "
                   f"forward-filled values; recommendation confidence reduced accordingly.")

acc_col, rej_col, note_col = st.columns([1, 1, 3])
with acc_col:
    accept_clicked = st.button("✅ Accept Recommendation", use_container_width=True)
with rej_col:
    reject_clicked = st.button("❌ Reject Recommendation", use_container_width=True)
with note_col:
    operator_note = st.text_input("Operator note (optional)", key="op_note", label_visibility="collapsed",
                                   placeholder="Optional note for the audit trail...")

def log_decision(decision: str):
    """Append-only audit trail -- never overwrite, only append, so model
    accuracy can be reviewed against every historical recommendation."""
    file_exists = os.path.isfile(AUDIT_LOG_PATH)
    with open(AUDIT_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "logged_at", "t_min", "decision", "basis_weight_pv", "pct_dev",
                "risk_score", "recommended_setpoints", "operator_note"
            ])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            f"{latest['t_min']:.2f}",
            decision,
            f"{latest['basis_weight_pv']:.2f}",
            f"{pct_dev_now:.2f}",
            f"{current_risk:.3f}",
            str({k: round(v, 2) for k, v in opt_result.setpoints.items()}),
            operator_note,
        ])

if accept_clicked:
    log_decision("ACCEPTED")
    st.session_state.last_decision = "ACCEPTED"
    st.success("Recommendation accepted and logged to the audit trail.")
if reject_clicked:
    log_decision("REJECTED")
    st.session_state.last_decision = "REJECTED"
    st.info("Recommendation rejected and logged to the audit trail. This will be weighed in "
            "the next accuracy review pass.")

st.divider()


left, right = st.columns([3, 2])

with left:
    st.subheader("Live Trajectory vs. Future-State Projection")
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    fig.add_trace(go.Scatter(
        x=df_clean["t_min"], y=df_clean["basis_weight_pv"],
        name="Basis Weight (live)", line=dict(color="#3ddc84", width=2.2)
    ))
    fig.add_trace(go.Scatter(
        x=df_clean["t_min"], y=df_clean["basis_weight_target"],
        name="Recipe Target", line=dict(color="#8899aa", width=1.4, dash="dot")
    ))
    band_hi = df_clean["basis_weight_target"] * (1 + SPEC_BAND_PCT / 100)
    band_lo = df_clean["basis_weight_target"] * (1 - SPEC_BAND_PCT / 100)
    fig.add_trace(go.Scatter(x=df_clean["t_min"], y=band_hi, line=dict(width=0),
                              showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df_clean["t_min"], y=band_lo, line=dict(width=0),
                              fill="tonexty", fillcolor="rgba(61,220,132,0.08)",
                              name=f"±{SPEC_BAND_PCT}% spec band", hoverinfo="skip"))

    future_t = latest["t_min"] + np.arange(len(opt_result.trajectory_baseline)) * (
        df_full.attrs["sample_rate_sec"] / 60.0)
    fig.add_trace(go.Scatter(
        x=future_t, y=opt_result.trajectory_baseline,
        name="Projected (uncorrected)", line=dict(color="#ff4b4b", width=2, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=future_t, y=opt_result.trajectory_optimized,
        name="Projected (optimized)", line=dict(color="#4b9bff", width=2, dash="dash")
    ))
    fig.update_layout(
        height=430, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="Time (min)", yaxis_title="Basis Weight (gsm)",
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Baseline projection settles in **{opt_result.baseline_settling_min:.1f} min**. "
        f"Optimized setpoints project settling in **{opt_result.projected_settling_min:.1f} min** "
        f"-- a **{max(0, opt_result.baseline_settling_min - opt_result.projected_settling_min):.1f} min** "
        "reduction in stabilization time."
    )

with right:
    st.subheader("Recommended Safe Setpoints")
    sp_rows = []
    for k, v in opt_result.setpoints.items():
        current_val = latest[{
            "Stock Flow (L/min)": "stock_flow_pv",
            "Filler Flow (L/min)": "filler_flow_pv",
            "Dryer Steam Pressure (bar)": "steam_pressure_pv",
            "Machine Speed (m/min)": "machine_speed_pv",
        }[k]]
        sp_rows.append({"Loop": k, "Current": round(current_val, 2), "Recommended": round(v, 2)})
    st.dataframe(pd.DataFrame(sp_rows), hide_index=True, use_container_width=True)

    setpoint_expl = explain("SETPOINT_RECOMMENDATION", {})
    for line in opt_result.rationale:
        st.markdown(f'<div class="rationale-box">💡 {line}</div>', unsafe_allow_html=True)
    st.caption(f"Source: {setpoint_expl['source']}")

st.divider()


c1, c2 = st.columns([3, 2])

with c1:
    st.subheader("Latent Correlation Map (rolling 60-sample window)")
    if len(df_clean) >= 15:
        corr = correlation_engine(df_clean, window=min(60, len(df_clean)))
        heat = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            colorbar=dict(title="ρ"),
        ))
        heat.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark")
        st.plotly_chart(heat, use_container_width=True)
    else:
        st.info("Advance the run a little further to accumulate enough samples for correlation analysis.")

with c2:
    st.subheader("Highest-Impact Loops on Instability")
    if len(df_clean) >= 15:
        impacts = top_impact_loops(corr, target="Basis Weight", top_n=4)
        for loop, corr_val in impacts:
            bar_pct = int(abs(corr_val) * 100)
            direction = "↑ same direction" if corr_val > 0 else "↓ opposite direction"
            st.markdown(f"**{loop}** -- ρ = {corr_val:+.2f} ({direction})")
            st.progress(min(1.0, abs(corr_val)))
            insight = explain("CORRELATION_INSIGHT", {"loop": loop, "corr": corr_val, "window": 60})
            st.caption(insight["why"])
    else:
        st.info("Waiting on data...")

st.divider()


st.subheader("Secondary Loop Trends")
loop_cols = st.columns(4)
loop_defs = [
    ("Stock Flow", "stock_flow_pv", "stock_flow_sp", "#4b9bff"),
    ("Filler Flow", "filler_flow_pv", "filler_flow_sp", "#f0a500"),
    ("Steam Pressure", "steam_pressure_pv", "steam_pressure_sp", "#e05fdb"),
    ("Machine Speed", "machine_speed_pv", "machine_speed_sp", "#3ddc84"),
]
for col, (label, pv, sp, color) in zip(loop_cols, loop_defs):
    with col:
        f = go.Figure()
        f.add_trace(go.Scatter(x=df_clean["t_min"], y=df_clean[pv], name="PV",
                                line=dict(color=color, width=1.8)))
        f.add_trace(go.Scatter(x=df_clean["t_min"], y=df_clean[sp], name="SP",
                                line=dict(color="#8899aa", width=1, dash="dot")))
        f.update_layout(height=220, margin=dict(l=5, r=5, t=25, b=5),
                         template="plotly_dark", showlegend=False, title=label)
        st.plotly_chart(f, use_container_width=True)

st.divider()


st.subheader("Operator Feedback Audit Trail")
if os.path.isfile(AUDIT_LOG_PATH):
    audit_df = pd.read_csv(AUDIT_LOG_PATH)
    st.dataframe(audit_df.sort_values("logged_at", ascending=False), hide_index=True,
                 use_container_width=True, height=220)
    acc_rate = (audit_df["decision"] == "ACCEPTED").mean() * 100 if len(audit_df) else 0
    st.caption(f"{len(audit_df)} decisions logged · {acc_rate:.0f}% acceptance rate so far. "
               "This file is append-only and forms the basis for periodic model-accuracy review.")
else:
    st.caption("No operator decisions logged yet -- Accept or Reject a recommendation above to begin.")


if play and cursor < len(df_full) - 1:
    time.sleep(0.6)
    st.session_state.cursor = min(len(df_full) - 1, cursor + 3)
    st.rerun()
