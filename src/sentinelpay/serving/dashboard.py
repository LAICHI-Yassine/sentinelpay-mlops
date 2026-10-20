import json
import pathlib
import random
import time
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="SentinelPay | Enterprise Fraud & Risk Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"
ARTIFACTS_DIR = pathlib.Path("artifacts")

# -------------------------------------------------------------------------
# ENTERPRISE FINTECH CSS DESIGN SYSTEM (STRIPE / DATADOG DARK THEME)
# -------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-primary: #090D16;
    --bg-card: #111827;
    --bg-card-hover: #1E293B;
    --border-subtle: rgba(255, 255, 255, 0.08);
    --text-primary: #F8FAFC;
    --text-muted: #94A3B8;
    --accent-indigo: #6366F1;
    --status-approve: #10B981;
    --status-review: #F59E0B;
    --status-block: #EF4444;
}

.stApp {
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header {visibility: hidden;}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 96% !important;
}

/* Custom Top Navigation Bar */
.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: linear-gradient(90deg, #0F172A 0%, #111827 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 14px 24px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
}
.brand-box {
    display: flex;
    align-items: center;
    gap: 14px;
}
.brand-logo {
    width: 34px;
    height: 34px;
    border-radius: 8px;
    background: linear-gradient(135deg, #6366F1 0%, #3B82F6 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 16px;
    color: white;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.4);
}
.brand-title {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #F8FAFC;
    margin: 0;
}
.brand-sub {
    font-size: 12px;
    color: #94A3B8;
    font-family: 'JetBrains Mono', monospace;
}
.status-cluster {
    display: flex;
    gap: 12px;
    align-items: center;
}
.pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    padding: 5px 12px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--border-subtle);
    color: #CBD5E1;
}
.pill-live {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34D399;
}

/* Telemetry KPI Grid */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 14px;
    margin-bottom: 22px;
}
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 16px 18px;
    transition: border-color 0.2s;
}
.kpi-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
}
.kpi-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin-bottom: 6px;
    font-weight: 600;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 23px;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 4px;
}
.kpi-delta {
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    color: #38BDF8;
}

/* Enterprise Panel Containers */
.panel-header {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 8px;
    margin-bottom: 14px;
}

/* Decision Verdict Banner */
.verdict-box {
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.verdict-approve {
    background: linear-gradient(90deg, rgba(16, 185, 129, 0.14) 0%, rgba(17, 24, 39, 0.9) 100%);
    border: 1px solid rgba(16, 185, 129, 0.4);
}
.verdict-review {
    background: linear-gradient(90deg, rgba(245, 158, 11, 0.15) 0%, rgba(17, 24, 39, 0.9) 100%);
    border: 1px solid rgba(245, 158, 11, 0.45);
}
.verdict-block {
    background: linear-gradient(90deg, rgba(239, 68, 68, 0.18) 0%, rgba(17, 24, 39, 0.9) 100%);
    border: 1px solid rgba(239, 68, 68, 0.5);
}
.verdict-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 0.03em;
}
.verdict-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #94A3B8;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0B111E !important;
    border-right: 1px solid var(--border-subtle);
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------------
# LOAD ARTIFACT TELEMETRY & API STATUS
# -------------------------------------------------------------------------
meta_path = ARTIFACTS_DIR / "model_metadata.json"
drift_path = ARTIFACTS_DIR / "reports" / "drift_report.json"

meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
drift_data = json.loads(drift_path.read_text()) if drift_path.exists() else None

api_online = False
store_mode = "OFFLINE"
try:
    h = requests.get(f"{API_URL}/health", timeout=0.6).json()
    api_online = h.get("status") == "healthy"
    store_mode = h.get("feature_store", "in_memory").upper()
except Exception:
    pass

# -------------------------------------------------------------------------
# RENDER TOP NAVIGATION HEADER
# -------------------------------------------------------------------------
status_html = (
    '<span class="pill pill-live">● ENGINE ONLINE</span>'
    if api_online
    else '<span class="pill" style="color:#F87171;border-color:#EF4444;">● API OFFLINE</span>'
)

st.markdown(
    f"""
<div class="topbar">
    <div class="brand-box">
        <div class="brand-logo">SP</div>
        <div>
            <div class="brand-title">SENTINELPAY // RISK & MLOPS CONSOLE</div>
            <div class="brand-sub">IEEE-CIS Production Benchmark (N=590,540) • LightGBM → ONNX C++ Graph • TreeSHAP</div>
        </div>
    </div>
    <div class="status-cluster">
        <span class="pill">STORE: {store_mode}</span>
        <span class="pill">MODEL: v1.0-CHAMPION</span>
        {status_html}
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------------
# RENDER KPI TELEMETRY CARDS
# -------------------------------------------------------------------------
if meta:
    st.markdown(
        f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Out-of-Time ROC-AUC</div>
            <div class="kpi-value">{meta['metrics']['test_roc_auc']:.4f}</div>
            <div class="kpi-delta">Val Split: {meta['metrics']['val_roc_auc']:.4f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">PR-AUC Lift</div>
            <div class="kpi-value">14.0x</div>
            <div class="kpi-delta">PR-AUC: {meta['metrics']['test_pr_auc']:.4f} (Base: 0.034)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">ONNX C++ Mean Latency</div>
            <div class="kpi-value">{meta['benchmark']['onnx_mean_ms']:.3f} ms</div>
            <div class="kpi-delta" style="color:#34D399;">⚡ {meta['benchmark']['speedup_factor']}x vs Python LGBM</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">ONNX p99 Tail Latency</div>
            <div class="kpi-value">{meta['benchmark']['onnx_p99_ms']:.3f} ms</div>
            <div class="kpi-delta">SLA Budget: &lt; 15.00 ms</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">F2 Decision Threshold</div>
            <div class="kpi-value">{meta['optimal_threshold']:.4f}</div>
            <div class="kpi-delta">Recall@1%FPR: {meta['metrics']['test_recall_at_1pct_fpr']*100:.1f}%</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------------------
# SIDEBAR: PRESET TEST VECTORS & NAVIGATION
# -------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="panel-header">CONSOLE MODULE</div>', unsafe_allow_html=True)
    nav_mode = st.radio(
        "Module",
        [
            "01 // Real-Time Transaction Inspector",
            "02 // Redis Velocity Stress Test",
            "03 // PSI & KS Covariate Drift",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="panel-header">PRE-CALIBRATED TEST VECTORS</div>', unsafe_allow_html=True)
    scenario = st.selectbox(
        "Load Test Case Profile",
        [
            "Case A: Domestic Retail — APPROVE",
            "Case B: Grey-Zone Cross-Domain — REVIEW",
            "Case C: Account Takeover (ATO) — BLOCK",
            "Case D: Custom Manual Vector",
        ],
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size:11px; color:#64748B; font-family:'JetBrains Mono', monospace; line-height:1.6;">
        <b>DECISION POLICY MATRIX</b><br>
        • P(Fraud) &lt; 0.1526 → <span style="color:#10B981;">APPROVE</span><br>
        • 0.1526 ≤ P &lt; 0.50 → <span style="color:#F59E0B;">REVIEW</span><br>
        • P(Fraud) ≥ 0.5000 → <span style="color:#EF4444;">BLOCK</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------------------
# MODULE 01: REAL-TIME TRANSACTION INSPECTOR & TREESHAP
# -------------------------------------------------------------------------
if nav_mode.startswith("01"):
    if scenario.startswith("Case A"):
        d_amt, d_prod, d_c1, d_c2, d_c3 = 38.50, "W", 13926, 150.0, 150.0
        d_c4, d_c6, d_addr1, d_dist1 = "visa", "debit", 315.0, 5.0
        d_p_em, d_r_em, d_dev, d_info = "gmail.com", "gmail.com", "desktop", "Windows"
    elif scenario.startswith("Case B"):
        d_amt, d_prod, d_c1, d_c2, d_c3 = 310.00, "H", 12500, 321.0, 150.0
        d_c4, d_c6, d_addr1, d_dist1 = "mastercard", "credit", 204.0, 180.0
        d_p_em, d_r_em, d_dev, d_info = "yahoo.com", "outlook.com", "mobile", "iOS Device"
    elif scenario.startswith("Case C"):
        d_amt, d_prod, d_c1, d_c2, d_c3 = 2950.00, "C", 9500, 500.0, 185.0
        d_c4, d_c6, d_addr1, d_dist1 = "visa", "credit", 299.0, 850.0
        d_p_em, d_r_em, d_dev, d_info = "protonmail.com", "anonymous.ru", "mobile", "SM-G960U"
    else:
        d_amt, d_prod, d_c1, d_c2, d_c3 = 450.00, "C", 10400, 300.0, 185.0
        d_c4, d_c6, d_addr1, d_dist1 = "visa", "credit", 299.0, 50.0
        d_p_em, d_r_em, d_dev, d_info = "gmail.com", "yahoo.com", "mobile", "Android"

    col_form, col_result = st.columns([1.05, 1.45], gap="large")

    with col_form:
        st.markdown('<div class="panel-header">PAYMENT TELEMETRY PAYLOAD</div>', unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        tx_amt = f1.number_input("Amount (USD)", value=float(d_amt), min_value=0.5, step=25.0)
        prod_cd = f2.selectbox("ProductCD", ["W", "C", "H", "R", "S"], index=["W", "C", "H", "R", "S"].index(d_prod))
        card6 = f3.selectbox("Card Category", ["debit", "credit"], index=0 if d_c6 == "debit" else 1)

        f4, f5, f6 = st.columns(3)
        card1 = f4.number_input("Card BIN (card1)", value=int(d_c1), step=10)
        card2 = f5.number_input("Sub-Bank (card2)", value=float(d_c2), step=10.0)
        card3 = f6.number_input("Country ISO (card3)", value=float(d_c3), step=5.0, help="150.0 = Domestic US | 185.0 = High-Risk Cross-Border")

        f7, f8, f9 = st.columns(3)
        card4 = f7.selectbox("Network", ["visa", "mastercard", "discover", "american express"], index=["visa", "mastercard", "discover", "american express"].index(d_c4))
        addr1 = f8.number_input("Billing Region (addr1)", value=float(d_addr1), step=5.0)
        dist1 = f9.number_input("IP Distance (km)", value=float(d_dist1), step=25.0)

        f10, f11, f12 = st.columns(3)
        p_email = f10.text_input("Purchaser Domain", value=d_p_em)
        r_email = f11.text_input("Recipient Domain", value=d_r_em)
        dev_type = f12.selectbox("Device Class", ["desktop", "mobile"], index=0 if d_dev == "desktop" else 1)

        execute_btn = st.button("EXECUTE ONNX INFERENCE & SHAP ATTRIBUTION", type="primary", use_container_width=True)

    with col_result:
        st.markdown('<div class="panel-header">REAL-TIME RISK VERDICT & LOCAL EXPLAINABILITY</div>', unsafe_allow_html=True)
        if execute_btn or "last_response" not in st.session_state:
            payload = {
                "TransactionID": random.randint(5000000, 9999999),
                "TransactionAmt": tx_amt,
                "ProductCD": prod_cd,
                "card1": int(card1),
                "card2": float(card2),
                "card3": float(card3),
                "card4": card4,
                "card5": 137.0,
                "card6": card6,
                "addr1": float(addr1),
                "addr2": 87.0,
                "dist1": float(dist1),
                "P_emaildomain": p_email,
                "R_emaildomain": r_email,
                "DeviceType": dev_type,
                "DeviceInfo": d_info,
            }
            try:
                res = requests.post(f"{API_URL}/v1/score-transaction?explain_all=true", json=payload, timeout=5)
                st.session_state["last_response"] = res.json()
                st.session_state["last_payload"] = payload
            except Exception as e:
                st.error(f"Cannot reach FastAPI engine at {API_URL}. Run `uvicorn sentinelpay.serving.app:app --port 8000` ({e})")

        if "last_response" in st.session_state:
            data = st.session_state["last_response"]
            dec = data["decision"]
            prob = data["fraud_probability"]
            thresh = data["threshold_used"]

            v_class = "verdict-block" if dec == "BLOCK" else ("verdict-review" if dec == "REVIEW" else "verdict-approve")
            v_color = "#EF4444" if dec == "BLOCK" else ("#F59E0B" if dec == "REVIEW" else "#10B981")

            st.markdown(
                f"""
                <div class="verdict-box {v_class}">
                    <div>
                        <div style="font-size:11px; color:#94A3B8; text-transform:uppercase; font-weight:600;">ENGINE DISPOSITION</div>
                        <div class="verdict-title" style="color:{v_color};">{dec} // P(FRAUD) = {prob*100:.2f}%</div>
                    </div>
                    <div style="text-align:right;">
                        <div class="verdict-meta">TX_ID: #{data['transaction_id']}</div>
                        <div class="verdict-meta">E2E LATENCY: {data['inference_latency_ms']:.2f} ms</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            g_col, s_col = st.columns([1, 1.35])

            with g_col:
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        number={"suffix": "%", "font": {"size": 28, "color": "#F8FAFC", "family": "JetBrains Mono"}},
                        title={"text": "CALIBRATED RISK SCORE", "font": {"size": 11, "color": "#94A3B8"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#64748B"},
                            "bar": {"color": v_color, "thickness": 0.35},
                            "bgcolor": "#111827",
                            "borderwidth": 1,
                            "bordercolor": "rgba(255,255,255,0.08)",
                            "steps": [
                                {"range": [0, thresh * 100], "color": "rgba(16, 185, 129, 0.12)"},
                                {"range": [thresh * 100, 50], "color": "rgba(245, 158, 11, 0.12)"},
                                {"range": [50, 100], "color": "rgba(239, 68, 68, 0.15)"},
                            ],
                            "threshold": {
                                "line": {"color": "#F8FAFC", "width": 2},
                                "thickness": 0.8,
                                "value": thresh * 100,
                            },
                        },
                    )
                )
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=230,
                    margin=dict(l=20, r=20, t=35, b=10),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with s_col:
                reasons = data.get("top_risk_factors", [])
                if reasons:
                    labels = [f"{r['feature']} = {r['feature_value']}" for r in reasons][::-1]
                    impacts = [r["shap_log_odds_impact"] for r in reasons][::-1]
                    colors = ["#EF4444" if val > 0 else "#10B981" for val in impacts]

                    fig_shap = go.Figure(
                        go.Bar(
                            x=impacts,
                            y=labels,
                            orientation="h",
                            marker=dict(color=colors, line=dict(width=0)),
                            text=[f"{val:+.4f} log-odds" for val in impacts],
                            textposition="auto",
                            textfont=dict(family="JetBrains Mono", size=11, color="#F8FAFC"),
                        )
                    )
                    fig_shap.update_layout(
                        title=dict(text="TOP-3 TREESHAP FEATURE ATTRIBUTIONS", font=dict(size=11, color="#94A3B8")),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(17,24,39,0.6)",
                        font=dict(family="JetBrains Mono", color="#CBD5E1", size=11),
                        xaxis=dict(title="SHAP Impact (Log-Odds)", gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.2)"),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
                        height=230,
                        margin=dict(l=10, r=15, t=35, b=20),
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)

            with st.expander("INSPECT RAW JSON TELEMETRY & REQUEST VECTOR"):
                c_req, c_res = st.columns(2)
                c_req.json(st.session_state.get("last_payload", {}))
                c_res.json(data)

# -------------------------------------------------------------------------
# MODULE 02: REDIS SLIDING-WINDOW VELOCITY ATTACK SIMULATOR
# -------------------------------------------------------------------------
elif nav_mode.startswith("02"):
    st.markdown('<div class="panel-header">HIGH-FREQUENCY CARD TESTING SIMULATOR (REDIS SLIDING WINDOW)</div>', unsafe_allow_html=True)
    c_cfg, c_chart = st.columns([1, 2], gap="large")

    with c_cfg:
        target_card = st.number_input("Target Card BIN (card1)", value=77410, step=1)
        base_swipe_amt = st.number_input("Initial Swipe Amount ($)", value=95.0, step=25.0)
        swipe_Increment = st.number_input("Escalation Multiplier per Swipe", value=1.6, min_value=1.0, max_value=3.0, step=0.1)
        n_swipes = st.slider("Burst Swipe Count (within 1h window)", min_value=3, max_value=10, value=6)
        run_burst = st.button("LAUNCH BURST VELOCITY SIMULATION", type="primary", use_container_width=True)

    with c_chart:
        if run_burst:
            swipes, probs, latencies, rows = [], [], [], []
            for i in range(1, n_swipes + 1):
                amt = round(base_swipe_amt * (swipe_Increment ** (i - 1)), 2)
                payload = {
                    "TransactionID": 900000 + i,
                    "TransactionAmt": amt,
                    "ProductCD": "C",
                    "card1": int(target_card),
                    "card2": 500.0,
                    "card3": 185.0,
                    "card4": "visa",
                    "card6": "credit",
                    "addr1": 299.0,
                    "P_emaildomain": "gmail.com",
                    "R_emaildomain": "anonymous.de",
                    "DeviceType": "mobile",
                }
                res = requests.post(f"{API_URL}/v1/score-transaction?explain_all=true", json=payload, timeout=5).json()
                swipes.append(f"Swipe #{i} (${amt})")
                probs.append(round(res["fraud_probability"] * 100, 2))
                latencies.append(res["inference_latency_ms"])
                rows.append(
                    {
                        "Sequence": f"#{i}",
                        "Amount ($)": amt,
                        "P(Fraud) %": round(res["fraud_probability"] * 100, 2),
                        "Decision": res["decision"],
                        "Latency (ms)": res["inference_latency_ms"],
                        "Primary SHAP Driver": f"{res['top_risk_factors'][0]['feature']} ({res['top_risk_factors'][0]['shap_log_odds_impact']:+.3f})",
                    }
                )
                time.sleep(0.08)

            fig_vel = go.Figure()
            fig_vel.add_trace(
                go.Scatter(
                    x=swipes,
                    y=probs,
                    mode="lines+markers+text",
                    text=[f"{p:.1f}%" for p in probs],
                    textposition="top center",
                    line=dict(color="#6366F1", width=3),
                    marker=dict(size=10, color=["#EF4444" if p >= 50 else ("#F59E0B" if p >= 15.26 else "#10B981") for p in probs]),
                )
            )
            fig_vel.add_hline(y=15.26, line_dash="dot", line_color="#F59E0B", annotation_text="REVIEW Threshold (15.26%)")
            fig_vel.add_hline(y=50.0, line_dash="dash", line_color="#EF4444", annotation_text="BLOCK Threshold (50.0%)")
            fig_vel.update_layout(
                title="REAL-TIME RISK ESCALATION ACROSS CONSECUTIVE SWIPES",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.6)",
                font=dict(family="JetBrains Mono", color="#CBD5E1", size=11),
                yaxis=dict(title="Fraud Probability (%)", gridcolor="rgba(255,255,255,0.05)"),
                height=310,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_vel, use_container_width=True)
            st.dataframe(rows, use_container_width=True)

# -------------------------------------------------------------------------
# MODULE 03: PSI & KS COVARIATE DRIFT MONITOR
# -------------------------------------------------------------------------
elif nav_mode.startswith("03"):
    st.markdown('<div class="panel-header">POPULATION STABILITY INDEX (PSI) & KOLMOGOROV-SMIRNOV DRIFT MONITOR</div>', unsafe_allow_html=True)
    if drift_data:
        features = list(drift_data["features"].keys())
        psi_vals = [drift_data["features"][f]["psi"] for f in features]
        colors = ["#EF4444" if drift_data["features"][f]["drift_detected"] else "#10B981" for f in features]

        fig_psi = go.Figure(
            go.Bar(
                x=features,
                y=psi_vals,
                marker_color=colors,
                text=[f"PSI: {v:.3f}" for v in psi_vals],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=11, color="#F8FAFC"),
            )
        )
        fig_psi.add_hline(y=0.20, line_dash="dash", line_color="#EF4444", annotation_text="Banking Retraining Trigger (PSI ≥ 0.20)")
        fig_psi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,0.6)",
            font=dict(family="JetBrains Mono", color="#CBD5E1", size=11),
            yaxis=dict(title="Population Stability Index (PSI)", gridcolor="rgba(255,255,255,0.05)", range=[0, max(psi_vals) * 1.25]),
            height=380,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_psi, use_container_width=True)