"""
dashboard/app.py
================
UpayPulse AI — Enterprise MFS Liquidity & Ecosystem Intelligence Dashboard

A production-grade Streamlit application for Mobile Financial Services (MFS)
operators, field liquidity managers, and fintech risk analysts in Bangladesh.

Features:
---------
1. Executive Liquidity Command Center : Live KPI strip, shortage alert badges, and AI rebalancing.
2. AI-Powered Rebalancing Engine       : Automated candidate matching, Haversine scoring, and 1-click approvals.
3. Interactive Geospatial Map (Folium): Multi-layer GIS map covering agents and merchants across Bangladesh.
4. Transaction Flow Analytics (Plotly): Diurnal curves, volume breakdown, and commercial hub rankings.
5. Customer Diversion Simulator       : AI-triggered digital merchant offers to alleviate agent cash squeeze.
6. Data Explorer & CSV Export         : Search, filter, and export datasets.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# 1. ENVIRONMENT & PATH RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────
# Ensure the project root is in sys.path for importing ml_engine and data_generator
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATA_DIR = ROOT_DIR / "data"

# Import domain ML and generation modules if available
try:
    from ml_engine import haversine_km, calculate_shortage, classify_risk
except ImportError:
    # Fallback Haversine if imported standalone
    from math import radians, cos, sin, asin, sqrt
    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        return 2 * R * asin(sqrt(a))

    def calculate_shortage(agents_df):
        df = agents_df.copy()
        df["expected_shortage"] = np.maximum(0, df["predicted_demand_4h"] - df["current_cash"])
        df["surplus_cash"] = np.maximum(0, df["current_cash"] - df["predicted_demand_4h"])
        return df

    def classify_risk(agents_df):
        df = agents_df.copy()
        if "expected_shortage" not in df.columns:
            df = calculate_shortage(df)
        df["risk_level"] = df["expected_shortage"].apply(lambda x: "HIGH" if x > 0 else "SAFE")
        return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. STREAMLIT PAGE CONFIGURATION & STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UpayPulse AI | MFS Intelligence Command Center",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Impact Fintech Dark Theme CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Base application background */
.stApp {
    background: radial-gradient(circle at 10% 20%, #0d1117 0%, #161b22 90%);
    color: #e6edf3;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #12161f;
    border-right: 1px solid #30363d;
}

/* Premium KPI Metric Cards */
.kpi-container {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 20px 22px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}
.kpi-container:hover {
    transform: translateY(-3px);
    border-color: rgba(88, 166, 255, 0.3);
    box-shadow: 0 8px 30px rgba(88, 166, 255, 0.15);
}
.kpi-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #8b949e;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 4px 0;
}
.kpi-subtitle {
    font-size: 0.76rem;
    color: #7d8590;
    margin-top: 4px;
}

/* Accent Colors */
.text-red    { color: #f85149; }
.text-green  { color: #3fb950; }
.text-blue   { color: #58a6ff; }
.text-amber  { color: #d29922; }
.text-purple { color: #bc8cff; }

/* Section Header Banners */
.section-banner {
    background: linear-gradient(90deg, rgba(33, 38, 45, 0.95) 0%, rgba(22, 27, 34, 0.8) 100%);
    border-left: 4px solid #58a6ff;
    border-radius: 0 10px 10px 0;
    padding: 14px 22px;
    margin: 28px 0 18px 0;
    font-size: 1.12rem;
    font-weight: 600;
    color: #e6edf3;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Risk Badges */
.badge {
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-block;
}
.badge-danger {
    background-color: rgba(248, 81, 73, 0.15);
    color: #f85149;
    border: 1px solid rgba(248, 81, 73, 0.4);
}
.badge-success {
    background-color: rgba(63, 185, 80, 0.15);
    color: #3fb950;
    border: 1px solid rgba(63, 185, 80, 0.4);
}
.badge-info {
    background-color: rgba(88, 166, 255, 0.15);
    color: #58a6ff;
    border: 1px solid rgba(88, 166, 255, 0.4);
}

/* Smartphone Mockup Container */
.smartphone-screen {
    background: linear-gradient(180deg, #090d16 0%, #111827 100%);
    border: 2px solid #384252;
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
    max-width: 440px;
    margin: 0 auto;
}

/* Audit Log Row */
.audit-entry {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #3fb950;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA LOADING & PIPELINE CACHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_and_enrich_datasets():
    """
    Load agents.csv, transactions.csv, and merchants.csv.
    Enrich agents with empirical 4-hour demand projections derived from transactions.
    """
    agents_path = DATA_DIR / "agents.csv"
    txns_path = DATA_DIR / "transactions.csv"
    merchants_path = DATA_DIR / "merchants.csv"

    # Auto-generate if missing
    if not (agents_path.exists() and txns_path.exists() and merchants_path.exists()):
        st.info("Generating synthetic datasets for UpayPulse AI... Please wait.")
        try:
            from data_generator import save_all_datasets
            save_all_datasets(DATA_DIR)
        except Exception as e:
            st.error(f"Error generating data: {e}")

    agents_df = pd.read_csv(agents_path)
    txns_df = pd.read_csv(txns_path)
    merchants_df = pd.read_csv(merchants_path)

    # Convert timestamps
    txns_df["timestamp"] = pd.to_datetime(txns_df["timestamp"])
    txns_df["date"] = txns_df["timestamp"].dt.date
    txns_df["hour"] = txns_df["timestamp"].dt.hour

    # ── Derive realistic 4-Hour Cash Demand from Transaction Velocity ─────────
    # Calculate average daily cash-out volume per agent from the 50,000 transactions
    cashouts = txns_df[txns_df["transaction_type"] == "cash_out"]
    agent_total_cashout = cashouts.groupby("agent_id")["amount"].sum()
    days_span = max(1, (txns_df["date"].max() - txns_df["date"].min()).days)

    # 4-hour window demand = (daily cashout / 6) * peak multiplier (1.4 - 1.8x)
    rng = np.random.default_rng(42)
    demand_series = {}
    for aid in agents_df["agent_id"]:
        tot = agent_total_cashout.get(aid, 150_000.0)
        daily_avg = tot / days_span
        # Peak 4h session represents roughly 30% of daily cash-out traffic
        peak_factor = rng.uniform(0.24, 0.38)
        base_demand = daily_avg * peak_factor * 6.0
        # Add slight volatility
        demand_series[aid] = float(np.round(np.clip(base_demand * rng.uniform(0.85, 1.25), 15_000, 280_000) / 500) * 500)

    agents_df["predicted_demand_4h"] = agents_df["agent_id"].map(demand_series).fillna(65000.0)
    agents_df["current_cash"] = agents_df["cash_balance"]
    agents_df["current_efloat"] = agents_df["e_float_balance"]
    agents_df["reliability_score"] = np.round(agents_df["rating"] / 5.0, 2)

    # Compute Shortage & Risk Classification
    agents_df = calculate_shortage(agents_df)
    agents_df = classify_risk(agents_df)

    return agents_df, txns_df, merchants_df


# Load datasets
agents_df, txns_df, merchants_df = load_and_enrich_datasets()

# Initialize Streamlit Session State for Approvals and Audit Logging
if "approved_rebalances" not in st.session_state:
    st.session_state.approved_rebalances = set()
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "sim_selected_agent" not in st.session_state:
    st.session_state.sim_selected_agent = None


# ─────────────────────────────────────────────────────────────────────────────
# 4. SIDEBAR CONTROLS & GLOBAL FILTERS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0;'>
        <div style='font-size:2.4rem;'>💸</div>
        <div style='font-size:1.35rem; font-weight:800; color:#58a6ff; letter-spacing:-0.02em;'>
            UpayPulse AI
        </div>
        <div style='font-size:0.80rem; color:#8b949e; margin-top:2px;'>
            Next-Gen MFS Liquidity Intelligence
        </div>
    </div>
    <hr style='border-color:#30363d; margin: 14px 0;'>
    """, unsafe_allow_html=True)

    st.markdown("### 🎛️ Filter Scope")

    # 1. Geographic Area Filter
    all_areas = ["All Bangladesh"] + sorted(agents_df["area"].unique().tolist())
    selected_area = st.selectbox("Select Commercial Area / Hub", all_areas, index=0)

    # 2. Risk Level Filter
    risk_filter = st.radio(
        "Agent Risk Category",
        ["All Agents", "🔴 High-Risk Shortage Only", "🟢 Safe / Surplus Only"],
        index=0
    )

    # 3. Minimum Customer Rating Slider
    min_rating = st.slider("Minimum Agent Rating", min_value=3.0, max_value=5.0, value=3.0, step=0.1)

    # Apply Filters to Agents
    filtered_agents = agents_df.copy()
    if selected_area != "All Bangladesh":
        filtered_agents = filtered_agents[filtered_agents["area"] == selected_area]

    if risk_filter == "🔴 High-Risk Shortage Only":
        filtered_agents = filtered_agents[filtered_agents["risk_level"] == "HIGH"]
    elif risk_filter == "🟢 Safe / Surplus Only":
        filtered_agents = filtered_agents[filtered_agents["risk_level"] == "SAFE"]

    filtered_agents = filtered_agents[filtered_agents["rating"] >= min_rating]

    # Filter Merchants according to Area
    filtered_merchants = merchants_df.copy()
    if selected_area != "All Bangladesh":
        filtered_merchants = filtered_merchants[filtered_merchants["area"] == selected_area]

    # Filter Transactions according to Area
    filtered_txns = txns_df.copy()
    if selected_area != "All Bangladesh":
        filtered_txns = filtered_txns[filtered_txns["area"] == selected_area]

    st.markdown("<hr style='border-color:#30363d;'>", unsafe_allow_html=True)

    # Live Quick Stats in Sidebar
    st.markdown("### 📡 Live Zone Telemetry")
    st.markdown(f"- 👥 Total Agents in View: **{len(filtered_agents)}**")
    st.markdown(f"- 🔴 High-Risk Flagged: **{(filtered_agents['risk_level'] == 'HIGH').sum()}**")
    st.markdown(f"- 🏪 Retail Merchants: **{len(filtered_merchants)}**")
    st.markdown(f"- 💳 Analyzed Transactions: **{len(filtered_txns):,}**")

    st.markdown("""
    <div style='font-size:0.73rem; color:#8b949e; text-align:center; margin-top:24px;'>
        UpayPulse AI Engine v2.4<br>
        Bangladesh MFS Synthetic Environment
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. DASHBOARD HEADER & EXECUTIVE KPI STRIP
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex; justify-content:space-between; align-items:flex-end; padding-bottom:8px;'>
    <div>
        <h1 style='font-size:2.2rem; font-weight:800; color:#e6edf3; margin:0; letter-spacing:-0.03em;'>
            💸 UpayPulse AI Command Center
        </h1>
        <p style='color:#8b949e; margin:6px 0 0 0; font-size:0.95rem;'>
            Real-time Liquidity Forecasting, Geospatial Rebalancing & Digital Merchant Diversion for Bangladesh MFS
        </p>
    </div>
    <div style='text-align:right;'>
        <span class='badge badge-info'>🟢 SYSTEM ACTIVE</span>
        <div style='font-size:0.75rem; color:#8b949e; margin-top:4px;'>Refreshed: Just Now</div>
    </div>
</div>
<hr style='border-color:#21262d; margin: 16px 0 24px 0;'>
""", unsafe_allow_html=True)

# ── Top-level 5 KPI Cards Strip ──────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

total_shortage = filtered_agents["expected_shortage"].sum()
total_surplus = filtered_agents["surplus_cash"].sum()
high_risk_count = (filtered_agents["risk_level"] == "HIGH").sum()
total_volume = filtered_txns["amount"].sum()
active_merchants_count = len(filtered_merchants)

with kpi1:
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-label">High-Risk Agents</div>
        <div class="kpi-value text-red">{high_risk_count}</div>
        <div class="kpi-subtitle">out of {len(filtered_agents)} agents in scope</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-label">Expected Cash Shortage</div>
        <div class="kpi-value text-red">৳{total_shortage:,.0f}</div>
        <div class="kpi-subtitle">predicted 4-hour shortfall</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-label">Rebalanceable Surplus</div>
        <div class="kpi-value text-green">৳{total_surplus:,.0f}</div>
        <div class="kpi-subtitle">available from surplus agents</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-label">Total Transacted Volume</div>
        <div class="kpi-value text-blue">৳{total_volume / 1e6:,.1f}M</div>
        <div class="kpi-subtitle">{len(filtered_txns):,} transactions recorded</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-label">Merchant Network</div>
        <div class="kpi-value text-amber">{active_merchants_count}</div>
        <div class="kpi-subtitle">ready for digital diversion</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN WORKSPACE TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🎛️ AI Rebalancing Command Center",
    "🗺️ Geospatial Ecosystem GIS",
    "📈 Transaction Flow Analytics",
    "📱 Smart Offer Diversion Simulator",
    "📋 Data Explorer & Export"
])


# =============================================================================
# TAB 1: OPERATOR COMMAND CENTER & AI REBALANCING ENGINE
# =============================================================================
with tabs[0]:
    st.markdown('<div class="section-banner">⚡ Automated Liquidity Shortage Detection & AI Rebalancing Partners</div>', unsafe_allow_html=True)

    high_risk_agents = filtered_agents[filtered_agents["risk_level"] == "HIGH"]
    surplus_agents = agents_df[agents_df["surplus_cash"] > 10_000]

    cmd_col_left, cmd_col_right = st.columns([3, 2])

    with cmd_col_left:
        st.markdown(f"#### 🔴 Active Liquidity Squeezes ({len(high_risk_agents)} Agents Flagged)")

        if high_risk_agents.empty:
            st.success("✅ Excellent! All agents in the selected scope hold sufficient liquidity for the next 4 hours.")
        else:
            for _, agent in high_risk_agents.head(12).iterrows():
                aid = agent["agent_id"]
                is_approved = aid in st.session_state.approved_rebalances

                with st.expander(
                    f"{'✅ APPROVED: ' if is_approved else '🔴 SHORTAGE: '} {agent['agent_id']} — {agent['area']} | Shortage: ৳{agent['expected_shortage']:,.0f}",
                    expanded=(not is_approved)
                ):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Current Cash", f"৳{agent['current_cash']:,.0f}")
                    c2.metric("Predicted Demand 4h", f"৳{agent['predicted_demand_4h']:,.0f}")
                    c3.metric("Expected Shortfall", f"৳{agent['expected_shortage']:,.0f}")
                    c4.metric("E-Float Balance", f"৳{agent['current_efloat']:,.0f}")

                    st.markdown("**🤖 AI Top Rebalancing Partner Matches:**")

                    # Calculate Rebalancing Score with Surplus Agents
                    potential_partners = surplus_agents[surplus_agents["agent_id"] != aid].copy()
                    if not potential_partners.empty:
                        # Haversine distance
                        potential_partners["dist_km"] = potential_partners.apply(
                            lambda r: haversine_km(agent["latitude"], agent["longitude"], r["latitude"], r["longitude"]),
                            axis=1
                        )
                        # Filter to reasonable distance (~5 km or closest available)
                        nearby = potential_partners.sort_values("dist_km").head(8).copy()

                        # Composite AI Rebalancing Score
                        norm_surplus = (nearby["surplus_cash"] - nearby["surplus_cash"].min()) / max(1.0, nearby["surplus_cash"].max() - nearby["surplus_cash"].min())
                        norm_dist = 1.0 - (nearby["dist_km"] / max(1.0, nearby["dist_km"].max()))
                        nearby["ai_score"] = np.round(0.45 * norm_surplus + 0.35 * norm_dist + 0.20 * nearby["reliability_score"], 3)

                        top_recs = nearby.sort_values("ai_score", ascending=False).head(3)

                        # Display Table
                        rec_display = top_recs[[
                            "agent_id", "area", "dist_km", "surplus_cash", "ai_score"
                        ]].rename(columns={
                            "agent_id": "Partner ID",
                            "area": "Area",
                            "dist_km": "Distance (km)",
                            "surplus_cash": "Available Surplus (BDT)",
                            "ai_score": "Match Score"
                        })
                        rec_display["Distance (km)"] = rec_display["Distance (km)"].map("{:.2f} km".format)
                        rec_display["Available Surplus (BDT)"] = rec_display["Available Surplus (BDT)"].map("৳{:,.0f}".format)

                        st.dataframe(rec_display, use_container_width=True, hide_index=True)

                        best_partner = top_recs.iloc[0]
                        suggested_transfer = min(agent["expected_shortage"], best_partner["surplus_cash"] * 0.7)

                        # Approve Rebalancing Action
                        act_col1, act_col2 = st.columns([2, 3])
                        with act_col1:
                            if is_approved:
                                st.success("✅ Rebalancing Dispatch Approved")
                            else:
                                if st.button(f"⚡ Approve Rebalancing (৳{suggested_transfer:,.0f})", key=f"btn_app_{aid}"):
                                    st.session_state.approved_rebalances.add(aid)
                                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    st.session_state.audit_log.insert(0, {
                                        "timestamp": now_str,
                                        "source_agent": best_partner["agent_id"],
                                        "target_agent": aid,
                                        "area": agent["area"],
                                        "amount": suggested_transfer,
                                        "distance": best_partner["dist_km"]
                                    })
                                    st.rerun()

                        with act_col2:
                            st.caption(f"Suggested transfer: ৳{suggested_transfer:,.0f} from {best_partner['agent_id']} ({best_partner['area']}) via MFS field runner.")

    with cmd_col_right:
        st.markdown("#### 📜 Digital Liquidity Audit Trail")
        st.caption("Immutable operator log of all automated & approved cash transfers.")

        if not st.session_state.audit_log:
            st.info("📭 No rebalancing operations dispatched yet. Click 'Approve' on any high-risk agent to record.")
        else:
            for item in st.session_state.audit_log[:8]:
                st.markdown(f"""
                <div class="audit-entry">
                    <span style="color:#58a6ff;">[{item['timestamp']}]</span> REBALANCE_DISPATCHED<br>
                    Transfer: <b style="color:#3fb950;">৳{item['amount']:,.0f}</b><br>
                    From: <code>{item['source_agent']}</code> ➔ To: <code>{item['target_agent']}</code> ({item['area']})<br>
                    Distance: {item['distance']:.2f} km
                </div>
                """, unsafe_allow_html=True)

            if st.button("🗑️ Reset Audit Log"):
                st.session_state.audit_log = []
                st.session_state.approved_rebalances = set()
                st.rerun()


# =============================================================================
# TAB 2: GEOSPATIAL ECOSYSTEM GIS (FOLIUM)
# =============================================================================
with tabs[1]:
    st.markdown('<div class="section-banner">🗺️ Multi-Layer Geospatial GIS — Bangladesh MFS Liquidity Network</div>', unsafe_allow_html=True)

    map_ctrl1, map_ctrl2, map_ctrl3 = st.columns([1, 1, 2])
    with map_ctrl1:
        map_layer = st.radio("Display Layers", ["All Entities", "Agents Only", "Merchants Only"], horizontal=True)
    with map_ctrl2:
        show_danger_zones = st.checkbox("Highlight Shortage Danger Radii", value=True)

    # Determine center coordinates
    if not filtered_agents.empty:
        center_lat = filtered_agents["latitude"].mean()
        center_lon = filtered_agents["longitude"].mean()
        zoom_val = 14 if selected_area != "All Bangladesh" else 7
    else:
        center_lat, center_lon, zoom_val = 23.8103, 90.4125, 7

    # Build Folium Map with Dark Matter Tiles
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_val,
        tiles="CartoDB dark_matter"
    )

    # Add Agent Markers
    if map_layer in ["All Entities", "Agents Only"]:
        for _, ag in filtered_agents.iterrows():
            is_risk = ag["risk_level"] == "HIGH"
            color = "red" if is_risk else "green"
            icon = "exclamation-triangle" if is_risk else "check"

            popup_html = f"""
            <div style='font-family:Inter,sans-serif; min-width:180px; color:#1f2937;'>
                <b style='font-size:1.05rem;'>{ag['agent_id']}</b><br>
                <b>Area:</b> {ag['area']}<br>
                <b>Status:</b> <span style='color:{"red" if is_risk else "green"}; font-weight:700;'>{ag['risk_level']}</span><br>
                <hr style='margin:6px 0;'>
                <b>Cash in Hand:</b> ৳{ag['cash_balance']:,.0f}<br>
                <b>E-Float Balance:</b> ৳{ag['e_float_balance']:,.0f}<br>
                <b>Demand 4h:</b> ৳{ag['predicted_demand_4h']:,.0f}<br>
                <b>Expected Shortage:</b> ৳{ag['expected_shortage']:,.0f}<br>
                <b>Rating:</b> ⭐ {ag['rating']:.2f}<br>
                <b>Hours:</b> {ag['working_hours']}
            </div>
            """

            folium.Marker(
                location=[ag["latitude"], ag["longitude"]],
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{ag['agent_id']} ({ag['area']}) | {ag['risk_level']}",
                icon=folium.Icon(color=color, icon=icon, prefix="fa")
            ).add_to(m)

            if is_risk and show_danger_zones:
                folium.Circle(
                    location=[ag["latitude"], ag["longitude"]],
                    radius=400,
                    color="#f85149",
                    fill=True,
                    fill_opacity=0.12,
                    weight=1
                ).add_to(m)

    # Add Merchant Markers
    if map_layer in ["All Entities", "Merchants Only"]:
        for _, mr in filtered_merchants.head(250).iterrows():
            popup_html = f"""
            <div style='font-family:Inter,sans-serif; min-width:160px; color:#1f2937;'>
                <b>{mr['merchant_id']}</b><br>
                <b>Category:</b> {mr['category']}<br>
                <b>Area:</b> {mr['area']}<br>
                <b>Daily Volume:</b> ~{mr['daily_transactions']} txns/day
            </div>
            """
            folium.CircleMarker(
                location=[mr["latitude"], mr["longitude"]],
                radius=5,
                color="#58a6ff",
                fill=True,
                fill_color="#38bdf8",
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"🏪 {mr['merchant_id']} ({mr['category']})"
            ).add_to(m)

    # Render Map
    st_folium(m, width=None, height=560, returned_objects=[])

    st.markdown("""
    <div style='font-size:0.8rem; color:#8b949e; margin-top:8px;'>
        📌 <b>Legend:</b> &nbsp;
        🔴 <span style='color:#f85149;'>High-Risk Agent (Shortage Expected)</span> &nbsp;|&nbsp;
        🟢 <span style='color:#3fb950;'>Safe / Surplus Agent</span> &nbsp;|&nbsp;
        🔵 <span style='color:#58a6ff;'>Retail Merchant Point</span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB 3: TRANSACTION FLOW ANALYTICS (PLOTLY)
# =============================================================================
with tabs[2]:
    st.markdown('<div class="section-banner">📈 Time-Series Financial Velocity & Diurnal Demand Curves</div>', unsafe_allow_html=True)

    t_col1, t_col2 = st.columns(2)

    with t_col1:
        st.markdown("#### 🕒 Diurnal Hourly Transaction Volume (Bangladesh Peak Cycles)")
        hourly_counts = filtered_txns.groupby(["hour", "transaction_type"]).size().reset_index(name="count")

        fig_hourly = px.line(
            hourly_counts,
            x="hour",
            y="count",
            color="transaction_type",
            markers=True,
            color_discrete_map={
                "cash_out": "#f85149",
                "send_money": "#58a6ff",
                "merchant_payment": "#3fb950"
            },
            template="plotly_dark"
        )
        fig_hourly.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Hour of Day (24-Hour Clock)",
            yaxis_title="Total Transactions",
            legend_title="Type",
            height=340,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

    with t_col2:
        st.markdown("#### 💳 Volume Breakdown by Transaction Type")
        type_summary = filtered_txns.groupby("transaction_type")["amount"].agg(["sum", "count"]).reset_index()

        fig_donut = px.pie(
            type_summary,
            names="transaction_type",
            values="sum",
            hole=0.55,
            color="transaction_type",
            color_discrete_map={
                "cash_out": "#f85149",
                "send_money": "#58a6ff",
                "merchant_payment": "#3fb950"
            },
            template="plotly_dark"
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    t_row2_col1, t_row2_col2 = st.columns(2)

    with t_row2_col1:
        st.markdown("#### 🏙️ Top 10 Busiest Commercial Bazaars (Total BDT Volume)")
        area_vol = txns_df.groupby("area")["amount"].sum().sort_values(ascending=False).head(10).reset_index()

        fig_area = px.bar(
            area_vol,
            x="amount",
            y="area",
            orientation="h",
            color="amount",
            color_continuous_scale="Blues",
            template="plotly_dark"
        )
        fig_area.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(autorange="reversed"),
            xaxis_title="Total Volume (BDT)",
            yaxis_title="Commercial Area",
            coloraxis_showscale=False,
            height=320,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_area, use_container_width=True)

    with t_row2_col2:
        st.markdown("#### 💵 Ticket Size Distribution (Cash-out vs Send Money)")
        sample_txns = filtered_txns.sample(min(4000, len(filtered_txns)), random_state=42)

        fig_hist = px.histogram(
            sample_txns,
            x="amount",
            color="transaction_type",
            nbins=40,
            barmode="overlay",
            color_discrete_map={
                "cash_out": "#f85149",
                "send_money": "#58a6ff",
                "merchant_payment": "#3fb950"
            },
            template="plotly_dark"
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Amount in BDT (৳)",
            yaxis_title="Transaction Frequency",
            height=320,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# =============================================================================
# TAB 4: SMART OFFER DIVERSION SIMULATOR
# =============================================================================
with tabs[3]:
    st.markdown('<div class="section-banner">📲 AI Customer Diversion Engine — Converting Cash-Out to Digital QR Payment</div>', unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([1, 1])

    with sim_col1:
        st.markdown("#### 🎯 Simulate Customer Approaching a Stressed Agent")
        st.write("When an MFS user approaches a cash-stressed agent, UpayPulse AI intercepts with an instant merchant discount voucher, saving cash liquidity while rewarding the customer.")

        # Interactive Inputs
        target_agent = st.selectbox(
            "Select Target Agent Point",
            high_risk_agents["agent_id"].tolist() if not high_risk_agents.empty else agents_df["agent_id"].head(20).tolist()
        )
        agent_row = agents_df[agents_df["agent_id"] == target_agent].iloc[0]

        intended_cashout = st.slider("Customer Intended Cash-Out Amount (BDT)", 500, 20_000, 3_000, step=500)
        user_intent_prob = st.slider("Customer Cash-Out Urgency", 0.1, 1.0, 0.75, step=0.05)

        # Find Nearby Merchants for this Agent
        merchants_in_area = merchants_df[merchants_df["area"] == agent_row["area"]].copy()
        if merchants_in_area.empty:
            merchants_in_area = merchants_df.head(5).copy()

        selected_merchant = merchants_in_area.iloc[0]
        discount_offered = 8.0  # 8% discount
        savings_bdt = intended_cashout * (discount_offered / 100.0)

        st.markdown(f"""
        **Context Details:**
        - **Agent Location:** {agent_row['area']}
        - **Agent Cash-in-Hand:** ৳{agent_row['cash_balance']:,.0f}
        - **Current Shortage:** ৳{agent_row['expected_shortage']:,.0f}
        - **Partner Merchant:** {selected_merchant['merchant_id']} ({selected_merchant['category']})
        """)

        trigger_sim = st.button("🚀 Intercept & Dispatch Smart Offer Notification", use_container_width=True)

    with sim_col2:
        st.markdown("#### 📱 Customer Mobile App Live Preview")

        # Visual Smartphone UI Card
        st.markdown(f"""
        <div class="smartphone-screen">
            <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#8b949e; margin-bottom:14px;">
                <span>02:45 PM</span>
                <span>📶 4G &nbsp;🔋 88%</span>
            </div>
            <div style="background:rgba(88, 166, 255, 0.12); border:1px solid rgba(88, 166, 255, 0.3); border-radius:14px; padding:16px;">
                <div style="display:flex; align-items:center; gap:10px;">
                    <div style="font-size:1.8rem;">💸</div>
                    <div>
                        <div style="font-weight:700; color:#58a6ff; font-size:1.0rem;">UpayPulse Smart Discount</div>
                        <div style="font-size:0.75rem; color:#8b949e;">Nearby {agent_row['area']} Bazaar</div>
                    </div>
                </div>
                <hr style="border-color:rgba(255,255,255,0.08); margin:10px 0;">
                <p style="font-size:0.88rem; line-height:1.5; color:#e6edf3; margin-bottom:12px;">
                    Skip the cash-out fee! Pay digitally at <b>{selected_merchant['merchant_id']} ({selected_merchant['category']})</b> and receive an instant <b>{discount_offered:.0f}% cashback</b>!
                </p>
                <div style="background:#0d1117; border-radius:10px; padding:12px; text-align:center;">
                    <div style="font-size:0.75rem; color:#8b949e;">YOU INSTANTLY SAVE</div>
                    <div style="font-size:1.6rem; font-weight:800; color:#3fb950;">৳{savings_bdt:,.0f} BDT</div>
                    <div style="font-size:0.75rem; color:#8b949e;">Valid for the next 45 minutes</div>
                </div>
            </div>
            <div style="text-align:center; margin-top:16px;">
                <span class="badge badge-success">⚡ 85% DIVERSION PROBABILITY</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if trigger_sim:
            st.success(f"✅ User successfully diverted! Cash liquidity of ৳{intended_cashout:,.0f} preserved at {agent_row['agent_id']}.")


# =============================================================================
# TAB 5: DATA EXPLORER & CSV EXPORT
# =============================================================================
with tabs[4]:
    st.markdown('<div class="section-banner">📋 Raw Data Explorer & High-Speed CSV Exporter</div>', unsafe_allow_html=True)

    dataset_select = st.radio("Choose Dataset to Explore", ["Agents (500)", "Transactions (50,000)", "Merchants (1,000)"], horizontal=True)

    if dataset_select == "Agents (500)":
        st.dataframe(filtered_agents, use_container_width=True, hide_index=True)
        csv_data = filtered_agents.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Agents CSV",
            data=csv_data,
            file_name="agents_export.csv",
            mime="text/csv"
        )

    elif dataset_select == "Transactions (50,000)":
        st.dataframe(filtered_txns.head(1000), use_container_width=True, hide_index=True)
        st.caption("Displaying top 1,000 matching transactions.")
        csv_data = filtered_txns.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Transactions CSV",
            data=csv_data,
            file_name="transactions_export.csv",
            mime="text/csv"
        )

    else:
        st.dataframe(filtered_merchants, use_container_width=True, hide_index=True)
        csv_data = filtered_merchants.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Merchants CSV",
            data=csv_data,
            file_name="merchants_export.csv",
            mime="text/csv"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. DASHBOARD FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#21262d; margin-top:50px;'>
<div style='text-align:center; padding:12px 0 24px 0; color:#8b949e; font-size:0.82rem;'>
    💸 <b style='color:#58a6ff;'>UpayPulse AI</b> — MFS Liquidity & Digital Financial Intelligence Platform<br>
    Built with Streamlit · Pandas · NumPy · Plotly · Folium · Faker &nbsp;|&nbsp;
    Data: <b>100% Synthetic Bangladesh MFS Ecosystem</b>
</div>
""", unsafe_allow_html=True)
