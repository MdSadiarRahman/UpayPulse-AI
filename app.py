"""
app.py
------
UpayPulse AI — Streamlit Dashboard
MFS Ecosystem Intelligence Tool | Dinajpur Sadar Demo

Sections
--------
1. Operator Command Center   : High-risk agents, shortage metrics, approve rebalancing
2. Interactive Ecosystem Map : Folium map with agent & merchant markers
3. Customer & Merchant Simulator : User entering zone + Smart Offer notification
"""

import datetime
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

from data_generator import load_all
from ml_engine import run_pipeline

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UpayPulse AI | MFS Intelligence",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Premium Dark Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ──────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Global dark background ───────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    color: #e6edf3;
}

/* ── Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}

/* ── KPI Metric Cards ─────────────────────────────────────────── */
.kpi-card {
    background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    margin: 4px 0;
}
.kpi-label {
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8b949e;
}
.kpi-red   { color: #f85149; }
.kpi-green { color: #3fb950; }
.kpi-blue  { color: #58a6ff; }
.kpi-amber { color: #d29922; }

/* ── Section Headers ──────────────────────────────────────────── */
.section-header {
    background: linear-gradient(90deg, #21262d 0%, #161b22 100%);
    border-left: 4px solid #58a6ff;
    border-radius: 0 8px 8px 0;
    padding: 14px 20px;
    margin: 24px 0 16px 0;
    font-size: 1.05rem;
    font-weight: 600;
    color: #e6edf3;
    letter-spacing: 0.02em;
}

/* ── Risk Badge ───────────────────────────────────────────────── */
.badge-high {
    background: rgba(248, 81, 73, 0.15);
    color: #f85149;
    border: 1px solid #f85149;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-safe {
    background: rgba(63, 185, 80, 0.15);
    color: #3fb950;
    border: 1px solid #3fb950;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* ── Agent Card ───────────────────────────────────────────────── */
.agent-card {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.agent-card:hover { border-color: #58a6ff; }
.agent-card-high  { border-left: 4px solid #f85149; }
.agent-card-safe  { border-left: 4px solid #3fb950; }

/* ── Smart Offer Notification ─────────────────────────────────── */
.smart-offer {
    background: linear-gradient(135deg, #0d2137 0%, #0d1f33 100%);
    border: 1px solid #1d4ed8;
    border-radius: 16px;
    padding: 24px;
    margin-top: 16px;
    box-shadow: 0 0 30px rgba(88, 166, 255, 0.12);
}

/* ── Audit Log ────────────────────────────────────────────────── */
.audit-entry {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    color: #8b949e;
}
.audit-timestamp { color: #3fb950; font-weight: 600; }
.audit-action    { color: #58a6ff; }

/* ── Streamlit overrides ──────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px;
}
.stButton > button {
    background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 22px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
}
.stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────────────────────────────────────
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "approved_agents" not in st.session_state:
    st.session_state.approved_agents = set()
if "sim_user_id" not in st.session_state:
    st.session_state.sim_user_id = None


# ─────────────────────────────────────────────────────────────────────────────
# Data & Pipeline
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_pipeline_data():
    agents, merchants, users = load_all()
    enriched, recs, div = run_pipeline(agents, merchants, users)
    return enriched, merchants, users, recs, div

agents_df, merchants_df, users_df, recommendations, diversion = get_pipeline_data()

high_risk_df = agents_df[agents_df["risk_level"] == "HIGH"]
safe_df      = agents_df[agents_df["risk_level"] == "SAFE"]


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-size:2.5rem;'>💸</div>
        <div style='font-size:1.3rem; font-weight:700; color:#58a6ff; letter-spacing:0.03em;'>
            UpayPulse AI
        </div>
        <div style='font-size:0.78rem; color:#8b949e; margin-top:4px;'>
            MFS Ecosystem Intelligence
        </div>
    </div>
    <hr style='border-color:#30363d; margin: 12px 0;'>
    """, unsafe_allow_html=True)

    st.markdown("**📍 Zone**")
    st.info("Dinajpur Sadar, Dinajpur, Bangladesh")

    st.markdown("**🕐 Prediction Window**")
    st.info("Next 4 Hours")

    st.markdown("**📊 Live Summary**")
    st.markdown(f"- 🔴 High-Risk Agents: **{len(high_risk_df)}**")
    st.markdown(f"- 🟢 Safe Agents: **{len(safe_df)}**")
    st.markdown(f"- 🏪 Active Merchant Offers: **{merchants_df['active_offer'].sum()}**")
    st.markdown(f"- 👤 Users in Zone: **{len(users_df)}**")

    st.markdown("<hr style='border-color:#30363d;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.72rem; color:#8b949e; text-align:center;'>"
        "MFS AI Hackathon Prototype v1.0<br>"
        "All data is synthetic — no PII used</div>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 10px 0 4px 0;'>
    <h1 style='font-size:2rem; font-weight:700; color:#e6edf3; margin:0;'>
        💸 UpayPulse AI Dashboard
    </h1>
    <p style='color:#8b949e; margin:6px 0 0 2px; font-size:0.92rem;'>
        Real-time agent liquidity intelligence for Dinajpur Sadar MFS zone
    </p>
</div>
<hr style='border-color:#21262d; margin-bottom: 24px;'>
""", unsafe_allow_html=True)

# ── Top-level KPI Strip ────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
total_shortage  = agents_df["expected_shortage"].sum()
total_surplus   = agents_df["surplus_cash"].sum()
diverted_users  = diversion["diverted_users"]
diversion_rate  = diversion["diversion_rate_pct"]
cash_saved      = diversion["total_cash_saved_bdt"]

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">High-Risk Agents</div>
        <div class="kpi-value kpi-red">{len(high_risk_df)}</div>
        <div class="kpi-label">of {len(agents_df)} total</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Cash Shortage</div>
        <div class="kpi-value kpi-red">৳{total_shortage:,.0f}</div>
        <div class="kpi-label">BDT at risk</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Surplus Available</div>
        <div class="kpi-value kpi-green">৳{total_surplus:,.0f}</div>
        <div class="kpi-label">BDT rebalanceable</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Users Diverted</div>
        <div class="kpi-value kpi-blue">{diverted_users}</div>
        <div class="kpi-label">{diversion_rate}% diversion rate</div>
    </div>""", unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Cash-out Prevented</div>
        <div class="kpi-value kpi-amber">৳{cash_saved:,.0f}</div>
        <div class="kpi-label">BDT stays digital</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 : OPERATOR COMMAND CENTER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🎛️ Section 1 — Operator Command Center</div>', unsafe_allow_html=True)

tab_high, tab_all, tab_log = st.tabs(["🔴 High-Risk Agents", "📋 All Agents", "📜 Audit Log"])

# ── Tab: High-Risk Agents ──────────────────────────────────────────────────
with tab_high:
    if high_risk_df.empty:
        st.success("✅ No high-risk agents detected at this time.")
    else:
        st.markdown(f"**{len(high_risk_df)} agent(s) require immediate attention.**")
        for _, agent in high_risk_df.iterrows():
            aid    = agent["agent_id"]
            recs   = recommendations.get(aid, [])
            already_approved = aid in st.session_state.approved_agents

            with st.expander(
                f"🔴 {agent['name']}  ({aid})  |  Shortage: ৳{agent['expected_shortage']:,.0f}  |  Reliability: {agent['reliability_score']}",
                expanded=True
            ):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Current Cash",        f"৳{agent['current_cash']:,.0f}")
                m2.metric("Predicted Demand 4h", f"৳{agent['predicted_demand_4h']:,.0f}")
                m3.metric("Expected Shortage",   f"৳{agent['expected_shortage']:,.0f}", delta_color="inverse")
                m4.metric("Reliability Score",   f"{agent['reliability_score']:.2f}")

                st.markdown("**🔀 AI-Recommended Rebalancing Partners:**")
                if recs:
                    rec_df = pd.DataFrame(recs)[
                        ["partner_id", "partner_name", "distance_km",
                         "score", "suggested_transfer"]
                    ].rename(columns={
                        "partner_id"        : "Partner ID",
                        "partner_name"      : "Agent Name",
                        "distance_km"       : "Distance (km)",
                        "score"             : "AI Score",
                        "suggested_transfer": "Suggested Transfer (BDT)",
                    })
                    rec_df["AI Score"] = rec_df["AI Score"].map("{:.4f}".format)
                    rec_df["Suggested Transfer (BDT)"] = rec_df["Suggested Transfer (BDT)"].map("৳{:,.0f}".format)
                    st.dataframe(rec_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("No surplus agents nearby for rebalancing.")

                # ── Approve Button ─────────────────────────────────────
                if already_approved:
                    st.success(f"✅ Rebalance for {aid} has been approved and logged.")
                else:
                    btn_col, _ = st.columns([1, 3])
                    with btn_col:
                        if st.button(f"✅ Approve Rebalance for {aid}", key=f"approve_{aid}"):
                            st.session_state.approved_agents.add(aid)
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            log_entry = {
                                "timestamp" : timestamp,
                                "action"    : f"REBALANCE_APPROVED",
                                "agent_id"  : aid,
                                "agent_name": agent["name"],
                                "shortage"  : agent["expected_shortage"],
                                "partners"  : [r["partner_name"] for r in recs],
                            }
                            st.session_state.audit_log.insert(0, log_entry)
                            st.success(f"✅ Approval logged at {timestamp}!")
                            st.rerun()

# ── Tab: All Agents ────────────────────────────────────────────────────────
with tab_all:
    display_cols = ["agent_id", "name", "current_cash", "current_efloat",
                    "predicted_demand_4h", "expected_shortage",
                    "surplus_cash", "reliability_score", "risk_level"]

    styled_df = agents_df[display_cols].copy()
    styled_df["current_cash"]        = styled_df["current_cash"].map("৳{:,.0f}".format)
    styled_df["current_efloat"]      = styled_df["current_efloat"].map("৳{:,.0f}".format)
    styled_df["predicted_demand_4h"] = styled_df["predicted_demand_4h"].map("৳{:,.0f}".format)
    styled_df["expected_shortage"]   = styled_df["expected_shortage"].map("৳{:,.0f}".format)
    styled_df["surplus_cash"]        = styled_df["surplus_cash"].map("৳{:,.0f}".format)

    def color_risk(val):
        if val == "HIGH":
            return "background-color: rgba(248,81,73,0.15); color: #f85149; font-weight: bold"
        return "background-color: rgba(63,185,80,0.12); color: #3fb950; font-weight: bold"

    st.dataframe(
        styled_df.style.applymap(color_risk, subset=["risk_level"]),
        use_container_width=True,
        hide_index=True
    )

# ── Tab: Audit Log ─────────────────────────────────────────────────────────
with tab_log:
    st.markdown("**Digital Audit Trail** — All operator actions are immutably logged.")
    if not st.session_state.audit_log:
        st.info("📭 No actions logged yet. Approve a rebalancing recommendation to create an entry.")
    else:
        for entry in st.session_state.audit_log:
            partners_str = ", ".join(entry["partners"]) if entry["partners"] else "None"
            st.markdown(f"""
            <div class="audit-entry">
                <span class="audit-timestamp">[{entry['timestamp']}]</span>
                &nbsp;|&nbsp;
                <span class="audit-action">{entry['action']}</span>
                &nbsp;|&nbsp;
                Agent: <b>{entry['agent_name']}</b> ({entry['agent_id']})
                &nbsp;|&nbsp;
                Shortage: ৳{entry['shortage']:,.0f}
                &nbsp;|&nbsp;
                Partners: {partners_str}
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.audit_log:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Audit Log (Demo Reset)"):
            st.session_state.audit_log = []
            st.session_state.approved_agents = set()
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 : INTERACTIVE ECOSYSTEM MAP
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🗺️ Section 2 — Interactive Ecosystem Map</div>', unsafe_allow_html=True)

# Map controls
map_col, legend_col = st.columns([3, 1])

with legend_col:
    st.markdown("**Map Legend**")
    st.markdown("🔴 &nbsp;HIGH-Risk Agent")
    st.markdown("🟢 &nbsp;SAFE / Surplus Agent")
    st.markdown("🔵 &nbsp;Merchant (offer active)")
    st.markdown("⚫ &nbsp;Merchant (no offer)")
    st.markdown("---")
    show_users = st.checkbox("Show User Density", value=False)
    map_zoom   = st.slider("Zoom Level", 13, 16, 14)

# Build Folium map
fmap = folium.Map(
    location=[25.6279, 88.6336],
    zoom_start=map_zoom,
    tiles="CartoDB dark_matter"
)

# ── Agent markers ──────────────────────────────────────────────────────────
for _, agent in agents_df.iterrows():
    is_high = agent["risk_level"] == "HIGH"
    color   = "red" if is_high else "green"
    icon    = "exclamation-triangle" if is_high else "check"

    popup_html = f"""
    <div style='font-family:Inter,sans-serif; min-width:200px;'>
        <b style='font-size:1rem;'>{agent['name']}</b><br>
        <span style='color:grey;'>{agent['agent_id']}</span><br><br>
        <b>Risk:</b> <span style='color:{"red" if is_high else "green"};'>{agent['risk_level']}</span><br>
        <b>Cash:</b> ৳{agent['current_cash']:,.0f}<br>
        <b>Demand 4h:</b> ৳{agent['predicted_demand_4h']:,.0f}<br>
        <b>Shortage:</b> ৳{agent['expected_shortage']:,.0f}<br>
        <b>Surplus:</b> ৳{agent['surplus_cash']:,.0f}<br>
        <b>Reliability:</b> {agent['reliability_score']}
    </div>
    """
    folium.Marker(
        location=[agent["latitude"], agent["longitude"]],
        popup=folium.Popup(popup_html, max_width=260),
        tooltip=f"{agent['name']} | {agent['risk_level']}",
        icon=folium.Icon(color=color, icon=icon, prefix="fa")
    ).add_to(fmap)

    # Danger radius circle for HIGH-risk
    if is_high:
        folium.Circle(
            location=[agent["latitude"], agent["longitude"]],
            radius=300,
            color="red",
            fill=True,
            fill_opacity=0.08,
        ).add_to(fmap)

# ── Merchant markers ───────────────────────────────────────────────────────
for _, merch in merchants_df.iterrows():
    has_offer = merch["active_offer"]
    color     = "blue" if has_offer else "gray"

    popup_html = f"""
    <div style='font-family:Inter,sans-serif; min-width:180px;'>
        <b>{merch['name']}</b><br>
        <span style='color:grey;'>{merch['category']}</span><br><br>
        <b>Active Offer:</b> {"✅ Yes" if has_offer else "❌ No"}<br>
        <b>Discount:</b> {merch['discount_pct']}%
    </div>
    """
    folium.Marker(
        location=[merch["latitude"], merch["longitude"]],
        popup=folium.Popup(popup_html, max_width=220),
        tooltip=f"🏪 {merch['name']} | {merch['discount_pct']}% off",
        icon=folium.Icon(color=color, icon="shopping-bag", prefix="fa")
    ).add_to(fmap)

# ── Optional user scatter ──────────────────────────────────────────────────
if show_users:
    for _, user in users_df.iterrows():
        folium.CircleMarker(
            location=[user["latitude"], user["longitude"]],
            radius=4,
            color="#f97316",
            fill=True,
            fill_opacity=user["cashout_intent"],
            tooltip=f"User | Intent: {user['cashout_intent']:.2f}",
        ).add_to(fmap)

with map_col:
    st_folium(fmap, width=None, height=500, returned_objects=[])


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 : CUSTOMER & MERCHANT SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📱 Section 3 — Customer & Merchant Simulator</div>', unsafe_allow_html=True)

sim_left, sim_right = st.columns([1, 1])

with sim_left:
    st.markdown("#### 👤 Simulate a User Entering the Zone")
    st.markdown("Select a user from the zone to see the personalised Smart Offer they receive.")

    user_options = [f"{r['user_id']}  (Intent: {r['cashout_intent']:.2f}  |  Typical txn: ৳{r['typical_txn_bdt']:,.0f})"
                    for _, r in users_df.iterrows()]
    selected_user_label = st.selectbox("Select User", user_options, index=0)
    selected_uid = selected_user_label.split("  ")[0].strip()
    selected_user = users_df[users_df["user_id"] == selected_uid].iloc[0]

    st.markdown("---")
    st.markdown("**User Profile**")
    u1, u2, u3 = st.columns(3)
    u1.metric("User ID",       selected_user["user_id"])
    u2.metric("Cash-out Intent", f"{selected_user['cashout_intent']:.0%}")
    u3.metric("Typical Txn",   f"৳{selected_user['typical_txn_bdt']:,.0f}")

    trigger_btn = st.button("📲 Simulate Zone Entry & Generate Smart Offer", use_container_width=True)

with sim_right:
    st.markdown("#### 🏪 Active Merchant Offers in Zone")
    active_merchants = merchants_df[merchants_df["active_offer"] == True]

    if active_merchants.empty:
        st.info("No merchants currently running offers.")
    else:
        for _, m in active_merchants.iterrows():
            discount_color = "#3fb950" if m["discount_pct"] >= 8 else "#d29922"
            st.markdown(f"""
            <div class="agent-card agent-card-safe" style="margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <b>🏪 {m['name']}</b><br>
                        <span style="color:#8b949e; font-size:0.85rem;">{m['category']}</span>
                    </div>
                    <div style="text-align:right;">
                        <span style="color:{discount_color}; font-size:1.4rem; font-weight:700;">
                            {m['discount_pct']}% OFF
                        </span><br>
                        <span style="font-size:0.75rem; color:#8b949e;">digital payment</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Smart Offer Notification ──────────────────────────────────────────────
if trigger_btn or st.session_state.sim_user_id == selected_uid:
    st.session_state.sim_user_id = selected_uid
    intent    = selected_user["cashout_intent"]
    txn_amt   = selected_user["typical_txn_bdt"]
    best_merch = active_merchants.sort_values("discount_pct", ascending=False).iloc[0] \
                 if not active_merchants.empty else None

    if best_merch is not None:
        saving = txn_amt * best_merch["discount_pct"] / 100
        diversion_likely = intent >= 0.35
        offer_headline   = "💡 Cash-out Alternative Smart Offer" if diversion_likely else "🎁 Exclusive Digital Offer For You"

        st.markdown(f"""
        <div class="smart-offer">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
                <div style="font-size:2rem;">📲</div>
                <div>
                    <div style="font-size:1.1rem; font-weight:700; color:#58a6ff;">{offer_headline}</div>
                    <div style="font-size:0.82rem; color:#8b949e;">Sent to {selected_uid} · Just now</div>
                </div>
            </div>
            <div style="background:#0d2137; border-radius:10px; padding:16px; margin-bottom:14px;">
                <div style="font-size:0.9rem; color:#e6edf3; line-height:1.6;">
                    Hi there! 👋 You're near <b style="color:#58a6ff;">{best_merch['name']}</b>.<br>
                    Skip the cash-out queue and <b style="color:#3fb950;">pay digitally to save
                    ৳{saving:,.0f}</b> instantly on your ৳{txn_amt:,.0f} transaction!
                </div>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-bottom:14px;">
                <div style="text-align:center; background:#1c2128; border-radius:8px; padding:12px;">
                    <div style="color:#3fb950; font-size:1.3rem; font-weight:700;">{best_merch['discount_pct']}%</div>
                    <div style="color:#8b949e; font-size:0.75rem;">Instant Discount</div>
                </div>
                <div style="text-align:center; background:#1c2128; border-radius:8px; padding:12px;">
                    <div style="color:#f97316; font-size:1.3rem; font-weight:700;">৳{saving:,.0f}</div>
                    <div style="color:#8b949e; font-size:0.75rem;">You Save (BDT)</div>
                </div>
                <div style="text-align:center; background:#1c2128; border-radius:8px; padding:12px;">
                    <div style="color:#58a6ff; font-size:1.3rem; font-weight:700;">{best_merch['category']}</div>
                    <div style="color:#8b949e; font-size:0.75rem;">Merchant Type</div>
                </div>
            </div>
            <div style="font-size:0.8rem; color:#8b949e; border-top:1px solid #1d4ed8; padding-top:12px; margin-top:4px;">
                ✅ <b>Why this offer?</b> AI detected local cash shortage risk. Your digital payment helps
                the MFS ecosystem stay liquid while you save money. Win-win! 🎯
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Diversion outcome ──────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        oc1, oc2, oc3 = st.columns(3)
        oc1.metric("Diversion Probability",
                   f"{min(best_merch['discount_pct'] / 10, 0.85):.0%}",
                   help="Probability user accepts the offer")
        oc2.metric("Cash-out Demand Reduced",
                   f"৳{txn_amt:,.0f}",
                   help="This much cash demand removed from agent network")
        oc3.metric("Ecosystem Impact",
                   "Positive ✅",
                   help="Agent liquidity pressure reduced")
    else:
        st.warning("No active merchant offers available to generate a smart offer.")

# ── Zone-wide Diversion Summary ────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 📊 Zone-Wide Diversion Summary")
d1, d2, d3, d4 = st.columns(4)
d1.metric("Total Users in Zone",     diversion["total_users"])
d2.metric("High-Intent Cash-out",    diversion["at_risk_users"])
d3.metric("Estimated Diversions",    diversion["diverted_users"])
d4.metric("Total Digital Shift",     f"৳{diversion['total_cash_saved_bdt']:,.0f}")

# Active merchant list
if diversion["active_merchants"]:
    st.success(
        "🏪 Merchants contributing to diversion: "
        + " · ".join(diversion["active_merchants"])
    )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#21262d; margin-top:40px;'>
<div style='text-align:center; padding:16px 0; color:#8b949e; font-size:0.8rem;'>
    💸 <b style='color:#58a6ff;'>UpayPulse AI</b> &nbsp;|&nbsp;
    MFS AI Hackathon Prototype &nbsp;|&nbsp;
    Built with Streamlit · Pandas · NumPy · Scikit-learn · Folium &nbsp;|&nbsp;
    All data is <b>100% synthetic</b> — no real personal information used.
</div>
""", unsafe_allow_html=True)
