
"""
dashboard/app.py
================
UpayPulse AI — Streamlit MFS Intelligence Dashboard

A clean, modern, and professional fintech dashboard for UpayPulse AI.
Analyzes Mobile Financial Services (MFS) agent liquidity, transaction flow,
and regional cash-out demand across Bangladesh.

Dashboard Pages:
----------------
1. Overview      : Total agents, total transactions, total cash-out amount & trends
2. Area Analysis : Cash-out demand by area & spatial / temporal heatmaps
3. Agent Status  : Lowest cash balance agents & highest transaction volume agents
"""

from pathlib import Path
import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.orchestrator import OrchestratorAgent
from i18n import get_text

# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIGURATION & THEME STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UpayPulse AI | MFS Dashboard",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Quality Fintech Dark Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Background */
.stApp {
    background-color: #0b0f17;
    color: #e6edf3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111622;
    border-right: 1px solid #1f2937;
}

/* Modern Glassmorphic KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, rgba(22, 27, 34, 0.85) 0%, rgba(13, 17, 23, 0.95) 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    border-color: #58a6ff;
}
.kpi-title {
    font-size: 0.80rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8b949e;
    margin-bottom: 6px;
}
.kpi-number {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 2px 0 6px 0;
}
.kpi-caption {
    font-size: 0.76rem;
    color: #8b949e;
}

/* Color Accents */
.c-blue   { color: #58a6ff; }
.c-green  { color: #3fb950; }
.c-red    { color: #f85149; }
.c-amber  { color: #d29922; }
.c-purple { color: #bc8cff; }

/* Section Header Banners */
.page-header {
    background: linear-gradient(90deg, #161b22 0%, #0d1117 100%);
    border-left: 4px solid #58a6ff;
    border-radius: 0 8px 8px 0;
    padding: 14px 20px;
    margin: 8px 0 24px 0;
}
.page-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #f0f6fc;
    margin: 0;
}
.page-subtitle {
    font-size: 0.85rem;
    color: #8b949e;
    margin: 4px 0 0 0;
}

/* Table styling */
.dataframe {
    font-family: 'JetBrains Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA LOADING & PATH DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_datasets():
    """
    Load agents.csv and transactions.csv from available project data paths.
    """
    # Search prospective paths
    possible_dirs = [
        Path(__file__).resolve().parent.parent / "data",
        Path(__file__).resolve().parent / "data",
        Path.cwd() / "data",
        Path("data"),
    ]

    agents_path = None
    txns_path = None

    for d in possible_dirs:
        a = d / "agents.csv"
        t = d / "transactions.csv"
        m = d / "merchants.csv"
        if a.exists() and t.exists() and m.exists():
            agents_path = a
            txns_path = t
            merchants_path = m
            break

    if not agents_path or not txns_path or not merchants_path:
        st.error("Error: Could not locate 'data/agents.csv', 'data/transactions.csv', and 'data/merchants.csv'.")
        st.stop()

    agents_df = pd.read_csv(agents_path)
    txns_df = pd.read_csv(txns_path)
    merchants_df = pd.read_csv(merchants_path)

    # Date and time parsing
    txns_df["timestamp"] = pd.to_datetime(txns_df["timestamp"])
    txns_df["date"] = txns_df["timestamp"].dt.date
    txns_df["hour"] = txns_df["timestamp"].dt.hour
    txns_df["day_name"] = txns_df["timestamp"].dt.day_name()

    return agents_df, txns_df, merchants_df


agents_df, txns_df, merchants_df = load_datasets()


# ─────────────────────────────────────────────────────────────────────────────
# 3. SIDEBAR NAVIGATION & FILTER CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 8px 0;'>
        <div style='font-size:2.2rem;'>💸</div>
        <div style='font-size:1.30rem; font-weight:800; color:#58a6ff; letter-spacing:-0.02em;'>
            UpayPulse AI
        </div>
        <div style='font-size:0.78rem; color:#8b949e;'>
            MFS Liquidity & Analytics
        </div>
    </div>
    <hr style='border-color:#21262d; margin: 12px 0 18px 0;'>
    """, unsafe_allow_html=True)

    # Set default language
    if "lang" not in st.session_state:
        st.session_state.lang = "en"
        
    st.markdown("### 🌐 Language / ভাষা")
    lang_choice = st.radio("Language", ["English", "বাংলা"], index=0 if st.session_state.lang == "en" else 1, horizontal=True, label_visibility="collapsed")
    if lang_choice == "English" and st.session_state.lang != "en":
        st.session_state.lang = "en"
        st.rerun()
    elif lang_choice == "বাংলা" and st.session_state.lang != "bn":
        st.session_state.lang = "bn"
        st.rerun()

    st.markdown(f"### {get_text('nav_title')}")
    page_options = [get_text('nav_overview'), get_text('nav_area_analysis'), get_text('nav_agent_status'), "AI Liquidity Radar", get_text('nav_ai_chatbot')]
    selected_page = st.radio(
        "Select Page",
        page_options,
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#21262d; margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown("### 🔍 Global Filter")

    # Area Filter
    all_areas = ["All Areas"] + sorted(agents_df["area"].unique().tolist())
    area_choice = st.selectbox("Filter by Area", all_areas, index=0)

    # Apply filter to copies used on views
    if area_choice != "All Areas":
        filtered_agents = agents_df[agents_df["area"] == area_choice].copy()
        filtered_txns = txns_df[txns_df["area"] == area_choice].copy()
        filtered_merchants = merchants_df[merchants_df["area"] == area_choice].copy()
    else:
        filtered_agents = agents_df.copy()
        filtered_txns = txns_df.copy()
        filtered_merchants = merchants_df.copy()

    st.markdown("<hr style='border-color:#21262d; margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#8b949e; line-height:1.6;'>
        <b>Data Source:</b> Bangladesh MFS Ecosystem<br>
        <b>Agents in View:</b> {agents_count}<br>
        <b>Transactions in View:</b> {txns_count:,}
    </div>
    """.format(
        agents_count=len(filtered_agents),
        txns_count=len(filtered_txns)
    ), unsafe_allow_html=True)


# Helper function for Plotly density maps with cross-version compatibility
def create_density_map(df, lat_col, lon_col, z_col, title, height=450):
    if hasattr(px, "density_map"):
        fig = px.density_map(
            df,
            lat=lat_col,
            lon=lon_col,
            z=z_col,
            radius=22,
            zoom=6.5 if area_choice == "All Areas" else 13,
            color_continuous_scale="Viridis",
            title=title
        )
    elif hasattr(px, "density_mapbox"):
        fig = px.density_mapbox(
            df,
            lat=lat_col,
            lon=lon_col,
            z=z_col,
            radius=22,
            zoom=6.5 if area_choice == "All Areas" else 13,
            mapbox_style="carto-darkmatter",
            color_continuous_scale="Viridis",
            title=title
        )
    else:
        fig = px.scatter(df, x=lon_col, y=lat_col, size=z_col, color=z_col, title=title)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        height=height
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if selected_page == get_text('nav_overview'):
    st.markdown(f"""
    <div class="page-header">
        <h2 class="page-title">{get_text('overview_title')}</h2>
        <p class="page-subtitle">{get_text('overview_subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Primary Required Metrics
    total_agents = len(filtered_agents)
    total_transactions = len(filtered_txns)
    cashout_subset = filtered_txns[filtered_txns["transaction_type"] == "cash_out"]
    total_cashout_amount = cashout_subset["amount"].sum()

    # Complementary Metrics
    total_volume_all = filtered_txns["amount"].sum()
    avg_cashout_ticket = cashout_subset["amount"].mean() if len(cashout_subset) > 0 else 0.0
    cashout_share = (total_cashout_amount / total_volume_all * 100) if total_volume_all > 0 else 0.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Agents</div>
            <div class="kpi-number c-blue">{total_agents:,}</div>
            <div class="kpi-caption">Active retail MFS touchpoints</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Transactions</div>
            <div class="kpi-number c-green">{total_transactions:,}</div>
            <div class="kpi-caption">Processed across network</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Cash-Out Amount</div>
            <div class="kpi-number c-red">৳{total_cashout_amount / 1e6:,.2f}M</div>
            <div class="kpi-caption">{cashout_share:.1f}% of total volume (৳{total_cashout_amount:,.0f})</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Cash-Out Ticket</div>
            <div class="kpi-number c-amber">৳{avg_cashout_ticket:,.0f}</div>
            <div class="kpi-caption">Average customer withdrawal size</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Charts Section
    row1_c1, row1_c2 = st.columns([1, 1])

    with row1_c1:
        st.markdown("#### 💳 Volume Breakdown by Transaction Type")
        type_agg = filtered_txns.groupby("transaction_type")["amount"].agg(["sum", "count"]).reset_index()
        type_agg["display_name"] = type_agg["transaction_type"].str.replace("_", " ").str.title()

        fig_donut = px.pie(
            type_agg,
            names="display_name",
            values="sum",
            hole=0.55,
            color="transaction_type",
            color_discrete_map={
                "cash_out": "#f85149",
                "send_money": "#58a6ff",
                "merchant_payment": "#3fb950"
            }
        )
        fig_donut.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=20),
            height=340,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with row1_c2:
        st.markdown("#### 📈 Daily Transacted Volume Trend (BDT)")
        daily_volume = filtered_txns.groupby("date")["amount"].sum().reset_index()

        fig_trend = px.area(
            daily_volume,
            x="date",
            y="amount",
            labels={"amount": "Volume (BDT)", "date": "Date"},
            color_discrete_sequence=["#58a6ff"]
        )
        fig_trend.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=20),
            height=340,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#21262d")
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # 3. Agent Liquidity Comparison
    st.markdown("#### 💼 Agent Network Liquidity Structure")
    col_bal1, col_bal2 = st.columns(2)

    with col_bal1:
        fig_bal = px.histogram(
            filtered_agents,
            x="cash_balance",
            nbins=35,
            color_discrete_sequence=["#3fb950"],
            labels={"cash_balance": "Physical Cash Balance (BDT)"},
            title="Distribution of Cash in Hand"
        )
        fig_bal.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_bal, use_container_width=True)

    with col_bal2:
        fig_efloat = px.histogram(
            filtered_agents,
            x="e_float_balance",
            nbins=35,
            color_discrete_sequence=["#bc8cff"],
            labels={"e_float_balance": "Digital E-Float Balance (BDT)"},
            title="Distribution of E-Float Balance"
        )
        fig_efloat.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_efloat, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: AREA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif selected_page == get_text('nav_area_analysis'):
    st.markdown(f"""
    <div class="page-header">
        <h2 class="page-title">{get_text('area_analysis_title')}</h2>
        <p class="page-subtitle">{get_text('area_analysis_subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🗺️ Interactive Liquidity & Merchant Map")
    st.caption("Agent Cash Shortage Risk (Red = High Risk, Green = Safe) & Merchant Locations (Blue markers)")
    
    # Initialize Map
    if not filtered_agents.empty:
        center_lat = filtered_agents["latitude"].mean()
        center_lon = filtered_agents["longitude"].mean()
    else:
        center_lat, center_lon = 23.8103, 90.4125 # Default Dhaka
        
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12 if area_choice == "All Areas" else 14, tiles="CartoDB dark_matter")
    
    # Add Agent Markers
    for _, row in filtered_agents.iterrows():
        is_risk = row["cash_balance"] < 50000  # Threshold for risk
        color = "#f85149" if is_risk else "#3fb950"  # Red / Green
        status = "High Risk" if is_risk else "Safe"
        action = "Rebalance Required" if is_risk else "Optimal"
        
        popup_html = f"""
        <div style="font-family:sans-serif;font-size:12px;">
            <b>Agent:</b> {row['agent_id']}<br>
            <b>Area:</b> {row['area']}<br>
            <b>Cash Balance:</b> ৳{row['cash_balance']:,.0f}<br>
            <b>Risk Score:</b> {status}<br>
            <b>Recommended Action:</b> {action}
        </div>
        """
        
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            popup=folium.Popup(popup_html, max_width=250),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            tooltip=f"Agent: {row['agent_id']} ({status})"
        ).add_to(m)
        
    # Add Merchant Markers
    for _, row in filtered_merchants.iterrows():
        popup_html = f"""
        <div style="font-family:sans-serif;font-size:12px;">
            <b>Merchant:</b> {row['merchant_id']}<br>
            <b>Category:</b> {row['category']}<br>
            <b>Area:</b> {row['area']}<br>
            <b>Daily Txns:</b> {row['daily_transactions']}
        </div>
        """
        
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            icon=folium.Icon(color="blue", icon="shopping-cart", prefix="fa"),
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"Merchant: {row['merchant_id']}"
        ).add_to(m)
        
    st_folium(m, width="100%", height=500, returned_objects=[])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 1. Cash-Out Demand by Area
    st.markdown("#### 💰 Cash-Out Demand by Commercial Area")

    # Aggregate cash-out demand per area
    cashout_txns = txns_df[txns_df["transaction_type"] == "cash_out"]
    area_demand = cashout_txns.groupby("area")["amount"].agg(
        total_cashout="sum",
        txn_count="count",
        avg_ticket="mean"
    ).reset_index().sort_values("total_cashout", ascending=True)

    fig_area_bar = px.bar(
        area_demand,
        x="total_cashout",
        y="area",
        orientation="h",
        color="total_cashout",
        color_continuous_scale="Blues",
        labels={"total_cashout": "Total Cash-Out (BDT)", "area": "Commercial Hub / Area"},
        text="total_cashout"
    )
    fig_area_bar.update_traces(
        texttemplate="৳%{text:,.0f}",
        textposition="outside"
    )
    fig_area_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        height=max(480, len(area_demand) * 22),
        margin=dict(l=20, r=50, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor="#21262d"),
        yaxis=dict(tickfont=dict(size=12))
    )
    st.plotly_chart(fig_area_bar, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Transaction Heatmaps (Spatio-Temporal)
    st.markdown("#### 🔥 Transaction Heatmaps")
    hm_col1, hm_col2 = st.columns([1, 1])

    with hm_col1:
        st.markdown("##### 🗺️ Geographic Transaction Density Heatmap")
        st.caption("Visualizing spatial intensity of cash flows across agent coordinates")

        # Map agent coordinates with their total transaction volume
        agent_vols = txns_df.groupby("agent_id")["amount"].sum().reset_index()
        geo_data = agents_df.merge(agent_vols, on="agent_id", how="left").fillna({"amount": 0})

        density_fig = create_density_map(
            geo_data,
            lat_col="latitude",
            lon_col="longitude",
            z_col="amount",
            title="Spatial Transaction Heatmap (Bangladesh)"
        )
        st.plotly_chart(density_fig, use_container_width=True)

    with hm_col2:
        st.markdown("##### 🕒 Hourly Temporal Activity Heatmap")
        st.caption("Transaction concentration across hours of the day and days of the week")

        # Days of week in chronological order
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        hourly_matrix = txns_df.groupby(["day_name", "hour"])["amount"].sum().reset_index()
        hourly_pivot = hourly_matrix.pivot(index="day_name", columns="hour", values="amount").reindex(days_order).fillna(0)

        fig_matrix = px.imshow(
            hourly_pivot,
            labels=dict(x="Hour of Day (24-Hour)", y="Day of Week", color="Volume (BDT)"),
            color_continuous_scale="Purples",
            aspect="auto"
        )
        fig_matrix.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(tickmode="linear", tick0=0, dtick=2)
        )
        st.plotly_chart(fig_matrix, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: AGENT STATUS
# ─────────────────────────────────────────────────────────────────────────────
elif selected_page == get_text('nav_agent_status'):
    st.markdown(f"""
    <div class="page-header">
        <h2 class="page-title">{get_text('agent_status_title')}</h2>
        <p class="page-subtitle">{get_text('agent_status_subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)

    # Calculate agent transaction metrics
    txn_agg = txns_df.groupby("agent_id").agg(
        total_txns=("transaction_id", "count"),
        total_amount=("amount", "sum"),
        cashout_amount=("amount", lambda s: s[txns_df.loc[s.index, "transaction_type"] == "cash_out"].sum())
    ).reset_index()

    agents_full = agents_df.merge(txn_agg, on="agent_id", how="left").fillna({
        "total_txns": 0,
        "total_amount": 0,
        "cashout_amount": 0
    })

    # Section 1: Lowest Cash Balance Agents
    st.markdown("#### ⚠️ Lowest Cash Balance Agents (Liquidity Risk)")
    st.write("Agents with critically low physical cash in hand are vulnerable to transaction failure during peak cash-out demand.")

    lowest_cash_agents = agents_full.sort_values("cash_balance", ascending=True).head(15).copy()

    # Bar chart for lowest cash balance agents
    fig_low_cash = px.bar(
        lowest_cash_agents,
        x="cash_balance",
        y="agent_id",
        orientation="h",
        color="cash_balance",
        color_continuous_scale="Reds_r",
        hover_data=["area", "e_float_balance", "rating"],
        labels={"cash_balance": "Physical Cash Balance (BDT)", "agent_id": "Agent ID"},
        text="cash_balance"
    )
    fig_low_cash.update_traces(
        texttemplate="৳%{text:,.0f}",
        textposition="outside"
    )
    fig_low_cash.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        height=400,
        margin=dict(l=20, r=40, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor="#21262d")
    )
    st.plotly_chart(fig_low_cash, use_container_width=True)

    # Detailed Table for Lowest Cash Balance Agents
    table_lowest = lowest_cash_agents[[
        "agent_id", "area", "cash_balance", "e_float_balance", "rating", "working_hours"
    ]].copy()
    table_lowest["cash_balance"] = table_lowest["cash_balance"].map("৳{:,.0f}".format)
    table_lowest["e_float_balance"] = table_lowest["e_float_balance"].map("৳{:,.0f}".format)
    table_lowest["rating"] = table_lowest["rating"].map("⭐ {:.2f}".format)
    table_lowest.rename(columns={
        "agent_id": "Agent ID",
        "area": "Area",
        "cash_balance": "Cash in Hand",
        "e_float_balance": "Digital E-Float",
        "rating": "Rating",
        "working_hours": "Operating Hours"
    }, inplace=True)

    st.dataframe(table_lowest, use_container_width=True, hide_index=True)

    st.markdown("<br><hr style='border-color:#21262d;'><br>", unsafe_allow_html=True)

    # Section 2: Highest Transaction Agents
    st.markdown("#### 🏆 Highest Transaction Agents (Top Performers)")
    st.write("Top MFS retail agents processing the largest volume of consumer financial transactions.")

    highest_txn_agents = agents_full.sort_values("total_txns", ascending=False).head(15).copy()

    # Bar chart for highest transaction agents
    fig_high_txns = px.bar(
        highest_txn_agents,
        x="total_txns",
        y="agent_id",
        orientation="h",
        color="total_txns",
        color_continuous_scale="Greens",
        hover_data=["area", "total_amount", "rating"],
        labels={"total_txns": "Transaction Count", "agent_id": "Agent ID"},
        text="total_txns"
    )
    fig_high_txns.update_traces(
        texttemplate="%{text:,} txns",
        textposition="outside"
    )
    fig_high_txns.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        height=400,
        margin=dict(l=20, r=40, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor="#21262d")
    )
    st.plotly_chart(fig_high_txns, use_container_width=True)

    # Detailed Table for Highest Transaction Agents
    table_highest = highest_txn_agents[[
        "agent_id", "area", "total_txns", "total_amount", "cash_balance", "rating"
    ]].copy()
    table_highest["total_txns"] = table_highest["total_txns"].map("{:,}".format)
    table_highest["total_amount"] = table_highest["total_amount"].map("৳{:,.0f}".format)
    table_highest["cash_balance"] = table_highest["cash_balance"].map("৳{:,.0f}".format)
    table_highest["rating"] = table_highest["rating"].map("⭐ {:.2f}".format)
    table_highest.rename(columns={
        "agent_id": "Agent ID",
        "area": "Area",
        "total_txns": "Total Transactions",
        "total_amount": "Total Processed BDT",
        "cash_balance": "Current Cash",
        "rating": "Rating"
    }, inplace=True)

    st.dataframe(table_highest, use_container_width=True, hide_index=True)

    st.markdown("<br><hr style='border-color:#21262d;'><br>", unsafe_allow_html=True)
    st.markdown(f"{get_text('ai_risk_analysis')}")
    st.write(f"{get_text('ai_risk_desc')}")
    
    agent_id_to_check = st.selectbox(get_text('select_agent'), options=agents_full["agent_id"].tolist())
    if st.button(get_text('run_ai_analysis')):
        with st.spinner("Analyzing with UpayPulse AI Backend and SHAP Explainer..."):
            try:
                import requests
                # 1. Fetch Backend API Response
                res = requests.get(f"http://localhost:8000/agent-risk/{agent_id_to_check}", timeout=5)
                if res.status_code == 200:
                    risk_data = res.json()
                    st.success(f"Analysis Complete for {risk_data['agent_id']}")
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.metric(label="Risk Probability", value=f"{risk_data['risk_score']*100:.1f}%")
                        st.markdown(f"**Predicted Shortage Amount:** ৳{risk_data['shortage_prediction']:,.2f}")
                        st.info(f"**AI Explanation:** {risk_data['explanation']}")
                        
                    # 2. Run SHAP Explainability
                    with col2:
                        st.markdown("##### 🔬 Model Explainability (SHAP)")
                        st.markdown("**Why risk is high? Feature impact:**")
                        
                        shap_reasons = risk_data.get("shap_reasons", [])
                        if shap_reasons:
                            for reason in shap_reasons:
                                st.markdown(f"**- {reason}**")
                        else:
                            st.info("No explainability data available for this prediction.")
                else:
                    st.error(f"Backend API Error: {res.status_code}")
            except Exception as e:
                st.error(f"Error during analysis: {e}. Please ensure the FastAPI server is running (`uvicorn backend.api:app`).")



# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: AI LIQUIDITY RADAR
# ─────────────────────────────────────────────────────────────────────────────
elif selected_page == "AI Liquidity Radar":
    st.markdown("""
    <div class="page-header">
        <h2 class="page-title">🎯 AI Liquidity Radar</h2>
        <p class="page-subtitle">Predict real-time agent cash shortage risks using Advanced Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    agent_id = st.selectbox("Select Agent ID", options=agents_df["agent_id"].tolist(), key="radar_agent")

    if st.button("Run Liquidity Prediction"):
        with st.spinner("Loading ML Model & Analyzing Risk..."):
            from src.liquidity_predictor import LiquidityPredictor
            
            predictor = LiquidityPredictor()
            res = predictor.predict_risk(agent_id)
            
            if "error" in res:
                st.error(res["error"])
            else:
                agent_data = agents_df[agents_df["agent_id"] == agent_id].iloc[0]
                predicted_demand = agent_data['cash_balance'] + res['expected_shortage_amount']
                
                risk_color = "#f85149" if res["risk_level"] == "HIGH" else "#d29922" if res["risk_level"] == "MEDIUM" else "#3fb950"
                
                reasons_html = ''.join(f'<li>{r}</li>' for r in res['main_reasons'])
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(22, 27, 34, 0.95) 0%, rgba(13, 17, 23, 0.95) 100%);
                            border: 1px solid #30363d; border-radius: 12px; padding: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.35);">
                    <h3 style="margin-top: 0; color: #58a6ff;">Agent: {agent_id}</h3>
                    <hr style="border-color:#30363d;">
                    
                    <div style="display: flex; justify-content: space-between; margin-bottom: 20px; flex-wrap: wrap;">
                        <div style="min-width: 150px; margin-bottom: 10px;">
                            <div style="color: #8b949e; font-size: 0.85rem; text-transform: uppercase;">Current Cash</div>
                            <div style="font-size: 1.5rem; font-weight: bold;">৳{agent_data['cash_balance']:,.0f}</div>
                        </div>
                        <div style="min-width: 150px; margin-bottom: 10px;">
                            <div style="color: #8b949e; font-size: 0.85rem; text-transform: uppercase;">Predicted Demand</div>
                            <div style="font-size: 1.5rem; font-weight: bold;">৳{predicted_demand:,.0f}</div>
                        </div>
                        <div style="min-width: 150px; margin-bottom: 10px;">
                            <div style="color: #8b949e; font-size: 0.85rem; text-transform: uppercase;">Risk Level</div>
                            <div style="font-size: 1.5rem; font-weight: bold; color: {risk_color};">{res['risk_level']}</div>
                        </div>
                    </div>
                    
                    <div style="display: flex; justify-content: space-between; margin-bottom: 20px; flex-wrap: wrap;">
                        <div style="min-width: 150px; margin-bottom: 10px;">
                            <div style="color: #8b949e; font-size: 0.85rem; text-transform: uppercase;">Probability</div>
                            <div style="font-size: 1.5rem; font-weight: bold; color: {risk_color};">{res['risk_probability']*100:.1f}%</div>
                        </div>
                        <div style="min-width: 150px; margin-bottom: 10px;">
                            <div style="color: #8b949e; font-size: 0.85rem; text-transform: uppercase;">Expected Shortage</div>
                            <div style="font-size: 1.5rem; font-weight: bold; color: #f85149;">৳{res['expected_shortage_amount']:,.0f}</div>
                        </div>
                        <div style="min-width: 150px; margin-bottom: 10px;"></div>
                    </div>
                    
                    <hr style="border-color:#30363d;">
                    <div style="margin-bottom: 15px;">
                        <div style="color: #8b949e; font-size: 0.85rem; text-transform: uppercase; margin-bottom: 8px;">Reasons</div>
                        <ul style="margin: 0; padding-left: 20px; line-height: 1.6;">
                            {reasons_html}
                        </ul>
                    </div>
                    
                    <div style="background: rgba(88, 166, 255, 0.1); border-left: 4px solid #58a6ff; padding: 12px 16px; border-radius: 4px;">
                        <div style="color: #58a6ff; font-weight: bold; margin-bottom: 4px;">💡 Recommendation</div>
                        Find nearby liquidity partner
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: AI CHATBOT
# ─────────────────────────────────────────────────────────────────────────────
elif selected_page == get_text('nav_ai_chatbot'):
    st.markdown(f"""
    <div class="page-header">
        <h2 class="page-title">{get_text('chatbot_title')}</h2>
        <p class="page-subtitle">{get_text('chatbot_subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am the UpayPulse AI Orchestrator. How can I assist you today?\n\nYou can ask me things like:\n- *Which agent needs cash?*\n- *How can a merchant increase sales?*\n- *Where can I get an offer?*"}
        ]

    # Initialize orchestrator
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = OrchestratorAgent()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    import requests

    # Accept user input
    if prompt := st.chat_input("Type your question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Try hitting the FastAPI backend
                    res = requests.post("http://localhost:8000/ask", json={"query": prompt, "lang": st.session_state.lang}, timeout=10)
                    if res.status_code == 200:
                        response = res.json().get("response", "No response.")
                    else:
                        response = f"API Error: {res.status_code}"
                except requests.exceptions.RequestException:
                    # Fallback to local orchestrator if backend is not running
                    response = st.session_state.orchestrator.handle_query(prompt, lang=st.session_state.lang)
                
                st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


# ─────────────────────────────────────────────────────────────────────────────
# 5. FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#21262d; margin-top:40px;'>
<div style='text-align:center; padding:12px 0 20px 0; color:#8b949e; font-size:0.80rem;'>
    💸 <b style='color:#58a6ff;'>UpayPulse AI</b> &nbsp;|&nbsp;
    Enterprise MFS Liquidity & Transaction Intelligence &nbsp;|&nbsp;
    Powered by Streamlit · Pandas · Plotly
</div>
""", unsafe_allow_html=True)
