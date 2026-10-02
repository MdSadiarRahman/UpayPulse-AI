"""
dashboard/app.py
================
UpayPulse AI — Executive MFS Liquidity & Transaction Intelligence Platform

An ultra-modern, high-fidelity FinTech analytics dashboard for Mobile Financial
Services (MFS) in Bangladesh. Features real-time liquidity surveillance,
spatio-temporal cash flow intelligence, AI-powered agent solvency radar,
merchant growth engine, and autonomous conversational assistant.
"""

from pathlib import Path
import datetime
import math
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

# Add root and dashboard directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

# Import AI Engines & Utilities
from agents.orchestrator import OrchestratorAgent
from src.liquidity_predictor import LiquidityPredictor
from src.agent_recommender import AgentRecommender
from src.merchant_engine import MerchantGrowthEngine
from src.customer_offer import CustomerOfferEngine
from i18n import get_text

# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIGURATION & LUXURY FINTECH DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UpayPulse AI | Enterprise MFS Intelligence",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Modern Glassmorphism & Cyber FinTech Dark Theme CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #070b13;
    --bg-surface: #0e1524;
    --bg-card: rgba(16, 24, 40, 0.75);
    --border-subtle: rgba(255, 255, 255, 0.07);
    --border-focus: rgba(14, 165, 233, 0.4);
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --cyan-primary: #0ea5e9;
    --cyan-glow: rgba(14, 165, 233, 0.25);
    --emerald-safe: #10b981;
    --amber-warning: #f59e0b;
    --rose-danger: #f43f5e;
    --violet-ai: #8b5cf6;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif;
    color: var(--text-primary);
}

/* Background */
.stApp {
    background-color: var(--bg-primary);
    background-image: 
        radial-gradient(circle at 15% 10%, rgba(14, 165, 233, 0.06) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(99, 102, 241, 0.05) 0%, transparent 40%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0a0f1a !important;
    border-right: 1px solid var(--border-subtle) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.90rem;
}

/* Metric KPI Cards with Glassmorphic Elevation */
.kpi-card {
    background: linear-gradient(145deg, rgba(20, 29, 47, 0.85) 0%, rgba(12, 18, 31, 0.95) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.03);
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}
.kpi-card:hover {
    transform: translateY(-3px);
    border-color: var(--border-focus);
    box-shadow: 0 16px 32px -8px rgba(0, 0, 0, 0.6), 0 0 20px var(--cyan-glow);
}
.kpi-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.kpi-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-secondary);
}
.kpi-icon {
    font-size: 1.15rem;
    padding: 6px 9px;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.04);
}
.kpi-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.15rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 4px 0 8px 0;
}
.kpi-footer {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.76rem;
    color: var(--text-muted);
}
.kpi-pill {
    padding: 2px 7px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.70rem;
    font-family: 'JetBrains Mono', monospace;
}
.pill-up { background: rgba(16, 185, 129, 0.15); color: #34d399; }
.pill-down { background: rgba(244, 63, 94, 0.15); color: #fb7185; }
.pill-info { background: rgba(14, 165, 233, 0.15); color: #38bdf8; }

/* Text Accents */
.c-cyan   { color: #38bdf8; }
.c-green  { color: #34d399; }
.c-rose   { color: #fb7185; }
.c-amber  { color: #fbbf24; }
.c-purple { color: #c084fc; }

/* Page Header Banner */
.page-banner {
    background: linear-gradient(90deg, rgba(20, 29, 47, 0.9) 0%, rgba(12, 18, 31, 0.6) 100%);
    border: 1px solid var(--border-subtle);
    border-left: 4px solid var(--cyan-primary);
    border-radius: 12px;
    padding: 18px 24px;
    margin: 4px 0 24px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.page-banner-title {
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #ffffff;
    margin: 0;
}
.page-banner-subtitle {
    font-size: 0.88rem;
    color: var(--text-secondary);
    margin: 4px 0 0 0;
}

/* System Live Status Pill */
.live-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    color: #34d399;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #10b981;
    box-shadow: 0 0 10px #10b981;
    animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0% { transform: scale(0.9); opacity: 1; }
    50% { transform: scale(1.3); opacity: 0.4; }
    100% { transform: scale(0.9); opacity: 1; }
}

/* Glass Card Container */
.glass-box {
    background: linear-gradient(145deg, rgba(20, 29, 47, 0.7) 0%, rgba(12, 18, 31, 0.85) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 22px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    margin-bottom: 20px;
}

/* Status Badges */
.badge-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-family: 'JetBrains Mono', monospace;
}
.badge-safe { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }
.badge-warning { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }
.badge-danger { background: rgba(244, 63, 94, 0.2); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.35); }
.badge-ai { background: rgba(139, 92, 246, 0.2); color: #c084fc; border: 1px solid rgba(139, 92, 246, 0.35); }

/* Chat Bubble Enhancements */
[data-testid="stChatMessage"] {
    background: rgba(16, 24, 40, 0.7) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
}

/* Streamlit Native Adjustments */
.stButton > button {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    color: #ffffff;
    font-weight: 600;
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 8px;
    padding: 8px 18px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
    box-shadow: 0 0 15px var(--cyan-glow);
    border-color: #38bdf8;
    color: #ffffff;
}

/* Prompt Pills */
.prompt-pill {
    display: inline-block;
    background: rgba(14, 165, 233, 0.1);
    border: 1px solid rgba(14, 165, 233, 0.3);
    color: #38bdf8;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.80rem;
    margin: 4px 6px 4px 0;
    cursor: pointer;
    transition: all 0.2s;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET CACHING & PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_datasets():
    """Load agents.csv, transactions.csv, and merchants.csv."""
    possible_dirs = [
        ROOT_DIR / "data",
        Path(__file__).resolve().parent / "data",
        Path.cwd() / "data",
        Path("data"),
    ]

    agents_path = None
    txns_path = None
    merchants_path = None

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
        st.error("Error: Could not locate datasets in project data folder.")
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

# Initialize AI Service Instances in Session State (with local instant fallback)
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = OrchestratorAgent()
if "liquidity_predictor" not in st.session_state:
    st.session_state.liquidity_predictor = LiquidityPredictor()
if "agent_recommender" not in st.session_state:
    st.session_state.agent_recommender = AgentRecommender()
if "merchant_engine" not in st.session_state:
    st.session_state.merchant_engine = MerchantGrowthEngine()
if "customer_engine" not in st.session_state:
    st.session_state.customer_engine = CustomerOfferEngine()


# ─────────────────────────────────────────────────────────────────────────────
# 3. SIDEBAR NAVIGATION & EXECUTIVE CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 14px 0 10px 0;'>
        <div style='font-size: 2.3rem; margin-bottom: 2px;'>💸</div>
        <div style='font-size: 1.35rem; font-weight: 800; color: #38bdf8; letter-spacing: -0.03em;'>
            UpayPulse AI
        </div>
        <div style='display: flex; justify-content: center; gap: 6px; margin-top: 6px;'>
            <span class='badge-tag badge-ai'>v2.5 PRO</span>
            <span class='badge-tag badge-safe'>AI LIVE</span>
        </div>
    </div>
    <hr style='border-color: rgba(255,255,255,0.08); margin: 12px 0 18px 0;'>
    """, unsafe_allow_html=True)

    # Language Switcher
    if "lang" not in st.session_state:
        st.session_state.lang = "en"

    col_lang1, col_lang2 = st.columns(2)
    with col_lang1:
        if st.button("🇺🇸 English", use_container_width=True, type="primary" if st.session_state.lang == "en" else "secondary"):
            st.session_state.lang = "en"
            st.rerun()
    with col_lang2:
        if st.button("🇧🇩 বাংলা", use_container_width=True, type="primary" if st.session_state.lang == "bn" else "secondary"):
            st.session_state.lang = "bn"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.75rem; font-weight:700; color:#64748b; letter-spacing:0.08em; text-transform:uppercase;'>{get_text('nav_title')}</div>", unsafe_allow_html=True)

    # Navigation Menu
    nav_items = [
        ("📊 " + get_text("nav_overview"), "overview"),
        ("📍 " + get_text("nav_area_analysis"), "area_analysis"),
        ("👤 " + get_text("nav_agent_status"), "agent_status"),
        ("🎯 " + get_text("nav_liquidity_radar"), "liquidity_radar"),
        ("🏪 " + get_text("nav_merchant_offers"), "merchant_offers"),
        ("🤖 " + get_text("nav_ai_chatbot"), "ai_chatbot")
    ]
    nav_labels = [item[0] for item in nav_items]
    
    if "current_nav" not in st.session_state:
        st.session_state.current_nav = nav_labels[0]

    selected_nav_label = st.radio(
        "Navigation Menu",
        nav_labels,
        index=nav_labels.index(st.session_state.current_nav) if st.session_state.current_nav in nav_labels else 0,
        label_visibility="collapsed"
    )
    st.session_state.current_nav = selected_nav_label

    # Find the active page key
    active_page_key = next((item[1] for item in nav_items if item[0] == selected_nav_label), "overview")

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown(f"**{get_text('filter_title')}**")

    # Global Area Filter
    all_areas = [get_text("filter_all_areas")] + sorted(agents_df["area"].unique().tolist())
    area_choice = st.selectbox(get_text("filter_area"), all_areas, index=0)

    # Filter Datasets
    if area_choice != get_text("filter_all_areas"):
        filtered_agents = agents_df[agents_df["area"] == area_choice].copy()
        filtered_txns = txns_df[txns_df["area"] == area_choice].copy()
        filtered_merchants = merchants_df[merchants_df["area"] == area_choice].copy()
    else:
        filtered_agents = agents_df.copy()
        filtered_txns = txns_df.copy()
        filtered_merchants = merchants_df.copy()

    # Network Telemetry in Sidebar
    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 12px; font-size: 0.80rem; line-height: 1.6;'>
        <div style='color: #94a3b8; font-weight: 600; margin-bottom: 4px;'>📡 {get_text('data_source')}</div>
        <div style='display:flex; justify-content:space-between;'>
            <span style='color: #64748b;'>{get_text('agents_in_view')}</span>
            <b style='color: #38bdf8; font-family: monospace;'>{len(filtered_agents):,}</b>
        </div>
        <div style='display:flex; justify-content:space-between;'>
            <span style='color: #64748b;'>{get_text('txns_in_view')}</span>
            <b style='color: #34d399; font-family: monospace;'>{len(filtered_txns):,}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Helper function for Plotly density maps with cross-version compatibility
def create_density_map(df, lat_col, lon_col, z_col, title, height=450):
    if hasattr(px, "density_map"):
        fig = px.density_map(
            df,
            lat=lat_col,
            lon=lon_col,
            z=z_col,
            radius=22,
            zoom=6.5 if area_choice == get_text("filter_all_areas") else 12.5,
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
            zoom=6.5 if area_choice == get_text("filter_all_areas") else 12.5,
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
# PAGE 1: 📊 EXECUTIVE OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if active_page_key == "overview":
    st.markdown(f"""
    <div class="page-banner">
        <div>
            <h1 class="page-banner-title">{get_text('overview_title')}</h1>
            <p class="page-banner-subtitle">{get_text('overview_subtitle')}</p>
        </div>
        <div>
            <span class="live-pill"><span class="pulse-dot"></span> LIVE MFS TELEMETRY</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Core Metrics Calculations
    total_agents = len(filtered_agents)
    total_transactions = len(filtered_txns)
    cashout_subset = filtered_txns[filtered_txns["transaction_type"] == "cash_out"]
    total_cashout_amount = cashout_subset["amount"].sum()
    total_volume_all = filtered_txns["amount"].sum()
    avg_cashout_ticket = cashout_subset["amount"].mean() if len(cashout_subset) > 0 else 0.0
    cashout_share = (total_cashout_amount / total_volume_all * 100) if total_volume_all > 0 else 0.0
    
    # Critical Risk Count
    low_cash_count = len(filtered_agents[filtered_agents["cash_balance"] < 40000])

    # 4 Hero KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">{get_text('total_agents')}</span>
                <span class="kpi-icon">👥</span>
            </div>
            <div class="kpi-number c-cyan">{total_agents:,}</div>
            <div class="kpi-footer">
                <span class="kpi-pill pill-info">100% ONLINE</span>
                <span>{get_text('total_agents_caption')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">{get_text('total_txns')}</span>
                <span class="kpi-icon">⚡</span>
            </div>
            <div class="kpi-number c-green">{total_transactions:,}</div>
            <div class="kpi-footer">
                <span class="kpi-pill pill-up">↑ 14.8%</span>
                <span>{get_text('total_txns_caption')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">{get_text('total_cashout')}</span>
                <span class="kpi-icon">💸</span>
            </div>
            <div class="kpi-number c-rose">৳{total_cashout_amount / 1e6:,.2f}M</div>
            <div class="kpi-footer">
                <span class="kpi-pill pill-down">{cashout_share:.1f}%</span>
                <span>{get_text('total_cashout_caption')} (৳{total_cashout_amount:,.0f})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">{get_text('avg_cashout')}</span>
                <span class="kpi-icon">🏷️</span>
            </div>
            <div class="kpi-number c-amber">৳{avg_cashout_ticket:,.0f}</div>
            <div class="kpi-footer">
                <span class="kpi-pill pill-info">STABLE</span>
                <span>{get_text('avg_cashout_caption')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Real-Time Liquidity Solvency Banner
    st.markdown(f"""
    <div style='background: linear-gradient(90deg, rgba(244, 63, 94, 0.12) 0%, rgba(20, 29, 47, 0.6) 100%);
                border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 12px; padding: 14px 20px;
                display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;'>
        <div style='display: flex; align-items: center; gap: 14px;'>
            <div style='font-size: 1.8rem;'>🚨</div>
            <div>
                <b style='color: #fb7185; font-size: 0.95rem;'>NETWORK LIQUIDITY SURVEILLANCE ALERT:</b>
                <span style='color: #e2e8f0; font-size: 0.88rem; margin-left: 6px;'>
                    <b>{low_cash_count} agents</b> in this view currently hold physical cash reserves under <b>৳40,000</b>.
                </span>
            </div>
        </div>
        <div>
            <span class='badge-tag badge-danger'>ACTION REQUIRED</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Row 1: Charts (Donut & Area Trend)
    row1_c1, row1_c2 = st.columns([1, 1])

    with row1_c1:
        st.markdown(f"#### {get_text('vol_breakdown')}")
        type_agg = filtered_txns.groupby("transaction_type")["amount"].agg(["sum", "count"]).reset_index()
        type_agg["display_name"] = type_agg["transaction_type"].str.replace("_", " ").str.title()

        fig_donut = px.pie(
            type_agg,
            names="display_name",
            values="sum",
            hole=0.62,
            color="transaction_type",
            color_discrete_map={
                "cash_out": "#f43f5e",
                "send_money": "#0ea5e9",
                "merchant_payment": "#10b981"
            }
        )
        fig_donut.update_traces(
            textposition='inside',
            textinfo='percent',
            marker=dict(line=dict(color='#0b0f17', width=2))
        )
        fig_donut.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=340,
            legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5, font=dict(size=12))
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with row1_c2:
        st.markdown(f"#### {get_text('daily_trend')}")
        daily_volume = filtered_txns.groupby("date")["amount"].sum().reset_index()

        fig_trend = px.area(
            daily_volume,
            x="date",
            y="amount",
            labels={"amount": "Volume (BDT)", "date": "Date"},
            color_discrete_sequence=["#0ea5e9"]
        )
        fig_trend.update_traces(
            line=dict(width=2.5, color='#38bdf8'),
            fillcolor='rgba(14, 165, 233, 0.18)'
        )
        fig_trend.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=20),
            height=340,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickprefix="৳")
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # Row 2: Agent Liquidity Structure
    st.markdown(f"#### {get_text('agent_liquidity')}")
    col_bal1, col_bal2 = st.columns(2)

    with col_bal1:
        fig_bal = px.histogram(
            filtered_agents,
            x="cash_balance",
            nbins=35,
            color_discrete_sequence=["#10b981"],
            labels={"cash_balance": "Physical Cash Balance (BDT)"},
            title="Physical Cash in Hand Distribution"
        )
        fig_bal.update_traces(marker=dict(line=dict(color='#0b0f17', width=1)))
        fig_bal.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(tickprefix="৳", gridcolor="rgba(255,255,255,0.06)")
        )
        st.plotly_chart(fig_bal, use_container_width=True)

    with col_bal2:
        fig_efloat = px.histogram(
            filtered_agents,
            x="e_float_balance",
            nbins=35,
            color_discrete_sequence=["#c084fc"],
            labels={"e_float_balance": "Digital E-Float Balance (BDT)"},
            title="Digital E-Float Balance Distribution"
        )
        fig_efloat.update_traces(marker=dict(line=dict(color='#0b0f17', width=1)))
        fig_efloat.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(tickprefix="৳", gridcolor="rgba(255,255,255,0.06)")
        )
        st.plotly_chart(fig_efloat, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: 📍 SPATIAL & AREA INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────
elif active_page_key == "area_analysis":
    st.markdown(f"""
    <div class="page-banner">
        <div>
            <h1 class="page-banner-title">{get_text('area_analysis_title')}</h1>
            <p class="page-banner-subtitle">{get_text('area_analysis_subtitle')}</p>
        </div>
        <div>
            <span class="badge-tag badge-ai">GEOSPATIAL AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"#### {get_text('map_title')}")
    st.caption(get_text('map_caption'))

    # Initialize Folium Map
    if not filtered_agents.empty:
        center_lat = filtered_agents["latitude"].mean()
        center_lon = filtered_agents["longitude"].mean()
    else:
        center_lat, center_lon = 23.8103, 90.4125

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12 if area_choice == get_text("filter_all_areas") else 14,
        tiles="CartoDB dark_matter"
    )

    # Add Agent Markers with rich HTML popups
    for _, row in filtered_agents.iterrows():
        is_risk = row["cash_balance"] < 50000
        color = "#f43f5e" if is_risk else "#10b981"
        status_label = "HIGH SHORTAGE RISK" if is_risk else "HEALTHY SURPLUS"
        action = "Initiate Urgent Cash Transfer" if is_risk else "Optimal Liquidity"

        popup_html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 13px; color: #0f172a; width: 220px; line-height: 1.5;">
            <div style="font-weight: 800; font-size: 14px; border-bottom: 2px solid {color}; padding-bottom: 4px; margin-bottom: 6px;">
                Agent: {row['agent_id']}
            </div>
            <div><b>Area:</b> {row['area']}</div>
            <div><b>Cash Balance:</b> ৳{row['cash_balance']:,.0f}</div>
            <div><b>E-Float:</b> ৳{row['e_float_balance']:,.0f}</div>
            <div><b>Rating:</b> ⭐ {row['rating']}</div>
            <div style="margin-top: 6px; padding: 4px 6px; border-radius: 4px; background: {'#fee2e2' if is_risk else '#dcfce7'}; color: {'#b91c1c' if is_risk else '#15803d'}; font-weight: bold; font-size: 11px;">
                {status_label}
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=7,
            popup=folium.Popup(popup_html, max_width=260),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=f"{row['agent_id']} — {status_label}"
        ).add_to(m)

    # Add Merchant Markers
    for _, row in filtered_merchants.head(150).iterrows():
        popup_html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 12px; color: #0f172a; width: 190px;">
            <div style="font-weight: bold; color: #0284c7;">Merchant: {row['merchant_id']}</div>
            <div><b>Category:</b> {row['category']}</div>
            <div><b>Area:</b> {row['area']}</div>
            <div><b>Daily Txns:</b> {row['daily_transactions']}</div>
        </div>
        """
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            icon=folium.Icon(color="blue", icon="shopping-cart", prefix="fa"),
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"Merchant: {row['merchant_id']} ({row['category']})"
        ).add_to(m)

    st_folium(m, width=700, height=500, use_container_width=True, returned_objects=[])

    st.markdown("<br>", unsafe_allow_html=True)

    # Cash-Out Demand by Area Ranking
    st.markdown(f"#### {get_text('cashout_demand_area')}")

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
        color_continuous_scale="Teal",
        labels={"total_cashout": "Total Cash-Out (BDT)", "area": "Commercial Hub / Area"},
        text="total_cashout"
    )
    fig_area_bar.update_traces(
        texttemplate="৳%{text:,.0f}",
        textposition="outside",
        marker=dict(line=dict(color='#0b0f17', width=1))
    )
    fig_area_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        height=max(480, len(area_demand) * 24),
        margin=dict(l=20, r=60, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickprefix="৳"),
        yaxis=dict(tickfont=dict(size=12))
    )
    st.plotly_chart(fig_area_bar, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Spatio-Temporal Heatmaps
    st.markdown(f"#### {get_text('heatmaps_title')}")
    hm_col1, hm_col2 = st.columns([1, 1])

    with hm_col1:
        st.markdown(f"##### {get_text('geo_heatmap_title')}")
        st.caption(get_text('geo_heatmap_caption'))

        agent_vols = txns_df.groupby("agent_id")["amount"].sum().reset_index()
        geo_data = agents_df.merge(agent_vols, on="agent_id", how="left").fillna({"amount": 0})

        density_fig = create_density_map(
            geo_data,
            lat_col="latitude",
            lon_col="longitude",
            z_col="amount",
            title="Spatial Intensity Matrix"
        )
        st.plotly_chart(density_fig, use_container_width=True)

    with hm_col2:
        st.markdown(f"##### {get_text('temporal_heatmap_title')}")
        st.caption(get_text('temporal_heatmap_caption'))

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
# PAGE 3: 👤 AGENT OPERATIONS & SOLVENCY HEALTH
# ─────────────────────────────────────────────────────────────────────────────
elif active_page_key == "agent_status":
    st.markdown(f"""
    <div class="page-banner">
        <div>
            <h1 class="page-banner-title">{get_text('agent_status_title')}</h1>
            <p class="page-banner-subtitle">{get_text('agent_status_subtitle')}</p>
        </div>
        <div>
            <span class="badge-tag badge-warning">OPERATIONAL AUDIT</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Compute agent aggregated performance metrics
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

    # Summary Row
    low_bal_count = len(agents_full[agents_full["cash_balance"] < 40000])
    mid_bal_count = len(agents_full[(agents_full["cash_balance"] >= 40000) & (agents_full["cash_balance"] < 80000)])
    safe_bal_count = len(agents_full[agents_full["cash_balance"] >= 80000])

    c_stat1, c_stat2, c_stat3 = st.columns(3)
    with c_stat1:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #f43f5e;">
            <div class="kpi-title">Critical Deficit Risk</div>
            <div class="kpi-number c-rose">{low_bal_count} Agents</div>
            <div class="kpi-caption">Cash balance &lt; ৳40,000</div>
        </div>
        """, unsafe_allow_html=True)
    with c_stat2:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #f59e0b;">
            <div class="kpi-title">Moderate Reserve</div>
            <div class="kpi-number c-amber">{mid_bal_count} Agents</div>
            <div class="kpi-caption">Cash balance ৳40k – ৳80k</div>
        </div>
        """, unsafe_allow_html=True)
    with c_stat3:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #10b981;">
            <div class="kpi-title">Optimal Surplus</div>
            <div class="kpi-number c-green">{safe_bal_count} Agents</div>
            <div class="kpi-caption">Cash balance &gt; ৳80,000</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 1: Lowest Cash Balance Agents
    st.markdown(f"#### {get_text('lowest_cash_title')}")
    st.caption(get_text('lowest_cash_desc'))

    lowest_cash_agents = agents_full.sort_values("cash_balance", ascending=True).head(12).copy()

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
        textposition="outside",
        marker=dict(line=dict(color='#0b0f17', width=1))
    )
    fig_low_cash.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        height=380,
        margin=dict(l=20, r=40, t=10, b=10),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickprefix="৳")
    )
    st.plotly_chart(fig_low_cash, use_container_width=True)

    # Table for Lowest Cash Balance Agents
    table_lowest = lowest_cash_agents[[
        "agent_id", "area", "cash_balance", "e_float_balance", "rating", "working_hours"
    ]].copy()
    table_lowest["cash_balance"] = table_lowest["cash_balance"].map("৳{:,.0f}".format)
    table_lowest["e_float_balance"] = table_lowest["e_float_balance"].map("৳{:,.0f}".format)
    table_lowest["rating"] = table_lowest["rating"].map("⭐ {:.2f}".format)
    table_lowest.rename(columns={
        "agent_id": "Agent ID",
        "area": "Commercial Area",
        "cash_balance": "Cash in Hand",
        "e_float_balance": "Digital E-Float",
        "rating": "Trust Score",
        "working_hours": "Operating Hours"
    }, inplace=True)

    st.dataframe(table_lowest, use_container_width=True, hide_index=True)

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 30px 0;'>", unsafe_allow_html=True)

    # Section 2: Top Transaction Volume Agents
    st.markdown(f"#### {get_text('highest_txn_title')}")
    st.caption(get_text('highest_txn_desc'))

    highest_txn_agents = agents_full.sort_values("total_txns", ascending=False).head(12).copy()

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
        textposition="outside",
        marker=dict(line=dict(color='#0b0f17', width=1))
    )
    fig_high_txns.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        height=380,
        margin=dict(l=20, r=40, t=10, b=10),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    )
    st.plotly_chart(fig_high_txns, use_container_width=True)

    # Table for Highest Transaction Agents
    table_highest = highest_txn_agents[[
        "agent_id", "area", "total_txns", "total_amount", "cash_balance", "rating"
    ]].copy()
    table_highest["total_txns"] = table_highest["total_txns"].map("{:,}".format)
    table_highest["total_amount"] = table_highest["total_amount"].map("৳{:,.0f}".format)
    table_highest["cash_balance"] = table_highest["cash_balance"].map("৳{:,.0f}".format)
    table_highest["rating"] = table_highest["rating"].map("⭐ {:.2f}".format)
    table_highest.rename(columns={
        "agent_id": "Agent ID",
        "area": "Commercial Area",
        "total_txns": "Total Txns Count",
        "total_amount": "Total Transacted BDT",
        "cash_balance": "Current Cash",
        "rating": "Trust Score"
    }, inplace=True)

    st.dataframe(table_highest, use_container_width=True, hide_index=True)

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 30px 0;'>", unsafe_allow_html=True)

    # Deep AI Diagnosis Section
    st.markdown(f"#### 🔬 {get_text('ai_risk_analysis')}")
    st.caption(get_text('ai_risk_desc'))

    c_sel, c_btn = st.columns([3, 1])
    with c_sel:
        target_audit_agent = st.selectbox(get_text("select_agent"), options=agents_full["agent_id"].tolist())
    with c_btn:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        run_audit = st.button(get_text("run_ai_analysis"), use_container_width=True)

    if run_audit:
        with st.spinner("Executing UpayPulse AI ML Predictor & SHAP Explainer..."):
            try:
                # Direct instant prediction using engine
                risk_data = st.session_state.liquidity_predictor.predict_risk(target_audit_agent)
                if "error" not in risk_data:
                    res_col1, res_col2 = st.columns([1, 1])

                    with res_col1:
                        st.markdown(f"""
                        <div class="glass-box" style="border-left: 4px solid {'#f43f5e' if risk_data['risk_level'] == 'HIGH' else '#f59e0b' if risk_data['risk_level'] == 'MEDIUM' else '#10b981'};">
                            <div style="font-size:0.85rem; color:#94a3b8; text-transform:uppercase; font-weight:700;">Agent Diagnostic Dossier</div>
                            <h2 style="margin: 4px 0 12px 0; color: #38bdf8;">{target_audit_agent}</h2>
                            <div style="display:flex; gap: 20px; margin-bottom: 12px;">
                                <div>
                                    <div style="font-size:0.78rem; color:#64748b;">Risk Level</div>
                                    <div style="font-size:1.4rem; font-weight:800; color:{'#f43f5e' if risk_data['risk_level'] == 'HIGH' else '#f59e0b' if risk_data['risk_level'] == 'MEDIUM' else '#10b981'};">
                                        {risk_data['risk_level']}
                                    </div>
                                </div>
                                <div>
                                    <div style="font-size:0.78rem; color:#64748b;">Probability</div>
                                    <div style="font-size:1.4rem; font-weight:800; color:#38bdf8;">
                                        {risk_data['risk_probability']*100:.1f}%
                                    </div>
                                </div>
                                <div>
                                    <div style="font-size:0.78rem; color:#64748b;">Expected Shortage</div>
                                    <div style="font-size:1.4rem; font-weight:800; color:#f43f5e;">
                                        ৳{risk_data['expected_shortage_amount']:,.0f}
                                    </div>
                                </div>
                            </div>
                            <hr style="border-color: rgba(255,255,255,0.08); margin: 10px 0;">
                            <b style="font-size: 0.85rem; color: #e2e8f0;">Primary Drivers:</b>
                            <ul style="margin: 6px 0; padding-left: 20px; font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
                                {''.join(f'<li>{r}</li>' for r in risk_data['main_reasons'])}
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

                    with res_col2:
                        st.markdown(f"""
                        <div class="glass-box">
                            <div style="font-size:0.85rem; color:#94a3b8; text-transform:uppercase; font-weight:700;">{get_text('model_explainability')}</div>
                            <p style="font-size:0.82rem; color:#64748b; margin-top: 4px;">Contribution of key parameters to cash shortage likelihood:</p>
                        """, unsafe_allow_html=True)

                        shap_reasons = risk_data.get("shap_reasons", [])
                        if shap_reasons:
                            for sr in shap_reasons:
                                st.markdown(f"""
                                <div style="display:flex; justify-content:space-between; align-items:center; background: rgba(255,255,255,0.03); padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; border: 1px solid rgba(255,255,255,0.05);">
                                    <span style="font-size:0.85rem; font-weight:600;">{sr.split(':')[0]}</span>
                                    <span class="badge-tag {'badge-danger' if '+' in sr else 'badge-safe'}">{sr.split(':')[1].strip()}</span>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("Feature weights balanced for this low-risk agent profile.")
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error(risk_data["error"])
            except Exception as e:
                st.error(f"Error during diagnosis: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: 🎯 AI LIQUIDITY RADAR (MISSION CONTROL)
# ─────────────────────────────────────────────────────────────────────────────
elif active_page_key == "liquidity_radar":
    st.markdown(f"""
    <div class="page-header">
        <h2 class="page-title">🎯 {get_text('nav_ai_liquidity_radar')}</h2>
        <p class="page-subtitle">{get_text('ai_liquidity_radar_subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)

    agent_id = st.selectbox(get_text('radar_select_agent'), options=agents_df["agent_id"].tolist(), key="radar_agent")

    if st.button(get_text('radar_run_prediction')):
        with st.spinner(get_text('radar_analyzing')):
            import requests
            try:
                res = requests.get(f"http://localhost:8000/agent-risk/{agent_id}", timeout=10)
                if res.status_code == 200:
                    risk_res = res.json()
                    
                    agent_data = agents_df[agents_df["agent_id"] == agent_id].iloc[0]
                    predicted_demand = agent_data['cash_balance'] + risk_res['shortage_prediction']
                    risk_color = "#f85149" if risk_res["risk_level"] == "HIGH" else "#d29922" if risk_res["risk_level"] == "MEDIUM" else "#3fb950"
                    
                    reasons_html = ''.join(f'<li>{r}</li>' for r in risk_res['explanation']['analysis'].split(' | '))
                    
                    shap_reasons = risk_res.get("shap_reasons", [])
                    shap_html = ""
                    if shap_reasons:
                        shap_html = "<ul>" + "".join(f'<li>{r}</li>' for r in shap_reasons) + "</ul>"
                    else:
                        shap_html = f"<i>{get_text('radar_no_shap')}</i>"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(22, 27, 34, 0.95) 0%, rgba(13, 17, 23, 0.95) 100%);
                                border: 1px solid #30363d; border-radius: 12px; padding: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.35);">
                        <h3 style="margin-top: 0; color: #58a6ff;">{get_text('radar_agent_label')}: {agent_id}</h3>
                        <hr style="border-color:#30363d;">
                        
                        <div style="display: flex; justify-content: space-between; margin-bottom: 20px; flex-wrap: wrap;">
                            <div style="min-width: 150px; margin-bottom: 10px;">
                                <div style="color: #8b949e; font-size: 0.85rem; text-transform: uppercase;">{get_text('radar_current_cash')}</div>
                                <div style="font-size: 1.5rem; font-weight: bold;">৳{agent_data['cash_balance']:,.0f}</div>
                            </div>
                            <div style="min-width: 150px; margin-bottom: 10px;">
                                <div style="color: #8b949e; font-size: 0.85rem; text-transform: uppercase;">{get_text('radar_risk_level')}</div>
                                <div style="font-size: 1.5rem; font-weight: bold; color: {risk_color};">{risk_res['risk_level']}</div>
                            </div>
                            <div style="min-width: 150px; margin-bottom: 10px;">
                                <div style="color: #8b949e; font-size: 0.85rem; text-transform: uppercase;">{get_text('radar_probability')}</div>
                                <div style="font-size: 1.5rem; font-weight: bold; color: {risk_color};">{risk_res['risk_score']*100:.1f}%</div>
                            </div>
                            <div style="min-width: 150px; margin-bottom: 10px;">
                                <div style="color: #8b949e; font-size: 0.85rem; text-transform: uppercase;">{get_text('radar_expected_shortage')}</div>
                                <div style="font-size: 1.5rem; font-weight: bold; color: #f85149;">৳{risk_res['shortage_prediction']:,.0f}</div>
                            </div>
                        </div>
                        
                        <hr style="border-color:#30363d;">
                        <div style="display: flex; gap: 20px;">
                            <div style="flex: 1;">
                                <div style="color: #8b949e; font-size: 0.85rem; text-transform: uppercase; margin-bottom: 8px;">{get_text('radar_main_reasons')}</div>
                                <ul style="margin: 0; padding-left: 20px; line-height: 1.6;">
                                    {reasons_html}
                                </ul>
                            </div>
                            <div style="flex: 1;">
                                <div style="color: #8b949e; font-size: 0.85rem; text-transform: uppercase; margin-bottom: 8px;">{get_text('radar_shap_explanation')}</div>
                                {shap_html}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"### 🤝 {get_text('radar_recommended_partners')}")
                    
                    partners_res = requests.get(f"http://localhost:8000/liquidity-partners/{agent_id}", timeout=10)
                    if partners_res.status_code == 200:
                        recs = partners_res.json().get("recommended_agents", [])
                        if recs:
                            df_recs = pd.DataFrame(recs)
                            # Rename columns for display
                            df_display = df_recs.rename(columns={
                                "partner_id": get_text('radar_partner_agent_id'),
                                "score": get_text('radar_partner_score'),
                                "distance_km": get_text('radar_distance'),
                                "available_liquidity": get_text('radar_available_cash'),
                                "rating": get_text('radar_reliability'),
                                "explanation": get_text('radar_recommendation_reason')
                            })
                            # Keep only the requested columns
                            df_display = df_display[[get_text('radar_partner_agent_id'), get_text('radar_partner_score'), get_text('radar_distance'), get_text('radar_available_cash'), get_text('radar_reliability'), get_text('radar_recommendation_reason')]]
                            st.dataframe(df_display, use_container_width=True, hide_index=True)
                            
                            best_partner = recs[0]["partner_id"]
                            st.markdown(f"""
                            <div style="background: rgba(88, 166, 255, 0.1); border-left: 4px solid #58a6ff; padding: 16px; border-radius: 4px; margin-top: 10px;">
                                <div style="color: #58a6ff; font-weight: bold; font-size: 1.1rem; margin-bottom: 4px;">{get_text('radar_recommended_action')}</div>
                                {get_text('radar_request_support')} {best_partner}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info(get_text('radar_no_partners'))
                    else:
                        st.error(f"{get_text('radar_failed_fetch')} {partners_res.status_code}")
                        
                else:
                    st.error(f"{get_text('radar_backend_error')} {res.status_code}")
            except Exception as e:
                st.error(f"{get_text('radar_connection_failed')} {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: 🏪 MERCHANT GROWTH & CUSTOMER OFFERS HUB
# ─────────────────────────────────────────────────────────────────────────────
elif active_page_key == "merchant_offers":
    st.markdown(f"""
    <div class="page-banner">
        <div>
            <h1 class="page-banner-title">{get_text('merchant_offers_title')}</h1>
            <p class="page-banner-subtitle">{get_text('merchant_offers_subtitle')}</p>
        </div>
        <div>
            <span class="badge-tag badge-ai">OFFER ENGINE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_merchant, tab_customer = st.tabs([
        "🏪 Merchant Sales Acceleration Engine",
        "🎁 Hyper-Local Customer Reward Simulator"
    ])

    # Tab 1: Merchant Growth Engine
    with tab_merchant:
        st.markdown("#### 🚀 AI Merchant Revenue Optimization")
        st.caption("Generate tailored digital payment discount offers and identify optimal promotional time windows.")

        m_col1, m_col2 = st.columns([1, 2])

        with m_col1:
            selected_merchant_id = st.selectbox(
                "Select Retail Merchant",
                options=merchants_df["merchant_id"].tolist(),
                index=0
            )
            merchant_row = merchants_df[merchants_df["merchant_id"] == selected_merchant_id].iloc[0]

            st.markdown(f"""
            <div class="glass-box" style="margin-top: 14px;">
                <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase;">Merchant Profile</div>
                <h3 style="margin:4px 0 2px 0; color:#38bdf8;">{merchant_row['merchant_id']}</h3>
                <div style="font-size:0.88rem; color:#cbd5e1; margin-bottom:10px;"><b>{merchant_row['category']}</b></div>
                <div style="font-size:0.82rem; color:#94a3b8; line-height:1.6;">
                    📍 <b>Hub:</b> {merchant_row['area']}<br>
                    ⚡ <b>Daily Txns:</b> {merchant_row['daily_transactions']:,}<br>
                    🌐 <b>Digital QR:</b> Active (Upay QR)
                </div>
            </div>
            """, unsafe_allow_html=True)

        with m_col2:
            m_rec = st.session_state.merchant_engine.generate_recommendation(selected_merchant_id)

            if "error" not in m_rec:
                rec_data = m_rec["recommendation"]
                st.markdown(f"""
                <div class="glass-box" style="border-left: 4px solid #38bdf8; margin-top: 14px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="badge-tag badge-safe">AI RECOMMENDATION ACTIVE</span>
                        <span class="badge-tag badge-info">GROWTH SURGE</span>
                    </div>

                    <h3 style="color:#ffffff; margin: 12px 0 6px 0;">🎯 {rec_data['offer']}</h3>
                    <p style="color:#94a3b8; font-size:0.88rem; line-height:1.5;">{rec_data['reason']}</p>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 16px;">
                        <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                            <div style="font-size:0.72rem; color:#64748b; text-transform:uppercase;">Optimal Launch Window</div>
                            <div style="font-size:1.05rem; font-weight:700; color:#fbbf24; margin-top:2px;">
                                🕒 {rec_data['best_time']}
                            </div>
                        </div>
                        <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                            <div style="font-size:0.72rem; color:#64748b; text-transform:uppercase;">Target Audience</div>
                            <div style="font-size:1.05rem; font-weight:700; color:#38bdf8; margin-top:2px;">
                                👥 {rec_data['target_segment']}
                            </div>
                        </div>
                    </div>

                    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 12px; margin-top: 14px; display:flex; justify-content:space-between; align-items:center;">
                        <span style="color:#6ee7b7; font-size:0.85rem; font-weight:600;">Expected Digital Volume Lift:</span>
                        <b style="color:#34d399; font-size:1.15rem; font-family:'JetBrains Mono', monospace;">{rec_data['expected_impact']}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(m_rec["error"])

    # Tab 2: Customer Offers Engine
    with tab_customer:
        st.markdown("#### 🎁 Proximity Customer Discount Engine")
        st.caption("Personalized digital micro-offers delivered to users based on real-time location and spend preference.")

        c_col1, c_col2 = st.columns([1, 2])

        with c_col1:
            st.markdown("**Simulate Customer Context**")
            customer_area = st.selectbox("Customer Current Location", options=sorted(merchants_df["area"].unique().tolist()), index=0)
            target_cat = st.selectbox("Preferred Shopping Category", ["Pharmacy & Healthcare", "Restaurant & Fast Food", "Grocery & Superstore", "Any"])

            # Approximate coordinates from the selected area's first merchant
            area_merchants = merchants_df[merchants_df["area"] == customer_area]
            sim_lat = area_merchants["latitude"].mean() if not area_merchants.empty else 23.788
            sim_lon = area_merchants["longitude"].mean() if not area_merchants.empty else 90.405

        with c_col2:
            pref = None if target_cat == "Any" else target_cat
            cust_res = st.session_state.customer_engine.generate_offer("CUST-LIVE", sim_lat, sim_lon, preferred_category=pref)

            if "error" not in cust_res:
                rec_m = cust_res["recommended_merchant"]
                st.markdown(f"""
                <div class="glass-box" style="border: 1px solid rgba(139, 92, 246, 0.4); background: linear-gradient(135deg, rgba(22, 18, 38, 0.85) 0%, rgba(13, 17, 23, 0.95) 100%);">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="badge-tag badge-ai">NEARBY EXCLUSIVE OFFER</span>
                        <span style="font-size:0.80rem; color:#a78bfa; font-family: monospace;">📍 {rec_m['distance']} away</span>
                    </div>

                    <h3 style="color:#ffffff; margin: 12px 0 4px 0;">🎉 {rec_m['offer']}</h3>
                    <div style="color:#c084fc; font-weight:600; font-size:0.92rem; margin-bottom:8px;">
                        Merchant: {rec_m['merchant_id']} &nbsp;|&nbsp; Category: {rec_m['category']}
                    </div>
                    <p style="color:#94a3b8; font-size:0.85rem; line-height:1.5;">{rec_m['reason']}</p>

                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px 14px; margin-top: 12px; font-size: 0.82rem; color: #cbd5e1;">
                        💡 <b>Expected Platform Impact:</b> {rec_m['expected_benefit']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(cust_res["error"])


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6: 🤖 UPAYPULSE AI ASSISTANT (CHATBOT)
# ─────────────────────────────────────────────────────────────────────────────
elif active_page_key == "ai_chatbot":
    st.markdown(f"""
    <div class="page-banner">
        <div>
            <h1 class="page-banner-title">{get_text('chatbot_title')}</h1>
            <p class="page-banner-subtitle">{get_text('chatbot_subtitle')}</p>
        </div>
        <div>
            <span class="live-pill"><span class="pulse-dot"></span> MULTI-AGENT ORCHESTRATOR</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": get_text('chatbot_greeting')}
        ]

    # Quick Prompt Chips
    st.markdown("<div style='margin-bottom: 12px;'>", unsafe_allow_html=True)
    chip1, chip2, chip3, chip4 = st.columns(4)

    prompt_to_send = None
    with chip1:
        if st.button("🚨 Check Cash Shortage (AGT-0435)", use_container_width=True):
            prompt_to_send = "Which agents have liquidity risk? Check AGT-0435"
    with chip2:
        if st.button("🏪 Merchant Promo (MRC-0001)", use_container_width=True):
            prompt_to_send = "How can merchant MRC-0001 increase sales?"
    with chip3:
        if st.button("🎁 Nearest Pharmacy Offer", use_container_width=True):
            prompt_to_send = "Where can I find a pharmacy discount offer?"
    with chip4:
        if st.button("🇧🇩 বাংলাতে ঝুঁকি জানতে চাই", use_container_width=True):
            prompt_to_send = "AGT-0002 এর ক্যাশ সংকট ঝুঁকি কেমন?"

    st.markdown("</div>", unsafe_allow_html=True)

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    # Chat Input
    user_query = st.chat_input(get_text('type_question')) or prompt_to_send

    if user_query:
        # Add user query to history
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query, unsafe_allow_html=True)

        # Generate response using Backend or In-Memory Orchestrator fallback
        with st.chat_message("assistant"):
            with st.spinner(get_text('thinking')):
                bot_response = None
                try:
                    import requests
                    res = requests.post(
                        "http://localhost:8000/chat",
                        json={"query": user_query, "lang": st.session_state.lang},
                        timeout=3
                    )
                    if res.status_code == 200:
                        bot_response = res.json().get("response")
                except Exception:
                    # Seamless local fallback
                    bot_response = st.session_state.orchestrator.handle_query(
                        user_query,
                        lang=st.session_state.lang
                    )

                if not bot_response:
                    bot_response = st.session_state.orchestrator.handle_query(
                        user_query,
                        lang=st.session_state.lang
                    )

                st.markdown(bot_response, unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": bot_response})


# ─────────────────────────────────────────────────────────────────────────────
# 4. LUXURY FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<hr style='border-color: rgba(255,255,255,0.08); margin-top: 50px;'>
<div style='display: flex; justify-content: space-between; align-items: center; color: #64748b; font-size: 0.80rem; padding: 12px 0 24px 0; flex-wrap: wrap;'>
    <div>
        💸 <b style='color: #38bdf8;'>UpayPulse AI</b> &nbsp;|&nbsp;
        {get_text('footer')}
    </div>
    <div style='font-family: monospace;'>
        NETWORK STATUS: <span style='color: #34d399;'>ONLINE</span> &nbsp;|&nbsp;
        BUILD: <span style='color: #94a3b8;'>v2.5.0</span>
    </div>
</div>
""", unsafe_allow_html=True)
