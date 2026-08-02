import asyncio

# --- EVENT LOOP FIX FÜR STREAMLIT / PYTHON 3.10+ ---
try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, date, timedelta

# Versuche ib_async oder ib_insync zu importieren
try:
    from ib_async import IB, Option, util
    IBKR_AVAILABLE = True
except ImportError:
    try:
        from ib_insync import IB, Option, util
        IBKR_AVAILABLE = True
    except ImportError:
        IBKR_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="OptionNet Explorer - Chain & DTE Comparison", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
        font-weight: bold;
    }
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- US MARKET HOLIDAYS (2025 - 2026) ---
US_HOLIDAYS_2025_2026 = {
    # 2025
    date(2025, 1, 1): "New Year's Day (Closed)",
    date(2025, 1, 20): "Martin Luther King Jr. Day (Closed)",
    date(2025, 2, 17): "Presidents' Day (Closed)",
    date(2025, 4, 18): "Good Friday (Closed)",
    date(2025, 5, 26): "Memorial Day (Closed)",
    date(2025, 6, 19): "Juneteenth (Closed)",
    date(2025, 7, 3): "Early Close (1:00 PM ET)",
    date(2025, 7, 4): "Independence Day (Closed)",
    date(2025, 9, 1): "Labor Day (Closed)",
    date(2025, 11, 27): "Thanksgiving Day (Closed)",
    date(2025, 11, 28): "Early Close (1:00 PM ET)",
    date(2025, 12, 24): "Early Close (1:00 PM ET)",
    date(2025, 12, 25): "Christmas Day (Closed)",
    # 2026
    date(2026, 1, 1): "New Year's Day (Closed)",
    date(2026, 1, 19): "Martin Luther King Jr. Day (Closed)",
    date(2026, 2, 16): "Presidents' Day (Closed)",
    date(2026, 4, 3): "Good Friday (Closed)",
    date(2026, 5, 25): "Memorial Day (Closed)",
    date(2026, 6, 19): "Juneteenth (Closed)",
    date(2026, 7, 3): "Independence Day (Closed)",
    date(2026, 9, 7): "Labor Day (Closed)",
    date(2026, 11, 26): "Thanksgiving Day (Closed)",
    date(2026, 11, 27): "Early Close (1:00 PM ET)",
    date(2026, 12, 24): "Early Close (1:00 PM ET)",
    date(2026, 12, 25): "Christmas Day (Closed)"
}

def get_market_status(target_date):
    if target_date in US_HOLIDAYS_2025_2026:
        return f"🔴 {US_HOLIDAYS_2025_2026[target_date]}"
    elif target_date.weekday() >= 5:
        return "🟡 Weekend (Closed)"
    else:
        return "🟢 Open"

# --- BLACK-SCHOLES & GREEKS HELPER FUNCTIONS ---
def bs_price(option_type, S, K, T, r, sigma):
    if T <= 0.00001:
        return np.maximum(0.0, S - K) if option_type == 'C' else np.maximum(0.0, K - S)
            
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'C':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_greeks(option_type, S, K, T, r, sigma):
    if T <= 0.00001:
        return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    
    if option_type == 'C':
        delta = norm.cdf(d1)
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:
        delta = -norm.cdf(-d1)
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
        
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = (S * pdf_d1 * np.sqrt(T)) / 100.0
    
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}

# --- OPTION CHAIN GENERATOR (WITH VOL SKEW) ---
def build_option_chain(spot, dte, base_iv, r=0.045, strike_step=5, num_strikes=15):
    T = max(0.00001, dte / 365.0)
    atm_strike = round(spot / strike_step) * strike_step
    half_strikes = num_strikes // 2
    
    strikes = [atm_strike + i * strike_step for i in range(-half_strikes, half_strikes + 1)]
    chain = []
    
    for K in strikes:
        # Modellierung von Volatility Skew (Put Skew typisch höher für Indizes)
        moneyness = np.log(K / spot)
        skew_iv = base_iv - (moneyness * 15.0) + (moneyness**2 * 25.0)
        sigma = max(0.05, skew_iv / 100.0)
        
        c_price = bs_price('C', spot, K, T, r, sigma)
        c_greeks = bs_greeks('C', spot, K, T, r, sigma)
        
        p_price = bs_price('P', spot, K, T, r, sigma)
        p_greeks = bs_greeks('P', spot, K, T, r, sigma)
        
        chain.append({
            "Call Delta": round(c_greeks['delta'], 2),
            "Call Price ($)": round(c_price, 2),
            "Call IV (%)": round(skew_iv, 2),
            "Strike": float(K),
            "Put IV (%)": round(skew_iv, 2),
            "Put Price ($)": round(p_price, 2),
            "Put Delta": round(p_greeks['delta'], 2)
        })
        
    return pd.DataFrame(chain)

# --- LIVE DATA FETCHER ---
TICKER_MAP = {
    "SPX": "^GSPC",
    "DAX": "^GDAXI",
    "SPY": "SPY",
    "QQQ": "QQQ",
    "RUT": "^RUT"
}

@st.cache_data(ttl=300)
def fetch_delayed_spot(ticker_symbol):
    try:
        t = yf.Ticker(ticker_symbol)
        data = t.history(period="1d", interval="1m")
        if not data.empty:
            return round(float(data["Close"].iloc[-1]), 2)
    except Exception:
        pass
    return None

# --- SIDEBAR CONTROLS ---
st.sidebar.title("⚙️ Base & Data Settings")

underlying_symbol = st.sidebar.selectbox("Underlying Symbol", list(TICKER_MAP.keys()), index=0)
ticker = TICKER_MAP[underlying_symbol]

live_spot = fetch_delayed_spot(ticker)
default_spot = live_spot if live_spot is not None else 600.0

spot_price = st.sidebar.number_input(
    f"Current Spot Price ({'Live/Delayed' if live_spot else 'Manual'})", 
    value=default_spot, 
    step=1.0
)

base_iv = st.sidebar.number_input("Base IV (%)", value=18.0, step=0.5)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=4.5, step=0.1) / 100.0

def get_default_legs(current_spot, current_iv):
    return [
        {"Enable": True, "Phase": "Initial", "Type": "C", "Strike": round(current_spot, -1), "DTE": 30, "IV_%": current_iv, "Qty": 10, "Entry_Price": round(current_spot * 0.02, 2)},
        {"Enable": True, "Phase": "Initial", "Type": "P", "Strike": round(current_spot * 0.95, -1), "DTE": 30, "IV_%": current_iv, "Qty": -10, "Entry_Price": round(current_spot * 0.01, 2)},
    ]

if "last_symbol" not in st.session_state:
    st.session_state["last_symbol"] = underlying_symbol
    st.session_state["legs_df"] = pd.DataFrame(get_default_legs(spot_price, base_iv))

if st.session_state["last_symbol"] != underlying_symbol:
    st.session_state["last_symbol"] = underlying_symbol
    st.session_state["legs_df"] = pd.DataFrame(get_default_legs(spot_price, base_iv))
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Portfolio Positions Manager")

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("🔄 Sync/Reset"):
    st.session_state["legs_df"] = pd.DataFrame(get_default_legs(spot_price, base_iv))
    st.rerun()

if col_btn2.button("🗑️ Clear All"):
    st.session_state["legs_df"] = pd.DataFrame(columns=["Enable", "Phase", "Type", "Strike", "DTE", "IV_%", "Qty", "Entry_Price"])
    st.rerun()

edited_df = st.sidebar.data_editor(
    st.session_state["legs_df"],
    num_rows="dynamic",
    key="portfolio_editor",
    column_config={
        "Enable": st.column_config.CheckboxColumn("Aktiv", default=True),
        "Phase": st.column_config.SelectboxColumn("Phase", options=["Initial", "Adjustment"], default="Initial"),
        "Type": st.column_config.SelectboxColumn("Typ", options=["C", "P"], required=True),
        "Strike": st.column_config.NumberColumn("Strike", min_value=1.0, step=5.0),
        "DTE": st.column_config.NumberColumn("DTE", min_value=0, step=1),
        "IV_%": st.column_config.NumberColumn("IV %", min_value=1.0, max_value=300.0, step=0.5),
        "Qty": st.column_config.NumberColumn("Qty (+/-)", step=1),
        "Entry_Price": st.column_config.NumberColumn("Entry ($)", step=0.05),
    }
)

st.session_state["legs_df"] = edited_df

# --- MAIN DASHBOARD INTERFACE ---
st.title(f"📈 OptionNet Explorer - Option Chain & DTE Comparison ({underlying_symbol})")

tab1, tab2 = st.tabs(["⚔️ DTE Comparison Matrix (z.B. DTE 15 vs DTE 22)", "🔍 Single Option Chain Explorer"])

# --- TAB 1: DTE COMPARISON MATRIX ---
with tab1:
    st.subheader("📊 Direktvergleich zweier Verfallstage (DTE A vs. DTE B)")
    
    col_a, col_b = st.columns(2)
    with col_a:
        dte_a = st.number_input("Wähle DTE A", value=15, min_value=0, max_value=120, step=1)
        dt_a = date.today() + timedelta(days=int(dte_a))
        st.caption(f"Verfall Datum A: `{dt_a.strftime('%Y-%m-%d (%A)')}` | Status: {get_market_status(dt_a)}")
        
    with col_b:
        dte_b = st.number_input("Wähle DTE B", value=22, min_value=0, max_value=120, step=1)
        dt_b = date.today() + timedelta(days=int(dte_b))
        st.caption(f"Verfall Datum B: `{dt_b.strftime('%Y-%m-%d (%A)')}` | Status: {get_market_status(dt_b)}")

    chain_a = build_option_chain(spot_price, dte_a, base_iv, r=risk_free_rate)
    chain_b = build_option_chain(spot_price, dte_b, base_iv, r=risk_free_rate)
    
    # Merge für direkten Vergleich
    comp_df = pd.merge(
        chain_a[["Strike", "Put IV (%)", "Put Price ($)", "Call Price ($)", "Call IV (%)"]],
        chain_b[["Strike", "Put IV (%)", "Put Price ($)", "Call Price ($)", "Call IV (%)"]],
        on="Strike",
        suffixes=(f" (DTE {dte_a})", f" (DTE {dte_b})")
    )
    
    # IV Diff Spalten hinzufügen
    comp_df["Put IV Diff (%)"] = round(comp_df[f"Put IV (%) (DTE {dte_b})"] - comp_df[f"Put IV (%) (DTE {dte_a})"], 2)
    comp_df["Call IV Diff (%)"] = round(comp_df[f"Call IV (%) (DTE {dte_b})"] - comp_df[f"Call IV (%) (DTE {dte_a})"], 2)
    
    st.markdown(f"#### Vergleichstabelle Spot: `${spot_price}`")
    st.dataframe(comp_df, use_container_width=True, height=450)

# --- TAB 2: SINGLE OPTION CHAIN EXPLORER ---
with tab2:
    st.subheader("🔍 Option Chain Explorer")
    selected_dte = st.slider("DTE auswählen", 0, 60, value=30)
    
    single_chain = build_option_chain(spot_price, selected_dte, base_iv, r=risk_free_rate, num_strikes=21)
    
    st.dataframe(
        single_chain,
        use_container_width=True,
        height=550
    )
