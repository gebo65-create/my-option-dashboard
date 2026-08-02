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
    page_title="OptionNet Explorer - Full Chain, Risk & Margin", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
        font-weight: bold;
    }
    .main .block-container {
        padding-top: 1rem;
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

# --- BLACK-SCHOLES & GREEKS ---
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

# --- MARGIN CALCULATION HELPER ---
def calculate_phase_margin(legs_subset, spot_price):
    if legs_subset.empty:
        return 0.0
    
    total_margin = 0.0
    for _, row in legs_subset.iterrows():
        try:
            qty = float(row["Qty"])
            strike = float(row["Strike"])
            opt_type = str(row["Type"])
            entry = float(row["Entry_Price"])
            
            # Nur Short-Positionen erfordern explizite Margin nach Standard-Regeln
            if qty < 0:
                abs_qty = abs(qty)
                if opt_type == 'P':
                    otm_amount = max(0.0, spot_price - strike)
                    reg_t_margin = (0.20 * spot_price - otm_amount + entry) * 100.0 * abs_qty
                    min_margin = (0.10 * strike + entry) * 100.0 * abs_qty
                    leg_margin = max(reg_t_margin, min_margin)
                else: # Call
                    otm_amount = max(0.0, strike - spot_price)
                    reg_t_margin = (0.20 * spot_price - otm_amount + entry) * 100.0 * abs_qty
                    min_margin = (0.10 * spot_price + entry) * 100.0 * abs_qty
                    leg_margin = max(reg_t_margin, min_margin)
                total_margin += leg_margin
            else:
                # Long Option: Margin ist die gezahlte Prämie
                total_margin += (entry * 100.0 * qty)
        except (ValueError, KeyError):
            continue
            
    return round(total_margin, 2)

# --- OPTION CHAIN BUILDER ---
def build_classic_option_chain(spot, dte, base_iv, r=0.045, strike_step=5, num_strikes=17):
    T = max(0.00001, dte / 365.0)
    atm_strike = round(spot / strike_step) * strike_step
    half_strikes = num_strikes // 2
    
    strikes = [atm_strike + i * strike_step for i in range(-half_strikes, half_strikes + 1)]
    chain = []
    
    for K in strikes:
        moneyness = np.log(K / spot)
        skew_iv = base_iv - (moneyness * 12.0) + (moneyness**2 * 20.0)
        sigma = max(0.05, skew_iv / 100.0)
        
        c_price = bs_price('C', spot, K, T, r, sigma)
        c_greeks = bs_greeks('C', spot, K, T, r, sigma)
        
        p_price = bs_price('P', spot, K, T, r, sigma)
        p_greeks = bs_greeks('P', spot, K, T, r, sigma)
        
        chain.append({
            "C_Theta": round(c_greeks['theta'], 2),
            "C_Gamma": round(c_greeks['gamma'], 4),
            "C_Delta": round(c_greeks['delta'], 2),
            "C_IV_%": round(skew_iv, 1),
            "C_Bid": round(c_price * 0.98, 2),
            "C_Ask": round(c_price * 1.02, 2),
            "STRIKE": float(K),
            "P_Ask": round(p_price * 1.02, 2),
            "P_Bid": round(p_price * 0.98, 2),
            "P_IV_%": round(skew_iv, 1),
            "P_Delta": round(p_greeks['delta'], 2),
            "P_Gamma": round(p_greeks['gamma'], 4),
            "P_Theta": round(p_greeks['theta'], 2),
        })
        
    return pd.DataFrame(chain)

# --- LIVE SPOT DATA ---
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

# --- SIDEBAR & PORTFOLIO ---
st.sidebar.title("⚙️ Base Settings & Positions")

underlying_symbol = st.sidebar.selectbox("Underlying Symbol", list(TICKER_MAP.keys()), index=0)
ticker = TICKER_MAP[underlying_symbol]

live_spot = fetch_delayed_spot(ticker)
default_spot = live_spot if live_spot is not None else 600.0

spot_price = st.sidebar.number_input(
    f"Spot Price ({'Live/Delayed' if live_spot else 'Manual'})", 
    value=default_spot, 
    step=1.0
)

base_iv = st.sidebar.number_input("Base IV (%)", value=18.0, step=0.5)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=4.5, step=0.1) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Display Options")
show_margin = st.sidebar.checkbox("Margin Details anzeigen", value=True)

def get_default_legs(current_spot, current_iv):
    return [
        {"Enable": True, "Phase": "Initial", "Type": "C", "Strike": round(current_spot, -1), "DTE": 30, "IV_%": current_iv, "Qty": 10, "Entry_Price": round(current_spot * 0.02, 2)},
        {"Enable": True, "Phase": "Initial", "Type": "P", "Strike": round(current_spot * 0.95, -1), "DTE": 30, "IV_%": current_iv, "Qty": -10, "Entry_Price": round(current_spot * 0.01, 2)},
        {"Enable": True, "Phase": "Adjustment", "Type": "P", "Strike": round(current_spot * 0.90, -1), "DTE": 20, "IV_%": current_iv + 2, "Qty": -5, "Entry_Price": round(current_spot * 0.008, 2)},
    ]

if "last_symbol" not in st.session_state:
    st.session_state["last_symbol"] = underlying_symbol
    st.session_state["legs_df"] = pd.DataFrame(get_default_legs(spot_price, base_iv))

if st.session_state["last_symbol"] != underlying_symbol:
    st.session_state["last_symbol"] = underlying_symbol
    st.session_state["legs_df"] = pd.DataFrame(get_default_legs(spot_price, base_iv))
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Portfolio Manager")

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("🔄 Reset Portfolio"):
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

# --- PNL & RISK CALCULATION ---
active_legs = edited_df[edited_df["Enable"] == True].copy() if not edited_df.empty else pd.DataFrame()

# Separate Phasen für Margin
initial_legs = active_legs[active_legs["Phase"] == "Initial"] if not active_legs.empty else pd.DataFrame()
adj_legs = active_legs[active_legs["Phase"] == "Adjustment"] if not active_legs.empty else pd.DataFrame()

margin_initial = calculate_phase_margin(initial_legs, spot_price)
margin_adj = calculate_phase_margin(adj_legs, spot_price)
margin_total = calculate_phase_margin(active_legs, spot_price)

if not active_legs.empty and "Strike" in active_legs and len(active_legs["Strike"].dropna()) > 0:
    min_strike = active_legs["Strike"].min()
    max_strike = active_legs["Strike"].max()
    x_min = min(spot_price * 0.85, min_strike * 0.90)
    x_max = max(spot_price * 1.15, max_strike * 1.10)
else:
    x_min = spot_price * 0.85
    x_max = spot_price * 1.15

spot_range = np.linspace(x_min, x_max, 400)

pnl_t0 = np.zeros_like(spot_range)
pnl_t1 = np.zeros_like(spot_range)
pnl_exp = np.zeros_like(spot_range)

spot_pnl_t0, spot_pnl_t1 = 0.0, 0.0
tot_delta, tot_gamma, tot_theta, tot_vega = 0.0, 0.0, 0.0, 0.0

if not active_legs.empty:
    for _, row in active_legs.iterrows():
        try:
            opt_type = str(row["Type"])
            strike = float(row["Strike"])
            dte = float(row["DTE"])
            iv = float(row["IV_%"]) / 100.0
            qty = float(row["Qty"])
            entry = float(row["Entry_Price"])
            
            t_t0 = max(0.00001, dte / 365.0)
            t_t1 = max(0.00001, max(0.0, dte - 1) / 365.0)
            
            exp_prices = np.where(opt_type == 'C', np.maximum(0, spot_range - strike), np.maximum(0, strike - spot_range))
            pnl_exp += (exp_prices - entry) * qty * 100.0
            
            t0_prices = np.array([bs_price(opt_type, s, strike, t_t0, risk_free_rate, iv) for s in spot_range])
            pnl_t0 += (t0_prices - entry) * qty * 100.0
            
            t1_prices = np.array([bs_price(opt_type, s, strike, t_t1, risk_free_rate, iv) for s in spot_range])
            pnl_t1 += (t1_prices - entry) * qty * 100.0
            
            val_t0 = bs_price(opt_type, spot_price, strike, t_t0, risk_free_rate, iv)
            val_t1 = bs_price(opt_type, spot_price, strike, t_t1, risk_free_rate, iv)
            spot_pnl_t0 += (val_t0 - entry) * qty * 100.0
            spot_pnl_t1 += (val_t1 - entry) * qty * 100.0
            
            g = bs_greeks(opt_type, spot_price, strike, t_t0, risk_free_rate, iv)
            tot_delta += g['delta'] * qty * 100.0
            tot_gamma += g['gamma'] * qty * 100.0
            tot_theta += g['theta'] * qty * 100.0
            tot_vega += g['vega'] * qty * 100.0
        except (ValueError, KeyError):
            continue

# --- MAIN WORKSPACE ---
st.title(f"📈 OptionNet Explorer Professional ({underlying_symbol})")

# Dynamische KPI Spaltenanzahl je nach Margin Toggle
if show_margin:
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("PnL T+0 (Heute)", f"${spot_pnl_t0:,.2f}")
    m2.metric("PnL T+1 (Morgen)", f"${spot_pnl_t1:,.2f}", f"${(spot_pnl_t1 - spot_pnl_t0):,.2f}")
    m3.metric("Position Delta", f"{tot_delta:,.2f}")
    m4.metric("Position Theta", f"${tot_theta:,.2f}")
    m5.metric("Position Vega", f"{tot_vega:,.2f}")
    m6.metric("Gesamt Margin", f"${margin_total:,.2f}")
else:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("PnL T+0 (Heute)", f"${spot_pnl_t0:,.2f}")
    m2.metric("PnL T+1 (Morgen)", f"${spot_pnl_t1:,.2f}", f"${(spot_pnl_t1 - spot_pnl_t0):,.2f}")
    m3.metric("Position Delta", f"{tot_delta:,.2f}")
    m4.metric("Position Theta", f"${tot_theta:,.2f}")
    m5.metric("Position Vega", f"{tot_vega:,.2f}")

st.markdown("---")

# SECTION 1: MARGIN OVERVIEW PANEL (NUR WENN EIN GESCHALTET)
if show_margin:
    st.subheader("💵 Margin Breakdown (Original vs. Adjustiert)")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    col_m1.metric("Initial Margin (Original)", f"${margin_initial:,.2f}")
    col_m2.metric("Adjustment Margin", f"${margin_adj:,.2f}")
    col_m3.metric("Gesamt Margin", f"${margin_total:,.2f}")
    
    # Return on Margin Berechnen
    rom_t0 = (spot_pnl_t0 / margin_total * 100.0) if margin_total > 0 else 0.0
    col_m4.metric("Return on Margin (T+0)", f"{rom_t0:.2f}%")
    
    st.markdown("---")

# SECTION 2: RISK GRAPH
st.subheader("📉 Risk Profile Graph")

fig = go.Figure()
fig.add_trace(go.Scatter(x=spot_range, y=pnl_t0, mode='lines', name='T+0 (Heute)', line=dict(color='#ff5252', width=3)))
fig.add_trace(go.Scatter(x=spot_range, y=pnl_t1, mode='lines', name='T+1 (Morgen)', line=dict(color='#66bb6a', width=2, dash='dash')))
fig.add_trace(go.Scatter(x=spot_range, y=pnl_exp, mode='lines', name='Expiration PnL', line=dict(color='#29b6f6', width=1.5)))

fig.add_hline(y=0, line_dash="solid", line_color="#555555", line_width=1)
fig.add_vline(x=spot_price, line_dash="dot", line_color="#ffffff", line_width=1.5)

fig.update_layout(
    xaxis_title="Underlying Price",
    yaxis_title="Profit / Loss ($)",
    template="plotly_dark",
    height=420,
    hovermode="x unified",
    margin=dict(l=20, r=20, t=30, b=20)
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# SECTION 3: CLASSIC OPTION CHAIN & DTE COMPARISON
st.subheader("⛓️ Option Chain & Multi-DTE Analysis")

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    selected_dte_1 = st.number_input("DTE A (z.B. Primary)", value=15, min_value=0, max_value=120)
    dt1 = date.today() + timedelta(days=int(selected_dte_1))
    st.caption(f"Datum A: `{dt1.strftime('%Y-%m-%d')}` | Status: {get_market_status(dt1)}")

with c2:
    selected_dte_2 = st.number_input("DTE B (z.B. Vergleich)", value=22, min_value=0, max_value=120)
    dt2 = date.today() + timedelta(days=int(selected_dte_2))
    st.caption(f"Datum B: `{dt2.strftime('%Y-%m-%d')}` | Status: {get_market_status(dt2)}")

with c3:
    num_strikes_view = st.slider("Anzahl Strikes anzeigen", 11, 31, 17, step=2)

chain_df_1 = build_classic_option_chain(spot_price, selected_dte_1, base_iv, r=risk_free_rate, num_strikes=num_strikes_view)
chain_df_2 = build_classic_option_chain(spot_price, selected_dte_2, base_iv, r=risk_free_rate, num_strikes=num_strikes_view)

tab_chain1, tab_comp = st.tabs([f"🏛️ Option Chain (DTE {selected_dte_1})", f"⚔️ DTE {selected_dte_1} vs DTE {selected_dte_2} Matrix"])

with tab_chain1:
    st.markdown(f"**Option Chain Layout (CALLS | STRIKE | PUTS) - Spot @ `{spot_price}`**")
    st.dataframe(
        chain_df_1,
        use_container_width=True,
        height=480
    )

with tab_comp:
    st.markdown(f"**Direktvergleich: DTE {selected_dte_1} vs. DTE {selected_dte_2}**")
    
    comp_merged = pd.merge(
        chain_df_1[["STRIKE", "C_IV_%", "C_Ask", "P_IV_%", "P_Ask"]],
        chain_df_2[["STRIKE", "C_IV_%", "C_Ask", "P_IV_%", "P_Ask"]],
        on="STRIKE",
        suffixes=(f" (DTE {selected_dte_1})", f" (DTE {selected_dte_2})")
    )
    
    comp_merged["Call IV Diff (%)"] = round(comp_merged[f"C_IV_% (DTE {selected_dte_2})"] - comp_merged[f"C_IV_% (DTE {selected_dte_1})"], 1)
    comp_merged["Put IV Diff (%)"] = round(comp_merged[f"P_IV_% (DTE {selected_dte_2})"] - comp_merged[f"P_IV_% (DTE {selected_dte_1})"], 1)
    comp_merged["Call Price Diff ($)"] = round(comp_merged[f"C_Ask (DTE {selected_dte_2})"] - comp_merged[f"C_Ask (DTE {selected_dte_1})"], 2)
    comp_merged["Put Price Diff ($)"] = round(comp_merged[f"P_Ask (DTE {selected_dte_2})"] - comp_merged[f"P_Ask (DTE {selected_dte_1})"], 2)

    st.dataframe(
        comp_merged,
        use_container_width=True,
        height=480
    )
