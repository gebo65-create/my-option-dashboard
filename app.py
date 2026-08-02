import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

# Page Configuration
st.set_page_config(
    page_title="OptionNet Explorer - Risk & PnL Analytics", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom Styling for ONE UI Feel
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: bold;
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- BLACK-SCHOLES PRICING ENGINE ---
def bs_price(option_type, S, K, T, r, sigma):
    """Calculates Black-Scholes Option Price."""
    if T <= 0.00001:
        if option_type == 'C':
            return np.maximum(0.0, S - K)
        else:
            return np.maximum(0.0, K - S)
            
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'C':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
    return price

def bs_greeks(option_type, S, K, T, r, sigma):
    """Calculates Option Greeks."""
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
    vega = (S * pdf_d1 * np.sqrt(T)) / 100.0  # Per 1% IV change
    
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}


# --- SIDEBAR: CONTROLS & TRADES ---
st.sidebar.title("⚙️ Base Settings")

underlying_symbol = st.sidebar.selectbox("Underlying", ["SPX", "SPY", "QQQ", "DAX", "RUT"], index=1)
spot_price = st.sidebar.number_input("Current Spot Price", value=600.0, step=1.0)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=4.5, step=0.1) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Time Travel (T+X Days)")
days_forward = st.sidebar.slider("Days Elapsed (Forward in Time)", min_value=0, max_value=30, value=0, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Trade Legs Configuration")

# Korrekter Iron Condor (Sell 580P/620C, Buy 560P/640C)
default_legs = pd.DataFrame([
    {"Type": "P", "Strike": 560.0, "DTE": 30, "IV_%": 22.0, "Qty": 10, "Entry_Price": 2.10},   # Long Put (Absicherung)
    {"Type": "P", "Strike": 580.0, "DTE": 30, "IV_%": 18.0, "Qty": -10, "Entry_Price": 4.50},  # Short Put (Prämie)
    {"Type": "C", "Strike": 620.0, "DTE": 30, "IV_%": 16.0, "Qty": -10, "Entry_Price": 2.50},  # Short Call (Prämie)
    {"Type": "C", "Strike": 640.0, "DTE": 30, "IV_%": 19.0, "Qty": 10, "Entry_Price": 1.10},   # Long Call (Absicherung)
])

edited_df = st.sidebar.data_editor(
    default_legs,
    num_rows="dynamic",
    column_config={
        "Type": st.column_config.SelectboxColumn("Type", options=["C", "P"], required=True),
        "Strike": st.column_config.NumberColumn("Strike", min_value=1.0, step=5.0),
        "DTE": st.column_config.NumberColumn("DTE (Days)", min_value=1, step=1),
        "IV_%": st.column_config.NumberColumn("IV %", min_value=1.0, max_value=300.0, step=0.5),
        "Qty": st.column_config.NumberColumn("Qty (+/ -)", step=1),
        "Entry_Price": st.column_config.NumberColumn("Entry ($)", step=0.05),
    }
)

# --- CALCULATION ENGINE ---
min_strike = edited_df["Strike"].min() if not edited_df.empty else spot_price * 0.8
max_strike = edited_df["Strike"].max() if not edited_df.empty else spot_price * 1.2

x_min = min(spot_price * 0.85, min_strike * 0.92)
x_max = max(spot_price * 1.15, max_strike * 1.08)
spot_range = np.linspace(x_min, x_max, 500)

pnl_exp = np.zeros_like(spot_range)
pnl_t0 = np.zeros_like(spot_range)
pnl_tx = np.zeros_like(spot_range)

tot_delta, tot_gamma, tot_theta, tot_vega = 0.0, 0.0, 0.0, 0.0

if not edited_df.empty:
    for _, row in edited_df.iterrows():
        opt_type = str(row["Type"])
        strike = float(row["Strike"])
        dte = float(row["DTE"])
        iv = float(row["IV_%"]) / 100.0
        qty = float(row["Qty"])
        entry = float(row["Entry_Price"])
        
        t_exp = dte / 365.0
        t_t0 = max(0.00001, dte / 365.0)
        t_tx = max(0.00001, (dte - days_forward) / 365.0)
        
        # 1. PnL at Expiration
        exp_prices = np.where(opt_type == 'C', np.maximum(0, spot_range - strike), np.maximum(0, strike - spot_range))
        pnl_exp += (exp_prices - entry) * qty * 100.0
        
        # 2. PnL at T+0 (Today)
        t0_prices = np.array([bs_price(opt_type, s, strike, t_t0, risk_free_rate, iv) for s in spot_range])
        pnl_t0 += (t0_prices - entry) * qty * 100.0
        
        # 3. PnL at T+X (Elapsed Days)
        tx_prices = np.array([bs_price(opt_type, s, strike, t_tx, risk_free_rate, iv) for s in spot_range])
        pnl_tx += (tx_prices - entry) * qty * 100.0
        
        # Net Greeks calculation at current spot price
        greeks = bs_greeks(opt_type, spot_price, strike, t_tx, risk_free_rate, iv)
        tot_delta += greeks['delta'] * qty * 100.0
        tot_gamma += greeks['gamma'] * qty * 100.0
        tot_theta += greeks['theta'] * qty * 100.0
        tot_vega += greeks['vega'] * qty * 100.0

# --- MAIN DASHBOARD LAYOUT ---
st.title("📈 Option Analytics & Adjustment Center")

# Top KPI Bar (Greeks)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Net Delta", f"{tot_delta:,.2f}")
kpi2.metric("Net Gamma", f"{tot_gamma:,.4f}")
kpi3.metric("Net Theta ($/Tag)", f"{tot_theta:,.2f}")
kpi4.metric("Net Vega", f"{tot_vega:,.2f}")

st.markdown("---")

# Plotly High-Precision Risk Chart
fig = go.Figure()

# Expiration Line (Blue solid)
fig.add_trace(go.Scatter(
    x=spot_range, y=pnl_exp,
    mode='lines',
    name='Expiration PnL',
    line=dict(color='#29b6f6', width=2.5)
))

# T+0 Line (Red solid)
fig.add_trace(go.Scatter(
    x=spot_range, y=pnl_t0,
    mode='lines',
    name='T+0 (Heute)',
    line=dict(color='#ff5252', width=2.5)
))

# T+X Line (Orange dashed) - if time shifted
if days_forward > 0:
    fig.add_trace(go.Scatter(
        x=spot_range, y=pnl_tx,
        mode='lines',
        name=f'T+{days_forward} Tage',
        line=dict(color='#ffa726', width=2, dash='dash')
    ))

# Zero PnL Baseline
fig.add_hline(y=0, line_dash="dash", line_color="#78909c", line_width=1)

# Current Spot Vertical Line
fig.add_vline(
    x=spot_price, line_dash="dot", line_color="#ffffff", line_width=1.5, 
    annotation_text=f" Spot: {spot_price}", annotation_position="top right"
)

fig.update_layout(
    title=f"Risk Chart ({underlying_symbol}) - Expiration vs. T+0",
    xaxis_title=f"Underlying Price ({underlying_symbol})",
    yaxis_title="PnL ($)",
    template="plotly_dark",
    height=550,
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

# Render Chart
st.plotly_chart(fig, use_container_width=True)

# Position Table Display
st.subheader("📋 Current Position Legs")
st.dataframe(edited_df, use_container_width=True)
