import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

# Page Configuration
st.set_page_config(
    page_title="OptionNet Explorer - Dynamic Multi-Leg Analytics", 
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

# --- BLACK-SCHOLES, DELTA & EXPECTED MOVE HELPER FUNCTIONS ---
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

def strike_from_delta(option_type, target_delta_pct, S, T, r, sigma):
    target_delta = target_delta_pct / 100.0
    if option_type == 'P':
        target_delta = -target_delta
        
    if option_type == 'C':
        d1 = norm.ppf(target_delta)
    else:
        d1 = norm.ppf(1 + target_delta)
        
    K = S * np.exp(-d1 * sigma * np.sqrt(T) + (r + 0.5 * sigma**2) * T)
    return round(K / 5.0) * 5.0

def calculate_expected_move(S, iv_pct, dte):
    T = dte / 365.0
    sigma = iv_pct / 100.0
    em_points = S * sigma * np.sqrt(T)
    return em_points, S - em_points, S + em_points

def get_moneyness_label(opt_type, strike, spot):
    diff_pct = (strike - spot) / spot * 100.0
    if abs(diff_pct) <= 1.5:
        return "ATM"
    if opt_type == "C":
        return "ITM" if strike < spot else "OTM"
    else:
        return "ITM" if strike > spot else "OTM"

# --- MARGIN CALCULATION ENGINE ---
def calculate_leg_margin(legs_df, spot_price):
    if legs_df.empty:
        return 0.0

    total_margin = 0.0
    debit_credit = 0.0
    
    for _, row in legs_df.iterrows():
        qty = float(row["Qty"])
        price = float(row["Entry_Price"])
        debit_credit += price * qty * 100.0

    if debit_credit > 0:
        total_margin += debit_credit

    for _, row in legs_df.iterrows():
        qty = float(row["Qty"])
        strike = float(row["Strike"])
        opt_type = str(row["Type"])
        price = float(row["Entry_Price"])
        
        if qty < 0:
            abs_qty = abs(qty)
            if opt_type == "C":
                otm = max(0.0, strike - spot_price)
                margin_req = (0.20 * spot_price - otm + price) * 100.0 * abs_qty
            else:
                otm = max(0.0, spot_price - strike)
                margin_req = (0.20 * spot_price - otm + price) * 100.0 * abs_qty
            
            min_margin = (0.10 * strike + price) * 100.0 * abs_qty
            total_margin += max(margin_req, min_margin)

    return max(total_margin, 0.0)

# --- DYNAMIC MULTI-LEG STRATEGY BUILDER ---
def build_custom_strategy(strategy_type, S, iv_default=18.0):
    iv = iv_default / 100.0
    r = 0.045
    legs = []
    
    k_atm_c = strike_from_delta("C", 50, S, 30/365, r, iv)
    k_itm_c = strike_from_delta("C", 70, S, 60/365, r, iv)
    k_otm_c = strike_from_delta("C", 30, S, 30/365, r, iv)
    
    k_atm_p = strike_from_delta("P", 50, S, 30/365, r, iv)
    k_itm_p = strike_from_delta("P", 70, S, 60/365, r, iv)
    k_otm_p = strike_from_delta("P", 30, S, 30/365, r, iv)

    if strategy_type == "Multi-Leg Combo (ITM/ATM/OTM)":
        legs.extend([
            # ITM Long Leg (Back Month)
            {"Enable": True, "Phase": "Initial", "Status": "Open", "Type": "C", "Strike": k_itm_c, "DTE": 60, "IV_%": iv_default, "Qty": 10, "Entry_Price": 22.0, "Close_Price": 0.0},
            # ATM Short Leg (Front Month)
            {"Enable": True, "Phase": "Initial", "Status": "Open", "Type": "C", "Strike": k_atm_c, "DTE": 30, "IV_%": iv_default, "Qty": -10, "Entry_Price": 12.0, "Close_Price": 0.0},
            # OTM Short Leg (Front Month - Wing / Extra Income)
            {"Enable": True, "Phase": "Initial", "Status": "Open", "Type": "C", "Strike": k_otm_c, "DTE": 30, "IV_%": iv_default, "Qty": -5, "Entry_Price": 5.50, "Close_Price": 0.0},
            # OTM Put Hedge (Downside Protection)
            {"Enable": True, "Phase": "Adjustment", "Status": "Open", "Type": "P", "Strike": k_otm_p, "DTE": 30, "IV_%": iv_default, "Qty": 5, "Entry_Price": 4.20, "Close_Price": 0.0},
        ])
    elif strategy_type == "Diagonal Spread":
        legs.extend([
            {"Enable": True, "Phase": "Initial", "Status": "Open", "Type": "C", "Strike": k_itm_c, "DTE": 60, "IV_%": iv_default, "Qty": 10, "Entry_Price": 22.0, "Close_Price": 0.0},
            {"Enable": True, "Phase": "Initial", "Status": "Open", "Type": "C", "Strike": k_otm_c, "DTE": 30, "IV_%": iv_default, "Qty": -10, "Entry_Price": 5.50, "Close_Price": 0.0},
        ])
    elif strategy_type == "Calendar Spread":
        legs.extend([
            {"Enable": True, "Phase": "Initial", "Status": "Open", "Type": "C", "Strike": k_atm_c, "DTE": 60, "IV_%": iv_default, "Qty": 10, "Entry_Price": 18.5, "Close_Price": 0.0},
            {"Enable": True, "Phase": "Initial", "Status": "Open", "Type": "C", "Strike": k_atm_c, "DTE": 30, "IV_%": iv_default, "Qty": -10, "Entry_Price": 12.0, "Close_Price": 0.0},
        ])
    else:  # Blank / Custom Template
        legs.extend([
            {"Enable": True, "Phase": "Initial", "Status": "Open", "Type": "C", "Strike": k_atm_c, "DTE": 30, "IV_%": iv_default, "Qty": 10, "Entry_Price": 12.0, "Close_Price": 0.0},
            {"Enable": True, "Phase": "Initial", "Status": "Open", "Type": "P", "Strike": k_atm_p, "DTE": 30, "IV_%": iv_default, "Qty": -10, "Entry_Price": 11.5, "Close_Price": 0.0},
        ])

    return legs


# --- SIDEBAR CONTROLS ---
st.sidebar.title("⚙️ Base Settings")

underlying_symbol = st.sidebar.selectbox("Underlying", ["SPX", "SPY", "QQQ", "DAX", "RUT"], index=1)
spot_price = st.sidebar.number_input("Current Spot Price", value=600.0, step=1.0)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=4.5, step=0.1) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Strategy Selector")

strategy_choice = st.sidebar.selectbox(
    "Preset Template",
    ["Multi-Leg Combo (ITM/ATM/OTM)", "Diagonal Spread", "Calendar Spread", "Custom / Blank"]
)

config_key = f"{strategy_choice}_{spot_price}"
if "last_config" not in st.session_state or st.session_state["last_config"] != config_key or st.sidebar.button("Reset Strategy"):
    st.session_state["last_config"] = config_key
    st.session_state["legs_data"] = build_custom_strategy(strategy_choice, spot_price)

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Time Travel & Overlays")
show_t1 = st.sidebar.checkbox("Show T+1 Line (Morgen)", value=True)
show_initial_overlay = st.sidebar.checkbox("Overlay 'Initial Position' Curve", value=True)
days_forward = st.sidebar.slider("Days Elapsed (Forward in Time)", min_value=0, max_value=60, value=0, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Portfolio & Adjustment Legs")
st.sidebar.caption("💡 Füge beliebig viele Legs hinzu (über `+`). Trage bei geschlossenen Legs den `Close_Price` ein.")

legs_df = pd.DataFrame(st.session_state["legs_data"])

edited_df = st.sidebar.data_editor(
    legs_df,
    num_rows="dynamic",
    column_config={
        "Enable": st.column_config.CheckboxColumn("Active?", default=True),
        "Phase": st.column_config.SelectboxColumn("Phase", options=["Initial", "Adjustment"], default="Initial"),
        "Status": st.column_config.SelectboxColumn("Status", options=["Open", "Closed"], default="Open"),
        "Type": st.column_config.SelectboxColumn("Type", options=["C", "P"], required=True),
        "Strike": st.column_config.NumberColumn("Strike", min_value=1.0, step=5.0),
        "DTE": st.column_config.NumberColumn("DTE", min_value=1, step=1),
        "IV_%": st.column_config.NumberColumn("IV %", min_value=1.0, max_value=300.0, step=0.5),
        "Qty": st.column_config.NumberColumn("Qty (+/-)", step=1),
        "Entry_Price": st.column_config.NumberColumn("Entry ($)", step=0.05),
        "Close_Price": st.column_config.NumberColumn("Close ($)", step=0.05),
    }
)


# --- CALCULATION ENGINE ---
active_legs = edited_df[edited_df["Enable"] == True].copy() if not edited_df.empty else edited_df.copy()

if not active_legs.empty:
    active_legs["Moneyness"] = active_legs.apply(lambda r: get_moneyness_label(r["Type"], float(r["Strike"]), spot_price), axis=1)

open_legs = active_legs[active_legs["Status"] == "Open"] if not active_legs.empty else pd.DataFrame()
closed_legs = active_legs[active_legs["Status"] == "Closed"] if not active_legs.empty else pd.DataFrame()

initial_open = open_legs[open_legs["Phase"] == "Initial"] if not open_legs.empty else pd.DataFrame()
adj_open = open_legs[open_legs["Phase"] == "Adjustment"] if not open_legs.empty else pd.DataFrame()

# Margin Calculations
margin_initial = calculate_leg_margin(initial_open, spot_price)
margin_adj = calculate_leg_margin(adj_open, spot_price)
margin_total = margin_initial + margin_adj

# Realized PnL
realized_pnl = 0.0
if not closed_legs.empty:
    for _, row in closed_legs.iterrows():
        entry = float(row["Entry_Price"])
        close_p = float(row["Close_Price"])
        qty = float(row["Qty"])
        realized_pnl += (close_p - entry) * qty * 100.0

# Expected Move
if not open_legs.empty:
    avg_iv = open_legs["IV_%"].mean()
    min_dte = open_legs["DTE"].min()
    em_pts, em_lower, em_upper = calculate_expected_move(spot_price, avg_iv, min_dte)
else:
    em_pts, em_lower, em_upper = 0.0, spot_price, spot_price
    min_dte = 30

min_strike = active_legs["Strike"].min() if not active_legs.empty else spot_price * 0.8
max_strike = active_legs["Strike"].max() if not active_legs.empty else spot_price * 1.2

x_min = min(spot_price * 0.85, min_strike * 0.90, em_lower * 0.95)
x_max = max(spot_price * 1.15, max_strike * 1.10, em_upper * 1.05)
spot_range = np.linspace(x_min, x_max, 500)

pnl_exp = np.full_like(spot_range, realized_pnl)
pnl_t0 = np.full_like(spot_range, realized_pnl)
pnl_t1 = np.full_like(spot_range, realized_pnl)
pnl_tx = np.full_like(spot_range, realized_pnl)
pnl_initial_t0 = np.zeros_like(spot_range)

tot_delta, tot_gamma, tot_theta, tot_vega = 0.0, 0.0, 0.0, 0.0

if not open_legs.empty:
    for _, row in open_legs.iterrows():
        opt_type = str(row["Type"])
        strike = float(row["Strike"])
        dte = float(row["DTE"])
        iv = float(row["IV_%"]) / 100.0
        qty = float(row["Qty"])
        entry = float(row["Entry_Price"])
        
        t_t0 = max(0.00001, dte / 365.0)
        t_t1 = max(0.00001, (dte - 1) / 365.0)
        t_tx = max(0.00001, (dte - days_forward) / 365.0)
        
        # Expiration & T-Lines
        exp_prices = np.where(opt_type == 'C', np.maximum(0, spot_range - strike), np.maximum(0, strike - spot_range))
        pnl_exp += (exp_prices - entry) * qty * 100.0
        
        t0_prices = np.array([bs_price(opt_type, s, strike, t_t0, risk_free_rate, iv) for s in spot_range])
        pnl_t0 += (t0_prices - entry) * qty * 100.0
        
        t1_prices = np.array([bs_price(opt_type, s, strike, t_t1, risk_free_rate, iv) for s in spot_range])
        pnl_t1 += (t1_prices - entry) * qty * 100.0
        
        tx_prices = np.array([bs_price(opt_type, s, strike, t_tx, risk_free_rate, iv) for s in spot_range])
        pnl_tx += (tx_prices - entry) * qty * 100.0
        
        # Net Greeks
        greeks = bs_greeks(opt_type, spot_price, strike, t_tx, risk_free_rate, iv)
        tot_delta += greeks['delta'] * qty * 100.0
        tot_gamma += greeks['gamma'] * qty * 100.0
        tot_theta += greeks['theta'] * qty * 100.0
        tot_vega += greeks['vega'] * qty * 100.0

if show_initial_overlay and not initial_open.empty:
    for _, row in initial_open.iterrows():
        opt_type = str(row["Type"])
        strike = float(row["Strike"])
        dte = float(row["DTE"])
        iv = float(row["IV_%"]) / 100.0
        qty = float(row["Qty"])
        entry = float(row["Entry_Price"])
        t_t0 = max(0.00001, dte / 365.0)
        t0_prices = np.array([bs_price(opt_type, s, strike, t_t0, risk_free_rate, iv) for s in spot_range])
        pnl_initial_t0 += (t0_prices - entry) * qty * 100.0


# --- MAIN DASHBOARD LAYOUT ---
st.title(f"📈 Dynamic Multi-Leg Option Analytics ({underlying_symbol})")

# Top KPI Bar - Row 1 (Greeks & PnL)
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Realized PnL", f"${realized_pnl:,.2f}")
m2.metric("Net Delta", f"{tot_delta:,.2f}")
m3.metric("Net Gamma", f"{tot_gamma:,.4f}")
m4.metric("Net Theta ($/Tag)", f"{tot_theta:,.2f}")
m5.metric("Net Vega", f"{tot_vega:,.2f}")
m6.metric(f"Expected Move ({min_dte} DTE)", f"±{em_pts:,.2f} pts", f"{em_lower:,.1f} - {em_upper:,.1f}")

st.markdown("---")

# Top KPI Bar - Row 2 (Margin Metrics)
margin_col1, margin_col2, margin_col3, margin_col4 = st.columns(4)
margin_col1.metric("Margin Base Position", f"${margin_initial:,.2f}")
margin_col2.metric("Margin Adjustment", f"${margin_adj:,.2f}")
margin_col3.metric("TOTAL MARGIN REQUIRED", f"${margin_total:,.2f}")
margin_col4.metric("Margin Return (Theta/Margin)", f"{(tot_theta/margin_total*100 if margin_total > 0 else 0):,.2f}% / Tag")

st.markdown("---")

# Plotly High-Precision Risk Chart
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=spot_range, y=pnl_exp, mode='lines', name='Expiration PnL', line=dict(color='#29b6f6', width=2.5)
))

fig.add_trace(go.Scatter(
    x=spot_range, y=pnl_t0, mode='lines', name='T+0 (Heute)', line=dict(color='#ff5252', width=2.5)
))

if show_t1:
    fig.add_trace(go.Scatter(
        x=spot_range, y=pnl_t1, mode='lines', name='T+1 (Morgen)', line=dict(color='#66bb6a', width=1.8, dash='dash')
    ))

if days_forward > 1:
    fig.add_trace(go.Scatter(
        x=spot_range, y=pnl_tx, mode='lines', name=f'T+{days_forward} Tage', line=dict(color='#ffa726', width=2, dash='dot')
    ))

if show_initial_overlay and not initial_open.empty and len(open_legs) > len(initial_open):
    fig.add_trace(go.Scatter(
        x=spot_range, y=pnl_initial_t0, mode='lines', name='Initial Position (Pre-ADJ)', line=dict(color='#9e9e9e', width=1.5, dash='dashdot')
    ))

fig.add_hline(y=0, line_dash="dash", line_color="#78909c", line_width=1)

fig.add_vline(
    x=spot_price, line_dash="dot", line_color="#ffffff", line_width=1.5, 
    annotation_text=f" Spot: {spot_price}", annotation_position="top right"
)

fig.add_vline(
    x=em_lower, line_dash="dash", line_color="#ab47bc", line_width=1.5,
    annotation_text=f"-EM: {em_lower:,.1f}", annotation_position="bottom left"
)
fig.add_vline(
    x=em_upper, line_dash="dash", line_color="#ab47bc", line_width=1.5,
    annotation_text=f"+EM: {em_upper:,.1f}", annotation_position="bottom right"
)

fig.update_layout(
    title=f"Risk Chart ONE Style ({underlying_symbol}) - Total Margin: ${margin_total:,.2f}",
    xaxis_title=f"Underlying Price ({underlying_symbol})",
    yaxis_title="Total PnL ($)",
    template="plotly_dark",
    height=580,
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig, use_container_width=True)

# Detailed Multi-Leg Table with ITM/ATM/OTM Breakdown
st.subheader("📋 Complete Portfolio Legs (with ITM/ATM/OTM Classification)")
if not active_legs.empty:
    st.dataframe(
        active_legs[["Phase", "Status", "Moneyness", "Type", "Strike", "DTE", "IV_%", "Qty", "Entry_Price", "Close_Price"]],
        use_container_width=True
    )
else:
    st.info("Keine aktiven Beine vorhanden. Verwende das Sidebar-Menü, um welche hinzuzufügen.")
