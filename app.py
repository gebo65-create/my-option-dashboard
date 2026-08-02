import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
import yfinance as yf
from datetime import datetime

# Versuche ib_insync zu importieren (falls installiert)
try:
    from ib_insync import IB, Option, util
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="OptionNet Explorer - Live Data & IBKR Integration", 
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

# --- LIVE DATA FETCHER (YFINANCE - 5 MIN CACHE) ---
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

def generate_dte_iv_table(spot_price, base_iv, r=0.045):
    records = []
    for dte in range(0, 41):
        if dte == 0:
            iv_dte = base_iv * 1.35
        elif dte <= 7:
            iv_dte = base_iv * (1.20 - (dte * 0.02))
        else:
            iv_dte = base_iv * (1.0 + np.sin(dte / 10.0) * 0.03)

        sigma = iv_dte / 100.0
        T = max(0.00001, dte / 365.0)
        
        em_pts = spot_price * sigma * np.sqrt(T) if dte > 0 else 0.0
        em_pct = (em_pts / spot_price) * 100.0
        
        records.append({
            "DTE": dte,
            "IV (%)": round(iv_dte, 2),
            "Exp. Move (± Pts)": round(em_pts, 2),
            "Exp. Move (± %)": round(em_pct, 2),
            "Call ITM (Δ70)": int(strike_from_delta("C", 70, spot_price, T, r, sigma)),
            "Call ATM (Δ50)": int(strike_from_delta("C", 50, spot_price, T, r, sigma)),
            "Call OTM (Δ30)": int(strike_from_delta("C", 30, spot_price, T, r, sigma)),
            "Put ITM (Δ70)": int(strike_from_delta("P", 70, spot_price, T, r, sigma)),
            "Put ATM (Δ50)": int(strike_from_delta("P", 50, spot_price, T, r, sigma)),
            "Put OTM (Δ30)": int(strike_from_delta("P", 30, spot_price, T, r, sigma)),
        })
    return pd.DataFrame(records)


# --- IBKR POSITIONS IMPORT ENGINE ---
def fetch_ibkr_positions(host='127.0.0.1', port=7497, client_id=1):
    if not IBKR_AVAILABLE:
        st.error("Das Paket 'ib_insync' ist nicht installiert. Bitte 'pip install ib-insync' ausführen.")
        return None
        
    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=5)
        positions = ib.positions()
        
        imported_legs = []
        today = datetime.now().date()
        
        for pos in positions:
            contract = pos.contract
            # Nur Optionskontrakte filtern
            if contract.secType == 'OPT':
                # DTE Berechnung
                exp_date = datetime.strptime(contract.lastTradeDateOrContractMonth, "%Y%m%d").date()
                dte = max(0, (exp_date - today).days)
                
                multiplier = float(contract.multiplier) if contract.multiplier else 100.0
                entry_price = round(pos.avgCost / multiplier, 2)
                
                imported_legs.append({
                    "Enable": True,
                    "Phase": "Initial",
                    "Type": "C" if contract.right == "C" or contract.right == "CALL" else "P",
                    "Strike": float(contract.strike),
                    "DTE": int(dte),
                    "IV_%": 18.0, # Standardwert (kann angepasst werden)
                    "Qty": int(pos.position),
                    "Entry_Price": entry_price
                })
                
        ib.disconnect()
        return pd.DataFrame(imported_legs) if imported_legs else pd.DataFrame()
        
    except Exception as e:
        st.error(f"Verbindungsfehler zu IBKR TWS: {e}")
        if ib.isConnected():
            ib.disconnect()
        return None


# --- SIDEBAR CONTROLS ---
st.sidebar.title("⚙️ Base & Data Settings")

underlying_symbol = st.sidebar.selectbox("Underlying Symbol", list(TICKER_MAP.keys()), index=0)
ticker = TICKER_MAP[underlying_symbol]

# Automatischer 5-Minuten Abruf
live_spot = fetch_delayed_spot(ticker)
default_spot = live_spot if live_spot is not None else 600.0

spot_price = st.sidebar.number_input(
    f"Current Spot Price ({'Live/Delayed' if live_spot else 'Manual'})", 
    value=default_spot, 
    step=1.0
)

base_iv = st.sidebar.number_input("Base IV (%)", value=18.0, step=0.5)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=4.5, step=0.1) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("🔌 Interactive Brokers (TWS) Import")

ib_port = st.sidebar.number_input("TWS Port (7497 Paper / 7496 Live)", value=7497, step=1)

if st.sidebar.button("📥 Positions aus TWS Laden"):
    with st.spinner("Verbinde mit TWS und lade Positionen..."):
        ib_df = fetch_ibkr_positions(port=int(ib_port))
        if ib_df is not None:
            if not ib_df.empty:
                st.session_state["legs_df"] = ib_df
                st.sidebar.success(f"{len(ib_df)} Beine von IBKR importiert!")
                st.rerun()
            else:
                st.sidebar.warning("Keine offenen Optionspositionen in TWS gefunden.")

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Portfolio Positions Manager")

default_legs = [
    {"Enable": True, "Phase": "Initial", "Type": "C", "Strike": spot_price, "DTE": 30, "IV_%": base_iv, "Qty": 10, "Entry_Price": 12.0},
    {"Enable": True, "Phase": "Initial", "Type": "P", "Strike": spot_price * 0.95, "DTE": 30, "IV_%": base_iv, "Qty": -10, "Entry_Price": 6.0},
]

if "legs_df" not in st.session_state:
    st.session_state["legs_df"] = pd.DataFrame(default_legs)

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("🔄 Reset Default"):
    st.session_state["legs_df"] = pd.DataFrame(default_legs)
    st.rerun()

if col_btn2.button("🗑️ Alle Löschen"):
    st.session_state["legs_df"] = pd.DataFrame(columns=["Enable", "Phase", "Type", "Strike", "DTE", "IV_%", "Qty", "Entry_Price"])
    st.rerun()

# Interaktiver Table-Editor
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


# --- MAIN CALCULATIONS ---
active_legs = edited_df[edited_df["Enable"] == True].copy() if not edited_df.empty else pd.DataFrame()
initial_legs = active_legs[active_legs["Phase"] == "Initial"] if not active_legs.empty else pd.DataFrame()

min_strike = active_legs["Strike"].min() if not active_legs.empty and "Strike" in active_legs else spot_price * 0.8
max_strike = active_legs["Strike"].max() if not active_legs.empty and "Strike" in active_legs else spot_price * 1.2

x_min = min(spot_price * 0.85, min_strike * 0.90)
x_max = max(spot_price * 1.15, max_strike * 1.10)
spot_range = np.linspace(x_min, x_max, 500)

pnl_t0 = np.zeros_like(spot_range)
pnl_t1 = np.zeros_like(spot_range)
pnl_exp = np.zeros_like(spot_range)
pnl_initial_t0 = np.zeros_like(spot_range)

spot_pnl_t0 = 0.0
spot_pnl_t1 = 0.0
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

if not initial_legs.empty and len(active_legs) > len(initial_legs):
    for _, row in initial_legs.iterrows():
        try:
            opt_type = str(row["Type"])
            strike = float(row["Strike"])
            dte = float(row["DTE"])
            iv = float(row["IV_%"]) / 100.0
            qty = float(row["Qty"])
            entry = float(row["Entry_Price"])
            
            t_t0 = max(0.00001, dte / 365.0)
            t0_prices = np.array([bs_price(opt_type, s, strike, t_t0, risk_free_rate, iv) for s in spot_range])
            pnl_initial_t0 += (t0_prices - entry) * qty * 100.0
        except (ValueError, KeyError):
            continue


# --- DASHBOARD LAYOUT ---
st.title(f"📈 OptionNet Explorer ({underlying_symbol})")

# Header KPI Metrics
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("T+0 Current PnL", f"${spot_pnl_t0:,.2f}")
k2.metric("T+1 Projected PnL", f"${spot_pnl_t1:,.2f}", f"${(spot_pnl_t1 - spot_pnl_t0):,.2f}")
k3.metric("Net Delta", f"{tot_delta:,.2f}")
k4.metric("Net Theta ($/Tag)", f"{tot_theta:,.2f}")
k5.metric("Net Vega", f"{tot_vega:,.2f}")

st.markdown("---")

# SECTION 1: RISK CHART
st.subheader("📉 Position Risk Graph (T+0 & T+1 mit Spot Price Marker)")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=spot_range, y=pnl_t0, mode='lines', name='T+0 (Gesamte Position Heute)',
    line=dict(color='#ff5252', width=3),
    hovertemplate="Spot: %{x:.2f}<br>PnL T+0: $%{y:,.2f}"
))

fig.add_trace(go.Scatter(
    x=spot_range, y=pnl_t1, mode='lines', name='T+1 (Gesamte Position Morgen)',
    line=dict(color='#66bb6a', width=2, dash='dash'),
    hovertemplate="Spot: %{x:.2f}<br>PnL T+1: $%{y:,.2f}"
))

if not initial_legs.empty and len(active_legs) > len(initial_legs):
    fig.add_trace(go.Scatter(
        x=spot_range, y=pnl_initial_t0, mode='lines', name='Initial Position (Vor Adjustment)',
        line=dict(color='#9e9e9e', width=1.5, dash='dot'),
        hovertemplate="Spot: %{x:.2f}<br>Initial PnL: $%{y:,.2f}"
    ))

fig.add_trace(go.Scatter(
    x=spot_range, y=pnl_exp, mode='lines', name='Expiration PnL',
    line=dict(color='#29b6f6', width=1.5),
    hovertemplate="Spot: %{x:.2f}<br>Exp PnL: $%{y:,.2f}"
))

fig.add_hline(y=0, line_dash="solid", line_color="#455a64", line_width=1)
fig.add_vline(x=spot_price, line_dash="dot", line_color="#ffffff", line_width=1.5)

# T+0 Marker Annotation am Spot Price
fig.add_annotation(
    x=spot_price, y=spot_pnl_t0,
    text=f" <b>T+0 Spot: ${spot_pnl_t0:,.2f}</b>",
    showarrow=True, arrowhead=2, ax=60, ay=-30,
    bgcolor="#ff5252", bordercolor="#ffffff", font=dict(color="white", size=11)
)

# T+1 Marker Annotation am Spot Price
fig.add_annotation(
    x=spot_price, y=spot_pnl_t1,
    text=f" <b>T+1 Spot: ${spot_pnl_t1:,.2f}</b>",
    showarrow=True, arrowhead=2, ax=60, ay=30,
    bgcolor="#66bb6a", bordercolor="#ffffff", font=dict(color="white", size=11)
)

fig.update_layout(
    title=f"Risk Profile - {underlying_symbol} @ {spot_price}",
    xaxis_title="Underlying Price",
    yaxis_title="Profit / Loss ($)",
    template="plotly_dark",
    height=550,
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# SECTION 2: ALWAYS-VISIBLE DTE 0 - 40 OVERVIEW TABLE
st.subheader("📅 DTE 0 bis 40 - Implizierte Volatilität, Expected Move & Strike Grid")

dte_df = generate_dte_iv_table(spot_price, base_iv, r=risk_free_rate)

f_col1, _ = st.columns([3, 1])
dte_range = f_col1.slider("DTE Bereich eingrenzen", 0, 40, (0, 40))
filtered_dte_df = dte_df[(dte_df["DTE"] >= dte_range[0]) & (dte_df["DTE"] <= dte_range[1])]

st.dataframe(
    filtered_dte_df,
    use_container_width=True,
    height=450
)
