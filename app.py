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
    from ib_async import IB, Index, Stock, Option, util
    IBKR_AVAILABLE = True
except ImportError:
    try:
        from ib_insync import IB, Index, Stock, Option, util
        IBKR_AVAILABLE = True
    except ImportError:
        IBKR_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="OptionNet Explorer - Multi-Strategy Dashboard", 
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

# --- IBKR CLIENT INITIALISIERUNG ---
if "ib_client" not in st.session_state:
    st.session_state["ib_client"] = None
if "ib_connected" not in st.session_state:
    st.session_state["ib_connected"] = False

def connect_ibkr(host, port, client_id):
    if not IBKR_AVAILABLE:
        st.sidebar.error("❌ 'ib_async' oder 'ib_insync' ist nicht installiert!")
        return False
    try:
        ib = IB()
        ib.connect(host, port, clientId=client_id, timeout=5)
        st.session_state["ib_client"] = ib
        st.session_state["ib_connected"] = True
        return True
    except Exception as e:
        st.sidebar.error(f"Verbindungsfehler: {e}")
        st.session_state["ib_connected"] = False
        return False

def disconnect_ibkr():
    if st.session_state.get("ib_client") and st.session_state["ib_connected"]:
        try:
            st.session_state["ib_client"].disconnect()
        except Exception:
            pass
    st.session_state["ib_connected"] = False
    st.session_state["ib_client"] = None

def fetch_ibkr_spot(symbol):
    ib = st.session_state.get("ib_client")
    if not ib or not st.session_state.get("ib_connected"):
        return None
    try:
        # Ticker-Spezifikation für IBKR
        if symbol in ["SPX", "RUT"]:
            contract = Index(symbol, 'CBOE', 'USD')
        elif symbol in ["SPY", "QQQ"]:
            contract = Stock(symbol, 'SMART', 'USD')
        else:
            contract = Stock(symbol, 'SMART', 'USD')
            
        ib.qualifyContracts(contract)
        ticker_data = ib.reqMktData(contract, '', False, False)
        ib.sleep(1)  # Kurze Pause für Datenempfang
        
        price = ticker_data.marketPrice()
        if np.isnan(price) or price <= 0:
            price = ticker_data.close
            
        if not np.isnan(price) and price > 0:
            return round(float(price), 2)
    except Exception as e:
        st.sidebar.warning(f"IBKR Spot Abruf fehlgeschlagen für {symbol}: {e}")
    return None

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
            if pd.isna(row.get("Qty")) or pd.isna(row.get("Strike")) or pd.isna(row.get("Entry_Price")):
                continue

            qty = float(row["Qty"])
            strike = float(row["Strike"])
            opt_type = str(row["Type"])
            entry = float(row["Entry_Price"])
            
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
                total_margin += (entry * 100.0 * qty)
        except (ValueError, KeyError, TypeError):
            continue
            
    return round(total_margin, 2)

# --- OPTION CHAIN BUILDER ---
def build_classic_option_chain(spot, dte, base_iv, r=0.045, strike_step=1.0, num_strikes=21):
    T = max(0.00001, dte / 365.0)
    atm_strike = round(spot / strike_step) * strike_step
    half_strikes = num_strikes // 2
    
    strikes = [round(atm_strike + i * strike_step, 2) for i in range(-half_strikes, half_strikes + 1)]
    chain = []
    
    for K in strikes:
        moneyness = np.log(K / spot)
        skew_iv = base_iv - (moneyness * 12.0) + (moneyness**2 * 20.0)
        sigma = max(0.05, skew_iv / 100.0)
        
        c_price = bs_price('C', spot, K, T, r, sigma)
        c_greeks = bs_greeks('C', spot, K, T, r, sigma)
        
        p_price = bs_price('P', spot, K, T, r, sigma)
        p_greeks = bs_greeks('P', spot, K, T, r, sigma)
        
        call_itm = K < spot
        put_itm = K > spot
        is_atm = (abs(K - atm_strike) < 0.001)
        
        c_intrinsic = max(0.0, spot - K)
        c_extrinsic = max(0.0, c_price - c_intrinsic)
        
        p_intrinsic = max(0.0, K - spot)
        p_extrinsic = max(0.0, p_price - p_intrinsic)
        
        chain.append({
            "C_ITM": call_itm,
            "C_Intrinsic": round(c_intrinsic, 2),
            "C_Extrinsic": round(c_extrinsic, 2),
            "C_Theta": round(c_greeks['theta'], 2),
            "C_Gamma": round(c_greeks['gamma'], 4),
            "C_Delta": round(c_greeks['delta'], 2),
            "C_IV_%": round(skew_iv, 1),
            "C_Bid": round(c_price * 0.98, 2),
            "C_Ask": round(c_price * 1.02, 2),
            "STRIKE": float(K),
            "ATM_Marker": "🎯 ATM" if is_atm else "",
            "P_Ask": round(p_price * 1.02, 2),
            "P_Bid": round(p_price * 0.98, 2),
            "P_IV_%": round(skew_iv, 1),
            "P_Delta": round(p_greeks['delta'], 2),
            "P_Gamma": round(p_greeks['gamma'], 4),
            "P_Theta": round(p_greeks['theta'], 2),
            "P_Intrinsic": round(p_intrinsic, 2),
            "P_Extrinsic": round(p_extrinsic, 2),
            "P_ITM": put_itm,
            "IS_ATM": is_atm
        })
        
    return pd.DataFrame(chain)

def highlight_option_chain(row):
    styles = [''] * len(row)
    is_atm = row.get("IS_ATM", False)
    c_itm = row.get("C_ITM", False)
    p_itm = row.get("P_ITM", False)
    
    if is_atm:
        return ['background-color: #8b0000; color: white; font-weight: bold;'] * len(row)
    
    call_cols_idx = [i for i, col in enumerate(row.index) if col.startswith("C_")]
    put_cols_idx = [i for i, col in enumerate(row.index) if col.startswith("P_")]
    
    if c_itm:
        for idx in call_cols_idx:
            styles[idx] = 'background-color: #1a365d; color: #90caf9;'
            
    if p_itm:
        for idx in put_cols_idx:
            styles[idx] = 'background-color: #1b4332; color: #a7f3d0;'
            
    return styles

# --- TICKER MAP (OHNE DAX) ---
TICKER_MAP = {
    "SPX": "^GSPC",
    "SPY": "SPY",
    "RUT": "^RUT",
    "QQQ": "QQQ"
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

def get_default_legs(current_spot, current_iv):
    return [
        {"Enable": True, "Phase": "Initial", "Type": "C", "Strike": round(current_spot, 0), "DTE": 30, "IV_%": current_iv, "Qty": 10, "Entry_Price": round(current_spot * 0.02, 2)},
        {"Enable": True, "Phase": "Initial", "Type": "P", "Strike": round(current_spot * 0.95, 0), "DTE": 30, "IV_%": current_iv, "Qty": -10, "Entry_Price": round(current_spot * 0.01, 2)},
        {"Enable": True, "Phase": "Adjustment", "Type": "P", "Strike": round(current_spot * 0.90, 0), "DTE": 20, "IV_%": current_iv + 2, "Qty": -5, "Entry_Price": round(current_spot * 0.008, 2)},
    ]

# --- SIDEBAR & PORTFOLIO ---
st.sidebar.title("⚙️ Base Settings & Positions")

# --- IBKR VERBINDUNGS-PANEL ---
with st.sidebar.expander("🔌 Interactive Brokers (IBKR) Connection", expanded=False):
    ib_host = st.text_input("IP / Host", value="127.0.0.1")
    ib_port = st.number_input("Port (7497 Paper / 7496 Live)", value=7497, step=1)
    ib_client_id = st.number_input("Client ID", value=1, step=1)
    
    if not st.session_state["ib_connected"]:
        if st.button("🔌 IBKR Verbinden"):
            if connect_ibkr(ib_host, ib_port, ib_client_id):
                st.success("🟢 Verbunden mit IBKR TWS/Gateway")
                st.rerun()
    else:
        st.success("🟢 IBKR Verbunden")
        if st.button("🔴 Trennen"):
            disconnect_ibkr()
            st.rerun()

underlying_symbol = st.sidebar.selectbox("Underlying Symbol", list(TICKER_MAP.keys()), index=0)
ticker = TICKER_MAP[underlying_symbol]

# Spot Abruf: Priorität IBKR -> Fallback yfinance
live_spot = None
if st.session_state.get("ib_connected"):
    live_spot = fetch_ibkr_spot(underlying_symbol)

if live_spot is None:
    live_spot = fetch_delayed_spot(ticker)

default_spot = live_spot if live_spot is not None else (600.0 if underlying_symbol == "SPY" else 6000.0)

spot_source_str = "IBKR Live" if st.session_state.get("ib_connected") and live_spot else ("Delayed/yFinance" if live_spot else "Manuell")
spot_price = st.sidebar.number_input(
    f"Spot Price ({spot_source_str})", 
    value=default_spot, 
    step=1.0
)

default_step_idx = 0 if underlying_symbol in ["SPX", "SPY", "RUT"] else 2
strike_step_val = st.sidebar.selectbox(
    "Strike Schrittweite (Option Chain)",
    options=[1.0, 2.5, 5.0, 10.0],
    index=default_step_idx,
    help="1.0 wählen für 1er-Schritte bei SPY, SPX, RUT"
)

base_iv = st.sidebar.number_input("Base IV (%)", value=18.0, step=0.5)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=4.5, step=0.1) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Display Options")
show_margin = st.sidebar.checkbox("Margin Details anzeigen", value=True)
show_individual_legs_graph = st.sidebar.checkbox("Einzel-Legs im Risikograph einblenden", value=True)

# --- MULTI-POSITION MANAGER ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔀 Strategie / Position Manager")

if "portfolios" not in st.session_state:
    st.session_state["portfolios"] = {
        "Position 1 (Hauptposition)": pd.DataFrame(get_default_legs(spot_price, base_iv)),
        "Position 2 (Hedge/Spread)": pd.DataFrame([
            {"Enable": True, "Phase": "Initial", "Type": "P", "Strike": round(spot_price * 0.92, 0), "DTE": 45, "IV_%": base_iv, "Qty": 5, "Entry_Price": round(spot_price * 0.015, 2)}
        ])
    }
    st.session_state["active_portfolio_name"] = "Position 1 (Hauptposition)"

portfolio_names = list(st.session_state["portfolios"].keys())

selected_pos = st.sidebar.selectbox(
    "Aktive Position wählen:", 
    options=portfolio_names, 
    index=portfolio_names.index(st.session_state["active_portfolio_name"]) if st.session_state["active_portfolio_name"] in portfolio_names else 0
)

st.session_state["active_portfolio_name"] = selected_pos

with st.sidebar.expander("➕ Neue Position erstellen"):
    new_pos_name = st.text_input("Name der neuen Position:", placeholder="z.B. Iron Condor SPX")
    if st.button("Position anlegen"):
        if new_pos_name and new_pos_name not in st.session_state["portfolios"]:
            st.session_state["portfolios"][new_pos_name] = pd.DataFrame(columns=["Enable", "Phase", "Type", "Strike", "DTE", "IV_%", "Qty", "Entry_Price"])
            st.session_state["active_portfolio_name"] = new_pos_name
            st.rerun()
        elif new_pos_name in st.session_state["portfolios"]:
            st.warning("Eine Position mit diesem Namen existiert bereits!")

if len(st.session_state["portfolios"]) > 1:
    if st.sidebar.button(f"🗑️ '{selected_pos}' löschen"):
        del st.session_state["portfolios"][selected_pos]
        st.session_state["active_portfolio_name"] = list(st.session_state["portfolios"].keys())[0]
        st.rerun()

st.sidebar.markdown(f"**Bearbeite: `{selected_pos}`**")

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("🔄 Reset Legs"):
    st.session_state["portfolios"][selected_pos] = pd.DataFrame(get_default_legs(spot_price, base_iv))
    st.rerun()

if col_btn2.button("🗑️ Clear Legs"):
    st.session_state["portfolios"][selected_pos] = pd.DataFrame(columns=["Enable", "Phase", "Type", "Strike", "DTE", "IV_%", "Qty", "Entry_Price"])
    st.rerun()

edited_df = st.sidebar.data_editor(
    st.session_state["portfolios"][selected_pos],
    num_rows="dynamic",
    key=f"portfolio_editor_{selected_pos}",
    column_config={
        "Enable": st.column_config.CheckboxColumn("Aktiv", default=True),
        "Phase": st.column_config.SelectboxColumn("Phase", options=["Initial", "Adjustment"], default="Initial"),
        "Type": st.column_config.SelectboxColumn("Typ", options=["C", "P"], required=True),
        "Strike": st.column_config.NumberColumn("Strike", min_value=1.0, step=1.0),
        "DTE": st.column_config.NumberColumn("DTE", min_value=0, step=1),
        "IV_%": st.column_config.NumberColumn("IV %", min_value=1.0, max_value=300.0, step=0.5),
        "Qty": st.column_config.NumberColumn("Qty (+/-)", step=1),
        "Entry_Price": st.column_config.NumberColumn("Entry ($)", step=0.05),
    }
)

st.session_state["portfolios"][selected_pos] = edited_df

# --- PNL & RISK CALCULATION FOR ACTIVE PORTFOLIO ---
active_legs = edited_df[edited_df["Enable"] == True].copy() if not edited_df.empty else pd.DataFrame()

initial_legs = active_legs[active_legs["Phase"] == "Initial"] if not active_legs.empty else pd.DataFrame()
adj_legs = active_legs[active_legs["Phase"] == "Adjustment"] if not active_legs.empty else pd.DataFrame()

margin_initial = calculate_phase_margin(initial_legs, spot_price)
margin_adj = calculate_phase_margin(adj_legs, spot_price)
margin_total = calculate_phase_margin(active_legs, spot_price)

valid_strikes = []
if not active_legs.empty and "Strike" in active_legs:
    valid_strikes = [float(s) for s in active_legs["Strike"].dropna() if pd.notna(s)]

if len(valid_strikes) > 0:
    min_strike = min(valid_strikes)
    max_strike = max(valid_strikes)
    x_min = min(spot_price * 0.85, min_strike * 0.90)
    x_max = max(spot_price * 1.15, max_strike * 1.10)
else:
    x_min = spot_price * 0.85
    x_max = spot_price * 1.15

spot_range = np.linspace(x_min, x_max, 400)

pnl_t0 = np.zeros_like(spot_range)
pnl_t1 = np.zeros_like(spot_range)
pnl_exp = np.zeros_like(spot_range)

leg_exp_curves = []

spot_pnl_t0, spot_pnl_t1 = 0.0, 0.0
tot_delta, tot_gamma, tot_theta, tot_vega = 0.0, 0.0, 0.0, 0.0

summary_rows = []

if not active_legs.empty:
    for idx, row in active_legs.iterrows():
        try:
            if pd.isna(row.get("Qty")) or pd.isna(row.get("Strike")) or pd.isna(row.get("DTE")) or pd.isna(row.get("IV_%")) or pd.isna(row.get("Entry_Price")):
                continue

            phase = str(row.get("Phase", "Initial"))
            opt_type = str(row["Type"])
            strike = float(row["Strike"])
            dte = float(row["DTE"])
            iv = float(row["IV_%"]) / 100.0
            qty = float(row["Qty"])
            entry = float(row["Entry_Price"])
            
            t_t0 = max(0.00001, dte / 365.0)
            t_t1 = max(0.00001, max(0.0, dte - 1) / 365.0)
            
            leg_exp_prices = np.where(opt_type == 'C', np.maximum(0, spot_range - strike), np.maximum(0, strike - spot_range))
            single_leg_pnl_exp = (leg_exp_prices - entry) * qty * 100.0
            
            pnl_exp += single_leg_pnl_exp
            leg_exp_curves.append((f"Leg {len(leg_exp_curves)+1}: {qty:+g} {opt_type} @ {strike}", single_leg_pnl_exp))
            
            t0_prices = np.array([bs_price(opt_type, s, strike, t_t0, risk_free_rate, iv) for s in spot_range])
            pnl_t0 += (t0_prices - entry) * qty * 100.0
            
            t1_prices = np.array([bs_price(opt_type, s, strike, t_t1, risk_free_rate, iv) for s in spot_range])
            pnl_t1 += (t1_prices - entry) * qty * 100.0
            
            val_t0 = bs_price(opt_type, spot_price, strike, t_t0, risk_free_rate, iv)
            val_t1 = bs_price(opt_type, spot_price, strike, t_t1, risk_free_rate, iv)
            spot_pnl_t0 += (val_t0 - entry) * qty * 100.0
            spot_pnl_t1 += (val_t1 - entry) * qty * 100.0
            
            g = bs_greeks(opt_type, spot_price, strike, t_t0, risk_free_rate, iv)
            l_delta = g['delta'] * qty * 100.0
            l_gamma = g['gamma'] * qty * 100.0
            l_theta = g['theta'] * qty * 100.0
            l_vega = g['vega'] * qty * 100.0

            tot_delta += l_delta
            tot_gamma += l_gamma
            tot_theta += l_theta
            tot_vega += l_vega

            summary_rows.append({
                "Phase": phase,
                "Typ": opt_type,
                "Strike": strike,
                "DTE": int(dte),
                "Qty": int(qty),
                "Entry ($)": entry,
                "Total Value ($)": round(entry * qty * 100.0, 2),
                "Delta": round(l_delta, 2),
                "Gamma": round(l_gamma, 4),
                "Theta ($)": round(l_theta, 2),
                "Vega ($)": round(l_vega, 2)
            })
        except (ValueError, KeyError, TypeError):
            continue

# --- MAIN WORKSPACE ---
st.title(f"📈 OptionNet Explorer Professional ({underlying_symbol})")
st.caption(f"Aktuell angezeigte Strategie/Position: **{selected_pos}** | Datenquelle: `{spot_source_str}`")

# KPI Spalten
if show_margin:
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("PnL T+0 (Heute)", f"${spot_pnl_t0:,.2f}")
    m2.metric("PnL T+1 (Morgen)", f"${spot_pnl_t1:,.2f}", f"${(spot_pnl_t1 - spot_pnl_t0):,.2f}")
    m3.metric("Position Delta", f"{tot_delta:,.2f}")
    m4.metric("Position Theta", f"${tot_theta:,.2f}")
    m5.metric("Position Vega", f"${tot_vega:,.2f}")
    m6.metric("Gesamt Margin", f"${margin_total:,.2f}")
else:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("PnL T+0 (Heute)", f"${spot_pnl_t0:,.2f}")
    m2.metric("PnL T+1 (Morgen)", f"${spot_pnl_t1:,.2f}", f"${(spot_pnl_t1 - spot_pnl_t0):,.2f}")
    m3.metric("Position Delta", f"{tot_delta:,.2f}")
    m4.metric("Position Theta", f"${tot_theta:,.2f}")
    m5.metric("Position Vega", f"${tot_vega:,.2f}")

st.markdown("---")

# SECTION 0: GESAMTPOSITION & LEGS SUMMARY
st.subheader(f"📊 Übersicht Gesamtposition & Adjustierungen — {selected_pos}")

if summary_rows:
    df_summary = pd.DataFrame(summary_rows)
    
    total_qty = df_summary["Qty"].sum()
    total_val = df_summary["Total Value ($)"].sum()
    
    total_row = pd.DataFrame([{
        "Phase": "GESAMTPOSITION",
        "Typ": "ALL",
        "Strike": "-",
        "DTE": "-",
        "Qty": total_qty,
        "Entry ($)": "-",
        "Total Value ($)": round(total_val, 2),
        "Delta": round(tot_delta, 2),
        "Gamma": round(tot_gamma, 4),
        "Theta ($)": round(tot_theta, 2),
        "Vega ($)": round(tot_vega, 2)
    }])
    
    full_summary_df = pd.concat([df_summary, total_row], ignore_index=True)
    
    def highlight_total_row(row):
        if row["Phase"] == "GESAMTPOSITION":
            return ['background-color: #1e3d59; color: #ffffff; font-weight: bold;'] * len(row)
        elif row["Phase"] == "Adjustment":
            return ['background-color: #2b2b36; color: #ffab40;'] * len(row)
        else:
            return [''] * len(row)

    styled_summary = full_summary_df.style.apply(highlight_total_row, axis=1)
    st.dataframe(styled_summary, use_container_width=True)
else:
    st.info("Keine aktiven Positionen ausgewählt. Bitte erstelle Legs in der SideBar.")

st.markdown("---")

# SECTION 1: MARGIN OVERVIEW PANEL
if show_margin:
    st.subheader(f"💵 Margin Breakdown — {selected_pos}")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    col_m1.metric("Initial Margin (Original)", f"${margin_initial:,.2f}")
    col_m2.metric("Adjustment Margin", f"${margin_adj:,.2f}")
    col_m3.metric("Gesamt Margin", f"${margin_total:,.2f}")
    
    rom_t0 = (spot_pnl_t0 / margin_total * 100.0) if margin_total > 0 else 0.0
    col_m4.metric("Return on Margin (T+0)", f"{rom_t0:.2f}%")
    
    st.markdown("---")

# SECTION 2: RISIKOGRAPH (GESAMTPOSITION + EINZELNE LEGS)
st.subheader(f"📉 Risikograph Gesamtposition (Net Payoff) — {selected_pos}")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=spot_range, 
    y=pnl_exp, 
    mode='lines', 
    name='Gesamtposition Expiration PnL', 
    line=dict(color='#29b6f6', width=3)
))

fig.add_trace(go.Scatter(
    x=spot_range, 
    y=pnl_t0, 
    mode='lines', 
    name='Gesamtposition T+0 (Heute)', 
    line=dict(color='#ff5252', width=2)
))

fig.add_trace(go.Scatter(
    x=spot_range, 
    y=pnl_t1, 
    mode='lines', 
    name='Gesamtposition T+1 (Morgen)', 
    line=dict(color='#66bb6a', width=1.5, dash='dash')
))

if show_individual_legs_graph and len(leg_exp_curves) > 1:
    for leg_name, leg_pnl in leg_exp_curves:
        fig.add_trace(go.Scatter(
            x=spot_range, 
            y=leg_pnl, 
            mode='lines', 
            name=f"Leg: {leg_name}", 
            line=dict(width=1, dash='dot'),
            opacity=0.5
        ))

fig.add_hline(y=0, line_dash="solid", line_color="#777777", line_width=1)
fig.add_vline(x=spot_price, line_dash="dot", line_color="#ffffff", line_width=1.5)

fig.update_layout(
    xaxis_title=f"Underlying Price ({underlying_symbol})",
    yaxis_title="Profit / Loss ($)",
    template="plotly_dark",
    height=450,
    hovermode="x unified",
    margin=dict(l=20, r=20, t=30, b=20)
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# SECTION 3: CLASSIC OPTION CHAIN & DTE COMPARISON
st.subheader(f"⛓️ Option Chain & Multi-DTE Analysis ({strike_step_val}-Schritte)")

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
    num_strikes_view = st.slider("Anzahl Strikes anzeigen", 11, 51, 21, step=2)

chain_df_1 = build_classic_option_chain(spot_price, selected_dte_1, base_iv, r=risk_free_rate, strike_step=strike_step_val, num_strikes=num_strikes_view)
chain_df_2 = build_classic_option_chain(spot_price, selected_dte_2, base_iv, r=risk_free_rate, strike_step=strike_step_val, num_strikes=num_strikes_view)

tab_chain1, tab_comp = st.tabs([f"🏛️ Option Chain (DTE {selected_dte_1})", f"⚔️ DTE {selected_dte_1} vs DTE {selected_dte_2} Matrix"])

with tab_chain1:
    st.markdown(f"**Option Chain Layout (CALLS | STRIKE | PUTS) — Spot @ `{spot_price}` (Schrittweite: `{strike_step_val}`)**")
    st.caption("🟦 **Blau hinterlegt:** ITM Calls ($K < \\text{Spot}$) | 🟩 **Grün hinterlegt:** ITM Puts ($K > \\text{Spot}$) | 🟥 **Rot:** ATM Strike")
    
    cols_to_hide = ["C_ITM", "P_ITM", "IS_ATM"]
    styled_df_1 = chain_df_1.style.apply(highlight_option_chain, axis=1).hide(axis="columns", subset=cols_to_hide)
    
    st.dataframe(
        styled_df_1,
        use_container_width=True,
        height=550
    )

with tab_comp:
    st.markdown(f"**Direktvergleich: DTE {selected_dte_1} vs. DTE {selected_dte_2} (Schrittweite: `{strike_step_val}`)**")
    
    comp_merged = pd.merge(
        chain_df_1[["STRIKE", "ATM_Marker", "IS_ATM", "C_ITM", "P_ITM", "C_IV_%", "C_Ask", "P_IV_%", "P_Ask"]],
        chain_df_2[["STRIKE", "C_IV_%", "C_Ask", "P_IV_%", "P_Ask"]],
        on="STRIKE",
        suffixes=(f" (DTE {selected_dte_1})", f" (DTE {selected_dte_2})")
    )
    
    comp_merged["Call IV Diff (%)"] = round(comp_merged[f"C_IV_% (DTE {selected_dte_2})"] - comp_merged[f"C_IV_% (DTE {selected_dte_1})"], 1)
    comp_merged["Put IV Diff (%)"] = round(comp_merged[f"P_IV_% (DTE {selected_dte_2})"] - comp_merged[f"P_IV_% (DTE {selected_dte_1})"], 1)
    comp_merged["Call Price Diff ($)"] = round(comp_merged[f"C_Ask (DTE {selected_dte_2})"] - comp_merged[f"C_Ask (DTE {selected_dte_1})"], 2)
    comp_merged["Put Price Diff ($)"] = round(comp_merged[f"P_Ask (DTE {selected_dte_2})"] - comp_merged[f"P_Ask (DTE {selected_dte_1})"], 2)

    cols_to_hide_comp = ["C_ITM", "P_ITM", "IS_ATM"]
    styled_comp = comp_merged.style.apply(highlight_option_chain, axis=1).hide(axis="columns", subset=cols_to_hide_comp)

    st.dataframe(
        styled_comp,
        use_container_width=True,
        height=550
    )
