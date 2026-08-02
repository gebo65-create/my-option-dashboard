import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# ------------------------------------------------------------------------------
# 1. PAGE SETUP (ONE-Style Layout)
# ------------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="OptionNet Explorer Clone", page_icon="📈")

st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    .stMetric {background-color: #1e222d; padding: 10px; border-radius: 5px;}
    </style>
""", unsafe_allow_html=True)

st.title("📈 Option Analytics & Adjustment Center")

# ------------------------------------------------------------------------------
# 2. BLACK-SCHOLES ENGINE (Für T+0 Linie & Greeks)
# ------------------------------------------------------------------------------
def bs_price(S, K, T, r, sigma, option_type='C'):
    if T <= 0.001:
        return np.maximum(S - K, 0) if option_type == 'C' else np.maximum(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'C':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_greeks(S, K, T, r, sigma, option_type='C'):
    if T <= 0.001:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta = norm.cdf(d1) if option_type == 'C' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100.0
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
             - r * K * np.exp(-r * T) * (norm.cdf(d2) if option_type == 'C' else norm.cdf(-d2))) / 365.0
    
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}

# ------------------------------------------------------------------------------
# 3. SIDEBAR STEUERUNG
# ------------------------------------------------------------------------------
st.sidebar.header("⚙️ Basiswert & Einstellungen")
symbol = st.sidebar.selectbox("Underlying", ["SPX", "SPY", "QQQ"])

# Default Werte je nach Symbol
default_spot = 598.67 if symbol == "SPY" else (5980.0 if symbol == "SPX" else 480.0)
spot_price = st.sidebar.number_input("Aktueller Spot Price", value=float(default_spot), step=1.0)

r_rate = 0.045  # 4.5% risikofreier Zins

st.sidebar.markdown("---")
st.sidebar.header("🔀 Adjustierungs-Schalter")
show_adjustment = st.sidebar.checkbox("Adjustierung aktivieren / überlagern", value=True)

# ------------------------------------------------------------------------------
# 4. DEFAULT-POSITIONEN SPEICHERN (Session State)
# ------------------------------------------------------------------------------
if 'orig_legs' not in st.session_state:
    st.session_state.orig_legs = pd.DataFrame([
        {"Type": "P", "Strike": 606.0, "DTE": 14, "IV_%": 16.0, "Qty": -10, "Entry_Price": 5.74},
        {"Type": "P", "Strike": 607.0, "DTE": 21, "IV_%": 15.0, "Qty": 5, "Entry_Price": 11.11},
    ])

if 'adj_legs' not in st.session_state:
    st.session_state.adj_legs = pd.DataFrame([
        {"Type": "P", "Strike": 595.0, "DTE": 14, "IV_%": 17.0, "Qty": 5, "Entry_Price": 3.20},
    ])

# ------------------------------------------------------------------------------
# 5. MAIN LAYOUT (2 SPALTEN WIE OPTIONNET EXPLORER)
# ------------------------------------------------------------------------------
col_left, col_right = st.columns([0.42, 0.58])

# ==================== LINKES PANEL: POSITIONSEINGABE ====================
with col_left:
    st.subheader("📌 1. Original Position")
    edited_orig = st.data_editor(
        st.session_state.orig_legs, 
        num_rows="dynamic", 
        use_container_width=True,
        key="editor_orig"
    )

    if show_adjustment:
        st.subheader("🛠️ 2. Adjustierung (Zusätzliche Legs)")
        edited_adj = st.data_editor(
            st.session_state.adj_legs, 
            num_rows="dynamic", 
            use_container_width=True,
            key="editor_adj"
        )
    else:
        edited_adj = pd.DataFrame()

    st.markdown("---")
    st.subheader("💳 Margin-Schätzung")
    
    # Einfache Reg-T / Span Margin Approximation
    def calc_approx_margin(df, spot):
        if df.empty:
            return 0.0
        total_margin = 0
        mult = 100 if symbol in ["SPY", "QQQ"] else 100
        for _, row in df.iterrows():
            if row['Qty'] < 0: # Short Positionen
                total_margin += abs(row['Qty']) * (row['Strike'] * 0.15) * mult
        return max(total_margin, 2000.0 if not df.empty else 0.0)

    margin_orig = calc_approx_margin(edited_orig, spot_price)
    
    if show_adjustment and not edited_adj.empty:
        combined_df = pd.concat([edited_orig, edited_adj], ignore_index=True)
        margin_adj = calc_approx_margin(combined_df, spot_price)
    else:
        margin_adj = margin_orig

    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Margin Original", f"${margin_orig:,.0f}")
    if show_adjustment:
        m_col2.metric("Margin Adjustiert", f"${margin_adj:,.0f}", f"${margin_adj - margin_orig:+,.0f}")

# ==================== RECHTES PANEL: PAYOFF CHART & GREEKS ====================
with col_right:
    st.subheader("📈 Risk Chart (Expiration & T+0)")

    # Preisspektrum für X-Achse
    spot_range = np.linspace(spot_price * 0.90, spot_price * 1.10, 150)

    def compute_pnl_profile(df, s_range):
        pnl_exp = np.zeros_like(s_range)
        pnl_t0 = np.zeros_like(s_range)
        mult = 100

        if df.empty:
            return pnl_exp, pnl_t0

        for _, row in df.iterrows():
            K = row['Strike']
            T = max(row['DTE'], 0.001) / 365.0
            sigma = row['IV_%'] / 100.0
            qty = row['Qty']
            entry = row['Entry_Price']
            opt_type = row['Type']

            # Expiration Payoff
            if opt_type == 'C':
                payoff = np.maximum(s_range - K, 0)
            else:
                payoff = np.maximum(K - s_range, 0)
            pnl_exp += (payoff - entry) * qty * mult

            # T+0 Line (Heute)
            t0_prices = np.array([bs_price(S, K, T, r_rate, sigma, opt_type) for S in s_range])
            pnl_t0 += (t0_prices - entry) * qty * mult

        return pnl_exp, pnl_t0

    pnl_exp_orig, pnl_t0_orig = compute_pnl_profile(edited_orig, spot_range)

    fig = go.Figure()

    # Original Linien
    fig.add_trace(go.Scatter(x=spot_range, y=pnl_exp_orig, mode='lines', name='Original Expiration', line=dict(color='#2962FF', width=2)))
    fig.add_trace(go.Scatter(x=spot_range, y=pnl_t0_orig, mode='lines', name='Original T+0', line=dict(color='#FF5252', width=2)))

    # Adjustierte Linien
    if show_adjustment and not edited_adj.empty:
        combined_df = pd.concat([edited_orig, edited_adj], ignore_index=True)
        pnl_exp_adj, pnl_t0_adj = compute_pnl_profile(combined_df, spot_range)
        fig.add_trace(go.Scatter(x=spot_range, y=pnl_exp_adj, mode='lines', name='Adjusted Expiration', line=dict(color='#00E676', dash='dash', width=2)))
        fig.add_trace(go.Scatter(x=spot_range, y=pnl_t0_adj, mode='lines', name='Adjusted T+0', line=dict(color='#FFD600', dash='dash', width=2)))

    # Zero Line & Spot Marker
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.add_vline(x=spot_price, line_dash="dot", line_color="white", annotation_text=f"Spot: {spot_price}")

    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Underlying Price",
        yaxis_title="PnL ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------------------------
    # GREEKS DASHBOARD
    # --------------------------------------------------------------------------
    st.subheader("🛡️ Net Greeks Breakdown")

    def calc_net_greeks(df, S):
        net_d = net_g = net_t = net_v = 0
        mult = 100
        if df.empty:
            return net_d, net_g, net_t, net_v

        for _, row in df.iterrows():
            g = bs_greeks(S, row['Strike'], max(row['DTE'], 0.001)/365.0, r_rate, row['IV_%']/100.0, row['Type'])
            q = row['Qty'] * mult
            net_d += g['delta'] * q
            net_g += g['gamma'] * q
            net_t += g['theta'] * q
            net_v += g['vega'] * q
        return net_d, net_g, net_t, net_v

    d_orig, g_orig, t_orig, v_orig = calc_net_greeks(edited_orig, spot_price)

    if show_adjustment and not edited_adj.empty:
        combined_df = pd.concat([edited_orig, edited_adj], ignore_index=True)
        d_adj, g_adj, t_adj, v_adj = calc_net_greeks(combined_df, spot_price)
    else:
        d_adj, g_adj, t_adj, v_adj = d_orig, g_orig, t_orig, v_orig

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Delta", f"{d_orig:.2f}", f"{d_adj - d_orig:+.2f}" if show_adjustment else None)
    g2.metric("Gamma", f"{g_orig:.3f}", f"{g_adj - g_orig:+.3f}" if show_adjustment else None)
    g3.metric("Theta ($/Tag)", f"{t_orig:.2f}", f"{t_adj - t_orig:+.2f}" if show_adjustment else None)
    g4.metric("Vega", f"{v_orig:.2f}", f"{v_adj - v_orig:+.2f}" if show_adjustment else None)
