# =============================================================================
# Retail Demand Forecasting Tool
# Trained on 2017 Brazilian e-commerce data (Kaggle: Retail Price Optimization)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold

# =============================================================================
# Constants
# =============================================================================

DATA_PATH = "retail_price.csv"

CATEGORIES = [
    "bed_bath_table",
    "computers_accessories",
    "consoles_games",
    "cool_stuff",
    "furniture_decor",
    "garden_tools",
    "health_beauty",
    "perfumery",
    "watches_gifts",
]

CATEGORY_LABELS = {
    "bed_bath_table": "Bed & Bath",
    "computers_accessories": "Computers & Accessories",
    "consoles_games": "Consoles & Games",
    "cool_stuff": "Cool Stuff",
    "furniture_decor": "Furniture & Decor",
    "garden_tools": "Garden Tools",
    "health_beauty": "Health & Beauty",
    "perfumery": "Perfumery",
    "watches_gifts": "Watches & Gifts",
}

NUMERIC_FEATURES = [
    "unit_price",
    "freight_price",
    "pct_price_change",
    "product_score",
    "holiday",
    "sin_month",
    "cos_month",
]
CATEGORICAL_FEATURES = ["product_category_name"]

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# ── Palette ──────────────────────────────────────────────────────────────────
PRIMARY   = "#1B4F8A"   # deep navy
SECONDARY = "#2E7DD1"   # mid blue
ACCENT    = "#E8523A"   # warm red-orange
LIGHT_BG  = "#F4F7FB"   # off-white panel
MUTED     = "#6B7A8D"   # muted slate

# =============================================================================
# Global style injection
# =============================================================================

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Page background ── */
    .stApp {
        background-color: #F4F7FB;
    }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Top wordmark bar ── */
    .wordmark {
        display: flex;
        align-items: baseline;
        gap: 10px;
        padding: 28px 0 4px 0;
        border-bottom: 2px solid #1B4F8A;
        margin-bottom: 28px;
    }
    .wordmark-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        color: #1B4F8A;
        line-height: 1;
        margin: 0;
    }
    .wordmark-badge {
        background: #1B4F8A;
        color: #fff;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 3px 8px;
        border-radius: 3px;
    }
    .wordmark-sub {
        color: #6B7A8D;
        font-size: 0.85rem;
        margin-left: auto;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: transparent;
        border-bottom: 2px solid #D6E0EF;
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.82rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #6B7A8D;
        padding: 10px 22px;
        border-radius: 4px 4px 0 0;
        border: none;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #1B4F8A !important;
        background: #fff !important;
        border-top: 2px solid #1B4F8A !important;
        font-weight: 600 !important;
    }

    /* ── Section headers ── */
    .section-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #6B7A8D;
        margin-bottom: 12px;
        margin-top: 8px;
    }

    /* ── Input group card ── */
    .input-card {
        background: #fff;
        border: 1px solid #D6E0EF;
        border-radius: 10px;
        padding: 24px 22px 18px 22px;
    }

    /* ── Result panel ── */
    .result-panel {
        background: #1B4F8A;
        border-radius: 10px;
        padding: 28px 26px;
        color: #fff;
        height: 100%;
    }
    .result-panel-empty {
        background: #fff;
        border: 2px dashed #D6E0EF;
        border-radius: 10px;
        padding: 28px 26px;
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 260px;
    }
    .result-panel-empty p {
        color: #6B7A8D;
        font-size: 0.9rem;
        text-align: center;
    }
    .result-headline {
        font-family: 'DM Serif Display', serif;
        font-size: 3.2rem;
        color: #fff;
        line-height: 1;
        margin: 4px 0 0 0;
    }
    .result-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.65);
        margin-bottom: 2px;
    }
    .result-revenue {
        font-size: 1.4rem;
        font-weight: 300;
        color: rgba(255,255,255,0.9);
        margin-top: 14px;
    }
    .result-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.2);
        margin: 20px 0;
    }
    .nudge-box {
        background: #FFF4F2;
        border-left: 3px solid #E8523A;
        border-radius: 6px;
        padding: 11px 15px;
        font-size: 0.83rem;
        color: #3D4A5C;
        margin-top: 10px;
        line-height: 1.55;
    }
    .price-change-box {
        background: #EBF1FA;
        border-left: 3px solid #2E7DD1;
        border-radius: 6px;
        padding: 11px 15px;
        font-size: 0.83rem;
        color: #3D4A5C;
        margin-top: 10px;
    }

    /* ── Stat tiles (Key Insights) ── */
    .stat-tile {
        background: #fff;
        border: 1px solid #D6E0EF;
        border-radius: 10px;
        padding: 20px 20px 16px 20px;
        text-align: center;
    }
    .stat-tile-value {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        color: #1B4F8A;
        line-height: 1;
    }
    .stat-tile-label {
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #6B7A8D;
        margin-top: 6px;
    }

    /* ── Model card (How It Works) ── */
    .model-card {
        background: #fff;
        border: 1px solid #D6E0EF;
        border-radius: 10px;
        padding: 24px 24px 20px 24px;
        margin-bottom: 16px;
    }
    .model-card h4 {
        font-family: 'DM Serif Display', serif;
        font-size: 1.1rem;
        color: #1B4F8A;
        margin: 0 0 10px 0;
    }
    .model-card p, .model-card li {
        font-size: 0.88rem;
        color: #3D4A5C;
        line-height: 1.65;
    }

    /* ── Predict button ── */
    .stButton > button[kind="primary"] {
        background: #1B4F8A !important;
        color: #fff !important;
        border: none !important;
        border-radius: 6px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.83rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        padding: 0.6rem 1rem !important;
        transition: background 0.18s !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #2E7DD1 !important;
    }

    /* ── Inputs ── */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        color: #3D4A5C !important;
        letter-spacing: 0.02em !important;
    }

    /* ── Chart containers ── */
    .chart-card {
        background: #fff;
        border: 1px solid #D6E0EF;
        border-radius: 10px;
        padding: 20px 16px 12px 16px;
    }

    /* ── Divider ── */
    hr { border-color: #D6E0EF !important; }

    /* ── Insight pills ── */
    .insight-pill {
        display: inline-block;
        background: #EBF1FA;
        color: #1B4F8A;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
        margin: 3px 3px 3px 0;
    }

    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# Data & Feature Engineering
# =============================================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lag = df["lag_price"].fillna(0)
    df["pct_price_change"] = np.where(lag == 0, 0, (df["unit_price"] - lag) / lag)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    if "qty" in df.columns:
        df["log_qty"] = np.log(df["qty"].clip(lower=1))
    return df


# =============================================================================
# Model Training
# =============================================================================

@st.cache_resource
def train_model():
    df = load_data()
    df = engineer_features(df)

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df["log_qty"]

    preprocessor = ColumnTransformer(
        [("ohe", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("gbm", GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        )),
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring="r2")
    cv_r2 = cv_scores.mean()

    pipeline.fit(X, y)

    feature_names_ohe = pipeline.named_steps["prep"].transformers_[0][1].get_feature_names_out(CATEGORICAL_FEATURES)
    all_feature_names = list(feature_names_ohe) + NUMERIC_FEATURES
    importances = pipeline.named_steps["gbm"].feature_importances_
    importance_dict = dict(zip(all_feature_names, importances))

    return pipeline, importance_dict, cv_r2


# =============================================================================
# Prediction
# =============================================================================

def predict_quantity(pipeline, unit_price, lag_price, freight_price,
                     product_score, holiday, category, month) -> int:
    row = {
        "unit_price": unit_price, "lag_price": lag_price,
        "freight_price": freight_price, "product_score": product_score,
        "holiday": holiday, "product_category_name": category,
        "month": month, "qty": 1,
    }
    df_row = engineer_features(pd.DataFrame([row]))
    log_pred = pipeline.predict(df_row[CATEGORICAL_FEATURES + NUMERIC_FEATURES])[0]
    return max(1, round(np.exp(log_pred)))


# =============================================================================
# Chart helpers — styled to match palette
# =============================================================================

def _apply_chart_style(fig, ax):
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F4F7FB")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D6E0EF")
    ax.spines["bottom"].set_color("#D6E0EF")
    ax.tick_params(colors="#6B7A8D", labelsize=8.5)
    ax.xaxis.label.set_color("#6B7A8D")
    ax.yaxis.label.set_color("#6B7A8D")
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.title.set_color("#1B4F8A")
    ax.title.set_fontsize(11)
    ax.title.set_fontweight("bold")
    fig.tight_layout(pad=1.6)


def chart_feature_importance(importance_dict: dict):
    label_map = {
        "unit_price": "Your Price",
        "freight_price": "Shipping Cost",
        "pct_price_change": "MoM Price Change",
        "product_score": "Product Rating",
        "holiday": "Holidays in Period",
        "sin_month": "Seasonal Cycle (sin)",
        "cos_month": "Seasonal Cycle (cos)",
    }
    cat_importance = sum(v for k, v in importance_dict.items() if k.startswith("product_category_name"))
    filtered = {label_map.get(k, k): v for k, v in importance_dict.items()
                if not k.startswith("product_category_name")}
    filtered["Product Category"] = cat_importance

    df_plot = pd.DataFrame(list(filtered.items()), columns=["label", "importance"]).sort_values("importance")

    colors = [ACCENT if i == len(df_plot) - 1 else SECONDARY for i in range(len(df_plot))]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.barh(df_plot["label"], df_plot["importance"], color=colors, height=0.55)
    ax.set_xlabel("Relative Importance")
    ax.set_title("What Drives Demand?")
    ax.bar_label(bars, fmt="%.3f", padding=4, color="#6B7A8D", fontsize=8)
    ax.set_xlim(0, df_plot["importance"].max() * 1.22)
    _apply_chart_style(fig, ax)
    return fig


def chart_price_curve(pipeline, base_kwargs: dict):
    prices = np.linspace(base_kwargs["unit_price"] * 0.5, base_kwargs["unit_price"] * 1.5, 50)
    qtys = [predict_quantity(pipeline, **{**base_kwargs, "unit_price": p}) for p in prices]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.fill_between(prices, qtys, alpha=0.12, color=SECONDARY)
    ax.plot(prices, qtys, color=SECONDARY, linewidth=2.2)
    ax.axvline(base_kwargs["unit_price"], color=ACCENT, linewidth=1.6,
               linestyle="--", label=f"Current  R${base_kwargs['unit_price']:.0f}")
    ax.set_xlabel("Unit Price (R$)")
    ax.set_ylabel("Predicted Units Sold")
    ax.set_title("Demand vs. Price")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8.5, framealpha=0.7, edgecolor="#D6E0EF")
    _apply_chart_style(fig, ax)
    return fig


def chart_seasonal(pipeline, base_kwargs: dict):
    qtys = [predict_quantity(pipeline, **{**base_kwargs, "month": m}) for m in range(1, 13)]
    short_months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    x = range(12)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.fill_between(x, qtys, alpha=0.12, color=SECONDARY)
    ax.plot(x, qtys, marker="o", markersize=5, color=SECONDARY, linewidth=2.2)
    ax.axvline(base_kwargs["month"] - 1, color=ACCENT, linewidth=1.6,
               linestyle="--", label=MONTH_NAMES[base_kwargs["month"] - 1])
    ax.set_xticks(list(x))
    ax.set_xticklabels(short_months)
    ax.set_ylabel("Predicted Units Sold")
    ax.set_title("Seasonal Demand Pattern")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8.5, framealpha=0.7, edgecolor="#D6E0EF")
    _apply_chart_style(fig, ax)
    return fig


def chart_category_bars(pipeline, base_kwargs: dict):
    """Predicted demand across all categories at current price settings."""
    qtys = []
    for cat in CATEGORIES:
        q = predict_quantity(pipeline, **{**base_kwargs, "category": cat})
        qtys.append(q)

    labels = [CATEGORY_LABELS[c] for c in CATEGORIES]
    colors = [ACCENT if CATEGORIES[i] == base_kwargs["category"] else SECONDARY
              for i in range(len(CATEGORIES))]

    sorted_pairs = sorted(zip(qtys, labels, colors), key=lambda x: x[0])
    qtys_s, labels_s, colors_s = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.barh(list(labels_s), list(qtys_s), color=list(colors_s), height=0.55)
    ax.bar_label(bars, fmt="%d", padding=4, color="#6B7A8D", fontsize=8)
    ax.set_xlabel("Predicted Units Sold")
    ax.set_title("Demand by Category (Current Price)")
    ax.set_xlim(0, max(qtys_s) * 1.18)
    _apply_chart_style(fig, ax)
    return fig


# =============================================================================
# App shell
# =============================================================================

st.set_page_config(page_title="Retail Demand Forecaster", page_icon="🛒", layout="wide")
inject_css()

# Wordmark
st.markdown("""
<div class="wordmark">
  <span class="wordmark-title">Demand Forecaster</span>
  <span class="wordmark-badge">Beta</span>
  <span class="wordmark-sub">Retail Price Optimization · 2017 Brazilian E-Commerce</span>
</div>
""", unsafe_allow_html=True)

pipeline, importance_dict, cv_r2 = train_model()

tab1, tab2 = st.tabs(["  📊  Demand Predictor  ", "  💡  Insights & Methodology  "])


# =============================================================================
# TAB 1 — Predictor
# =============================================================================
with tab1:

    # ── Input / result split ──────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<p class="section-label">Product Inputs</p>', unsafe_allow_html=True)
        st.markdown('<div class="input-card">', unsafe_allow_html=True)

        category = st.selectbox(
            "Product Category",
            options=CATEGORIES,
            format_func=lambda x: CATEGORY_LABELS.get(x, x),
        )
        c1, c2 = st.columns(2)
        with c1:
            unit_price = st.number_input("Your Price (R$)", min_value=1.0, max_value=2000.0, value=89.90, step=0.50)
        with c2:
            lag_price  = st.number_input("Last Month's Price (R$)", min_value=0.0, max_value=2000.0, value=89.90, step=0.50)

        c3, c4 = st.columns(2)
        with c3:
            freight_price = st.number_input("Shipping Cost (R$)", min_value=0.0, max_value=200.0, value=14.0, step=0.50)
        with c4:
            holiday = st.number_input("Holidays in Period", min_value=0, max_value=10, value=1, step=1)

        product_score = st.slider("Product Rating", min_value=1.0, max_value=5.0, value=4.1, step=0.1)

        month = st.selectbox(
            "Forecast Month",
            options=list(range(1, 13)),
            format_func=lambda m: MONTH_NAMES[m - 1],
            index=4,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        predict_clicked = st.button("Run Forecast →", type="primary", use_container_width=True)

    # ── Results panel ─────────────────────────────────────────────────────
    with right:
        st.markdown('<p class="section-label">Forecast Output</p>', unsafe_allow_html=True)

        if predict_clicked:
            base_kwargs = dict(
                unit_price=unit_price, lag_price=lag_price,
                freight_price=freight_price, product_score=product_score,
                holiday=holiday, category=category, month=month,
            )
            qty_pred = predict_quantity(pipeline, **base_kwargs)
            revenue  = qty_pred * unit_price
            pct_change = ((unit_price - lag_price) / lag_price * 100) if lag_price > 0 else 0.0

            lower_price = max(1.0, unit_price * 0.95)
            qty_lower   = predict_quantity(pipeline, **{**base_kwargs, "unit_price": lower_price})
            rev_delta   = (qty_lower * lower_price) - revenue
            show_nudge  = qty_lower > qty_pred

            # ── Main result panel (no nested dynamic HTML) ──
            st.markdown(f"""
            <div class="result-panel">
              <div class="result-label">Estimated Units Sold</div>
              <div class="result-headline">{qty_pred:,}</div>
              <div class="result-revenue">R$ {revenue:,.0f} projected revenue</div>
              <hr class="result-divider">
              <div class="result-label">Category</div>
              <div style="color:#fff; font-size:0.92rem; margin-bottom:6px;">
                {CATEGORY_LABELS.get(category, category)} &nbsp;·&nbsp; {MONTH_NAMES[month-1]}
              </div>
              <div class="result-label">Inputs</div>
              <div style="color:rgba(255,255,255,0.8); font-size:0.82rem; line-height:1.8;">
                Price R${unit_price:.2f} &nbsp;|&nbsp;
                Shipping R${freight_price:.2f} &nbsp;|&nbsp;
                Rating {product_score:.1f} ★ &nbsp;|&nbsp;
                {holiday} holiday{'s' if holiday != 1 else ''}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Price change callout — separate render call ──
            if abs(pct_change) >= 0.01:
                direction = "increase" if pct_change > 0 else "decrease"
                icon = "↑" if pct_change > 0 else "↓"
                st.markdown(f"""
                <div class="price-change-box">
                  {icon} Price is a <strong>{abs(pct_change):.1f}% {direction}</strong>
                  from last month (R${lag_price:.2f})
                </div>""", unsafe_allow_html=True)

            # ── Nudge — separate render call ──
            if show_nudge:
                arrow = "▲" if rev_delta >= 0 else "▼"
                st.markdown(f"""
                <div class="nudge-box">
                  💡 <strong>Price sensitivity alert:</strong> A 5% cut to R${lower_price:.2f}
                  could lift volume to ~{qty_lower:,} units
                  ({arrow} R${abs(rev_delta):,.0f} net revenue impact).
                </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="result-panel-empty">
              <p>Configure your product inputs on the left,<br>then click <strong>Run Forecast</strong> to see results.</p>
            </div>""", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────
    if predict_clicked:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">Scenario Analysis</p>', unsafe_allow_html=True)
        ch1, ch2 = st.columns(2, gap="large")

        with ch1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig_curve = chart_price_curve(pipeline, base_kwargs)
            st.pyplot(fig_curve, use_container_width=True)
            plt.close(fig_curve)
            st.markdown('</div>', unsafe_allow_html=True)

        with ch2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig_season = chart_seasonal(pipeline, base_kwargs)
            st.pyplot(fig_season, use_container_width=True)
            plt.close(fig_season)
            st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 2 — Insights & Methodology (merged)
# =============================================================================
with tab2:

    # ── Stat tiles row ────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Model at a Glance</p>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4, gap="medium")

    with s1:
        st.markdown(f"""
        <div class="stat-tile">
          <div class="stat-tile-value">{cv_r2:.2f}</div>
          <div class="stat-tile-label">Cross-Val R²</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="stat-tile">
          <div class="stat-tile-value">676</div>
          <div class="stat-tile-label">Training Observations</div>
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="stat-tile">
          <div class="stat-tile-value">9</div>
          <div class="stat-tile-label">Product Categories</div>
        </div>""", unsafe_allow_html=True)
    with s4:
        st.markdown("""
        <div class="stat-tile">
          <div class="stat-tile-value">200</div>
          <div class="stat-tile-label">Boosting Estimators</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Feature Analysis</p>', unsafe_allow_html=True)
    ci1, ci2 = st.columns(2, gap="large")

    with ci1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig_imp = chart_feature_importance(importance_dict)
        st.pyplot(fig_imp, use_container_width=True)
        plt.close(fig_imp)
        st.markdown('</div>', unsafe_allow_html=True)

    with ci2:
        # Category comparison using default inputs
        default_kwargs = dict(
            unit_price=89.90, lag_price=89.90, freight_price=14.0,
            product_score=4.1, holiday=1, category="health_beauty", month=5,
        )
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig_cat = chart_category_bars(pipeline, default_kwargs)
        st.pyplot(fig_cat, use_container_width=True)
        plt.close(fig_cat)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── Methodology cards ─────────────────────────────────────────────────
    st.markdown('<p class="section-label">Methodology</p>', unsafe_allow_html=True)
    mc1, mc2, mc3 = st.columns(3, gap="medium")

    with mc1:
        st.markdown("""
        <div class="model-card">
          <h4>Algorithm</h4>
          <p>Gradient Boosting Regressor (GBM) via scikit-learn. Builds 200 shallow
          decision trees sequentially, each correcting the errors of the last.
          Captures non-linear price-demand relationships that linear models miss.</p>
          <br>
          <span class="insight-pill">n_estimators: 200</span>
          <span class="insight-pill">max_depth: 3</span>
          <span class="insight-pill">lr: 0.05</span>
          <span class="insight-pill">subsample: 0.8</span>
        </div>""", unsafe_allow_html=True)

    with mc2:
        st.markdown("""
        <div class="model-card">
          <h4>Target & Evaluation</h4>
          <p>Predicts <strong>log(quantity sold)</strong> to handle skewed sales
          distributions, converted back via <code>exp()</code> at output.
          Performance is measured via <strong>5-fold cross-validation R²</strong>
          — the average score on data the model never trained on.</p>
          <br>
          <span class="insight-pill">Log-transformed target</span>
          <span class="insight-pill">5-fold CV</span>
          <span class="insight-pill">R² metric</span>
        </div>""", unsafe_allow_html=True)

    with mc3:
        st.markdown("""
        <div class="model-card">
          <h4>Limitations</h4>
          <ul>
            <li>2017 Brazilian e-commerce data — directional, not absolute</li>
            <li>No competitor pricing, marketing, or brand signals</li>
            <li>Use for scenario comparison, not as a sales guarantee</li>
          </ul>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Input reference table ─────────────────────────────────────────────
    st.markdown('<p class="section-label">Input Reference</p>', unsafe_allow_html=True)
    st.markdown('<div class="model-card">', unsafe_allow_html=True)
    st.markdown("""
| Input | Type | What It Captures |
|---|---|---|
| **Product Category** | Categorical | One-hot encoded; each category has its own demand baseline |
| **Your Price** | Numeric | Primary demand driver — the price customers see |
| **Last Month's Price** | Numeric | Used to compute month-over-month % change |
| **Shipping Cost** | Numeric | Adds to effective price from the customer's perspective |
| **Product Rating** | Numeric (1–5) | Higher-rated products sell more at equivalent prices |
| **Forecast Month** | Cyclical | Encoded as sin/cos pair to treat Jan and Dec as adjacent |
| **Holidays in Period** | Numeric | More holidays in a period tend to lift demand |
""")
    st.markdown('</div>', unsafe_allow_html=True)
