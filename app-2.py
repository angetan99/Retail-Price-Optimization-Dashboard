# =============================================================================
# Retail Demand Forecasting Tool
# Trained on 2017 Brazilian e-commerce data (Kaggle: Retail Price Optimization)
# =============================================================================

# 1. Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.graph_objects as go
import joblib
import os
from sklearn.linear_model import LinearRegression

# =============================================================================
# 2. Data Loading and Preprocessing
# =============================================================================

DATA_PATH = "retail_price.csv"

# Fixed category list from training data (ensures prediction-time alignment)
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
# Reference category dropped to avoid dummy variable trap
REFERENCE_CATEGORY = "bed_bath_table"
CATEGORY_DUMMIES = [c for c in CATEGORIES if c != REFERENCE_CATEGORY]

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


@st.cache_data
def load_data():
    """Load and preprocess the dataset."""
    df = pd.read_csv(DATA_PATH)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps identically for training and prediction.
    This function is the single source of truth for feature creation.
    """
    df = df.copy()

    # --- comp_price_ratio ---
    # How this product's price compares to the average competitor price.
    # A ratio > 1 means the product costs more than the competitor.
    df["comp_price_ratio"] = (
        df["unit_price"] / df["comp_1"]
        + df["unit_price"] / df["comp_2"]
        + df["unit_price"] / df["comp_3"]
    ) / 3

    # --- log_qty ---
    # Log of quantity sold; used as the model target to reduce skew.
    # We reverse this with np.exp() when showing predictions.
    df["log_qty"] = np.log(df["qty"].clip(lower=1))

    # --- pct_price_change ---
    # How much the price changed compared to last month.
    # A value of 0.10 means a 10% price increase vs. last month.
    # Default to 0 if lag_price is missing or zero.
    lag = df["lag_price"].fillna(0)
    df["pct_price_change"] = np.where(
        lag == 0, 0, (df["unit_price"] - lag) / lag
    )

    # --- category_x_price interaction terms ---
    # Captures that different categories respond to price changes differently.
    # Each term = (1 if product is in that category else 0) × unit_price.
    for cat in CATEGORY_DUMMIES:
        df[f"cat_{cat}"] = (df["product_category_name"] == cat).astype(int) * df["unit_price"]

    # --- sin_month / cos_month ---
    # Cyclical encoding of the calendar month so January and December are
    # treated as adjacent rather than far apart.
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def get_feature_columns():
    """Return the ordered list of features used by the model."""
    cat_cols = [f"cat_{c}" for c in CATEGORY_DUMMIES]
    return [
        "unit_price",
        "freight_price",
        "comp_price_ratio",
        "pct_price_change",
        "sin_month",
        "cos_month",
        *cat_cols,
    ]


# =============================================================================
# 3. Model Training
# =============================================================================

@st.cache_resource
def train_model():
    """Load data, engineer features, and train the OLS model. Cached so it
    only runs once per session."""
    df = load_data()
    df_eng = engineer_features(df)

    feature_cols = get_feature_columns()
    X = df_eng[feature_cols]
    y = df_eng["log_qty"]

    model = LinearRegression()
    model.fit(X, y)

    # Build a coefficient lookup for the insights tab
    coef_dict = dict(zip(feature_cols, model.coef_))
    coef_dict["(Intercept)"] = model.intercept_

    return model, coef_dict


# =============================================================================
# 4. Prediction Function
# =============================================================================

def predict_quantity(
    model,
    unit_price: float,
    lag_price: float,
    comp_1: float,
    comp_2: float,
    comp_3: float,
    freight_price: float,
    category: str,
    month: int,
) -> float:
    """
    Build a single-row feature vector from user inputs and return
    the predicted quantity (reversed from log scale).
    """
    row = {
        "unit_price": unit_price,
        "lag_price": lag_price,
        "comp_1": comp_1,
        "comp_2": comp_2,
        "comp_3": comp_3,
        "freight_price": freight_price,
        "product_category_name": category,
        "month": month,
        "qty": 1,  # placeholder — not used in prediction
    }
    df_row = pd.DataFrame([row])
    df_row = engineer_features(df_row)

    X = df_row[get_feature_columns()]
    log_pred = model.predict(X)[0]
    return max(1, round(np.exp(log_pred)))


# =============================================================================
# 5. Chart Generation Functions
# =============================================================================

CHART_COLOR = "#4F8EF7"
ACCENT_COLOR = "#FF6B6B"

def chart_top_drivers(coef_dict: dict):
    """Bar chart: which variables most influence demand (by |coefficient|)."""
    skip = {"(Intercept)"}
    items = {k: v for k, v in coef_dict.items() if k not in skip}

    # Friendly labels
    label_map = {
        "unit_price": "Your Price",
        "freight_price": "Shipping Cost",
        "comp_price_ratio": "Price vs. Competitors",
        "pct_price_change": "Month-over-Month Price Change",
        "sin_month": "Seasonal Cycle (sine)",
        "cos_month": "Seasonal Cycle (cosine)",
    }
    for cat in CATEGORY_DUMMIES:
        label_map[f"cat_{cat}"] = f"{CATEGORY_LABELS.get(cat, cat)} × Price"

    labels = [label_map.get(k, k) for k in items]
    values = list(items.values())

    df_plot = pd.DataFrame({"label": labels, "coef": values})
    df_plot["abs_coef"] = df_plot["coef"].abs()
    df_plot = df_plot.sort_values("abs_coef", ascending=True).tail(12)
    colors = [ACCENT_COLOR if v < 0 else CHART_COLOR for v in df_plot["coef"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df_plot["label"], df_plot["coef"], color=colors)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Effect on Log(Quantity Sold)\n(positive = more units sold)")
    ax.set_title("What Drives Demand Most?", fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


def chart_category_sensitivity(coef_dict: dict):
    """Bar chart: category-price interaction coefficients."""
    cat_data = {}
    for cat in CATEGORY_DUMMIES:
        key = f"cat_{cat}"
        if key in coef_dict:
            cat_data[CATEGORY_LABELS.get(cat, cat)] = coef_dict[key]

    df_plot = pd.DataFrame.from_dict(cat_data, orient="index", columns=["coef"])
    df_plot = df_plot.sort_values("coef", ascending=True)
    colors = [ACCENT_COLOR if v < 0 else CHART_COLOR for v in df_plot["coef"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(df_plot.index, df_plot["coef"], color=colors)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Price Sensitivity (effect per unit increase in price)")
    ax.set_title("Price Sensitivity by Product Category\n(vs. Bed & Bath as baseline)",
                 fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


def chart_competitor_effect(coef_dict: dict):
    """Bar chart: your price effect vs. competitor ratio effect."""
    labels = ["Your Price\n(direct effect)", "Price vs.\nCompetitors"]
    values = [coef_dict.get("unit_price", 0), coef_dict.get("comp_price_ratio", 0)]
    colors = [ACCENT_COLOR if v < 0 else CHART_COLOR for v in values]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, values, color=colors, width=0.4)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Effect on Log(Quantity Sold)")
    ax.set_title("Your Price vs. Competitor Pricing:\nWhich Matters More?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def chart_seasonal_effect(coef_dict: dict):
    """Line chart: combined seasonal effect across calendar months."""
    sin_c = coef_dict.get("sin_month", 0)
    cos_c = coef_dict.get("cos_month", 0)
    months = np.arange(1, 13)
    effect = sin_c * np.sin(2 * np.pi * months / 12) + cos_c * np.cos(2 * np.pi * months / 12)

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(month_names, effect, marker="o", color=CHART_COLOR, linewidth=2)
    ax.fill_between(month_names, effect, alpha=0.15, color=CHART_COLOR)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Seasonal Boost / Drag on Log(Qty)")
    ax.set_title("When Are Customers Most Responsive?\n(Seasonal demand pattern)",
                 fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    return fig


# =============================================================================
# 6. Streamlit Interface
# =============================================================================

st.set_page_config(
    page_title="Retail Demand Forecaster",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Retail Demand Forecasting Tool")
st.caption(
    "Set your price, see your expected volume — before you commit. "
    "Trained on 2017 Brazilian e-commerce data."
)

# Load model once
model, coef_dict = train_model()

tab1, tab2, tab3 = st.tabs(["📊 Demand Predictor", "💡 Key Insights", "ℹ️ How It Works"])

# ---------------------------------------------------------------------------
# TAB 1 — Demand Predictor
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Enter Your Pricing & Product Details")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Your Product**")
        category = st.selectbox(
            "Product Category",
            options=CATEGORIES,
            format_func=lambda x: CATEGORY_LABELS.get(x, x),
        )
        unit_price = st.number_input(
            "Your Price (R$)",
            min_value=1.0, max_value=2000.0, value=89.90, step=0.50,
            help="The price you plan to charge for this product."
        )
        lag_price = st.number_input(
            "Last Month's Price (R$)",
            min_value=0.0, max_value=2000.0, value=89.90, step=0.50,
            help="Your price from the previous month. Used to calculate price momentum."
        )
        freight_price = st.number_input(
            "Shipping Cost (R$)",
            min_value=0.0, max_value=200.0, value=14.0, step=0.50,
            help="Average shipping cost passed on to the customer."
        )
        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            format_func=lambda m: [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ][m - 1],
            index=4,
        )

    with col_right:
        st.markdown("**Competitor Prices (R$)**")
        comp_1 = st.number_input("Competitor 1 Price", min_value=1.0, max_value=2000.0, value=99.90, step=0.50)
        comp_2 = st.number_input("Competitor 2 Price", min_value=1.0, max_value=2000.0, value=95.00, step=0.50)
        comp_3 = st.number_input("Competitor 3 Price", min_value=1.0, max_value=2000.0, value=89.00, step=0.50)

    st.divider()

    if st.button("🔮 Predict Demand", type="primary", use_container_width=True):
        qty_pred = predict_quantity(
            model, unit_price, lag_price, comp_1, comp_2, comp_3,
            freight_price, category, month
        )

        avg_comp = (comp_1 + comp_2 + comp_3) / 3
        pct_vs_comp = (unit_price - avg_comp) / avg_comp * 100
        direction = "above" if pct_vs_comp >= 0 else "below"
        pct_vs_comp_abs = abs(pct_vs_comp)

        pct_change = 0.0
        if lag_price > 0:
            pct_change = (unit_price - lag_price) / lag_price * 100

        st.metric(
            label="📦 Estimated Units Sold This Month",
            value=f"{qty_pred:,}",
            delta=None,
        )

        st.info(
            f"At **R$ {unit_price:.2f}**, you are priced **{pct_vs_comp_abs:.1f}% {direction}** "
            f"the average competitor price (R$ {avg_comp:.2f}). "
            + (
                f"Your price is **unchanged** from last month."
                if abs(pct_change) < 0.01 else
                f"Your price **{'increased' if pct_change > 0 else 'decreased'} "
                f"{abs(pct_change):.1f}%** compared to last month."
            )
            + f"\n\n**Estimated demand: {qty_pred:,} units.** "
            f"Projected monthly revenue: **R$ {qty_pred * unit_price:,.0f}**."
        )

        # Quick sensitivity nudge
        lower_price = max(1.0, unit_price * 0.95)
        qty_lower = predict_quantity(
            model, lower_price, lag_price, comp_1, comp_2, comp_3,
            freight_price, category, month
        )
        if qty_lower > qty_pred:
            rev_lower = qty_lower * lower_price
            rev_current = qty_pred * unit_price
            rev_delta = rev_lower - rev_current
            st.caption(
                f"💡 A 5% price reduction to R$ {lower_price:.2f} could lift volume to "
                f"~{qty_lower:,} units — estimated revenue impact: "
                f"{'▲' if rev_delta >= 0 else '▼'} R$ {abs(rev_delta):,.0f}."
            )

# ---------------------------------------------------------------------------
# TAB 2 — Key Insights Dashboard
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("What the Data Tells Us About Demand")

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        st.markdown("#### What Drives Demand Most?")
        st.caption("Bars show the direction and size of each factor's effect on units sold.")
        fig1 = chart_top_drivers(coef_dict)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

        st.markdown("#### Your Price vs. Competitors: Which Matters More?")
        st.caption(
            "Compares the direct effect of your own price against your position "
            "relative to competitors."
        )
        fig3 = chart_competitor_effect(coef_dict)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

    with insight_col2:
        st.markdown("#### Price Sensitivity by Category")
        st.caption(
            "How much does a price increase hurt demand in each category, "
            "relative to Bed & Bath (baseline)?"
        )
        fig2 = chart_category_sensitivity(coef_dict)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        st.markdown("#### When Are Customers Most Price-Responsive?")
        st.caption("Seasonal demand pattern derived from the cyclical month encoding.")
        fig4 = chart_seasonal_effect(coef_dict)
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

# ---------------------------------------------------------------------------
# TAB 3 — How It Works
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("How This Tool Works")

    st.markdown("""
### The Model

This tool uses **Ordinary Least Squares (OLS) linear regression** — a well-understood
statistical approach that finds the straight-line relationship between pricing inputs
and the quantity sold. It was trained on **676 monthly product observations** across
**9 categories** from a Brazilian e-commerce retailer in 2017.

The model predicts the **natural log of quantity sold** (a common technique to handle
skewed sales data), then converts the result back to a plain unit count using `exp()`.

---

### Engineered Features — Plain English

| Feature | What it means |
|---|---|
| **Your Price vs. Competitors** | The average ratio of your price to each of three competitors. A value above 1.0 means you're more expensive. |
| **Month-over-Month Price Change** | How much your price moved compared to last month. Customers notice price jumps. |
| **Category × Price Interactions** | Lets the model learn that, say, a R$ 10 price increase affects electronics differently than home décor. |
| **Seasonal Cycle (sin/cos month)** | Encodes the calendar month in a circular way — so December and January are treated as close together, not far apart. |
| **Log Quantity Target** | The model learns from log-scaled sales to avoid being skewed by rare huge months. All predictions are converted back to plain units. |

---

### What the Predicted Number Means

The output is the **model's best estimate** of how many units would sell in a typical
month at the price you entered, given the competitive environment and seasonal timing.
It is a directional guide — not a guaranteed forecast.

---

### Limitations to Keep in Mind

- **Data scope:** This model was trained on 2017 data from a single Brazilian retailer.
  Absolute numbers may not transfer directly to your context, but the directional
  relationships (higher price → lower volume, being expensive vs. competitors → lower volume)
  should hold broadly.
- **No promotions or inventory effects:** The model doesn't account for flash sales,
  stock-outs, or marketing spend.
- **Linear assumptions:** Real demand curves can be non-linear. This model provides
  a useful approximation, not a precise simulation.
- **Use directionally:** Treat predictions as one input into your pricing decision —
  not the final word.
    """)
