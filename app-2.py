# =============================================================================
# Retail Demand Forecasting Tool
# Trained on 2017 Brazilian e-commerce data (Kaggle: Retail Price Optimization)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
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

CHART_COLOR = "#4F8EF7"
ACCENT_COLOR = "#FF6B6B"

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# =============================================================================
# Data & Feature Engineering
# =============================================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    lag = df["lag_price"].fillna(0)
    df["pct_price_change"] = np.where(
        lag == 0, 0, (df["unit_price"] - lag) / lag
    )
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

def predict_quantity(
    pipeline,
    unit_price: float,
    lag_price: float,
    freight_price: float,
    product_score: float,
    holiday: int,
    category: str,
    month: int,
) -> int:
    row = {
        "unit_price": unit_price,
        "lag_price": lag_price,
        "freight_price": freight_price,
        "product_score": product_score,
        "holiday": holiday,
        "product_category_name": category,
        "month": month,
        "qty": 1,
    }
    df_row = pd.DataFrame([row])
    df_row = engineer_features(df_row)
    X = df_row[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    log_pred = pipeline.predict(X)[0]
    return max(1, round(np.exp(log_pred)))


# =============================================================================
# Charts
# =============================================================================

def chart_feature_importance(importance_dict: dict):
    label_map = {
        "unit_price": "Your Price",
        "freight_price": "Shipping Cost",
        "pct_price_change": "Month-over-Month Price Change",
        "product_score": "Product Rating",
        "holiday": "Holidays in Period",
        "sin_month": "Seasonal Cycle (sine)",
        "cos_month": "Seasonal Cycle (cosine)",
    }
    cat_importance = sum(v for k, v in importance_dict.items() if k.startswith("product_category_name"))
    filtered = {label_map.get(k, k): v for k, v in importance_dict.items()
                if not k.startswith("product_category_name")}
    filtered["Product Category"] = cat_importance

    df_plot = pd.DataFrame(list(filtered.items()), columns=["label", "importance"])
    df_plot = df_plot.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(df_plot["label"], df_plot["importance"], color=CHART_COLOR)
    ax.set_xlabel("Relative importance")
    ax.set_title("What Drives Demand?", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


def chart_price_curve(pipeline, base_kwargs: dict):
    prices = np.linspace(base_kwargs["unit_price"] * 0.5, base_kwargs["unit_price"] * 1.5, 50)
    qtys = [predict_quantity(pipeline, **{**base_kwargs, "unit_price": p}) for p in prices]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(prices, qtys, color=CHART_COLOR, linewidth=2)
    ax.axvline(base_kwargs["unit_price"], color=ACCENT_COLOR, linewidth=1.5,
               linestyle="--", label="Current price")
    ax.fill_between(prices, qtys, alpha=0.1, color=CHART_COLOR)
    ax.set_xlabel("Unit Price (R$)")
    ax.set_ylabel("Predicted Units Sold")
    ax.set_title("How Demand Changes with Price", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    return fig


def chart_seasonal(pipeline, base_kwargs: dict):
    qtys = [predict_quantity(pipeline, **{**base_kwargs, "month": m}) for m in range(1, 13)]
    short_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(short_months, qtys, marker="o", color=CHART_COLOR, linewidth=2)
    ax.fill_between(short_months, qtys, alpha=0.1, color=CHART_COLOR)
    ax.axvline(base_kwargs["month"] - 1, color=ACCENT_COLOR, linewidth=1.5,
               linestyle="--", label="Current month")
    ax.set_ylabel("Predicted Units Sold")
    ax.set_title("Seasonal Demand Pattern", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    return fig


# =============================================================================
# App
# =============================================================================

st.set_page_config(page_title="Retail Demand Forecaster", page_icon="🛒", layout="wide")

st.title("🛒 Retail Demand Forecasting Tool")
st.caption("Enter your product details to estimate how many units you'll sell this month.")

pipeline, importance_dict, cv_r2 = train_model()

tab1, tab2, tab3 = st.tabs(["📊 Demand Predictor", "💡 Key Insights", "ℹ️ How It Works"])

# ---------------------------------------------------------------------------
# TAB 1 — Predictor
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Product Details")

    col1, col2 = st.columns(2)

    with col1:
        category = st.selectbox(
            "Product Category",
            options=CATEGORIES,
            format_func=lambda x: CATEGORY_LABELS.get(x, x),
        )
        unit_price = st.number_input(
            "Your Price (R$)", min_value=1.0, max_value=2000.0, value=89.90, step=0.50,
        )
        lag_price = st.number_input(
            "Last Month's Price (R$)", min_value=0.0, max_value=2000.0, value=89.90, step=0.50,
        )
        freight_price = st.number_input(
            "Shipping Cost (R$)", min_value=0.0, max_value=200.0, value=14.0, step=0.50,
        )

    with col2:
        product_score = st.slider(
            "Product Rating", min_value=1.0, max_value=5.0, value=4.1, step=0.1,
        )
        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            format_func=lambda m: MONTH_NAMES[m - 1],
            index=4,
        )
        holiday = st.number_input(
            "Holidays in Period", min_value=0, max_value=10, value=1, step=1,
        )

    st.divider()

    if st.button("🔮 Predict Demand", type="primary", use_container_width=True):
        base_kwargs = dict(
            unit_price=unit_price,
            lag_price=lag_price,
            freight_price=freight_price,
            product_score=product_score,
            holiday=holiday,
            category=category,
            month=month,
        )
        qty_pred = predict_quantity(pipeline, **base_kwargs)
        revenue = qty_pred * unit_price

        pct_change = ((unit_price - lag_price) / lag_price * 100) if lag_price > 0 else 0.0

        m1, m2 = st.columns(2)
        m1.metric("📦 Estimated Units Sold", f"{qty_pred:,}")
        m2.metric("💰 Projected Revenue", f"R$ {revenue:,.0f}")

        if abs(pct_change) >= 0.01:
            direction = "increase" if pct_change > 0 else "decrease"
            st.info(f"Your price is a **{abs(pct_change):.1f}% {direction}** from last month (R$ {lag_price:.2f}).")

        # 5% reduction nudge
        lower_price = max(1.0, unit_price * 0.95)
        qty_lower = predict_quantity(pipeline, **{**base_kwargs, "unit_price": lower_price})
        if qty_lower > qty_pred:
            rev_delta = (qty_lower * lower_price) - revenue
            st.caption(
                f"💡 A 5% price cut to R$ {lower_price:.2f} could lift volume to ~{qty_lower:,} units "
                f"({'▲' if rev_delta >= 0 else '▼'} R$ {abs(rev_delta):,.0f} revenue)."
            )

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            fig_curve = chart_price_curve(pipeline, base_kwargs)
            st.pyplot(fig_curve, use_container_width=True)
            plt.close(fig_curve)
        with col_b:
            fig_season = chart_seasonal(pipeline, base_kwargs)
            st.pyplot(fig_season, use_container_width=True)
            plt.close(fig_season)

# ---------------------------------------------------------------------------
# TAB 2 — Key Insights
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("What the Data Tells Us About Demand")
    st.caption(f"Cross-validated R² = **{cv_r2:.2f}** — the model explains roughly {cv_r2*100:.0f}% of demand variation on unseen data.")

    fig_imp = chart_feature_importance(importance_dict)
    st.pyplot(fig_imp, use_container_width=True)
    plt.close(fig_imp)

# ---------------------------------------------------------------------------
# TAB 3 — How It Works
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("How This Tool Works")
    st.markdown(f"""
### The Model

This tool uses a **Gradient Boosting Regressor (GBM)** trained on **676 monthly product
observations** across **9 categories** from a Brazilian e-commerce retailer in 2017.
It predicts **log(quantity sold)**, then converts back to units via `exp()`.

Cross-validated R² = **{cv_r2:.2f}**, meaning the model explains roughly
{cv_r2*100:.0f}% of demand variation on data it hasn't seen. Treat predictions as
directional guidance, not guaranteed forecasts.

---

### Inputs & What They Mean

| Input | What it captures |
|---|---|
| **Product Category** | Different categories have very different price sensitivities |
| **Your Price** | The primary driver of demand |
| **Last Month's Price** | Captures whether you're raising or lowering prices |
| **Shipping Cost** | Customers pay this on top of the unit price |
| **Product Rating** | Higher-rated products sell more at the same price |
| **Month** | Captures seasonal demand patterns |
| **Holidays in Period** | More holidays tend to lift demand |

---

### Limitations

- **Data scope:** 2017 data from a single Brazilian retailer. Directional relationships
  transfer broadly; absolute unit counts may not match your context.
- **No competitor or marketing effects** are modelled — this reflects what a manager
  can control and observe directly.
- **Use directionally:** Treat the output as one input into your pricing decision,
  not the final word.
    """)
