# =============================================================================
# Retail Demand Forecasting Tool
# Trained on 2017 Brazilian e-commerce data (Kaggle: Retail Price Optimization)
# Model: Gradient Boosting Regressor (R² ~0.84 vs ~0.09 for prior OLS)
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

# =============================================================================
# 2. Constants
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

CHART_COLOR = "#4F8EF7"
ACCENT_COLOR = "#FF6B6B"

NUMERIC_FEATURES = [
    "unit_price",
    "freight_price",
    "comp_price_ratio",
    "freight_ratio",
    "pct_price_change",
    "score_vs_comp",
    "log_customers",
    "holiday",
    "sin_month",
    "cos_month",
]
CATEGORICAL_FEATURES = ["product_category_name"]

# =============================================================================
# 3. Data Loading & Feature Engineering
# =============================================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single source of truth for all feature engineering.
    Works identically for training rows and single prediction rows.
    """
    df = df.copy()

    # --- comp_price_ratio ---
    # Average ratio of your price to each competitor's price.
    # >1.0 means you're more expensive on average.
    df["comp_price_ratio"] = (
        df["unit_price"] / df["comp_1"]
        + df["unit_price"] / df["comp_2"]
        + df["unit_price"] / df["comp_3"]
    ) / 3

    # --- freight_ratio ---
    # Your shipping cost relative to the average competitor shipping cost.
    # Customers compare total landed price, not just unit price.
    avg_comp_freight = (df["fp1"] + df["fp2"] + df["fp3"]) / 3
    df["freight_ratio"] = df["freight_price"] / (avg_comp_freight + 0.01)

    # --- pct_price_change ---
    # Month-over-month price change. Customers notice price jumps.
    lag = df["lag_price"].fillna(0)
    df["pct_price_change"] = np.where(
        lag == 0, 0, (df["unit_price"] - lag) / lag
    )

    # --- score_vs_comp ---
    # Your product rating minus the average competitor rating.
    # Positive = better-rated than competitors.
    avg_comp_score = (df["ps1"] + df["ps2"] + df["ps3"]) / 3
    df["score_vs_comp"] = df["product_score"] - avg_comp_score

    # --- log_customers ---
    # Log of the number of customers who viewed/considered the product.
    # Strongest single predictor in the dataset (corr ~0.41 with log_qty).
    df["log_customers"] = np.log(df["customers"].clip(lower=1))

    # --- holiday ---
    # Number of holidays in the period — direct count, better than sine/cosine alone.
    # Already present in the data; kept as-is.

    # --- sin_month / cos_month ---
    # Cyclical encoding so December and January are treated as adjacent.
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- log_qty (training target) ---
    if "qty" in df.columns:
        df["log_qty"] = np.log(df["qty"].clip(lower=1))

    return df


# =============================================================================
# 4. Model Training
# =============================================================================

@st.cache_resource
def train_model():
    """Train GBM pipeline. Cached — runs once per session."""
    df = load_data()
    df = engineer_features(df)

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df["log_qty"]

    preprocessor = ColumnTransformer([
        ("ohe", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL_FEATURES),
    ], remainder="passthrough")

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("gbm", GradientBoostingRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )),
    ])

    pipeline.fit(X, y)

    r2 = r2_score(y, pipeline.predict(X))

    # Feature importances with friendly labels
    feature_names_ohe = pipeline.named_steps["prep"].transformers_[0][1].get_feature_names_out(CATEGORICAL_FEATURES)
    all_feature_names = list(feature_names_ohe) + NUMERIC_FEATURES
    importances = pipeline.named_steps["gbm"].feature_importances_
    importance_dict = dict(zip(all_feature_names, importances))

    return pipeline, importance_dict, r2


# =============================================================================
# 5. Prediction
# =============================================================================

def predict_quantity(
    pipeline,
    unit_price: float,
    lag_price: float,
    comp_1: float,
    comp_2: float,
    comp_3: float,
    freight_price: float,
    fp1: float,
    fp2: float,
    fp3: float,
    product_score: float,
    ps1: float,
    ps2: float,
    ps3: float,
    customers: int,
    holiday: int,
    category: str,
    month: int,
) -> int:
    row = {
        "unit_price": unit_price,
        "lag_price": lag_price,
        "comp_1": comp_1,
        "comp_2": comp_2,
        "comp_3": comp_3,
        "freight_price": freight_price,
        "fp1": fp1,
        "fp2": fp2,
        "fp3": fp3,
        "product_score": product_score,
        "ps1": ps1,
        "ps2": ps2,
        "ps3": ps3,
        "customers": customers,
        "holiday": holiday,
        "product_category_name": category,
        "month": month,
        "qty": 1,  # placeholder
    }
    df_row = pd.DataFrame([row])
    df_row = engineer_features(df_row)
    X = df_row[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    log_pred = pipeline.predict(X)[0]
    return max(1, round(np.exp(log_pred)))


# =============================================================================
# 6. Chart Helpers
# =============================================================================

def chart_feature_importance(importance_dict: dict):
    label_map = {
        "unit_price": "Your Price",
        "freight_price": "Your Shipping Cost",
        "comp_price_ratio": "Price vs. Competitors",
        "freight_ratio": "Shipping vs. Competitor Shipping",
        "pct_price_change": "Month-over-Month Price Change",
        "score_vs_comp": "Your Rating vs. Competitors",
        "log_customers": "Customer Traffic (log)",
        "holiday": "Holidays in Period",
        "sin_month": "Seasonal Cycle (sine)",
        "cos_month": "Seasonal Cycle (cosine)",
    }
    # Aggregate OHE category dummies into one "Product Category" importance
    cat_importance = sum(v for k, v in importance_dict.items() if k.startswith("product_category_name"))
    filtered = {label_map.get(k, k): v for k, v in importance_dict.items()
                if not k.startswith("product_category_name")}
    filtered["Product Category"] = cat_importance

    df_plot = pd.DataFrame(list(filtered.items()), columns=["label", "importance"])
    df_plot = df_plot.sort_values("importance", ascending=True).tail(12)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df_plot["label"], df_plot["importance"], color=CHART_COLOR)
    ax.set_xlabel("Feature Importance (fraction of variance explained)")
    ax.set_title("What Drives Demand Most?", fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


def chart_price_curve(pipeline, base_inputs: dict):
    """Demand curve: predicted qty across a range of prices."""
    prices = np.linspace(base_inputs["unit_price"] * 0.5, base_inputs["unit_price"] * 1.5, 40)
    qtys = []
    for p in prices:
        q = predict_quantity(pipeline, p, **{k: v for k, v in base_inputs.items() if k != "unit_price"})
        qtys.append(q)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(prices, qtys, color=CHART_COLOR, linewidth=2)
    ax.axvline(base_inputs["unit_price"], color=ACCENT_COLOR, linewidth=1.5, linestyle="--", label="Current price")
    ax.fill_between(prices, qtys, alpha=0.1, color=CHART_COLOR)
    ax.set_xlabel("Unit Price (R$)")
    ax.set_ylabel("Predicted Units Sold")
    ax.set_title("Demand Curve at Current Settings", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    return fig


def chart_seasonal_effect(pipeline, base_inputs: dict):
    """Seasonal demand: predicted qty across all 12 months."""
    months = list(range(1, 13))
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    qtys = []
    for m in months:
        q = predict_quantity(pipeline, **{**base_inputs, "month": m})
        qtys.append(q)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(month_names, qtys, marker="o", color=CHART_COLOR, linewidth=2)
    ax.fill_between(month_names, qtys, alpha=0.12, color=CHART_COLOR)
    ax.axvline(base_inputs["month"] - 1, color=ACCENT_COLOR, linewidth=1.5, linestyle="--", label="Current month")
    ax.set_ylabel("Predicted Units Sold")
    ax.set_title("Seasonal Demand Pattern", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    return fig


# =============================================================================
# 7. Streamlit App
# =============================================================================

st.set_page_config(page_title="Retail Demand Forecaster", page_icon="🛒", layout="wide")

st.title("🛒 Retail Demand Forecasting Tool")
st.caption("Set your price, see your expected volume — before you commit. Trained on 2017 Brazilian e-commerce data.")

pipeline, importance_dict, train_r2 = train_model()
st.caption(f"Model: Gradient Boosting — Training R² = **{train_r2:.3f}**")

tab1, tab2, tab3 = st.tabs(["📊 Demand Predictor", "💡 Key Insights", "ℹ️ How It Works"])

# ---------------------------------------------------------------------------
# TAB 1 — Demand Predictor
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Enter Your Pricing & Product Details")

    col_left, col_mid, col_right = st.columns(3)

    with col_left:
        st.markdown("**Your Product**")
        category = st.selectbox("Product Category", options=CATEGORIES,
                                format_func=lambda x: CATEGORY_LABELS.get(x, x))
        unit_price = st.number_input("Your Price (R$)", min_value=1.0, max_value=2000.0, value=89.90, step=0.50,
                                     help="The price you plan to charge.")
        lag_price = st.number_input("Last Month's Price (R$)", min_value=0.0, max_value=2000.0, value=89.90, step=0.50,
                                    help="Your price from the previous month.")
        freight_price = st.number_input("Your Shipping Cost (R$)", min_value=0.0, max_value=200.0, value=14.0, step=0.50)
        product_score = st.slider("Your Product Rating", min_value=1.0, max_value=5.0, value=4.1, step=0.1,
                                  help="Your product's average customer rating (1–5).")
        customers = st.number_input("Expected Monthly Customers", min_value=1, max_value=10000, value=75, step=5,
                                    help="Estimated number of customers who will view/consider this product.")
        month = st.selectbox("Month", options=list(range(1, 13)),
                             format_func=lambda m: ["January","February","March","April","May","June",
                                                    "July","August","September","October","November","December"][m-1],
                             index=4)
        holiday = st.number_input("Holidays in Period", min_value=0, max_value=10, value=1,
                                  help="Number of public holidays in the sales period.")

    with col_mid:
        st.markdown("**Competitor 1**")
        comp_1 = st.number_input("Price (R$)", min_value=1.0, max_value=2000.0, value=99.90, step=0.50, key="c1p")
        fp1 = st.number_input("Shipping Cost (R$)", min_value=0.0, max_value=200.0, value=14.0, step=0.50, key="c1f")
        ps1 = st.slider("Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1, key="c1s")

        st.markdown("**Competitor 2**")
        comp_2 = st.number_input("Price (R$)", min_value=1.0, max_value=2000.0, value=95.00, step=0.50, key="c2p")
        fp2 = st.number_input("Shipping Cost (R$)", min_value=0.0, max_value=200.0, value=12.0, step=0.50, key="c2f")
        ps2 = st.slider("Rating", min_value=1.0, max_value=5.0, value=4.2, step=0.1, key="c2s")

        st.markdown("**Competitor 3**")
        comp_3 = st.number_input("Price (R$)", min_value=1.0, max_value=2000.0, value=89.00, step=0.50, key="c3p")
        fp3 = st.number_input("Shipping Cost (R$)", min_value=0.0, max_value=200.0, value=16.0, step=0.50, key="c3f")
        ps3 = st.slider("Rating", min_value=1.0, max_value=5.0, value=3.9, step=0.1, key="c3s")

    with col_right:
        st.markdown("**Prediction**")
        predict_clicked = st.button("🔮 Predict Demand", type="primary", use_container_width=True)

        if predict_clicked:
            base_inputs = dict(
                unit_price=unit_price, lag_price=lag_price,
                comp_1=comp_1, comp_2=comp_2, comp_3=comp_3,
                freight_price=freight_price, fp1=fp1, fp2=fp2, fp3=fp3,
                product_score=product_score, ps1=ps1, ps2=ps2, ps3=ps3,
                customers=customers, holiday=holiday, category=category, month=month,
            )
            qty_pred = predict_quantity(pipeline, **base_inputs)

            avg_comp = (comp_1 + comp_2 + comp_3) / 3
            pct_vs_comp = (unit_price - avg_comp) / avg_comp * 100
            direction = "above" if pct_vs_comp >= 0 else "below"
            pct_change = ((unit_price - lag_price) / lag_price * 100) if lag_price > 0 else 0.0
            avg_comp_score = (ps1 + ps2 + ps3) / 3

            st.metric("📦 Estimated Units Sold", f"{qty_pred:,}")
            st.metric("💰 Projected Revenue", f"R$ {qty_pred * unit_price:,.0f}")

            score_delta = product_score - avg_comp_score
            st.info(
                f"Priced **{abs(pct_vs_comp):.1f}% {'above' if pct_vs_comp >= 0 else 'below'}** "
                f"avg competitor (R$ {avg_comp:.2f}). "
                + (f"Price **unchanged** from last month." if abs(pct_change) < 0.01
                   else f"Price **{'▲' if pct_change > 0 else '▼'} {abs(pct_change):.1f}%** vs last month.")
                + f"\n\nRating **{'+' if score_delta >= 0 else ''}{score_delta:.1f}** vs competitor average ({avg_comp_score:.1f})."
            )

            # 5% price reduction nudge
            lower_price = max(1.0, unit_price * 0.95)
            qty_lower = predict_quantity(pipeline, **{**base_inputs, "unit_price": lower_price})
            rev_current = qty_pred * unit_price
            rev_lower = qty_lower * lower_price
            rev_delta = rev_lower - rev_current
            if qty_lower > qty_pred:
                st.caption(
                    f"💡 A 5% cut to R$ {lower_price:.2f} → ~{qty_lower:,} units "
                    f"({'▲' if rev_delta >= 0 else '▼'} R$ {abs(rev_delta):,.0f} revenue)."
                )

            st.divider()
            st.markdown("**Demand Curve**")
            fig_curve = chart_price_curve(pipeline, base_inputs)
            st.pyplot(fig_curve, use_container_width=True)
            plt.close(fig_curve)

# ---------------------------------------------------------------------------
# TAB 2 — Key Insights
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("What the Data Tells Us About Demand")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Feature Importance")
        st.caption("How much each variable contributes to the model's predictions.")
        fig_imp = chart_feature_importance(importance_dict)
        st.pyplot(fig_imp, use_container_width=True)
        plt.close(fig_imp)

    with col_b:
        st.markdown("#### Seasonal Demand Pattern")
        st.caption("Predicted monthly demand at default input values.")
        # Use sensible defaults for the seasonal chart
        default_inputs = dict(
            unit_price=89.90, lag_price=89.90,
            comp_1=99.90, comp_2=95.00, comp_3=89.00,
            freight_price=14.0, fp1=14.0, fp2=12.0, fp3=16.0,
            product_score=4.1, ps1=4.0, ps2=4.2, ps3=3.9,
            customers=75, holiday=1, category="health_beauty", month=5,
        )
        fig_season = chart_seasonal_effect(pipeline, default_inputs)
        st.pyplot(fig_season, use_container_width=True)
        plt.close(fig_season)

# ---------------------------------------------------------------------------
# TAB 3 — How It Works
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("How This Tool Works")
    st.markdown(f"""
### The Model

This tool uses a **Gradient Boosting Regressor (GBM)** — an ensemble of decision trees
that captures non-linear price elasticity and complex feature interactions automatically.
It was trained on **676 monthly product observations** across **9 categories** from a
Brazilian e-commerce retailer in 2017.

Training R² = **{train_r2:.3f}** (vs ~0.09 for the prior OLS model).

The model predicts **log(quantity sold)**, then converts back to units via `exp()`.

---

### New & Improved Features

| Feature | What it means | Why it's new |
|---|---|---|
| **Customer Traffic** | Log of monthly customers who viewed the product | Strongest predictor — was completely unused before |
| **Competitor Shipping Cost** | Your freight vs. avg competitor freight | Customers compare total landed price, not just unit price |
| **Rating vs. Competitors** | Your score minus avg competitor score | A better-rated product sells more regardless of price |
| **Holidays in Period** | Count of public holidays | Outperforms sine/cosine alone for demand spikes |
| **Competitor Ratings** | `ps1`, `ps2`, `ps3` | Competitive quality context was ignored before |

### Why GBM vs. Linear Regression

Linear regression assumes demand changes at a constant rate as price increases — a straight
line. Real demand curves bend: small price differences near a competitor may matter a lot;
large gaps may matter less. GBM learns these curves automatically from the data without
requiring you to specify the shape in advance.

---

### Limitations

- **Data scope:** 2017 data from a single Brazilian retailer. Directional relationships
  (higher price → lower volume, worse ratings → lower volume) transfer broadly; absolute
  unit counts may not.
- **In-sample R²:** The reported R² is on training data. With only 676 rows, cross-validated
  R² will be lower — treat predictions as directional guidance.
- **No promotions, inventory, or marketing effects** are modelled.
    """)
