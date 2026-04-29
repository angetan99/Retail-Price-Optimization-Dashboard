# =============================================================================
# Retail Demand Forecasting App
# Predicts quantity sold based on pricing, competition, and seasonal inputs
# Built for retail managers and buyers — no data science background needed
# =============================================================================

# 1. IMPORTS
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression

# =============================================================================
# 2. DATA LOADING AND PREPROCESSING
# =============================================================================

DATA_PATH = os.path.join(os.path.dirname(__file__), "retail_price_csv.csv")

def load_and_engineer(path):
    """Load dataset and engineer all model features."""
    df = pd.read_csv(path)

    # --- Feature: log_qty ---
    # We predict the log of quantity sold (a common technique when demand
    # varies widely). The app converts this back to plain units at output time.
    df = df[df["qty"] > 0].copy()
    df["log_qty"] = np.log(df["qty"])

    # --- Feature: comp_price_ratio ---
    # How is our price positioned vs. competitors?
    # A value > 1 means we are more expensive than that competitor.
    df["ratio_1"] = df["unit_price"] / df["comp_1"].replace(0, np.nan)
    df["ratio_2"] = df["unit_price"] / df["comp_2"].replace(0, np.nan)
    df["ratio_3"] = df["unit_price"] / df["comp_3"].replace(0, np.nan)
    df["comp_price_ratio"] = df[["ratio_1", "ratio_2", "ratio_3"]].mean(axis=1)

    # --- Feature: pct_price_change ---
    # Did we raise or cut our price vs. last month?
    # Edge case: if lag_price is 0 or missing, set change to 0.
    safe_lag = df["lag_price"].replace(0, np.nan)
    df["pct_price_change"] = (df["unit_price"] - safe_lag) / safe_lag
    df["pct_price_change"] = df["pct_price_change"].fillna(0)

    # --- Feature: sin_month / cos_month ---
    # Cyclical encoding ensures December and January are treated as "close"
    # in time — regular month numbers (1–12) would not do that.
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


df_raw = load_and_engineer(DATA_PATH)

# --- Feature: category_x_price ---
# We interact each product category with the unit price so the model learns
# how price sensitivity differs across categories.
# One category is dropped (the reference level) to avoid redundancy.
CATEGORIES = sorted(df_raw["product_category_name"].dropna().unique().tolist())
REFERENCE_CATEGORY = CATEGORIES[0]          # e.g., bed_bath_table
DUMMY_CATEGORIES = CATEGORIES[1:]           # all other 8 categories

def add_category_interactions(df, categories):
    """One-hot encode product category and multiply each dummy by unit_price."""
    for cat in categories:
        col = f"cat_x_price_{cat}"
        df[col] = (df["product_category_name"] == cat).astype(float) * df["unit_price"]
    return df

df_raw = add_category_interactions(df_raw, DUMMY_CATEGORIES)

CAT_X_PRICE_COLS = [f"cat_x_price_{c}" for c in DUMMY_CATEGORIES]

# =============================================================================
# 3. MODEL TRAINING
# =============================================================================

FEATURE_COLS = (
    ["unit_price", "freight_price", "comp_price_ratio", "pct_price_change",
     "sin_month", "cos_month"]
    + CAT_X_PRICE_COLS
)

df_model = df_raw[FEATURE_COLS + ["log_qty"]].dropna()
X_train = df_model[FEATURE_COLS].values
y_train = df_model["log_qty"].values

model = LinearRegression()
model.fit(X_train, y_train)

# Save for optional external use
MODEL_PATH = os.path.join(os.path.dirname(__file__), "demand_model.joblib")
joblib.dump(model, MODEL_PATH)

# Build a readable coefficient table for the dashboard
coef_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "coefficient": model.coef_
}).sort_values("coefficient", key=abs, ascending=False)

# =============================================================================
# 4. PREDICTION FUNCTION
# =============================================================================

def predict_demand(unit_price, lag_price, comp_1, comp_2, comp_3,
                   freight_price, category, month):
    """
    Transform raw retail inputs with the same feature engineering used in
    training, then return predicted units sold (np.exp of log prediction).
    """
    # comp_price_ratio
    ratios = []
    for c in [comp_1, comp_2, comp_3]:
        if c and c > 0:
            ratios.append(unit_price / c)
    comp_price_ratio = np.mean(ratios) if ratios else 1.0

    # pct_price_change
    if lag_price and lag_price > 0:
        pct_price_change = (unit_price - lag_price) / lag_price
    else:
        pct_price_change = 0.0

    # cyclical month
    sin_month = np.sin(2 * np.pi * month / 12)
    cos_month = np.cos(2 * np.pi * month / 12)

    # category × price interactions
    cat_values = []
    for cat in DUMMY_CATEGORIES:
        cat_values.append(unit_price if category == cat else 0.0)

    row = np.array([[
        unit_price, freight_price, comp_price_ratio, pct_price_change,
        sin_month, cos_month
    ] + cat_values])

    log_pred = model.predict(row)[0]
    predicted_units = int(round(np.exp(log_pred)))

    # Plain-language interpretation
    avg_comp = np.mean([c for c in [comp_1, comp_2, comp_3] if c > 0])
    pct_vs_comp = ((unit_price - avg_comp) / avg_comp * 100) if avg_comp > 0 else 0
    direction = "above" if pct_vs_comp >= 0 else "below"
    pct_change_str = ""
    if abs(pct_price_change) > 0.001:
        verb = "raised" if pct_price_change > 0 else "cut"
        pct_change_str = (
            f" You {verb} your price by {abs(pct_price_change)*100:.1f}% "
            f"from last month."
        )

    interpretation = (
        f"At ${unit_price:.2f}, you are priced "
        f"{abs(pct_vs_comp):.1f}% {direction} competitor average "
        f"(${avg_comp:.2f}).{pct_change_str} "
        f"📦 Estimated monthly demand: {predicted_units:,} units."
    )

    return predicted_units, interpretation

# =============================================================================
# 5. CHART GENERATION FUNCTIONS
# =============================================================================

CHART_STYLE = {
    "bg": "#F8F9FA",
    "bar_main": "#2563EB",
    "bar_accent": "#F59E0B",
    "text": "#1E293B",
    "grid": "#E2E8F0",
}

def _base_fig(figsize=(9, 4.5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=CHART_STYLE["bg"])
    ax.set_facecolor(CHART_STYLE["bg"])
    ax.tick_params(colors=CHART_STYLE["text"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(CHART_STYLE["grid"])
    ax.yaxis.grid(True, color=CHART_STYLE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def chart_variable_influence():
    """Bar chart: which inputs have the biggest impact on demand?"""
    top = coef_df.head(12).copy()
    top["label"] = top["feature"].str.replace("cat_x_price_", "Category × Price: ", regex=False)
    top["label"] = top["label"].str.replace("_", " ").str.title()
    colors = [CHART_STYLE["bar_main"] if v >= 0 else CHART_STYLE["bar_accent"]
              for v in top["coefficient"]]

    fig, ax = _base_fig()
    bars = ax.barh(top["label"], top["coefficient"], color=colors)
    ax.axvline(0, color=CHART_STYLE["text"], linewidth=0.8)
    ax.set_xlabel("Effect on Demand (positive = more units sold)", color=CHART_STYLE["text"])
    ax.set_title("What Drives Demand Most?", color=CHART_STYLE["text"],
                 fontsize=13, fontweight="bold", pad=12)
    ax.invert_yaxis()
    # annotate
    for bar, val in zip(bars, top["coefficient"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}", va="center", ha="left",
                fontsize=8, color=CHART_STYLE["text"])
    plt.tight_layout()
    return fig


def chart_category_price_sensitivity():
    """Bar chart: how does price sensitivity differ across product categories?"""
    cat_coefs = []
    for cat in DUMMY_CATEGORIES:
        col = f"cat_x_price_{cat}"
        c = coef_df.loc[coef_df["feature"] == col, "coefficient"]
        val = c.values[0] if len(c) else 0.0
        cat_coefs.append({"category": cat.replace("_", " ").title(), "coef": val})
    # Add reference category as 0
    cat_coefs.append({"category": REFERENCE_CATEGORY.replace("_", " ").title() + " (ref)", "coef": 0.0})
    cat_df = pd.DataFrame(cat_coefs).sort_values("coef")

    fig, ax = _base_fig()
    colors = [CHART_STYLE["bar_main"] if v >= 0 else CHART_STYLE["bar_accent"]
              for v in cat_df["coef"]]
    ax.barh(cat_df["category"], cat_df["coef"], color=colors)
    ax.axvline(0, color=CHART_STYLE["text"], linewidth=0.8)
    ax.set_xlabel("Price Effect on Demand by Category", color=CHART_STYLE["text"])
    ax.set_title("Which Categories Are Most Price-Sensitive?",
                 color=CHART_STYLE["text"], fontsize=13, fontweight="bold", pad=12)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def chart_competitor_pricing_effect():
    """Side-by-side: effect of your own price vs. competitor price ratio."""
    own_price_coef = coef_df.loc[coef_df["feature"] == "unit_price", "coefficient"].values[0]
    comp_ratio_coef = coef_df.loc[coef_df["feature"] == "comp_price_ratio", "coefficient"].values[0]

    labels = ["Your Price Effect", "Competitor Price Ratio Effect"]
    values = [own_price_coef, comp_ratio_coef]
    colors = [CHART_STYLE["bar_main"] if v >= 0 else CHART_STYLE["bar_accent"]
              for v in values]

    fig, ax = _base_fig(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    ax.axhline(0, color=CHART_STYLE["text"], linewidth=0.8)
    ax.set_ylabel("Effect on log(Quantity Sold)", color=CHART_STYLE["text"])
    ax.set_title("Your Price vs. Competitor Pricing: Who Matters More?",
                 color=CHART_STYLE["text"], fontsize=12, fontweight="bold", pad=12)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.002 if val >= 0 else -0.005),
                f"{val:+.3f}", ha="center", fontsize=10, color=CHART_STYLE["text"])
    plt.tight_layout()
    return fig


def chart_seasonal_demand():
    """Line chart: predicted demand across each month of the year."""
    # Compute model-implied seasonal component for each calendar month
    # using median values for all other features (to isolate seasonality)
    median_vals = df_model[FEATURE_COLS].median()
    monthly_pred = []
    for m in range(1, 13):
        row = median_vals.copy()
        row["sin_month"] = np.sin(2 * np.pi * m / 12)
        row["cos_month"] = np.cos(2 * np.pi * m / 12)
        log_pred = model.predict(row.values.reshape(1, -1))[0]
        monthly_pred.append(np.exp(log_pred))

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = _base_fig()
    ax.plot(months, monthly_pred, color=CHART_STYLE["bar_main"],
            linewidth=2.5, marker="o", markersize=7)
    ax.fill_between(range(12), monthly_pred, alpha=0.12,
                    color=CHART_STYLE["bar_main"])
    ax.set_xticks(range(12))
    ax.set_xticklabels(months, color=CHART_STYLE["text"])
    ax.set_ylabel("Estimated Units Sold", color=CHART_STYLE["text"])
    ax.set_title("When Do Customers Buy the Most? (Seasonal Pattern)",
                 color=CHART_STYLE["text"], fontsize=13, fontweight="bold", pad=12)
    # Annotate peak and trough
    peak_m = np.argmax(monthly_pred)
    trough_m = np.argmin(monthly_pred)
    ax.annotate(f"Peak: {months[peak_m]}", xy=(peak_m, monthly_pred[peak_m]),
                xytext=(peak_m + 0.5, monthly_pred[peak_m] * 1.02),
                fontsize=9, color=CHART_STYLE["bar_main"])
    ax.annotate(f"Low: {months[trough_m]}", xy=(trough_m, monthly_pred[trough_m]),
                xytext=(trough_m + 0.4, monthly_pred[trough_m] * 0.97),
                fontsize=9, color=CHART_STYLE["bar_accent"])
    plt.tight_layout()
    return fig

# Pre-render all charts at startup
fig_influence    = chart_variable_influence()
fig_sensitivity  = chart_category_price_sensitivity()
fig_competitor   = chart_competitor_pricing_effect()
fig_seasonal     = chart_seasonal_demand()

# =============================================================================
# 6. GRADIO INTERFACE
# =============================================================================

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

HOW_IT_WORKS_MD = """
## How This Tool Works

### What the model does
This app uses a **linear regression model** trained on 676 months of sales data
from a Brazilian e-commerce retailer in 2017. You enter your price, competitor
prices, shipping cost, product category, and the month — and it estimates how
many units you'll likely sell.

> **Important:** Results are directional, not exact. Use this tool to compare
> scenarios ("What happens to demand if I raise my price 10%?"), not to
> produce a guaranteed sales figure.

---

### How each input is used

| Input | How it's used |
|---|---|
| **Your Price** | Core driver of demand — higher price generally means fewer sales |
| **Last Month's Price** | Captures whether customers notice a price change vs. last month |
| **Competitor Prices (1–3)** | Combined into a ratio of your price to theirs |
| **Shipping Cost** | Acts as an additional cost customers factor in |
| **Product Category** | Each category has a different baseline price sensitivity |
| **Month** | Encoded cyclically so Dec/Jan are treated as adjacent in time |

---

### Engineered features (behind the scenes)

- **Competitor Price Ratio** — Average of (your price ÷ each competitor's price). Above 1 = you're more expensive.
- **Month-over-Month Price Change** — How much your price moved from last month, as a percentage.
- **Category × Price Interactions** — The model learns that a $50 increase affects furniture differently than beauty products.
- **Seasonal Encoding (sin/cos)** — Converts the month number into a smooth wave so the model handles seasonality properly.
- **Log Quantity Target** — The model predicts the log of units sold; the app converts this back to plain units automatically.

---

### Limitations & honest caveats

- Trained on **2017 data from Brazil** — pricing norms and demand patterns may differ in your market.
- The model **does not account for promotions, advertising, or external events** (holidays, viral trends, etc.).
- Categories outside the 9 in the training data are not supported.
- A predicted quantity of, say, 12 units means "approximately 10–14 units at this price" — treat it as a range, not a point estimate.
"""

with gr.Blocks(title="Retail Demand Forecasting", theme=gr.themes.Soft()) as app:

    gr.Markdown(
        """
        # 🛒 Retail Demand Forecasting Tool
        ### Set smarter prices by predicting demand *before* you commit
        Enter your pricing details on the left and instantly see how many units
        you're likely to sell this month.
        """
    )

    # ── Tab 1: Demand Predictor ───────────────────────────────────────────────
    with gr.Tab("📊 Demand Predictor"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Your Pricing")
                unit_price_input = gr.Slider(
                    minimum=10, maximum=400, value=80, step=0.5,
                    label="Your Price ($)"
                )
                lag_price_input = gr.Slider(
                    minimum=10, maximum=400, value=80, step=0.5,
                    label="Last Month's Price ($)"
                )
                freight_input = gr.Slider(
                    minimum=0, maximum=80, value=15, step=0.5,
                    label="Shipping Cost ($)"
                )

                gr.Markdown("#### Competitor Prices")
                comp1_input = gr.Slider(
                    minimum=10, maximum=400, value=85, step=0.5,
                    label="Competitor 1 Price ($)"
                )
                comp2_input = gr.Slider(
                    minimum=10, maximum=400, value=90, step=0.5,
                    label="Competitor 2 Price ($)"
                )
                comp3_input = gr.Slider(
                    minimum=10, maximum=400, value=88, step=0.5,
                    label="Competitor 3 Price ($)"
                )

                gr.Markdown("#### Product Details")
                category_input = gr.Dropdown(
                    choices=CATEGORIES,
                    value=CATEGORIES[0],
                    label="Product Category"
                )
                month_input = gr.Dropdown(
                    choices=[(v, k) for k, v in MONTH_NAMES.items()],
                    value=6,
                    label="Month"
                )

                predict_btn = gr.Button("🔍 Predict Demand", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("#### Forecast Result")
                units_output = gr.Number(
                    label="Estimated Units Sold This Month",
                    precision=0
                )
                interpretation_output = gr.Textbox(
                    label="What this means for your business",
                    lines=4
                )

                gr.Markdown(
                    """
                    ---
                    **Tips for using this tool:**
                    - Try adjusting your price up and down to see how demand changes
                    - Compare your price to competitors — being significantly above them hurts volume
                    - Seasonal patterns matter: check the Insights tab for peak months
                    """
                )

        predict_btn.click(
            fn=predict_demand,
            inputs=[unit_price_input, lag_price_input,
                    comp1_input, comp2_input, comp3_input,
                    freight_input, category_input, month_input],
            outputs=[units_output, interpretation_output]
        )

    # ── Tab 2: Insights Dashboard ─────────────────────────────────────────────
    with gr.Tab("📈 Key Insights Dashboard"):
        gr.Markdown(
            """
            ### What the data tells us about pricing and demand
            These charts are derived from the regression model trained on all 676
            months of sales history. The **effect size** (shown on each chart's
            x-axis) tells you how strongly each factor shifts expected demand.
            """
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### What Drives Demand the Most?")
                gr.Markdown(
                    "Bars to the right mean more demand; bars to the left mean less. "
                    "Blue = positive effect, orange = negative effect."
                )
                plot1 = gr.Plot(value=fig_influence)

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Price Sensitivity by Product Category")
                gr.Markdown(
                    "A longer bar means price changes have a bigger effect on sales "
                    "in that category. The reference category (Bed Bath Table) is set to 0."
                )
                plot2 = gr.Plot(value=fig_sensitivity)

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Your Price vs. Competitor Pricing: Which Matters More?")
                gr.Markdown(
                    "Compares the direct effect of your price to how your price "
                    "*relative to competitors* shapes demand."
                )
                plot3 = gr.Plot(value=fig_competitor)

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Seasonal Demand Pattern (Jan – Dec)")
                gr.Markdown(
                    "Holding price and other factors constant, this shows which months "
                    "tend to see higher or lower sales volume."
                )
                plot4 = gr.Plot(value=fig_seasonal)

    # ── Tab 3: How It Works ───────────────────────────────────────────────────
    with gr.Tab("ℹ️ How It Works"):
        gr.Markdown(HOW_IT_WORKS_MD)

# =============================================================================
# 7. LAUNCH
# =============================================================================

if __name__ == "__main__":
    app.launch()
