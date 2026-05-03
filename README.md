---
title: Retail Demand Forecasting
emoji: 🛒
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# 🛒 Retail Demand Forecasting Tool

**Set smarter prices by predicting customer demand before you commit.**

Most retailers adjust prices after revenue has already taken a hit. This tool flips that — enter your pricing scenario and see estimated monthly demand *before* you publish a price. Compare options, stress-test your margin, and walk into buying decisions with data behind you.

---

## What It Does

The app takes your pricing inputs and returns a predicted quantity sold for the month. It also surfaces charts showing how demand shifts with price, which month of the year peaks for your category, and which factors drive demand most.

**Three tabs:**

| Tab | What You'll Find |
|---|---|
| 📊 Demand Predictor | Enter your price, shipping cost, product rating, category, and month — get a unit forecast, projected revenue, and two demand charts |
| 💡 Key Insights | Feature importance chart showing what actually drives demand, plus the model's cross-validated accuracy |
| ℹ️ How It Works | Non-technical explanation of the model, each input, engineered features, and important caveats |

---

## How to Use It

1. **Open the Demand Predictor tab**
2. Select your product category and set your current and last month's price
3. Enter your shipping cost, product rating, and number of holidays in the period
4. Select the month you're forecasting for
5. Hit **Predict Demand** — your estimated monthly unit sales and projected revenue will appear, along with a price-change callout and a 5% price-cut nudge if it would improve your outcome

**Pro tip:** Run multiple scenarios back-to-back. Try your current price, then raise it 10%, then lower it 10% — the demand curve chart updates each time and shows exactly how elastic your product is.

---

## The Model

**Algorithm:** Gradient Boosting Regressor (GBM)
**Library:** scikit-learn
**Target:** Log-transformed quantity sold (converted back to plain units in the output)
**Validation:** 5-fold cross-validation (R² reported in the Key Insights tab)
**Training data:** 676 monthly sales records across 9 categories — sourced from a Brazilian e-commerce retailer (2017)

### Features the model uses

| Feature | What it captures |
|---|---|
| `unit_price` | Your product's own price — the primary demand driver |
| `freight_price` | Shipping cost, which adds to the customer's effective price |
| `pct_price_change` | How much you raised or cut your price since last month, as a % |
| `product_score` | Product rating (1–5); higher-rated products sell more at the same price |
| `holiday` | Number of holidays in the period; more holidays tend to lift demand |
| `sin_month` / `cos_month` | Month of year encoded as a smooth wave so January and December are treated as adjacent |
| `product_category_name` | One-hot encoded; each category has its own demand baseline |

### Feature engineering at a glance

All engineered features are computed identically at training time and prediction time — no data leakage.

```
pct_price_change = (unit_price - lag_price) / lag_price   [0 if lag_price is missing or zero]
sin_month        = np.sin(2π × month / 12)
cos_month        = np.cos(2π × month / 12)
log_qty          = np.log(qty.clip(lower=1))   → reversed with np.exp() at output
```

---

## Product Categories

The model supports the 9 categories present in the training data:

| Display Name | Internal Key |
|---|---|
| Bed & Bath | bed_bath_table |
| Computers & Accessories | computers_accessories |
| Consoles & Games | consoles_games |
| Cool Stuff | cool_stuff |
| Furniture & Decor | furniture_decor |
| Garden Tools | garden_tools |
| Health & Beauty | health_beauty |
| Perfumery | perfumery |
| Watches & Gifts | watches_gifts |

---

## Limitations & Honest Caveats

This tool is designed for **directional scenario planning**, not guaranteed sales targets. Keep these in mind:

- **Geography & era:** Trained on 2017 Brazilian e-commerce data. Pricing norms, consumer behavior, and competitive dynamics in your market may differ.
- **No competitor or marketing signals:** The model doesn't know about competitor pricing, ad spend, promotions, or viral moments. These can shift demand significantly and aren't captured here.
- **Treat forecasts as ranges:** A prediction of 12 units realistically means "somewhere in that ballpark." Use it for comparison across scenarios, not as a guaranteed target.
- **R² reflects directional value:** The model's R² (shown in Key Insights) reflects how much demand variation price signals alone can explain. The real value lies in comparing *relative* scenarios — not reading absolute unit counts as gospel.

---

## File Structure

```
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── retail_price.csv      # Training dataset (676 rows)
└── README.md             # This file
```

---

## Requirements

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

---

## Running Locally

```bash
git clone https://github.com/your-username/retail-demand-forecasting
cd retail-demand-forecasting
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Data Source

Based on the **Retail Price Optimization** dataset originally published on Kaggle. The dataset contains monthly pricing and sales data from a Brazilian e-commerce platform across 12 months in 2017.

---

*Built for retail managers and buyers — no data science background required.*
