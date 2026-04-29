---
title: Retail Demand Forecasting
emoji: 🛒
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# 🛒 Retail Demand Forecasting Tool

**Set smarter prices by predicting customer demand before you commit.**

Most retailers adjust prices after revenue has already taken a hit. This tool flips that — enter your pricing scenario and see estimated monthly demand *before* you publish a price. Compare options, stress-test your margin, and walk into buying decisions with data behind you.

---

## What It Does

The app takes your pricing inputs and returns a predicted quantity sold for the month. It also surfaces four data-driven insights about what actually moves demand in retail — from competitor pricing to seasonal patterns to category-level price sensitivity.

**Three tabs:**

| Tab | What You'll Find |
|---|---|
| 📊 Demand Predictor | Enter your price, competitor prices, shipping cost, category, and month — get a unit forecast and plain-English interpretation |
| 📈 Key Insights Dashboard | Four charts showing what drives demand, which categories are most price-sensitive, how competitors affect you, and which months peak |
| ℹ️ How It Works | Non-technical explanation of the model, each feature, and important caveats |

---

## How to Use It

1. **Open the Demand Predictor tab**
2. Set your price and last month's price using the sliders
3. Enter your three main competitors' prices
4. Select your product category and the current month
5. Hit **Predict Demand** — your estimated monthly unit sales will appear on the right, along with a short business interpretation

**Pro tip:** Run multiple scenarios back-to-back. Try your current price, then raise it 10%, then lower it 10% — the difference in predicted units tells you how price-elastic your product is.

---

## The Model

**Algorithm:** Ordinary Least Squares (OLS) linear regression
**Library:** scikit-learn
**Target:** Log-transformed quantity sold (converted back to plain units in the output)
**Training data:** 676 monthly sales records, 52 products, 9 categories — sourced from a Brazilian e-commerce retailer (2017)

### Features the model uses

| Feature | What it captures |
|---|---|
| `unit_price` | Your product's own price |
| `freight_price` | Shipping cost (adds to the customer's effective price) |
| `comp_price_ratio` | Your price divided by the average competitor price — above 1.0 means you're more expensive |
| `pct_price_change` | How much you raised or cut your price since last month, as a % |
| `sin_month` / `cos_month` | Month of year, encoded as a smooth wave so Jan and Dec are treated as adjacent |
| `category × price` | Separate price-sensitivity coefficient for each of the 9 product categories |

### Feature engineering at a glance

All five engineered features are computed identically at training time and prediction time — no data leakage.

```
comp_price_ratio   = mean(unit_price / comp_1, unit_price / comp_2, unit_price / comp_3)
pct_price_change   = (unit_price - lag_price) / lag_price   [0 if lag_price is missing]
log_qty            = np.log(qty)   → reversed with np.exp() at output
sin_month          = np.sin(2π × month / 12)
cos_month          = np.cos(2π × month / 12)
category_x_price   = one-hot category dummies × unit_price  [reference: bed_bath_table]
```

---

## Product Categories

The model supports the 9 categories present in the training data:

- Bed Bath Table
- Computers Accessories
- Consoles Games
- Cool Stuff
- Furniture Decor
- Garden Tools
- Health Beauty
- Perfumery
- Watches Gifts

---

## Limitations & Honest Caveats

This tool is designed for **directional scenario planning**, not guaranteed sales targets. Keep these in mind:

- **Geography & era:** Trained on 2017 Brazilian e-commerce data. Pricing norms, consumer behavior, and competitive dynamics in your market may differ.
- **No promotions or marketing signals:** The model doesn't know about discounts, ad spend, influencer campaigns, or viral moments. These can shift demand significantly and aren't captured here.
- **Treat forecasts as ranges:** A prediction of 12 units realistically means "somewhere between 8 and 16 units." Use it for comparison, not as a contract.
- **Linear relationships only:** OLS assumes demand responds to price in a straight-line fashion. Real demand curves are often non-linear, especially at extreme price points.
- **R² of ~0.09:** The model explains about 9% of the variation in quantity sold using price signals alone. That's modest — it reflects how much of demand is driven by factors outside this dataset (brand, reviews, discovery, etc.). The value of the tool lies in comparing *relative* scenarios, not absolute predictions.

---

## File Structure

```
├── app.py                  # Main Gradio application
├── requirements.txt        # Python dependencies
├── retail_price_csv.csv    # Training dataset (676 rows)
└── README.md               # This file
```

---

## Requirements

```
gradio>=4.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
matplotlib>=3.7.0
plotly>=5.15.0
joblib>=1.3.0
xlrd>=2.0.1
```

---

## Data Source

Based on the **Retail Price Optimization** dataset originally published on Kaggle. The dataset contains monthly pricing and sales data scraped from a Brazilian e-commerce platform across 12 months in 2017.

---

## Running Locally

```bash
git clone https://huggingface.co/spaces/your-username/retail-demand-forecasting
cd retail-demand-forecasting
pip install -r requirements.txt
python app.py
```

The app will open at `http://localhost:7860`.

---

*Built for retail managers and buyers — no data science background required.*
