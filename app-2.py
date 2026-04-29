# app.py

# 1. Imports
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.express as px

# 2. Data loading and preprocessing
@st.cache_data
def load_data():
    df = pd.read_csv("retail_price.csv")
    
    # --- Feature Engineering ---
    df['log_qty'] = np.log(df['qty'])
    df['log_price'] = np.log(df['unit_price'])

    df['comp_price_ratio'] = (
        (df['unit_price'] / df['comp_1']) +
        (df['unit_price'] / df['comp_2']) +
        (df['unit_price'] / df['comp_3'])
    ) / 3

    df['comp_std'] = df[['comp_1','comp_2','comp_3']].std(axis=1)

    df['pct_price_change'] = np.where(
        df['lag_price'] > 0,
        (df['unit_price'] - df['lag_price']) / df['lag_price'],
        0
    )

    df['freight_ratio'] = df['freight_price'] / df['unit_price']

    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

    # Category interactions
    categories = df['product_category_name'].unique()
    for cat in categories:
        df[f'cat_{cat}'] = (df['product_category_name'] == cat).astype(int)
        df[f'{cat}_price'] = df[f'cat_{cat}'] * df['log_price']

    return df, categories

# 3. Model training
@st.cache_resource
def train_models():
    df, categories = load_data()

    features = [
        'comp_price_ratio','comp_std','pct_price_change',
        'freight_ratio','sin_month','cos_month'
    ]

    interaction_cols = [f'{cat}_price' for cat in categories]
    features += interaction_cols

    X = df[features]
    y = df['log_qty']

    # Ridge for prediction stability
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)

    # OLS for interpretability dashboard
    X_ols = sm.add_constant(X)
    ols = sm.OLS(y, X_ols).fit()

    return ridge, ols, features, categories

# 4. Prediction function
def predict_demand(model, features, input_dict, categories):
    df = pd.DataFrame([input_dict])

    df['log_price'] = np.log(df['unit_price'])

    df['comp_price_ratio'] = (
        (df['unit_price'] / df['comp_1']) +
        (df['unit_price'] / df['comp_2']) +
        (df['unit_price'] / df['comp_3'])
    ) / 3

    df['comp_std'] = df[['comp_1','comp_2','comp_3']].std(axis=1)

    df['pct_price_change'] = np.where(
        df['lag_price'] > 0,
        (df['unit_price'] - df['lag_price']) / df['lag_price'],
        0
    )

    df['freight_ratio'] = df['freight_price'] / df['unit_price']

    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

    for cat in categories:
        df[f'{cat}_price'] = (df['product_category_name'] == cat) * df['log_price']

    X = df[features]
    pred_log = model.predict(X)[0]
    return int(np.exp(pred_log))

# 5. Price optimization
def optimize_price(model, features, base_input, categories):
    prices = np.linspace(base_input['unit_price']*0.5, base_input['unit_price']*1.5, 50)
    results = []

    for p in prices:
        base_input['unit_price'] = p
        demand = predict_demand(model, features, base_input, categories)
        revenue = p * demand
        results.append((p, demand, revenue))

    df = pd.DataFrame(results, columns=['price','demand','revenue'])
    best_row = df.loc[df['revenue'].idxmax()]

    return df, best_row

# 6. Charts
def plot_coefficients(ols_model):
    coefs = ols_model.params.drop('const')
    coefs = coefs.sort_values()
    fig = px.bar(coefs, orientation='h', title='Driver Impact on Demand')
    return fig

# 7. Streamlit UI
st.title("Retail Demand & Pricing Tool")

df, categories = load_data()
ridge, ols, features = train_models(df, categories)

tab1, tab2, tab3 = st.tabs(["Demand Predictor","Insights","How It Works"])

with tab1:
    st.header("Demand Prediction")

    unit_price = st.number_input("Unit Price", value=100.0)
    lag_price = st.number_input("Last Month Price", value=100.0)
    comp_1 = st.number_input("Competitor 1 Price", value=100.0)
    comp_2 = st.number_input("Competitor 2 Price", value=100.0)
    comp_3 = st.number_input("Competitor 3 Price", value=100.0)
    freight_price = st.number_input("Freight Price", value=10.0)

    category = st.selectbox("Category", categories)
    month = st.slider("Month", 1, 12, 6)

    input_dict = {
        'unit_price': unit_price,
        'lag_price': lag_price,
        'comp_1': comp_1,
        'comp_2': comp_2,
        'comp_3': comp_3,
        'freight_price': freight_price,
        'product_category_name': category,
        'month': month
    }

    demand = predict_demand(ridge, features, input_dict, categories)

    st.metric("Predicted Units Sold", demand)

    comp_avg = (comp_1 + comp_2 + comp_3) / 3
    diff = (unit_price - comp_avg) / comp_avg * 100

    st.info(f"You are priced {diff:.1f}% vs competitors. Estimated demand: {demand} units.")

    # Optimization
    opt_df, best = optimize_price(ridge, features, input_dict.copy(), categories)
    st.success(f"Optimal price: {best['price']:.2f} for max revenue.")

    fig = px.line(opt_df, x='price', y='revenue', title='Revenue vs Price')
    st.plotly_chart(fig)

with tab2:
    st.header("Key Insights")
    fig = plot_coefficients(ols)
    st.plotly_chart(fig)

with tab3:
    st.header("How It Works")
    st.markdown("""
    This tool predicts how many units you will sell based on your pricing.

    - Uses historical pricing and competitor data
    - Adjusts for seasonality
    - Accounts for category-specific price sensitivity

    Predictions are based on past patterns and should be used directionally.
    """)
