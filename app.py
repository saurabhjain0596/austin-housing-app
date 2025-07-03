# app.py
import os
import zipfile
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt
import plotly.express as px

from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    import shap
    _has_shap = True
except ImportError:
    _has_shap = False

KAGGLE_DATASET = "ericpierce/austinhousingprices"
FILE_NAME = "austinHousingData.csv"
MODEL_PATH = "best_model.pkl"

@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/<your-username>/austin-housing-app/main/austinHousingData.csv"
    return pd.read_csv(url)

def select_features(df: pd.DataFrame, target: str = "latestPrice", thresh: float = 0.05):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    corrs = df[numeric_cols + [target]].corr()[target].abs()
    selected_numeric = corrs[corrs > thresh].index.tolist()
    selected_numeric = [c for c in selected_numeric if c != target]
    return selected_numeric, categorical_cols

def build_preprocessor(num_cols: List[str], cat_cols: List[str], use_pca: bool):
    if use_pca and len(num_cols) > 8:
        num_pipeline = Pipeline([('scale', StandardScaler()), ('pca', PCA(n_components=8))])
    else:
        num_pipeline = Pipeline([('scale', StandardScaler())])
    return ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

def train_models(X, y, preprocessor):
    models = {
        'XGBoost': XGBRegressor(random_state=42, learning_rate=0.05, n_estimators=600, max_depth=6, subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror'),
        'RandomForest': RandomForestRegressor(n_estimators=400, max_depth=15, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(random_state=42)
    }
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    best_rmse = np.inf
    best_name, best_pipe = None, None
    cv_results = {}
    for name, model in models.items():
        pipe = Pipeline([('prep', preprocessor), ('model', model)])
        cv_rmse = np.mean(np.sqrt(-cross_val_score(pipe, X, y, scoring='neg_mean_squared_error', cv=kf, n_jobs=-1)))
        cv_results[name] = cv_rmse
        if cv_rmse < best_rmse:
            best_rmse = cv_rmse
            best_name = name
            best_pipe = pipe
    return best_name, best_pipe, cv_results

@st.cache_resource(show_spinner=True)
def get_model(df: pd.DataFrame, use_pca: bool):
    if os.path.isfile(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    y = df['latestPrice']
    num_cols, cat_cols = select_features(df)
    preprocessor = build_preprocessor(num_cols, cat_cols, use_pca)
    X = df[num_cols + cat_cols]
    best_name, best_pipe, cv_results = train_models(X, y, preprocessor)
    best_pipe.fit(X, y)
    meta = {
        'best_model': best_name,
        'cv_results': cv_results,
        'numeric_cols': num_cols,
        'categorical_cols': cat_cols,
        'use_pca': use_pca
    }
    joblib.dump({'pipeline': best_pipe, 'meta': meta}, MODEL_PATH)
    return {'pipeline': best_pipe, 'meta': meta}

def build_sidebar(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]):
    st.sidebar.header('Property Features')
    user_data = {}
    for c in cat_cols:
        options = df[c].dropna().unique()
        user_data[c] = st.sidebar.selectbox(c, options)
    for n in num_cols:
        col_min, col_max = float(df[n].min()), float(df[n].max())
        default = float(df[n].median())
        if col_max - col_min > 1000:
            user_data[n] = st.sidebar.slider(n, col_min, col_max, default)
        else:
            user_data[n] = st.sidebar.number_input(n, col_min, col_max, default)
    return pd.DataFrame([user_data])

def plot_map(df: pd.DataFrame, lat, lon):
    radius_m = 4023
    train_layer = pdk.Layer(
        'ScatterplotLayer',
        data=df[['longitude', 'latitude', 'latestPrice']].rename(columns={'longitude': 'lon', 'latitude': 'lat'}),
        get_position='[lon, lat]',
        get_fill_color='[0,112,255,120]',
        get_radius=60
    )
    circle_layer = pdk.Layer(
        'ScatterplotLayer',
        data=pd.DataFrame({'lon': [lon], 'lat': [lat]}),
        get_position='[lon, lat]',
        get_fill_color='[255,0,0,60]',
        stroked=True,
        radius=radius_m
    )
    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=11, pitch=35)
    deck = pdk.Deck(layers=[train_layer, circle_layer], initial_view_state=view_state, tooltip={'text': 'Price: {latestPrice}'})
    st.pydeck_chart(deck, use_container_width=True)

def correlation_heatmap(df: pd.DataFrame):
    corr_df = df.select_dtypes(include=[np.number]).corr()
    corr_df = corr_df.stack().reset_index(name='value')
    chart = alt.Chart(corr_df).mark_rect().encode(
        x='level_0:O',
        y='level_1:O',
        color=alt.Color('value:Q', scale=alt.Scale(scheme='redblue')),
        tooltip=['level_0', 'level_1', alt.Tooltip('value:Q', format='.2f')]
    )
    st.altair_chart(chart, use_container_width=True)

def plot_cv_results(cv_results: dict):
    names = list(cv_results.keys())
    rmses = [cv_results[n] for n in names]
    fig = px.bar(x=names, y=rmses, labels={'x': 'Model', 'y': 'CV RMSE'}, title='Cross‑validated RMSE (lower is better)')
    st.plotly_chart(fig, use_container_width=True)

def shap_explanation(model, transformed_input):
    if not _has_shap:
        st.info('Install SHAP to see explanation.')
        return
    explainer = shap.Explainer(model)
    shap_values = explainer(transformed_input)
    st.subheader('Prediction explanation')
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(bbox_inches='tight', dpi=90)

def main():
    st.set_page_config(page_title='Austin Housing Price App', layout='wide')
    st.title("🏡 Austin Housing Price Prediction")

    with st.sidebar:
        st.header("Configuration")
        use_pca = st.checkbox("Use PCA on numeric features", value=False)

    df = load_data()
    model_dict = get_model(df, use_pca)
    pipeline = model_dict['pipeline']
    meta = model_dict['meta']

    tab1, tab2, tab3 = st.tabs(["📊 Explore Data", "🤖 Model Insights", "📍 Predict"])

    with tab1:
        st.subheader("Feature Correlation")
        correlation_heatmap(df)

        st.subheader("Price Distribution")
        fig = px.histogram(df, x='latestPrice', nbins=50)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Data Preview")
        st.dataframe(df.head(50))

    with tab2:
        st.subheader("Cross‑Validation RMSE")
        plot_cv_results(meta['cv_results'])

        st.markdown("### Model Metadata")
        st.json(meta)

    with tab3:
        st.subheader("🧮 Predict Home Price")
        input_df = build_sidebar(df, meta['numeric_cols'], meta['categorical_cols'])

        prediction = pipeline.predict(input_df)[0]
        st.metric("Predicted Price ($)", f"{prediction:,.0f}")

        lat, lon = input_df.get("latitude", [30.2672])[0], input_df.get("longitude", [-97.7431])[0]
        st.map(input_df[['latitude', 'longitude']])

        st.subheader("Nearby Listings Map")
        plot_map(df, lat, lon)

        if _has_shap:
            shap_explanation(pipeline.named_steps['model'], pipeline.named_steps['prep'].transform(input_df))

if __name__ == '__main__':
    main()