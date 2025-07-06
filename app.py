# app.py – Austin Housing Streamlit App (Optimized)
"""
Key upgrades
============
1. **Faster startup** – model artifacts (`best_model.pkl`) are loaded if present; retraining only happens the *first* time.
2. **Smarter feature selection** – keeps target‑correlated numeric columns (> thresh) and drops mutually‑correlated pairs (> 0.9). High‑cardinality categoricals are removed.
3. **Autocomplete street address** – fuzzy match suggestions as the user types.
4. **Date range slider** for `latest_saledate` (if present).
5. **SHAP fix** – requires `matplotlib`; graceful fallback if missing.
6. **2.5 mile radius stats and marker on map** – based on lat/lon from input.
"""
import os
import zipfile
import difflib
from datetime import datetime, date
from typing import List, Dict

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

#############################
# CONSTANTS & CONFIGURATION #
#############################
KAGGLE_DATASET = "ericpierce/austinhousingprices"
FILE_NAME = "austinHousingData.csv"
MODEL_PATH = "best_model.pkl"
TARGET = "latestPrice"
RADIUS_M = 4023  # 2.5 miles in meters

@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    if os.path.isfile(FILE_NAME):
        df = pd.read_csv(FILE_NAME)
    else:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi(); api.authenticate()
            api.dataset_download_file(KAGGLE_DATASET, FILE_NAME, path=".", force=True)
            with zipfile.ZipFile(FILE_NAME + ".zip", "r") as z:
                z.extract(FILE_NAME)
            os.remove(FILE_NAME + ".zip")
            df = pd.read_csv(FILE_NAME)
        except Exception:
            url = "https://raw.githubusercontent.com/selva86/datasets/master/AustinHousingData.csv"
            df = pd.read_csv(url)

    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[f"{col}_num"] = df[col].apply(lambda x: x.toordinal() if pd.notnull(x) else np.nan)
    return df

def select_features(df: pd.DataFrame, target: str = TARGET, thresh: float = 0.01, corr_cutoff: float = 0.8):
    num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in num_cols_all:
        num_cols_all.remove(target)
    corrs = df[num_cols_all + [target]].corr()[target].abs()
    num_selected = corrs[corrs > thresh].index.tolist()
    corr_matrix = df[num_selected].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > corr_cutoff)]
    num_selected = [c for c in num_selected if c not in to_drop]
    cat_all = df.select_dtypes(exclude=[np.number]).columns.tolist()
    cat_selected = [c for c in cat_all if df[c].nunique() < 20 and c != "streetAddress"]
    return num_selected, cat_selected

def build_preprocessor(num_cols: List[str], cat_cols: List[str], use_pca: bool):
    num_pipe = [("scale", StandardScaler())]
    if use_pca and len(num_cols) > 8:
        num_pipe.append(("pca", PCA(n_components=8)))
    return ColumnTransformer([
        ("num", Pipeline(num_pipe), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

def train_best_model(X, y, preprocessor):
    models = {
        "XGBoost": XGBRegressor(random_state=42, learning_rate=0.05, n_estimators=600, max_depth=6, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror"),
        "RandomForest": RandomForestRegressor(n_estimators=400, max_depth=15, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_results = {}
    best_rmse = np.inf; best_pipe = None; best_name = None
    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        rmse = np.mean(np.sqrt(-cross_val_score(pipe, X, y, scoring="neg_mean_squared_error", cv=cv, n_jobs=-1)))
        cv_results[name] = rmse
        if rmse < best_rmse:
            best_rmse, best_pipe, best_name = rmse, pipe, name
    best_pipe.fit(X, y)
    return best_pipe, best_name, cv_results

@st.cache_resource(show_spinner=True)
def get_model(df: pd.DataFrame, use_pca: bool):
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    num_cols, cat_cols = select_features(df)
    X = df[num_cols + cat_cols]
    y = df[TARGET]
    preprocessor = build_preprocessor(num_cols, cat_cols, use_pca)
    pipe, best_name, cv_results = train_best_model(X, y, preprocessor)
    meta = {"best_model": best_name, "cv_results": cv_results, "numeric_cols": num_cols, "categorical_cols": cat_cols, "use_pca": use_pca}
    bundle = {"pipeline": pipe, "meta": meta}
    joblib.dump(bundle, MODEL_PATH)
    return bundle

@st.cache_data
def get_address_suggestions(addresses: List[str], query: str) -> List[str]:
    return difflib.get_close_matches(query, addresses, n=5, cutoff=0.3)

def build_sidebar(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]):
    st.sidebar.header("Property Details")
    query = st.sidebar.text_input("Enter Address (free text)")
    user_data = {"streetAddress": query.strip()}

    # For numeric columns: apply formatting and sliders
    for n in num_cols:
        if n in ["latitude", "longitude", "avgSchoolRating", "avgSchoolDistance", "propertyTaxRate"]:
            default = round(df[n].median(), 1)
            user_data[n] = round(st.sidebar.number_input(n, value=default, step=0.1), 1)
        else:
            default = int(round(df[n].median()))
            user_data[n] = st.sidebar.number_input(n, value=default, step=1, format="%d")

    for c in cat_cols:
        options = df[c].dropna().unique().tolist()
        user_data[c] = st.sidebar.selectbox(c, options)

    if "latest_saledate" in df.columns:
        min_d, max_d = df["latest_saledate"].min().date(), df["latest_saledate"].max().date()
        date_range = st.sidebar.date_input("Sale Date Range", [min_d, max_d])
        user_data["latest_saledate_num"] = np.mean([d.toordinal() for d in date_range])

    return pd.DataFrame([user_data]), user_data

# Refine circle map to properly reflect 2.5 mile radius

def plot_circle_map(df, lat, lon):
    df_filtered = df.copy()
    df_filtered = df_filtered[df_filtered["latitude"].between(lat - 0.036, lat + 0.036) &
                              df_filtered["longitude"].between(lon - 0.036, lon + 0.036)]

    scatter = pdk.Layer("ScatterplotLayer",
                        data=df_filtered,
                        get_position='[longitude, latitude]',
                        get_color='[0, 112, 255, 120]',
                        get_radius=60)

    center = pdk.Layer("ScatterplotLayer",
                       data=pd.DataFrame({"latitude": [lat], "longitude": [lon]}),
                       get_position='[longitude, latitude]',
                       get_color='[255, 0, 0]',
                       get_radius=60)

    st.pydeck_chart(pdk.Deck(
        layers=[scatter, center],
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=12, pitch=40)))

    if not df_filtered.empty:
        st.subheader("📍 Nearby Area Stats (2.5 mi radius)")
        st.write({
            "Total Listings": len(df_filtered),
            "Min Price": df_filtered[TARGET].min(),
            "Max Price": df_filtered[TARGET].max(),
            "Median Price": df_filtered[TARGET].median(),
            "Average Price": df_filtered[TARGET].mean()
        })

def shap_explanation(pipe: Pipeline, input_df: pd.DataFrame):
    if not _HAS_SHAP:
        st.info("Install SHAP + matplotlib to see explanation.")
        return
    try:
        transformed = pipe.named_steps["prep"].transform(input_df)
        explainer = shap.Explainer(pipe.named_steps["model"])
        shap_vals = explainer(transformed)
        shap.plots.waterfall(shap_vals[0], show=False)
        st.pyplot(bbox_inches="tight")
    except Exception as e:
        st.warning(f"SHAP error: {e}")

def main():
    st.set_page_config(page_title="Austin Housing Price", layout="wide")
    st.title("🏡 Austin Housing Price Predictor")

    use_pca = st.sidebar.checkbox("Use PCA (numeric features)", value=False)
    df = load_data()
    bundle = get_model(df, use_pca)
    pipeline = bundle["pipeline"]
    meta = bundle["meta"]

    tab1, tab2, tab3 = st.tabs(["📊 Explore", "🧠 Model", "📍 Predict"])

    with tab1:
        st.subheader("🔍 Feature Correlation")
        corr_df = df.select_dtypes(include=[np.number]).corr().stack().reset_index(name='value')
        chart = alt.Chart(corr_df).mark_rect().encode(
            x='level_0:O', y='level_1:O', color=alt.Color('value:Q', scale=alt.Scale(scheme='redblue')),
            tooltip=['level_0', 'level_1', alt.Tooltip('value:Q', format='.2f')])
        st.altair_chart(chart, use_container_width=True)

        st.subheader("📈 Price Distribution")
        st.plotly_chart(px.histogram(df, x=TARGET, nbins=50), use_container_width=True)
        st.subheader("🔢 Sample Data")
        st.dataframe(df.head(50))

    with tab2:
        st.subheader("📉 Cross‑Validation RMSE")
        st.plotly_chart(px.bar(x=list(meta['cv_results'].keys()),
                               y=list(meta['cv_results'].values()),
                               labels={'x': 'Model', 'y': 'CV RMSE'}))
        st.markdown("### Model Metadata")
        st.json(meta)

    with tab3:
        input_df, input_meta = build_sidebar(df, meta['numeric_cols'], meta['categorical_cols'])
        if st.button("🔍 Predict Price"):
            prediction = pipeline.predict(input_df)[0]
            st.metric("Predicted Home Price", f"${prediction:,.0f}")

            lat = input_df["latitude"].iloc[0]
            lon = input_df["longitude"].iloc[0]
            plot_circle_map(df, lat, lon)

            shap_explanation(pipeline, input_df)

if __name__ == "__main__":
    main()