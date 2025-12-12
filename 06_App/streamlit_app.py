import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Project and source paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "04_Src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Service layer imports
from services.forecasting_service import (
    load_lstm_model_and_metadata,
    load_latest_feature_window,
    predict_next_return_with_lstm,
)
from services.generative_service import (
    load_trained_vae,
    load_recent_return_window,
    generate_return_scenarios,
)
from services.explainability_service import (
    load_tabular_data,
    train_random_forest_for_explainability,
    compute_local_shap_for_index,
)


def load_last_close_price(project_root: Path) -> float:
    """
    Load the last available close price from the daily features file.
    """
    features_path = project_root / "02_Data" / "Processed" / "sp500_features_daily.csv"
    df = pd.read_csv(features_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if "Close" not in df.columns:
        raise ValueError("Expected a 'Close' column in sp500_features_daily.csv")

    last_close = float(df["Close"].iloc[-1])
    return last_close


def returns_to_price_paths(
    base_price: float,
    return_paths: np.ndarray,
) -> np.ndarray:
    """
    Convert paths of daily log returns into price paths
    starting from base_price.

    return_paths shape:
        num_paths, seq_len
    """
    cum_returns = return_paths.cumsum(axis=1)
    prices = base_price * np.exp(cum_returns)
    return prices


@st.cache_resource
def get_forecasting_components():
    """
    Load and cache the LSTM forecasting model and its metadata.
    """
    model, meta = load_lstm_model_and_metadata(project_root=PROJECT_ROOT)
    return model, meta


@st.cache_resource
def get_generative_components():
    """
    Load and cache the trained VAE model.
    """
    vae_model = load_trained_vae(project_root=PROJECT_ROOT, seq_len=30)
    return vae_model


@st.cache_resource
def get_explainability_components():
    """
    Load and cache the random forest explainability model
    along with validation data.
    """
    (
        df_train,
        df_val,
        X_train,
        y_train,
        X_val,
        y_val,
        feature_cols,
    ) = load_tabular_data(project_root=PROJECT_ROOT)
    rf = train_random_forest_for_explainability(X_train, y_train)
    return df_val, X_val, feature_cols, rf


@st.cache_resource
def load_global_shap_importance() -> pd.DataFrame:
    """
    Load global SHAP feature importance values if available.
    """
    path = PROJECT_ROOT / "05_Results" / "rf_shap_global_importance.csv"
    if not path.exists():
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    df = pd.read_csv(path)
    return df


def page_forecast():
    st.header("Next day forecast")
    st.write(
        "This view shows the LSTM based next day log return forecast for the S and P five hundred, "
        "derived from the last thirty trading days of engineered features and news sentiment."
    )

    model, meta = get_forecasting_components()
    window, last_date = load_latest_feature_window(
        project_root=PROJECT_ROOT,
        window_size=meta["window_size"],
    )

    pred = predict_next_return_with_lstm(model, window)
    last_close = load_last_close_price(PROJECT_ROOT)
    next_price_estimate = last_close * float(np.exp(pred))

    st.subheader("Summary")
    st.write(f"Last available date: {last_date.date()}")
    st.write(f"Last close price: {last_close:.2f}")
    st.write(f"Predicted next day log return: {pred:.6f}")
    st.write(f"Implied next day price (approximate): {next_price_estimate:.2f}")

    st.subheader("Current model input window")
    st.write(
        "The model uses the last thirty trading days of engineered features as input."
    )
    df_window = pd.DataFrame(
        window,
        columns=meta["feature_cols"],
    )
    st.dataframe(df_window)


def page_scenarios():
    st.header("VAE scenarios")
    st.write(
        "This view uses a variational autoencoder trained on thirty day windows of daily log returns. "
        "Starting from the most recent window, it samples multiple plausible future return paths in latent space "
        "and converts them into implied price scenarios from the current index level."
    )

    vae_model = get_generative_components()
    returns_window, last_date = load_recent_return_window(
        project_root=PROJECT_ROOT,
        window_size=30,
    )
    last_close = load_last_close_price(PROJECT_ROOT)

    st.write(f"Last available date: {last_date.date()}")
    st.write(f"Last close price: {last_close:.2f}")

    num_paths = st.slider(
        "Number of scenarios",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
    )
    noise_scale = st.slider(
        "Latent noise scale",
        min_value=0.2,
        max_value=2.0,
        value=1.0,
        step=0.2,
    )

    scenarios = generate_return_scenarios(
        model=vae_model,
        last_window=returns_window,
        num_paths=num_paths,
        noise_scale=noise_scale,
    )

    st.subheader("Return paths in the window")
    fig_ret, ax_ret = plt.subplots(figsize=(8, 4))
    ax_ret.plot(returns_window, label="reference history", linewidth=2)
    for i in range(num_paths):
        ax_ret.plot(scenarios[i], alpha=0.5)
    ax_ret.set_xlabel("time step in window")
    ax_ret.set_ylabel("log return")
    ax_ret.legend()
    fig_ret.tight_layout()
    st.pyplot(fig_ret)

    st.subheader("Implied price paths from last close")
    price_paths = returns_to_price_paths(last_close, scenarios)

    fig_price, ax_price = plt.subplots(figsize=(8, 4))
    ax_price.axhline(
        last_close,
        color="black",
        linestyle="--",
        linewidth=1,
        label="current price",
    )
    for i in range(num_paths):
        ax_price.plot(price_paths[i], alpha=0.5)
    ax_price.set_xlabel("days ahead in scenario window")
    ax_price.set_ylabel("price level")
    ax_price.legend()
    fig_price.tight_layout()
    st.pyplot(fig_price)


def page_explainability():
    st.header("Explainability")

    st.write(
        "This view uses a random forest model trained on daily features and SHAP values. "
        "It shows which inputs contributed most to the predicted next day return for a chosen validation date, "
        "and also the average importance of each feature across the whole validation sample."
    )

    df_val, X_val, feature_cols, rf = get_explainability_components()

    st.write("Validation set size:", len(df_val))

    col_left, col_right = st.columns(2)

    with col_left:
        index = st.slider(
            "Pick a validation index",
            min_value=0,
            max_value=len(df_val) - 1,
            value=len(df_val) - 1,
        )

    with col_right:
        date_options = df_val["Date"].dt.date.astype(str).tolist()
        selected_date_str = st.selectbox(
            "Or pick a validation date",
            options=date_options,
            index=len(date_options) - 1,
        )

        # If date selection and slider differ, prefer the date selection
        date_mask = df_val["Date"].dt.date.astype(str) == selected_date_str
        if date_mask.any():
            index = int(df_val.index[date_mask][0])

    explanation = compute_local_shap_for_index(
        rf=rf,
        df_val=df_val,
        X_val=X_val,
        feature_cols=feature_cols,
        index=index,
    )

    st.subheader("Selected date and prediction")
    st.write("Date:", explanation["date"].date())
    st.write("True next return:", f"{explanation['true_target']:.6f}")
    st.write("Predicted next return:", f"{explanation['predicted_target']:.6f}")

    local_df = explanation["local_explanation"]

    st.subheader("Top contributing features for this day")
    st.dataframe(local_df.head(15))

    st.subheader("Global feature importance from SHAP mean absolute values")
    df_importance = load_global_shap_importance()

    if df_importance.empty:
        st.write(
            "Global importance file was not found. "
            "Run the explainability notebook to generate rf_shap_global_importance.csv in the results folder."
        )
    else:
        top_n = st.slider(
            "Number of top features to display",
            min_value=5,
            max_value=min(30, len(df_importance)),
            value=min(15, len(df_importance)),
        )

        top_imp = df_importance.head(top_n)

        fig_imp, ax_imp = plt.subplots(figsize=(6, 0.3 * top_n + 1))
        ax_imp.barh(top_imp["feature"][::-1], top_imp["mean_abs_shap"][::-1])
        ax_imp.set_xlabel("mean absolute SHAP value")
        ax_imp.set_ylabel("feature")
        fig_imp.tight_layout()

        st.pyplot(fig_imp)

        st.write("Global importance table for the top features:")
        st.dataframe(top_imp.reset_index(drop=True))


def main():
    st.set_page_config(
        page_title="Explainable generative financial forecasting",
        layout="wide",
    )

    st.title("Explainable Generative AI Dashboard")
    st.caption(
        "S and P five hundred daily forecasting with LSTM, VAE scenarios, and SHAP-based explainability."
    )

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select view",
        (
            "Forecast",
            "Scenarios",
            "Explainability",
        ),
    )

    if page == "Forecast":
        page_forecast()
    elif page == "Scenarios":
        page_scenarios()
    elif page == "Explainability":
        page_explainability()


if __name__ == "__main__":
    main()
