import os
import re
import io
import base64
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import requests
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import statsmodels.api as sm
from openai import OpenAI
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from xgboost import XGBRegressor
from datetime import timedelta


ART = None

# ============================================================
# CONFIG
# ============================================================
PATH_EQUIPOS = r"C:\Users\Sebastian.garzon.LAPTOP-G9J39UDS\Documents\Cosas\Cosas\Prueba tecnica Senior\Datos\historico_equipos.csv"
PATH_X = r"C:\Users\Sebastian.garzon.LAPTOP-G9J39UDS\Documents\Cosas\Cosas\Prueba tecnica Senior\Datos\X.csv"
PATH_Y = r"C:\Users\Sebastian.garzon.LAPTOP-G9J39UDS\Documents\Cosas\Cosas\Prueba tecnica Senior\Datos\Y.csv"
PATH_Z = r"C:\Users\Sebastian.garzon.LAPTOP-G9J39UDS\Documents\Cosas\Cosas\Prueba tecnica Senior\Datos\Z.csv"

ROLL_WINDOWS = [3, 6, 9, 12, 15, 30]
LAGS = [1, 3, 6, 12, 15, 30]
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)
DEFAULT_FORECAST_HORIZON = 30
XGB_RANDOM_STATE = 42


# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class TrainingArtifacts:
    df_raw: pd.DataFrame
    df_model: pd.DataFrame
    features: List[str]
    model1: XGBRegressor
    model2: XGBRegressor
    xgb_metrics: Dict[str, float]
    arima_metrics: Dict[str, float]
    adf_results: Dict[str, float]
    ols_full_summary: str
    ols_reduced_summary: str
    shap_model1: pd.DataFrame
    shap_model2: pd.DataFrame
    xgb_importance_model1: pd.DataFrame
    xgb_importance_model2: pd.DataFrame
    top_corr: pd.DataFrame
    seasonal_comment: str
    sarimax_x_summary: str
    sarimax_z_summary: str


# ============================================================
# LOAD + CLEAN
# ============================================================
def load_data(
    path_equipos: str = PATH_EQUIPOS,
    path_x: str = PATH_X,
    path_y: str = PATH_Y,
    path_z: str = PATH_Z,
) -> pd.DataFrame:
    equipos = pd.read_csv(path_equipos)
    x_df = pd.read_csv(path_x)
    y_df = pd.read_csv(path_y, sep=";")
    z_df = pd.read_csv(path_z)

    equipos["Date"] = pd.to_datetime(equipos["Date"], format="%Y-%m-%d")
    x_df["Date"] = pd.to_datetime(x_df["Date"], format="%Y-%m-%d")
    y_df["Date"] = pd.to_datetime(y_df["Date"], format="%d/%m/%Y")
    z_df["Date"] = pd.to_datetime(z_df["Date"], format="%Y-%m-%d")
    y_df["Price"] = y_df["Price"].astype(str).str.replace(",", ".", regex=False).astype(float)

    # Se toma el dataset consolidado de equipos como base del modelamiento final
    df = equipos.copy().sort_values("Date").reset_index(drop=True)
    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def create_features(df_base: pd.DataFrame) -> pd.DataFrame:
    df = df_base.copy().sort_values("Date").reset_index(drop=True)

    for w in ROLL_WINDOWS:
        df[f"X_prom_{w}"] = df["Price_X"].rolling(window=w).mean()
        df[f"Z_prom_{w}"] = df["Price_Z"].rolling(window=w).mean()

    for lag in LAGS:
        df[f"X_lag_{lag}"] = df["Price_X"].shift(lag)
        df[f"Z_lag_{lag}"] = df["Price_Z"].shift(lag)

    df["X_diff"] = df["Price_X"].diff()
    df["Z_diff"] = df["Price_Z"].diff()
    return df


# ============================================================
# TESTS + EDA
# ============================================================
def adf_pvalue(series: pd.Series) -> float:
    return float(adfuller(series.dropna())[1])


def compute_ols_summaries(df: pd.DataFrame) -> Tuple[str, str]:
    x_full = sm.add_constant(df[["Price_X", "Price_Y", "Price_Z"]])
    y = df["Price_Equipo2"]
    model_full = sm.OLS(y, x_full).fit()

    x_reduced = sm.add_constant(df[["Price_X", "Price_Z"]])
    model_reduced = sm.OLS(y, x_reduced).fit()
    return model_full.summary().as_text(), model_reduced.summary().as_text()


def seasonal_commentary(df: pd.DataFrame) -> str:
    comment = (
        "Se evaluó estacionalidad mediante inspección visual y descomposición aditiva sobre Price_Z. "
        "Como las materias primas muestran estructura temporal y no estacionariedad, se seleccionó SARIMA "
        "para capturar tendencia, diferenciación y componente estacional."
    )
    try:
        _ = seasonal_decompose(df.set_index("Date")["Price_Z"], model="additive", period=12)
    except Exception:
        comment += " No obstante, la descomposición puede ser sensible a ventanas y valores faltantes."
    return comment


def compute_top_correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Price_X", "Price_Y", "Price_Z", "Price_Equipo1", "Price_Equipo2"]
    corr = df[cols].corr().round(4)
    return corr


# ============================================================
# SARIMA
# ============================================================
def fit_sarima(series: pd.Series) -> Any:
    model = SARIMAX(series, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER)
    return model.fit(disp=False)


def sarima_forecast_with_ci(df: pd.DataFrame, steps: int) -> Tuple[pd.DataFrame, Any, Any]:
    df_sorted = df.sort_values("Date").reset_index(drop=True)
    x_model = fit_sarima(df_sorted["Price_X"])
    z_model = fit_sarima(df_sorted["Price_Z"])

    x_fcst = x_model.get_forecast(steps=steps)
    z_fcst = z_model.get_forecast(steps=steps)

    future_dates = pd.date_range(start=df_sorted["Date"].max(), periods=steps + 1, freq="D")[1:]
    x_ci = x_fcst.conf_int().reset_index(drop=True)
    z_ci = z_fcst.conf_int().reset_index(drop=True)

    out = pd.DataFrame(
        {
            "Date": future_dates,
            "Price_X": x_fcst.predicted_mean.reset_index(drop=True),
            "Price_Z": z_fcst.predicted_mean.reset_index(drop=True),
            "X_lower": x_ci.iloc[:, 0],
            "X_upper": x_ci.iloc[:, 1],
            "Z_lower": z_ci.iloc[:, 0],
            "Z_upper": z_ci.iloc[:, 1],
        }
    )
    return out, x_model, z_model


def backtest_sarima_price_x(df: pd.DataFrame, steps: int = 30) -> Dict[str, float]:
    # Replica la lógica de tu notebook: comparación con la cola real de X fuera de equipos
    x_df = pd.read_csv(PATH_X)
    x_df["Date"] = pd.to_datetime(x_df["Date"], format="%Y-%m-%d")
    x_df = x_df.sort_values("Date").reset_index(drop=True)

    fitted = fit_sarima(df.sort_values("Date")["Price_X"])
    fcst = fitted.get_forecast(steps=steps).predicted_mean.reset_index(drop=True)
    real = x_df[x_df["Date"] >= df["Date"].max()].head(steps).reset_index(drop=True)
    common = min(len(real), len(fcst))
    if common == 0:
        return {"mse_x": np.nan, "mape_x": np.nan}

    y_true = real["Price"].iloc[:common]
    y_pred = fcst.iloc[:common]
    return {
        "mse_x": float(mean_squared_error(y_true, y_pred)),
        "mape_x": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


# ============================================================
# XGBOOST + SHAP
# ============================================================
def build_model_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df_feat = create_features(df)
    drop_cols = ["Date", "Price_Y", "Price_Equipo1", "Price_Equipo2"]
    features = [c for c in df_feat.columns if c not in drop_cols]
    df_model = df_feat.dropna().reset_index(drop=True)
    return df_model, features


def train_xgb_models(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], XGBRegressor, XGBRegressor, Dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_model, features = build_model_dataset(df)
    x = df_model[features]
    y1 = df_model["Price_Equipo1"]
    y2 = df_model["Price_Equipo2"]

    x_train, x_test, y1_train, y1_test = train_test_split(x, y1, test_size=0.2, shuffle=False)
    _, _, y2_train, y2_test = train_test_split(x, y2, test_size=0.2, shuffle=False)

    model1 = XGBRegressor(random_state=XGB_RANDOM_STATE)
    model2 = XGBRegressor(random_state=XGB_RANDOM_STATE)
    model1.fit(x_train, y1_train)
    model2.fit(x_train, y2_train)

    y1_pred = model1.predict(x_test)
    y2_pred = model2.predict(x_test)

    metrics = {
        "rmse_equipo1": float(np.sqrt(mean_squared_error(y1_test, y1_pred))),
        "mape_equipo1": float(mean_absolute_percentage_error(y1_test, y1_pred)),
        "rmse_equipo2": float(np.sqrt(mean_squared_error(y2_test, y2_pred))),
        "mape_equipo2": float(mean_absolute_percentage_error(y2_test, y2_pred)),
    }

    imp1 = pd.DataFrame({"Feature": features, "Importance": model1.feature_importances_}).sort_values("Importance", ascending=False)
    imp2 = pd.DataFrame({"Feature": features, "Importance": model2.feature_importances_}).sort_values("Importance", ascending=False)

    shap1 = shap.Explainer(model1, x_train)(x_test)
    shap2 = shap.Explainer(model2, x_train)(x_test)
    shap1_df = pd.DataFrame({
        "Feature": x_test.columns,
        "Importance": np.abs(shap1.values).mean(axis=0)
    }).sort_values("Importance", ascending=False)
    shap2_df = pd.DataFrame({
        "Feature": x_test.columns,
        "Importance": np.abs(shap2.values).mean(axis=0)
    }).sort_values("Importance", ascending=False)

    return df_model, features, model1, model2, metrics, imp1, imp2, shap1_df, shap2_df


# ============================================================
# FULL TRAINING ARTIFACTS
# ============================================================
def build_training_artifacts() -> TrainingArtifacts:
    df = load_data()
    

    ols_full, ols_reduced = compute_ols_summaries(df)
    adf_results = {
        "Price_X": adf_pvalue(df["Price_X"]),
        "Price_Z": adf_pvalue(df["Price_Z"]),
        "Price_X_diff": adf_pvalue(df["Price_X"].diff()),
        "Price_Z_diff": adf_pvalue(df["Price_Z"].diff()),
        
    }
    arima_metrics = backtest_sarima_price_x(df, steps=30)

    df_model, features, model1, model2, xgb_metrics, imp1, imp2, shap1, shap2 = train_xgb_models(df)

    # Reentrenamiento con 100% de los datos modelables
    full_x = df_model[features]
    full_y1 = df_model["Price_Equipo1"]
    full_y2 = df_model["Price_Equipo2"]
    model1.fit(full_x, full_y1)
    model2.fit(full_x, full_y2)
    sarimax_x = fit_sarima(df["Price_X"])
    sarimax_z = fit_sarima(df["Price_Z"])

    return TrainingArtifacts(
        df_raw=df,
        df_model=df_model,
        features=features,
        model1=model1,
        model2=model2,
        xgb_metrics=xgb_metrics,
        arima_metrics=arima_metrics,
        adf_results=adf_results,
        ols_full_summary=ols_full,
        ols_reduced_summary=ols_reduced,
        shap_model1=shap1,
        shap_model2=shap2,
        xgb_importance_model1=imp1,
        xgb_importance_model2=imp2,
        top_corr=compute_top_correlations(df),
        seasonal_comment=seasonal_commentary(df),
        sarimax_x_summary=sarimax_x.summary().as_text(),
        sarimax_z_summary=sarimax_z.summary().as_text(),
        
    )


# ============================================================
# FORECASTS: MEAN / LOWER / UPPER FOR EQUIPOS
# ============================================================
def _build_future_features_from_sarima(base_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    hist = base_df[["Date", "Price_X", "Price_Z"]].copy()
    future = forecast_df[["Date", "Price_X", "Price_Z"]].copy()
    full = pd.concat([hist, future], axis=0, ignore_index=True)
    full = create_features(full)
    last_hist_date = base_df["Date"].max()
    future_feat = full[full["Date"] > last_hist_date].copy().reset_index(drop=True)
    return future_feat


def forecast_targets(artifacts: TrainingArtifacts, steps: int = DEFAULT_FORECAST_HORIZON) -> Dict[str, pd.DataFrame]:
    base_df = artifacts.df_raw.copy()
    fcst_mean, _, _ = sarima_forecast_with_ci(base_df, steps=steps)

    fcst_lower = fcst_mean.copy()
    fcst_upper = fcst_mean.copy()
    fcst_lower["Price_X"] = fcst_mean["X_lower"]
    fcst_lower["Price_Z"] = fcst_mean["Z_lower"]
    fcst_upper["Price_X"] = fcst_mean["X_upper"]
    fcst_upper["Price_Z"] = fcst_mean["Z_upper"]

    mean_feat = _build_future_features_from_sarima(base_df, fcst_mean)
    lower_feat = _build_future_features_from_sarima(base_df, fcst_lower)
    upper_feat = _build_future_features_from_sarima(base_df, fcst_upper)

    x_mean = mean_feat[artifacts.features]
    x_lower = lower_feat[artifacts.features]
    x_upper = upper_feat[artifacts.features]

    pred_mean = pd.DataFrame({
        "Date": mean_feat["Date"],
        "Price_Equipo1_pred": artifacts.model1.predict(x_mean),
        "Price_Equipo2_pred": artifacts.model2.predict(x_mean),
    })
    pred_lower = pd.DataFrame({
        "Date": lower_feat["Date"],
        "Equipo1_lower": artifacts.model1.predict(x_lower),
        "Equipo2_lower": artifacts.model2.predict(x_lower),
    })
    pred_upper = pd.DataFrame({
        "Date": upper_feat["Date"],
        "Equipo1_upper": artifacts.model1.predict(x_upper),
        "Equipo2_upper": artifacts.model2.predict(x_upper),
    })

    pred_final = pred_mean.merge(pred_lower, on="Date").merge(pred_upper, on="Date")

    # suavizado ligero de bandas para visualización
    for col in ["Equipo1_lower", "Equipo1_upper", "Equipo2_lower", "Equipo2_upper"]:
        pred_final[f"{col}_smooth"] = pred_final[col].rolling(3, min_periods=1).mean()

    return {
        "x_z_forecast": fcst_mean,
        "targets_forecast": pred_final,
    }


# ============================================================
# PLOTS
# ============================================================
def _fig_to_array(fig: plt.Figure):
    return fig


def plot_xz_forecast(artifacts: TrainingArtifacts, steps: int = 30, commodity: str = "X") -> plt.Figure:
    fcst = forecast_targets(artifacts, steps=steps)["x_z_forecast"]
    df = artifacts.df_raw.copy()

    fig, ax = plt.subplots(figsize=(12, 5))
    if commodity.upper() == "X":
        ax.plot(df["Date"].iloc[-60:], df["Price_X"].iloc[-60:], label="Real X", linewidth=2)
        ax.plot(fcst["Date"], fcst["Price_X"], linestyle="--", label="Forecast X", linewidth=2)
        ax.fill_between(fcst["Date"], fcst["X_lower"], fcst["X_upper"], alpha=0.2, label="Intervalo X")
        ax.set_title(f"Price X: pronóstico a {steps} días")
    else:
        ax.plot(df["Date"].iloc[-60:], df["Price_Z"].iloc[-60:], label="Real Z", linewidth=2)
        ax.plot(fcst["Date"], fcst["Price_Z"], linestyle="--", label="Forecast Z", linewidth=2)
        ax.fill_between(fcst["Date"], fcst["Z_lower"], fcst["Z_upper"], alpha=0.2, label="Intervalo Z")
        ax.set_title(f"Price Z: pronóstico a {steps} días")
    ax.axvline(x=df["Date"].max(), linestyle=":", label="Inicio forecast")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_target_forecast(artifacts: TrainingArtifacts, steps: int = 30, target: str = "Equipo2") -> plt.Figure:
    outputs = forecast_targets(artifacts, steps=steps)
    pred = outputs["targets_forecast"]
    df = artifacts.df_raw.copy()

    fig, ax = plt.subplots(figsize=(12, 5))
    if target.lower() == "equipo1":
        ax.plot(df["Date"].iloc[-60:], df["Price_Equipo1"].iloc[-60:], label="Real Equipo 1", linewidth=2)
        ax.plot(pred["Date"], pred["Price_Equipo1_pred"], linestyle="--", linewidth=2, label="Pronóstico Equipo 1")
        ax.fill_between(pred["Date"], pred["Equipo1_lower_smooth"], pred["Equipo1_upper_smooth"], alpha=0.15, label="Intervalo")
        ax.set_title(f"Equipo 1: pronóstico a {steps} días")
    else:
        ax.plot(df["Date"].iloc[-60:], df["Price_Equipo2"].iloc[-60:], label="Real Equipo 2", linewidth=2)
        ax.plot(pred["Date"], pred["Price_Equipo2_pred"], linestyle="--", linewidth=2, label="Pronóstico Equipo 2")
        ax.fill_between(pred["Date"], pred["Equipo2_lower_smooth"], pred["Equipo2_upper_smooth"], alpha=0.15, label="Intervalo")
        ax.set_title(f"Equipo 2: pronóstico a {steps} días")
    ax.axvline(x=df["Date"].max(), linestyle=":", label="Inicio forecast")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return fig


# ============================================================
# KNOWLEDGE BASE FOR THE AGENT
# ============================================================
def methodology_text(artifacts: TrainingArtifacts) -> str:
    return f"""
**Metodología utilizada**

1. **Datos de entrada**
   - Se usó como base el dataset consolidado `historico_equipos`, que ya contiene `Price_X`, `Price_Y`, `Price_Z`, `Price_Equipo1` y `Price_Equipo2`.
   - También se revisaron las fuentes individuales de X, Y y Z para validar consistencia temporal.

2. **EDA**
   - Se revisaron rangos de fechas, tipos de dato, calidad de fechas y precios.
   - Se graficaron las series de equipos y materias primas.
   - Se construyó matriz de correlación entre X, Y, Z y ambos equipos.

3. **Selección de variables**
   - `Price_Y` se eliminó del modelamiento final porque, aunque muestra correlación con Equipo 2, en regresión múltiple comparte información con Z.
   - Se comparó una OLS con X, Y, Z vs otra con X, Z. La versión reducida mantiene interpretación más estable y evita redundancia.

4. **Justificación de SARIMA**
   - Se aplicó Dickey-Fuller aumentado a X y Z. Los p-values iniciales fueron mayores a 0.05, indicando no estacionariedad.
   - Tras diferenciar, la serie se estabiliza mejor.
   - Además se inspeccionó estacionalidad y por eso se eligió SARIMA en lugar de ARIMA simple.

5. **Modelo final de targets**
   - Se generaron features temporales: promedios móviles y lags de X y Z.
   - Con esas variables se entrenaron dos modelos XGBoost: uno para Equipo 1 y otro para Equipo 2.
   - Se evaluó SHAP para interpretar importancia de variables.

6. **Pronóstico**
   - Primero se pronostican X y Z con SARIMA.
   - Luego se proyectan Equipo 1 y Equipo 2 usando XGBoost sobre las features futuras.
   - Se construyen escenarios mean, lower y upper propagando la incertidumbre de X y Z.
""".strip()


def build_answer_catalog(artifacts: TrainingArtifacts, steps: int) -> Dict[str, str]:
    xgb = artifacts.xgb_metrics
    arima = artifacts.arima_metrics
    adf = artifacts.adf_results

    return {
        "datos": (
            f"Se trabajó con un dataset consolidado que contiene Date, Price_X, Price_Y, Price_Z, Price_Equipo1 y Price_Equipo2. "
            f"El rango temporal del dataset base va de {artifacts.df_raw['Date'].min().date()} a {artifacts.df_raw['Date'].max().date()}."
        ),
        "eda": (
            "En el EDA se revisaron fechas, consistencia de variables, series históricas, correlación entre materias primas y equipos, "
            "y comportamiento temporal de X, Y y Z frente a Price_Equipo1 y Price_Equipo2."
        ),
        "variables_eliminadas": (
            "La variable eliminada del modelamiento final fue Price_Y. Se conservó X y Z como drivers principales y se añadieron lags y promedios móviles de ambas."
        ),
        "por_que_y": (
            "Price_Y se retiró porque en regresión múltiple con X, Y y Z parte de su efecto estaba absorbido por Z. "
            "Eso sugiere colinealidad y redundancia. Por estabilidad e interpretabilidad se priorizó el modelo con X y Z."
        ),
        "sarima": (
            f"Se usó SARIMA porque X y Z no eran estacionarias al inicio. Los p-values ADF fueron: X={adf['Price_X']:.4f}, Z={adf['Price_Z']:.4f}. "
            f"Tras diferenciar, mejoran a X_diff={adf['Price_X_diff']:.4f}, Z_diff={adf['Price_Z_diff']:.4f}. {artifacts.seasonal_comment}"
        ),
        "kpis_arima": (
            f"Backtest del modelo de X con horizonte 30: MSE={arima['mse_x']:.4f}, MAPE={arima['mape_x']:.2%}."
        ),
        "shap": (
            "SHAP se usó para medir la contribución de cada feature a las predicciones de XGBoost. "
            "Las variables temporales derivadas de X y Z tienden a dominar, lo que indica que el modelo aprende no solo del nivel actual de los insumos sino de su dinámica reciente."
        ),
        "xgb": (
            f"Resultados del modelo XGBoost en holdout temporal: Equipo 1 -> RMSE={xgb['rmse_equipo1']:.4f}, MAPE={xgb['mape_equipo1']:.2%}; "
            f"Equipo 2 -> RMSE={xgb['rmse_equipo2']:.4f}, MAPE={xgb['mape_equipo2']:.2%}."
        ),
        "forecast": (
            f"El agente puede proyectar Equipo 1 y Equipo 2 a {steps} días o cualquier horizonte solicitado. Además construye escenarios central, lower y upper."
        ),
        "metodologia": methodology_text(artifacts),
        "materias_primas": (
            f"Se pronosticaron Price_X y Price_Z con SARIMAX usando "
            f"SARIMA_ORDER={SARIMA_ORDER} y SARIMA_SEASONAL_ORDER={SARIMA_SEASONAL_ORDER}. "
            "Se incluyen escenarios mean, lower y upper, además de la justificación metodológica y resultados del ajuste."
        ),
        "correlacion": (
        "Se evaluó la relación entre Price_X, Price_Y, Price_Z, Price_Equipo1 y Price_Equipo2 mediante una matriz de correlación. "
        "Esto permite identificar asociaciones lineales relevantes, posibles redundancias y relaciones fuertes entre materias primas y targets."
         ),
    }

def interpret_results(question: str, artifacts, forecast_outputs=None) -> str:
    xgb = artifacts.xgb_metrics
    arima = artifacts.arima_metrics
    adf = artifacts.adf_results

    partes = []

    # Interpretación XGBoost
    partes.append(
        f"En el modelo final XGBoost, el desempeño del Equipo 1 fue RMSE={xgb['rmse_equipo1']:.2f} y MAPE={xgb['mape_equipo1']:.2%}, "
        f"mientras que para el Equipo 2 fue RMSE={xgb['rmse_equipo2']:.2f} y MAPE={xgb['mape_equipo2']:.2%}."
    )

    if xgb["mape_equipo2"] < xgb["mape_equipo1"]:
        partes.append(
            "Esto indica que el modelo predice mejor el Equipo 2 que el Equipo 1, probablemente porque la relación entre las variables explicativas y el precio del Equipo 2 es más estable o más fácil de capturar."
        )
    else:
        partes.append(
            "Esto indica que el modelo predice mejor el Equipo 1 que el Equipo 2, lo cual sugiere una estructura de señal más consistente para ese target."
        )

    # Interpretación ARIMA/SARIMA
    partes.append(
        f"En el caso de la materia prima X, el modelo SARIMA obtuvo MSE={arima['mse_x']:.2f} y MAPE={arima['mape_x']:.2%}."
    )

    if arima["mape_x"] < 0.05:
        partes.append(
            "Ese MAPE es excelente, por lo que el pronóstico de X puede considerarse confiable como insumo del modelo final."
        )
    elif arima["mape_x"] < 0.15:
        partes.append(
            "Ese MAPE es bueno y sugiere un desempeño adecuado para el horizonte evaluado."
        )
    else:
        partes.append(
            "Ese MAPE refleja un error relevante, por lo que conviene usar el forecast con cautela."
        )

    # ADF
    partes.append(
        f"Las pruebas ADF iniciales muestran no estacionariedad en X (p-value={adf['Price_X']:.4f}) y Z (p-value={adf['Price_Z']:.4f}), "
        f"mientras que tras diferenciar las series los p-values bajan a X_diff={adf['Price_X_diff']:.4f} y Z_diff={adf['Price_Z_diff']:.4f}."
    )
    partes.append(
        "Eso justifica el uso de diferenciación y respalda la elección de SARIMA para modelar la dinámica temporal."
    )

    # Si hay forecast, añadir lectura del horizonte
    if forecast_outputs is not None:
        pred = forecast_outputs["targets_forecast"]

        eq1_ini = pred["Price_Equipo1_pred"].iloc[0]
        eq1_fin = pred["Price_Equipo1_pred"].iloc[-1]
        eq2_ini = pred["Price_Equipo2_pred"].iloc[0]
        eq2_fin = pred["Price_Equipo2_pred"].iloc[-1]

        dir_eq1 = "al alza" if eq1_fin > eq1_ini else "a la baja"
        dir_eq2 = "al alza" if eq2_fin > eq2_ini else "a la baja"

        partes.append(
            f"En el horizonte proyectado, el Equipo 1 muestra una trayectoria {dir_eq1}, pasando de {eq1_ini:.2f} a {eq1_fin:.2f}, "
            f"mientras que el Equipo 2 muestra una trayectoria {dir_eq2}, pasando de {eq2_ini:.2f} a {eq2_fin:.2f}."
        )

        amp_eq1 = (pred["Equipo1_upper"].iloc[-1] - pred["Equipo1_lower"].iloc[-1])
        amp_eq2 = (pred["Equipo2_upper"].iloc[-1] - pred["Equipo2_lower"].iloc[-1])

        if amp_eq2 > amp_eq1:
            partes.append(
                "Además, el intervalo del Equipo 2 es más amplio al final del horizonte, lo que indica mayor incertidumbre acumulada en su forecast."
            )
        else:
            partes.append(
                "Además, el intervalo del Equipo 1 resulta más amplio al final del horizonte, lo que indica mayor sensibilidad o incertidumbre acumulada en ese forecast."
            )

    partes.append(
        "En conclusión, la solución híbrida SARIMA + XGBoost es consistente con la estructura del problema: SARIMA captura la dinámica temporal de los insumos y XGBoost transforma esa información en predicciones finales más flexibles para los equipos."
    )

    return "\n\n".join(partes)

# ============================================================
# OPTIONAL LLM LAYER
# ============================================================
def llm_analyze(
    question: str,
    artifacts: TrainingArtifacts,
    key: str,
    base_answer: str,
    table: Optional[pd.DataFrame] = None,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    print("OPENAI_API_KEY presente:", bool(api_key))
    print("Pregunta:", question)
    print("Key detectada:", key)

    market_context = {
        "news_price_x": get_market_news("commodity prices OR industrial input prices OR metals market"),
        "news_price_z": get_market_news("construction materials prices OR commodity z market"),
        "fred_cpi": get_fred_series("CPIAUCSL", limit=3),   # inflación USA
        "fred_dxy": get_fred_series("DTWEXBGS", limit=3),   # broad dollar index
        "fred_rate": get_fred_series("FEDFUNDS", limit=3),  # tasas
    }

    if not api_key:
        return "[DEBUG] OPENAI_API_KEY no encontrada. El agente está respondiendo sin LLM.\n\n" + base_answer
    
    try:
        client = OpenAI(api_key=api_key)
        external_context = build_external_context(artifacts, 30)
        contexto = {
            "xgb_metrics": artifacts.xgb_metrics,
            "arima_metrics": artifacts.arima_metrics,
            "adf_results": artifacts.adf_results,
            "top_corr": artifacts.top_corr.to_dict(),
            "top_shap_model1": artifacts.shap_model1.head(10).to_dict(orient="records"),
            "top_shap_model2": artifacts.shap_model2.head(10).to_dict(orient="records"),
            "top_xgb_model1": artifacts.xgb_importance_model1.head(10).to_dict(orient="records"),
            "top_xgb_model2": artifacts.xgb_importance_model2.head(10).to_dict(orient="records"),
            "seasonal_comment": artifacts.seasonal_comment,
            "contexto_mercado": market_context,
            "external_context": external_context
        }

        if table is not None and not table.empty:
            contexto["table_preview"] = table.head(15).to_dict(orient="records")

        prompt = f"""
        Pregunta del usuario:
        {question}

        Tipo de consulta detectado:
        {key}

        Respuesta base técnica:
        {base_answer}

        #Resultados disponibles del análisis:
        #{contexto}

        Instrucciones:
        - Responde como un analista de datos senior.
        - Analiza, compara y concluye.
        - Si hay métricas, interprétalas.
        - Si hay forecast, describe tendencia, riesgo e incertidumbre.
        - Si hay SHAP, explica qué variables dominan y qué implica eso.
        - Si hay ADF/SARIMA, explica por qué la metodología es consistente.
        - Si hay contexto de mercado, relaciónalo con las variaciones de X y Z.
        - Usa noticias y variables macro solo para complementar, no para inventar causalidad.
        - Explica si el contexto externo refuerza o contradice el forecast.
        - No inventes datos.
        - No repitas solo el paso a paso; da conclusión.
        - Sé claro y profesional.
        - Relaciona el comportamiento proyectado con el contexto macroeconómico de Colombia
        - Usa las noticias como evidencia reciente del mercado
        - NO asumas que las noticias ocurren en el futuro
        - Explica si el forecast es consistente con inflación, tasas o contexto económico
        - Identifica riesgos (inflación, tasas, energía, tipo de cambio)
        - Da conclusiones claras y ejecutivas
    NO repitas el paso a paso técnico.
    NO inventes datos.
        """

        resp = client.responses.create(
            model="gpt-5.4-mini",
            input=[
                {
                    "role": "system",
                    "content": (
                        "Eres un analista senior de forecasting y machine learning. "
                        "Tu tarea es interpretar resultados reales, no repetir plantillas. "
                        "Debes razonar sobre métricas, incertidumbre, drivers y conclusiones."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        ) 
        print("LLM respondió correctamente")
        return resp.output_text.strip()

    except Exception as e:
        return f"{base_answer}\n\n[No se pudo generar análisis LLM: {str(e)}]"


# ============================================================
# ROUTER
# ============================================================
def extract_horizon(text: str, default: int = DEFAULT_FORECAST_HORIZON) -> int:
    nums = re.findall(r"\d+", text)
    return int(nums[0]) if nums else default


def route_question(question: str) -> str:
    q = question.lower()

    if any(k in q for k in ["dato", "entrada", "input"]):
        return "datos"

    if "eda" in q or "explor" in q:
        return "eda"

    if "elimin" in q and "variable" in q:
        return "variables_eliminadas"

    if "price_y" in q or ("y" in q and "elim" in q):
        return "por_que_y"

    if "sarima" in q or "dickey" in q or "adf" in q or "estacion" in q or "estacional" in q:
        return "kpis_arima" if "kpi" in q else "sarima"

    if "shap" in q:
        return "shap"

    if "xgb" in q or "xgboost" in q or "modelo final" in q:
        return "xgb"

    if "metod" in q or "como se hizo" in q or "qué hicimos" in q or "que hicimos" in q:
        return "metodologia"
    if any(k in q for k in [
        "materia prima", "materias primas", "price_x", "price_z",
        "commodities", "commodity", "insumos", "x y z", "price x", "price z"
    ]):
        return "materias_primas"

    if any(k in q for k in ["correlacion", "correlación", "heatmap", "matriz de correlacion", "matriz de correlación"]):
        return "correlacion"

    if any(k in q for k in [
        "analiza", "analisis", "análisis",
        "conclusion", "conclusión",
        "interpreta", "interpretacion", "interpretación",
        "grafica", "gráfica", "graficas", "gráficas",
        "grafico", "gráfico", "graficos", "gráficos",
        "pronost", "forecast", "proyecc",
        "equipo 1", "equipo 2"
    ]):
        return "forecast"

    return "metodologia"

## Materias Primas
def plot_commodities_forecast(artifacts: TrainingArtifacts, steps: int = 30) -> plt.Figure:
    fcst = forecast_targets(artifacts, steps=steps)["x_z_forecast"]
    df = artifacts.df_raw.copy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(df["Date"].iloc[-60:], df["Price_X"].iloc[-60:], label="Real X", linewidth=2)
    axes[0].plot(fcst["Date"], fcst["Price_X"], linestyle="--", label="Forecast X", linewidth=2)
    axes[0].fill_between(fcst["Date"], fcst["X_lower"], fcst["X_upper"], alpha=0.2, label="Intervalo X")
    axes[0].axvline(x=df["Date"].max(), linestyle=":", label="Inicio forecast")
    axes[0].set_title(f"Price X: pronóstico a {steps} días")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(df["Date"].iloc[-60:], df["Price_Z"].iloc[-60:], label="Real Z", linewidth=2)
    axes[1].plot(fcst["Date"], fcst["Price_Z"], linestyle="--", label="Forecast Z", linewidth=2)
    axes[1].fill_between(fcst["Date"], fcst["Z_lower"], fcst["Z_upper"], alpha=0.2, label="Intervalo Z")
    axes[1].axvline(x=df["Date"].max(), linestyle=":", label="Inicio forecast")
    axes[1].set_title(f"Price Z: pronóstico a {steps} días")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    return fig
### Graficas Shaps
def plot_shap_bar(shap_df: pd.DataFrame, title: str) -> plt.Figure:
    top = shap_df.head(10).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top["Feature"], top["Importance"])
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return fig

##Correlacion
def plot_correlation_heatmap(corr_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Matriz de correlación")
    plt.tight_layout()
    return fig

### News

def get_market_news(query: str, page_size: int = 5) -> List[Dict[str, str]]:
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": api_key,
    }

    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        return [
            {
                "title": a.get("title", ""),
                "source": a.get("source", {}).get("name", ""),
                "publishedAt": a.get("publishedAt", ""),
                "description": a.get("description", ""),
                "url": a.get("url", ""),
            }
            for a in data.get("articles", [])[:page_size]
        ]
    except Exception:
        return []
#### News Colombia
def get_colombia_news(last_date, lookback_days=30):

    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return []

    from_date = (last_date - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    to_date = last_date.strftime("%Y-%m-%d")

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": "(Colombia OR colombiano) AND (inflacion OR tasas OR construccion OR energia OR commodities)",
        "from": from_date,
        "to": to_date,
        "sortBy": "publishedAt",
        "language": "es",
        "pageSize": 5,
        "apiKey": api_key,
    }

    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        return [
            {
                "title": a.get("title"),
                "source": a.get("source", {}).get("name"),
                "date": a.get("publishedAt"),
                "desc": a.get("description"),
            }
            for a in data.get("articles", [])
        ]

    except Exception:
        return []



##Macro
def get_fred_series(series_id: str, limit: int = 5) -> List[Dict[str, str]]:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        return []

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
    }

    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        return data.get("observations", [])
    except Exception:
        return []    

###Macro Colombia

def get_colombia_macro():

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        return {}

    url = "https://api.stlouisfed.org/fred/series/observations"

    def fetch(series_id):
        try:
            r = requests.get(url, params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 12
            }, timeout=20)

            return r.json().get("observations", [])
        except:
            return []

    return {
        "inflacion": fetch("FPCPITOTLZGCOL"),
        "cpi": fetch("COLCPALTT01IXNBM"),
    }


## Contexto Externo

def build_external_context(artifacts, steps):

    last_date = artifacts.df_raw["Date"].max()

    news = get_colombia_news(last_date)
    macro = get_colombia_macro()

    return {
        "fecha_corte": str(last_date),
        "horizonte_dias": steps,
        "noticias_colombia": news,
        "macro_colombia": macro
    }
# ============================================================
# AGENT RESPONSE
# ============================================================
def agent_answer(question: str, artifacts: TrainingArtifacts) -> Tuple[str, Optional[pd.DataFrame], Optional[plt.Figure], Optional[plt.Figure]]:
    steps = extract_horizon(question)
    catalog = build_answer_catalog(artifacts, steps=steps)
    key = route_question(question)
    base_answer = catalog[key]
    analysis_text = ""

    table = None
    fig1 = None
    fig2 = None

    if key == "forecast":
        outputs = forecast_targets(artifacts, steps=steps)
        table = outputs["targets_forecast"][
            ["Date", "Price_Equipo1_pred", "Equipo1_lower", "Equipo1_upper",
            "Price_Equipo2_pred", "Equipo2_lower", "Equipo2_upper"]
        ].copy()

        fig1 = plot_target_forecast(artifacts, steps=steps, target="Equipo1")
        fig2 = plot_target_forecast(artifacts, steps=steps, target="Equipo2")

        analysis_text = interpret_results(question, artifacts, outputs)

        base_answer = (
            f"{base_answer}\n\n"
            f"{analysis_text}\n\n"
            "Se generó el forecast de ambos equipos para el horizonte solicitado, incluyendo escenario central y bandas lower/upper."
        )

    elif key == "kpis_arima":
        outputs = forecast_targets(artifacts, steps=steps)
        table = outputs["x_z_forecast"][["Date", "Price_X", "X_lower", "X_upper", "Price_Z", "Z_lower", "Z_upper"]].copy()
        fig1 = plot_xz_forecast(artifacts, steps=steps, commodity="X")
        fig2 = plot_xz_forecast(artifacts, steps=steps, commodity="Z")

        analysis_text = interpret_results(question, artifacts, outputs)

        base_answer = (
            f"{base_answer}\n\n"
            f"{analysis_text}\n\n"
            "Se incluyen las trayectorias proyectadas de X y Z con sus intervalos de confianza."
        )

    elif key == "shap":
        table = pd.concat(
            [
                artifacts.shap_model1.head(10).assign(Modelo="Equipo 1"),
                artifacts.shap_model2.head(10).assign(Modelo="Equipo 2"),
            ],
            ignore_index=True,
        )

        fig1 = plot_shap_bar(artifacts.shap_model1, "Top SHAP - Equipo 1")
        fig2 = plot_shap_bar(artifacts.shap_model2, "Top SHAP - Equipo 2")

        analysis_text = interpret_results(question, artifacts)

        base_answer = (
            f"{base_answer}\n\n"
            f"{analysis_text}\n\n"
            "Además de la tabla, se muestran los gráficos de importancia SHAP para ambos modelos."
        )

    elif key == "xgb":
        table = pd.concat(
            [
                artifacts.xgb_importance_model1.head(10).assign(Modelo="Equipo 1"),
                artifacts.xgb_importance_model2.head(10).assign(Modelo="Equipo 2"),
            ],
            ignore_index=True,
        )

        analysis_text = interpret_results(question, artifacts)

        base_answer = (
            f"{base_answer}\n\n"
            f"{analysis_text}\n\n"
            "La tabla resume las importancias internas de XGBoost para ambos modelos."
        )

    elif key == "materias_primas":
        outputs = forecast_targets(artifacts, steps=steps)
        table = outputs["x_z_forecast"][["Date", "Price_X", "X_lower", "X_upper", "Price_Z", "Z_lower", "Z_upper"]].copy()
        fig1 = plot_commodities_forecast(artifacts, steps=steps)
        fig2 = None

        analysis_text = interpret_results(question, artifacts, outputs)

        base_answer = (
            f"{base_answer}\n\n"
            f"{analysis_text}\n\n"
            f"Se usó SARIMAX con order={SARIMA_ORDER} y seasonal_order={SARIMA_SEASONAL_ORDER}. "
            "El componente (1,1,1) captura autoregresión, diferenciación y media móvil; el término estacional (1,1,1,12) permite capturar un patrón repetitivo de periodicidad 12. "
            "Abajo se muestran las trayectorias proyectadas de X y Z con sus escenarios upper y lower."
            f"\n\nResumen SARIMAX X:\n{artifacts.sarimax_x_summary[:2000]}\n\nResumen SARIMAX Z:\n{artifacts.sarimax_z_summary[:2000]}"
        )

    elif key == "correlacion":
        table = artifacts.top_corr.copy()
        fig1 = plot_correlation_heatmap(artifacts.top_corr)
        fig2 = None

        base_answer = (
            f"{base_answer}\n\n"
            "Se muestra la matriz de correlación en tabla y heatmap para facilitar la interpretación visual de las relaciones entre X, Y, Z y los equipos."
        )

    elif key == "eda":
        table = artifacts.top_corr.copy()
        base_answer += "\n\nLa matriz de correlación resumida se muestra en la tabla."

    answer = llm_analyze(
        question=question,
        artifacts=artifacts,
        key=key,
        base_answer=base_answer,
        table=table,
    )
    return answer, table, fig1, fig2


# =========================
# GRADIO UI
# =========================
ARTIFACTS: Optional[TrainingArtifacts] = None

def responder(pregunta, history):
    global ART

    try:
        if ART is None:
            ART = build_training_artifacts()

        answer, table, fig1, fig2 = agent_answer(pregunta, ART)

        if history is None:
            history = []

        history = history + [
            {"role": "user", "content": pregunta},
            {"role": "assistant", "content": answer}
        ]

        if table is None:
            table = pd.DataFrame()

        return history, table, fig1, fig2, ""

    except Exception as e:
        if history is None:
            history = []

        history = history + [
            {"role": "user", "content": pregunta},
            {"role": "assistant", "content": f"Error: {str(e)}"}
        ]

        return history, pd.DataFrame(), None, None, ""


with gr.Blocks() as demo:

    gr.Markdown("# 🤖 Agente de Pronóstico")

    chatbot = gr.Chatbot()
    txt = gr.Textbox(label="Pregunta")

    table = gr.Dataframe()
    fig1 = gr.Plot()
    fig2 = gr.Plot()

    txt.submit(responder, [txt, chatbot], [chatbot, table, fig1, fig2, txt])

demo.launch()