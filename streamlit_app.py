
import pathlib
import io
import pandas as pd
import numpy as np
import streamlit as st

try:
    from src.data_utils import (
        load_sample_csv, remove_duplicates, handle_missing,
        detect_outliers_zscore, compute_kpis
    )
except Exception:
    # --- Fallbacks mínimos por si el import falla ---
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df

    def load_sample_csv(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["timestamp"], dayfirst=False, infer_datetime_format=True)
        return _standardize_columns(df)

    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()

    def handle_missing(df: pd.DataFrame, method: str = "interpolate") -> pd.DataFrame:
        df = df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if method == "drop":
            return df.dropna()
        elif method == "ffill":
            df[num_cols] = df[num_cols].ffill()
        elif method == "bfill":
            df[num_cols] = df[num_cols].bfill()
        else:
            # interpolate por defecto
            df[num_cols] = df[num_cols].interpolate(limit_direction="both")
        return df

    def detect_outliers_zscore(df: pd.DataFrame, cols=None, z: float = 3.0) -> pd.DataFrame:
        df = df.copy()
        if cols is None:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for c in cols:
            if c in df.columns:
                s = df[c].astype(float)
                mu = s.mean()
                sd = s.std(ddof=0)
                if sd == 0 or np.isnan(sd):
                    df[f"flag_out_{c}"] = 0
                else:
                    zsc = (s - mu) / sd
                    df[f"flag_out_{c}"] = (zsc.abs() > z).astype(int)
        return df

    def _max_null_streak(series: pd.Series) -> int:
        isna = series.isna().astype(int)
        # cuenta rachas consecutivas de 1s
        max_run, cur = 0, 0
        for v in isna:
            cur = cur + 1 if v == 1 else 0
            if cur > max_run:
                max_run = cur
        return int(max_run)

    def compute_kpis(df: pd.DataFrame, value_cols) -> dict:
        kpis = {}
        for col in value_cols:
            n = len(df)
            phys_col = f"flag_phys_{col}"
            out_col  = f"flag_out_{col}"

            phys = df[phys_col] if phys_col in df.columns else 0
            out  = df[out_col]  if out_col  in df.columns else 0

            valid_mask = df[col].notna() & (phys == 0) & (out == 0)
            pct_valid = float(valid_mask.mean() * 100.0)

            kpis[col] = {
                "pct_valid": round(pct_valid, 2),
                "max_null_streak": _max_null_streak(df[col]) if n > 0 else 0,
                "n_rows": n,
            }
        return kpis

try:
    from src.quality_checks import flag_physical_limits, flag_sensor_drift
except Exception:
    PHYSICAL_LIMITS = {
        "temp": (-50.0, 60.0),
        "hum":  (0.0, 100.0),
        "wind": (0.0, 75.0),
        "rain": (0.0, 500.0),
    }
    def flag_physical_limits(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col,(lo,hi) in PHYSICAL_LIMITS.items():
            if col in df.columns:
                df[f"flag_phys_{col}"] = (~df[col].between(lo, hi)).astype(int)
        return df

    def flag_sensor_drift(df: pd.DataFrame, col: str, window: int = 24, drift_threshold: float = 2.0) -> pd.DataFrame:
        if col not in df.columns:
            return df
        df = df.copy()
        roll = df[col].rolling(window=window, min_periods=max(2, window//2)).mean()
        drift = roll.diff().abs() > drift_threshold
        df[f"flag_drift_{col}"] = drift.fillna(False).astype(int)
        return df

# ------------- UI -------------
st.set_page_config(page_title="AEMET Data Quality", layout="wide")
st.title("AEMET — Data Quality Dashboard")

root = pathlib.Path(__file__).resolve().parent
sample_path = root / "data" / "raw" / "sample_measurements.csv"

with st.sidebar:
    st.header("Opciones")
    data_src = st.radio("Fuente de datos", ["CSV de ejemplo", "Subir CSV"])
    uploaded = None
    if data_src == "Subir CSV":
        uploaded = st.file_uploader("CSV con columnas (timestamp, ...)", type=["csv"])

    miss_method = st.selectbox("Tratamiento de nulos", ["interpolate", "ffill", "bfill", "drop"], index=0)
    z_thr = st.slider("Umbral z-score (outliers)", min_value=2.0, max_value=5.0, value=3.0, step=0.1)
    drift_col_hint = st.text_input("Columna para drift (ej. temp)", value="temp")
    drift_win = st.number_input("Ventana drift (p.ej. 24)", min_value=2, value=24, step=1)
    drift_thr = st.number_input("Umbral drift", min_value=0.1, value=2.0, step=0.1)

# ------------- Carga -------------
@st.cache_data(show_spinner=False)
def _load_df(_uploaded_bytes):
    if _uploaded_bytes is not None:
        df = pd.read_csv(io.BytesIO(_uploaded_bytes))
        # intentamos parsear timestamp si existe
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        # estandariza nombres
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df
    else:
        return load_sample_csv(str(sample_path))

if data_src == "Subir CSV" and uploaded is not None:
    df = _load_df(uploaded.read())
else:
    df = _load_df(None)

if df.empty:
    st.warning("No hay datos para mostrar.")
    st.stop()

# ------------- Limpieza + Flags -------------
df = remove_duplicates(df)
df = handle_missing(df, method=miss_method)

# Selección de columnas numéricas (por defecto temp, hum, wind, rain si existen)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
default_cols = [c for c in ["temp","hum","wind","rain"] if c in df.columns] or num_cols
sel_cols = st.multiselect("Columnas a validar", options=num_cols, default=default_cols)

if sel_cols:
    df = detect_outliers_zscore(df, cols=sel_cols, z=z_thr)
    df = flag_physical_limits(df)
    if drift_col_hint in df.columns:
        df = flag_sensor_drift(df, col=drift_col_hint, window=int(drift_win), drift_threshold=float(drift_thr))

# ------------- KPIs -------------
# ------------- KPIs -------------
if sel_cols:
    kpis = compute_kpis(df, sel_cols)
    kpi_df = pd.DataFrame.from_dict(kpis, orient="index")
    st.subheader("KPIs de calidad (válidos)")
    st.dataframe(kpi_df)

    # ----Conteo de NO válidos (por columna y global) ----
    rows = []
    invalid_masks = []

    for col in sel_cols:
        phys_col = f"flag_phys_{col}"
        out_col  = f"flag_out_{col}"

        phys = df[phys_col] if phys_col in df.columns else 0
        out  = df[out_col]  if out_col  in df.columns else 0
        nulls = df[col].isna()

        col_invalid_mask = nulls | (phys == 1) | (out == 1)
        invalid_masks.append(col_invalid_mask)

        rows.append({
            "variable": col,
            "n_invalid": int(col_invalid_mask.sum()),
            "pct_invalid": round(100 * col_invalid_mask.mean(), 2),
            "n_null": int(nulls.sum()),
            "n_outliers": int((out == 1).sum()) if isinstance(out, pd.Series) else 0,
            "n_phys": int((phys == 1).sum()) if isinstance(phys, pd.Series) else 0,
        })

    per_col_invalid_df = pd.DataFrame(rows).set_index("variable")

    # Métrica global (unión de no-válidos en cualquiera de las columnas seleccionadas)
    if invalid_masks:
        overall_invalid_mask = np.logical_or.reduce(invalid_masks)
    else:
        overall_invalid_mask = np.zeros(len(df), dtype=bool)

    n_invalid_global = int(overall_invalid_mask.sum())
    pct_invalid_global = 100.0 * (n_invalid_global / len(df)) if len(df) else 0.0

    st.subheader("NO válidos")
    m1, m2 = st.columns(2)
    m1.metric("Registros NO válidos (global)", f"{n_invalid_global}", f"{pct_invalid_global:.1f}%")
    m2.write(" ")  # separador visual

    st.write("**Desglose por columna** (n_invalid, %, n_null, n_outliers, n_phys):")
    st.dataframe(per_col_invalid_df)

    with st.expander("🔎 Ver filas NO válidas (global)"):
        st.dataframe(df.loc[overall_invalid_mask].head(300))
        csv_inv = df.loc[overall_invalid_mask].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar NO válidos (CSV)", data=csv_inv, file_name="invalid_rows.csv", mime="text/csv")

    # ---- (Opcional) Recordatorio de límites físicos aplicados ----
    st.caption("Los NO válidos incluyen: nulos, outliers (flag_out_*) y valores fuera de límites físicos (flag_phys_*). "
               "Ej.: hum<0% o hum>100% ya quedan marcados como físicos.")


# ------------- Gráficas -------------
st.subheader("Series temporales")
if "timestamp" in df.columns and sel_cols:
    cols = st.multiselect("Columnas a graficar", options=sel_cols, default=sel_cols[:1])
    if cols:
        for c in cols:
            st.line_chart(df.sort_values("timestamp").set_index("timestamp")[c], height=220)
else:
    st.info("Añade/normaliza una columna 'timestamp' para ver series temporales.")

# ------------- Descarga -------------
st.subheader("Datos procesados")
st.dataframe(df.head(100))
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar CSV procesado", data=csv, file_name="processed.csv", mime="text/csv")

st.caption("Tip: los KPIs excluyen nulos, *outliers* (flag_out_*) y valores fuera de límites físicos (flag_phys_*).")
