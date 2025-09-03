from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

# Para que funcione el import cuando ejecutas directamente la página
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from src.quality_checks import apply_meteorological_qc, qc_summary, DEFAULT_QC_CFG
except Exception as e:
    st.set_page_config(page_title="QC físico", layout="wide")
    st.error("No puedo importar src.quality_checks. Asegúrate de que 'src/quality_checks.py' existe.\n\n" + str(e))
    st.stop()

st.set_page_config(page_title="QC físico — AEMET", layout="wide")
st.title("🧪 QC físico — AEMET")

# --- Sidebar de configuración básica ---
cfg = DEFAULT_QC_CFG.copy()
cfg["time_col"] = st.sidebar.text_input("Columna temporal", value=cfg["time_col"])
station_col_input = st.sidebar.text_input("Columna estación (opcional)", value=cfg["station_col"] or "")
cfg["station_col"] = station_col_input or None

st.sidebar.subheader("Umbrales clave")
cfg["precip"]["rate_mm_per_hour_max"] = st.sidebar.number_input(
    "Precip máx (mm/h)", value=float(cfg["precip"]["rate_mm_per_hour_max"]), step=1.0
)
cfg["precip"]["is_cumulative"] = st.sidebar.checkbox(
    "Precip acumulada", value=cfg["precip"]["is_cumulative"]
)
cfg["pressure"]["max_step_hpa_per_hour"] = st.sidebar.number_input(
    "Salto presión máx (hPa/h)", value=float(cfg["pressure"]["max_step_hpa_per_hour"]), step=0.5
)
cfg["wind"]["calm_ws_threshold"] = st.sidebar.number_input(
    "Umbral calma viento (m/s)", value=float(cfg["wind"]["calm_ws_threshold"]), step=0.1
)
cfg["wind"]["max_wd_jump_deg"] = st.sidebar.number_input(
    "Salto dirección máx (°)", value=float(cfg["wind"]["max_wd_jump_deg"]), min_value=0.0, max_value=180.0, step=5.0
)

# --- 1) Cargar CSV ---
st.subheader("1) Cargar CSV")
file = st.file_uploader("Sube un CSV", type=["csv"])
use_sample = st.checkbox("Usar ejemplo del repo (data/raw/sample_measurements.csv)", value=not file)

if use_sample:
    csv_path = ROOT / "data" / "raw" / "sample_measurements.csv"
    if not csv_path.exists():
        st.error(f"No encuentro el archivo de ejemplo: {csv_path}")
        st.stop()
    df_raw = pd.read_csv(csv_path)
    st.caption(f"Ejemplo cargado: {csv_path}")
else:
    if not file:
        st.info("Sube un CSV o marca la casilla de ejemplo.")
        st.stop()
    df_raw = pd.read_csv(file)

st.dataframe(df_raw.head(), use_container_width=True)

# --- 2) Ejecutar QC ---
st.subheader("2) Ejecutar QC")
if st.button("Aplicar QC"):
    try:
        df_qc = apply_meteorological_qc(df_raw, cfg)
        summary = qc_summary(df_qc)
    except Exception as e:
        st.error(f"Error aplicando QC: {e}")
        st.stop()

    st.success("QC aplicado ✓")
    st.markdown("### Resumen de flags por variable")
    st.dataframe(summary, use_container_width=True)
