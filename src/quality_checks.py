from __future__ import annotations
import pandas as pd

# Límites físicos/operativos aproximados por variable
PHYSICAL_LIMITS = {
    'temp': (-50.0, 60.0),   # °C
    'hum': (0.0, 100.0),     # % de humedad relativa
    'wind': (0.0, 75.0),     # m/s aprox (tormentas muy fuertes)
    'rain': (0.0, 500.0),    # mm/h (valor extremo como placeholder)
}

def flag_physical_limits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea columnas flag_phys_* con True/False si el valor está fuera
    de los rangos definidos. No modifica los datos, solo etiqueta.
    """
    df = df.copy()
    for col, (lo, hi) in PHYSICAL_LIMITS.items():
        if col in df.columns:
            df[f'flag_phys_{col}'] = ~df[col].between(lo, hi)
    return df

def flag_sensor_drift(df: pd.DataFrame, col: str, window: int = 24, drift_threshold: float = 2.0) -> pd.DataFrame:
    """
    Marca drift cuando el cambio absoluto en la media móvil de tamaño 'window'
    supera el 'drift_threshold'. Ideal para series horarias.
    """
    if col not in df.columns or 'timestamp' not in df.columns:
        return df

    df = df.sort_values('timestamp').copy()
    roll_mean = df[col].rolling(window=window, min_periods=max(3, window//3)).mean()
    df[f'flag_drift_{col}'] = (roll_mean.diff().abs() > drift_threshold).astype(int)
    return df



import numpy as np
from typing import Dict, Optional, Tuple

# ---------------- Config por defecto  ----------------
DEFAULT_QC_CFG: Dict = {
    "time_col": "timestamp",
    "station_col": None,  # ej. "estacion"
    "freq": "H",
    "vars": {
        "t2m":   {"min": -40.0, "max": 55.0},
        "rh":    {"min": 0.0,   "max": 100.0},
        "td":    {"min": -60.0, "max": 40.0},
        "psl":   {"min": 850.0, "max": 1100.0},
        "ws":    {"min": 0.0,   "max": 75.0},
        "wd":    {"min": 0.0,   "max": 360.0},
        "precip":{"min": 0.0,   "max": 300.0}  # mm por intervalo
    },
    "precip": {
        "rate_mm_per_hour_max": 60.0,   # pico máximo razonable
        "is_cumulative": False          # True si el sensor reporta acumulado
    },
    "pressure": {
        "max_step_hpa_per_hour": 6.0
    },
    "t_rh_td": {
        "magnus_a": 17.625,
        "magnus_b": 243.04,   # °C
        "tol_td_deg": 2.0,    # incoherencia T vs Td
        "tol_rh_pct": 5.0
    },
    "wind": {
        "calm_ws_threshold": 0.5,      # m/s
        "max_wd_jump_deg": 90.0        # salto por paso (se analiza circularmente, informativo)
    }
}

# ---------------- Utilidades ----------------
def _to_datetime(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_datetime(df[col], errors="coerce", utc=True)
    return s.dt.tz_convert(None)

def _clip_range(s: pd.Series, vmin: Optional[float], vmax: Optional[float]) -> pd.Series:
    if vmin is not None and vmax is not None:
        return (s < vmin) | (s > vmax)
    if vmin is not None:
        return s < vmin
    if vmax is not None:
        return s > vmax
    return pd.Series(False, index=s.index)

def _circular_diff_deg(a: pd.Series, b: pd.Series) -> pd.Series:
    # diferencia circular mínima en grados
    d = (a - b + 180.0) % 360.0 - 180.0
    return d.abs()

# Magnus (T, RH) -> Td  y  (T, Td) -> RH; T y Td en °C, RH en %
def dewpoint_from_t_rh(t_c: pd.Series, rh_pct: pd.Series, a=17.625, b=243.04) -> pd.Series:
    rh = rh_pct.clip(lower=1e-6, upper=100) / 100.0
    gamma = np.log(rh) + (a * t_c) / (b + t_c)
    td = (b * gamma) / (a - gamma)
    return td

def rh_from_t_td(t_c: pd.Series, td_c: pd.Series, a=17.625, b=243.04) -> pd.Series:
    gamma_td = (a * td_c) / (b + td_c)
    gamma_t  = (a * t_c) / (b + t_c)
    rh = np.exp(gamma_td - gamma_t) * 100.0
    return rh

# ---------------- QC principal ----------------
def apply_meteorological_qc(df_in: pd.DataFrame, cfg: Dict = DEFAULT_QC_CFG) -> pd.DataFrame:
    df = df_in.copy()

    tcol = cfg["time_col"]
    if tcol not in df.columns:
        raise ValueError(f"Falta columna temporal '{tcol}'")
    df[tcol] = _to_datetime(df, tcol)

    # Asegurar columnas esperadas, convertir numéricas
    for v, r in cfg["vars"].items():
        if v in df.columns:
            df[v] = pd.to_numeric(df[v], errors="coerce")

    # --- Flags por variable: RANGE ---
    for v, r in cfg["vars"].items():
        if v in df.columns:
            rng_flag = _clip_range(df[v], r.get("min"), r.get("max"))
            df[f"{v}_qc_flag"] = np.where(rng_flag, "RANGE", "")

    # --- Coherencia T–RH–Td ---
    if "t2m" in df.columns and ("rh" in df.columns or "td" in df.columns):
        a = cfg["t_rh_td"]["magnus_a"]; b = cfg["t_rh_td"]["magnus_b"]
        tol_td = cfg["t_rh_td"]["tol_td_deg"]; tol_rh = cfg["t_rh_td"]["tol_rh_pct"]

        if "rh" in df.columns:
            td_est = dewpoint_from_t_rh(df["t2m"], df["rh"], a=a, b=b)
            df["td_est"] = td_est
            if "td" in df.columns:
                incoh_td = (df["td"] - td_est).abs() > tol_td
                _merge_flag(df, "td", incoh_td, "PHYSICS_T_RH_TD")
        if "td" in df.columns:
            rh_est = rh_from_t_td(df["t2m"], df["td"], a=a, b=b)
            df["rh_est"] = rh_est
            if "rh" in df.columns:
                incoh_rh = (df["rh"] - rh_est).abs() > tol_rh
                _merge_flag(df, "rh", incoh_rh, "PHYSICS_T_TD_RH")

    # --- Precipitación: no negativa + picos por intervalo + acumulado ---
    if "precip" in df.columns:
        neg = df["precip"] < 0
        _merge_flag(df, "precip", neg, "NEGATIVE")

        # tasa equivalente a mm/h según delta t observado
        dt = df[tcol].diff().dt.total_seconds().fillna(0).replace(0, np.nan)
        rate_h = df["precip"] / (dt / 3600.0)
        spike = rate_h > cfg["precip"]["rate_mm_per_hour_max"]
        _merge_flag(df, "precip", spike, "SPIKE_RATE")

        if cfg["precip"]["is_cumulative"]:
            decr = df["precip"].diff() < -1e-6
            _merge_flag(df, "precip", decr, "NON_MONOTONIC")

    # --- Viento: dirección con calma, saltos circulares grandes (informativo) ---
    if "ws" in df.columns and "wd" in df.columns:
        calm = df["ws"].fillna(0) < cfg["wind"]["calm_ws_threshold"]
        _merge_flag(df, "wd", calm, "DIR_UNCERTAIN_CALM")

        wd_jump = _circular_diff_deg(df["wd"], df["wd"].shift(1))
        big_jump = wd_jump > cfg["wind"]["max_wd_jump_deg"]
        # solo marcamos si la velocidad no es alta (turbulencia fuerte puede justificar)
        _merge_flag(df, "wd", big_jump & (df["ws"] < 3.0), "DIR_LARGE_JUMP")

    # --- Presión: saltos inverosímiles ---
    if "psl" in df.columns:
        # delta por hora (normalizamos por delta t real)
        dt_h = df[tcol].diff().dt.total_seconds().div(3600.0).replace(0, np.nan)
        dpsl_per_h = (df["psl"] - df["psl"].shift(1)).abs().div(dt_h)
        big_step = dpsl_per_h > cfg["pressure"]["max_step_hpa_per_hour"]
        _merge_flag(df, "psl", big_step, "PRESSURE_STEP")

    # --- Ensamblar razones globales por fila ---
    qc_cols = [c for c in df.columns if c.endswith("_qc_flag")]
    df["qc_reasons"] = (
        df[qc_cols]
        .apply(lambda row: ";".join(sorted({p for p in row.astype(str).tolist() if p and p != ""})), axis=1)
        .replace("", np.nan)
    )

    return df

def _merge_flag(df: pd.DataFrame, var: str, mask: pd.Series, label: str) -> None:
    col = f"{var}_qc_flag"
    if col not in df.columns:
        df[col] = ""
    df.loc[mask.fillna(False), col] = (
        df.loc[mask.fillna(False), col]
          .astype(str)
          .apply(lambda s: f"{s};{label}" if (s and s != "") else label)
    )

# ---------------- Resumen de métricas QC ----------------
def qc_summary(df: pd.DataFrame) -> pd.DataFrame:
    qc_cols = [c for c in df.columns if c.endswith("_qc_flag")]
    out = []
    n = len(df)

    for col in qc_cols:
        flags = (
            df[col]
            .dropna()
            .astype(str)
            .str.split(";")
            .explode()
            .str.strip()
        )
        flags = flags[flags != ""]
        total_flagged = (df[col].fillna("").astype(bool)).sum()
        out.append({
            "variable": col.replace("_qc_flag", ""),
            "rows_flagged": int(total_flagged),
            "pct_flagged": round(100 * total_flagged / max(n, 1), 2),
            "distinct_reasons": ",".join(sorted(flags.unique())) if not flags.empty else ""
        })

    # 🔧 caso sin flags/columnas: devolver DF vacío con columnas esperadas
    if not out:
        return pd.DataFrame(columns=["variable", "rows_flagged", "pct_flagged", "distinct_reasons"])

    return pd.DataFrame(out).sort_values("pct_flagged", ascending=False).reset_index(drop=True)
