from __future__ import annotations
import pandas as pd

def load_sample_csv(path: str) -> pd.DataFrame:
    """Carga CSV y parsea la columna timestamp como fecha-hora."""
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return standardize_columns(df)

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Pone nombres de columnas en minúsculas_con_guiones y asegura timestamp."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas duplicadas exactas."""
    return df.drop_duplicates()

def handle_missing(df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
    """
    Gestiona nulos:
    - 'drop' -> elimina filas con NaN
    - 'ffill'/'bfill' -> rellena hacia delante/detrás
    - 'interpolate' -> interpola numéricos
    """
    df = df.copy()
    if method == 'drop':
        df = df.dropna()
    elif method in ('ffill', 'bfill'):
        df = df.fillna(method=method)
    elif method == 'interpolate':
        df = df.interpolate(limit_direction='both')
    return df

def detect_outliers_zscore(df: pd.DataFrame, cols: list[str], z: float = 3.0) -> pd.DataFrame:
    """Crea banderas flag_out_* si el valor se aleja más de z desviaciones estándar."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            mu = df[c].mean()
            sigma = df[c].std(ddof=0)
            if sigma and sigma > 0:
                zscores = (df[c] - mu) / sigma
                df[f'flag_out_{c}'] = (zscores.abs() > z).astype(int)
            else:
                df[f'flag_out_{c}'] = 0
    return df

def compute_kpis(df: pd.DataFrame, value_cols: list[str]) -> dict:
    """Devuelve % de no-nulos por columna (sencillo pero útil como KPI)."""
    kpis = {}
    n = len(df)
    for c in value_cols:
        valid = df[c].notna().sum() if c in df.columns else 0
        kpis[f'valid_pct_{c}'] = round(100.0 * valid / n, 2) if n else 0.0
    return kpis
