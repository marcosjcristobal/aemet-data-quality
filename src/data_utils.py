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
    df = df.copy()

    if method == 'drop':
        return df.dropna()

    if method in ('ffill', 'bfill'):
        return df.fillna(method=method)

    if method == 'interpolate':
        # 1) Inferir tipos en todo el DF (convierte 'object' a numérico/datetime cuando pueda)
        df = df.infer_objects(copy=False)

        # 2) Asegurar que las columnas numéricas lo sean (convierte texto a NaN si hace falta)
        candidates = [c for c in ['temp', 'hum', 'wind', 'rain'] if c in df.columns]
        for c in candidates:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # 3) Interpolar **solo** las columnas numéricas para evitar el warning
        numeric_cols = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            df[numeric_cols] = df[numeric_cols].interpolate(limit_direction='both')

        return df

    # Si llega un método desconocido, devolvemos tal cual
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
    """Devuelve % de no-nulos y racha máxima de nulos por columna."""
    kpis = {}
    n = len(df)

    def max_nan_run(series: pd.Series) -> int:
        """Devuelve la racha más larga de NaN consecutivos."""
        max_run = run = 0
        for v in series.isna():
            run = run + 1 if v else 0
            max_run = max(max_run, run)
        return max_run

    for c in value_cols:
        valid = df[c].notna().sum() if c in df.columns else 0
        kpis[f'valid_pct_{c}'] = round(100.0 * valid / n, 2) if n else 0.0
        kpis[f'max_nan_run_{c}'] = max_nan_run(df[c]) if c in df.columns else None

    return kpis

