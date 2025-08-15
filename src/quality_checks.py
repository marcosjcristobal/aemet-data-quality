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
