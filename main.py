from src.data_utils import (
    load_sample_csv, remove_duplicates, handle_missing,
    detect_outliers_zscore, compute_kpis
)
from src.quality_checks import flag_physical_limits, flag_sensor_drift
from src.plotting import plot_timeseries

# 1) Cargar datos (CSV de ejemplo)
df = load_sample_csv('data/raw/sample_measurements.csv')

# 2) Limpieza básica
df = remove_duplicates(df)
df = handle_missing(df, method='interpolate')  # 'drop' | 'ffill' | 'bfill' | 'interpolate'

# 3) Flags de outliers (estadísticos) y de límites físicos
df = detect_outliers_zscore(df, cols=['temp','hum','wind','rain'], z=3.0)
df = flag_physical_limits(df)
df = flag_sensor_drift(df, col='temp', window=3, drift_threshold=2.0)
drift_cols = [c for c in df.columns if c.startswith('flag_drift_')]
print("Flags de drift totales:", {c: int(df[c].sum()) for c in drift_cols})

# 4) KPIs de calidad (porcentaje de válidos)
kpis = compute_kpis(df, ['temp','hum','wind','rain'])
print("KPIs de calidad:", kpis)

# 5) Conteo de flags físicos (para tener visibilidad rápida)
phys_cols = [c for c in df.columns if c.startswith('flag_phys_')]
print("Flags físicos totales:", {c: int(df[c].sum()) for c in phys_cols})

# 6) Guardar dataset procesado
df.to_csv('data/processed/sample_measurements_clean.csv', index=False)
print("Guardado: data/processed/sample_measurements_clean.csv")

# 7) Gráfico sencillo para visualizar la serie
plot_timeseries(df, 'temp', title='Temperatura – estación CBA001')
