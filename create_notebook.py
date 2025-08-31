# create_notebook.py
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

# Celda 1 — Markdown (título y resumen)
cells.append(nbf.v4.new_markdown_cell("""# AEMET Data Quality — Proyecto de Calidad de Datos Meteorológicos

Este proyecto demuestra un flujo de trabajo de **calidad de datos** aplicable a estaciones meteorológicas, usando Python.

Incluye:
- Limpieza y tratamiento de valores nulos
- Detección de *outliers* estadísticos (*z-score*)
- Validación contra límites físicos
- Cálculo de KPIs clave
- Detección de *sensor drift* con media móvil
- Visualización de series temporales

> Datos simulados a partir de un CSV de ejemplo.
"""))

# Celda 2 — Código (imports y carga)
cells.append(nbf.v4.new_code_cell("""import pandas as pd
from src.data_utils import (
    load_sample_csv, remove_duplicates, handle_missing,
    detect_outliers_zscore, compute_kpis
)
from src.quality_checks import flag_physical_limits, flag_sensor_drift
from src.plotting import plot_timeseries

df = load_sample_csv('data/raw/sample_measurements.csv')
df.head()
"""))

# Celda 3 — Código (limpieza y flags)
cells.append(nbf.v4.new_code_cell("""# Eliminar duplicados y tratar valores nulos mediante interpolación
df = remove_duplicates(df)
df = handle_missing(df, method='interpolate')

# Outliers estadísticos (z-score)
df = detect_outliers_zscore(df, cols=['temp','hum','wind','rain'], z=3.0)

# Validación contra límites físicos
df = flag_physical_limits(df)

# Sensor drift con media móvil (ventana corta para el ejemplo)
df = flag_sensor_drift(df, col='temp', window=3, drift_threshold=2.0)

# Ver columnas de banderas
df.filter(like='flag_').head()
"""))

# Celda 4 — Código (KPIs)
cells.append(nbf.v4.new_code_cell("""kpis = compute_kpis(df, ['temp','hum','wind','rain'])
import pandas as pd
pd.DataFrame([kpis])
"""))

# Celda 5 — Código (conteo de banderas)
cells.append(nbf.v4.new_code_cell("""phys_cols = [c for c in df.columns if c.startswith('flag_phys_')]
drift_cols = [c for c in df.columns if c.startswith('flag_drift_')]
{
    "flags_fisicos": {c: int(df[c].sum()) for c in phys_cols},
    "flags_drift":   {c: int(df[c].sum()) for c in drift_cols}
}
"""))

# Celda 6 — Código (gráfica)
cells.append(nbf.v4.new_code_cell("""plot_timeseries(df, 'temp', title='Temperatura – estación CBA001')"""))

# Celda 7 — Código (guardar procesado)
cells.append(nbf.v4.new_code_cell("""df.to_csv('data/processed/sample_measurements_clean.csv', index=False)
len(df), df.tail(3)
"""))

nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"}
}

out = Path("notebooks") / "aemet_quality.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print(f"Notebook creado en: {out.resolve()}")
