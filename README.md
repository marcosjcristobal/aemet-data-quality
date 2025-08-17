# AEMET Data Quality

Proyecto demostrativo de **calidad de datos meteorológicos** desarrollado en Python, con un enfoque reproducible y modular.  

Incluye ingestión de datos (CSV de ejemplo), limpieza de valores nulos, detección de *outliers* estadísticos (z-score), validación contra límites físicos y cálculo de KPIs clave.

------------------------------------------------------------------------------

## Objetivos

- Ingesta de datos desde CSV (simulación de datos meteorológicos de estación).
- Limpieza y tratamiento de valores nulos (interpolación, relleno, eliminación).
- Señalización de *outliers* estadísticos mediante z-score.
- Validación contra límites físicos por variable (temperatura, humedad, viento, lluvia).
- Cálculo de KPIs: % de valores válidos y racha máxima de nulos.
- Detección de *sensor drift* mediante media móvil.
- Visualización de series temporales.

------------------------------------------------------------------------------

## Estructura del proyecto

- **notebooks/**: cuadernos Jupyter con análisis, KPIs y gráficas.
- **src/**: código Python (utilidades, validaciones, visualizaciones).
- **data/raw** y **data/processed**: datos originales y procesados (no versionados).
- **sql/**: consultas SQL de calidad de datos.

------------------------------------------------------------------------------

## Resultados

- **KPIs**: % de válidos por variable y racha máxima de nulos.
- **Banderas**: outliers estadísticos (z-score), límites físicos y *sensor drift* (media móvil).
- **Visualización**: serie temporal de temperatura como ejemplo.

![KPIs](assets/kpis_example.png)
![Gráfica de temperatura](assets/plot_temp_example.png)

------------------------------------------------------------------------------

## Cómo reproducir


# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/aemet-data-quality.git
cd aemet-data-quality

# 2. Crear entorno virtual e instalar dependencias
python -m venv .venv

# Activar entorno en Windows
. .venv\Scripts\Activate.ps1

pip install -r requirements.txt

# 3. Ejecutar el script principal
python main.py


------------------------------------------------------------------------------

