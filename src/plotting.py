from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt

def plot_timeseries(df: pd.DataFrame, col: str, title: str = "") -> None:
    """Grafica una serie temporal simple de la columna dada."""
    if 'timestamp' not in df.columns or col not in df.columns:
        return
    df = df.sort_values('timestamp')
    plt.figure()
    plt.plot(df['timestamp'], df[col])
    plt.title(title or col)
    plt.xlabel('timestamp')
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()
