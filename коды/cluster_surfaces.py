import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

df = pd.read_csv(r"C:\UIRS\surface-classification\dct2_ready_data.csv")
pivot_ke = df.pivot(index='omega_center', columns='surface', values='median_Ke')
pivot_ke = pivot_ke.interpolate(method='linear').fillna(0)

# Транспонируем, чтобы кластеризовать поверхности (строки), а не бины (столбцы)
X = pivot_ke.T.values
surfaces = pivot_ke.columns.tolist()

# Кластеризация
linked = linkage(X, method='ward')

plt.figure(figsize=(14, 8))
dendrogram(
    linked,
    orientation='top',
    labels=surfaces,
    distance_sort='descending',
    show_leaf_counts=True,
    leaf_font_size=12
)
plt.title('Схожесть поверхностей по профилю Ke\n', fontsize=14)
plt.ylabel('Расстояние (мера непохожести)', fontsize=12)
plt.xlabel('Поверхности', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\UIRS\surface-classification\surface_dendrogram.png", dpi=300)
print("Дендрограмма сохранена: surface_dendrogram.png")