import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\UIRS\surface-classification\dct2_ready_data.csv")

# строки = omega_center, столбцы = surface, значения = median_Ke
pivot_ke = df.pivot(index='omega_center', columns='surface', values='median_Ke')

# Заполняем пропуски (если в каком-то бине нет данных) методом интерполяции или нулем
pivot_ke = pivot_ke.interpolate(method='linear').fillna(0)
 
# Считаем матрицу корреляции Пирсона между поверхностями
# Если корреляция близка к 1.0, поверхности ведут себя одинаково
correlation_matrix = pivot_ke.corr()


plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", 
            square=True, cbar_kws={'label': 'Корреляция профилей Ke'})
plt.title('Схожесть профилей энергопотребления (Ke) между поверхностями\n')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(r"C:\UIRS\surface-classification\ke_correlation_heatmap.png", dpi=300)
print("Тепловая карта сохранена: ke_correlation_heatmap.png")
