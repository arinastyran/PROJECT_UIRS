import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct

INPUT_FILE = r"C:\UIRS\surface-classification\dct2_ready_data.csv"
df = pd.read_csv(INPUT_FILE)

SURFACE_COLORS = {
    'artificial_grass':     '#1f77b4',
    'ceramic_tiles':        '#ff7f0e',
    'eva_foam_tiles':       '#2ca02c',
    'foam_underlayment':    '#d62728',
    'laminate_flooring':    '#9467bd',
    'linoleum':             '#8c564b',
    'long_carpet':          '#e377c2',
    'osb':                  '#7f7f7f',
    'pvc_foamboard':        '#bcbd22',
    'short_carpet':         '#17becf',
}

surfaces = sorted(df['surface'].unique())

# Класс DCT

class DCT:
    
    def __init__(self, data, cutoff_amount=None, range_min=None, range_max=None):
        """
        :param data: дискретный набор амплитуд сигнала для преобразования
        :param cutoff_amount: количество наибольших коэффициентов DCT для сохранения. 
                              Все остальные обнуляются. Если None — сохраняются все.
        :param range_min: минимальная X-координата данных
        :param range_max: максимальная X-координата данных
        """
        self.range_min = range_min
        self.range_max = range_max
        self.N = len(data)
        # Вычисляем коэффициенты DCT 
        self.coefficients = dct(data, type=2)
        # Фильтруем коэффициенты
        if cutoff_amount is not None:
            # Сортируем индексы по убыванию абсолютного значения коэффициентов
            indices = sorted(range(len(self.coefficients)), 
                           key=lambda i: abs(self.coefficients[i]), 
                           reverse=True)
            # Обнуляем все, кроме top-N
            for idx in indices[cutoff_amount:]:
                self.coefficients[idx] = 0
    
    def numpy_func(self, x, scaled=False):
        """
        Вычисляет аналитическую сумму косинусов в заданных точках
        
        :param x: точка(и) для вычисления. Может быть числом или массивом
        :param scaled: если True, масштабирует X-координаты в пространство индексов DCT
        :return: Y-значение(я) в точке x
        """
        x = np.atleast_1d(x).astype(float)
        
        if scaled:
            # Масштабируем x
            x = (x - self.range_min) / (self.range_max - self.range_min) * (self.N - 1)
        # Формула обратного DCT
        result = self.coefficients[0] / 2
        for n in range(1, self.N):
            result += self.coefficients[n] * np.cos((n / self.N) * np.pi * (x + 0.5))
        result *= (1 / self.N)
        return result if len(result) > 1 else result[0]
    def __call__(self, x, scaled=False):
        return self.numpy_func(x, scaled)


dct_models = {}          # Словарь моделей DCT
dct_models_std = {}      # Средний std для каждой поверхности (для классификатора)

for surface in surfaces:
    surf_data = df[df['surface'] == surface].sort_values('omega_bin')
    
    omega = surf_data['omega_center'].values
    median_ke = surf_data['median_Ke'].values
    std_ke = surf_data['std_Ke'].fillna(0).values
    color = SURFACE_COLORS[surface]
    label = surface.replace('_', ' ')
    
    dct_model = DCT(
        data=median_ke,
        cutoff_amount=3,  # Оставляем топ-3 коэффициента по модулю
        range_min=omega.min(),
        range_max=omega.max()
    )
    
    # Вычисляем модель на плотной сетке для гладкой линии
    x_smooth = np.linspace(omega.min(), omega.max(), 356)
    model_ke_smooth = dct_model.numpy_func(x_smooth, scaled=True)
    
    # Сохраняем модель и средний std
    dct_models[surface] = dct_model
    dct_models_std[surface] = np.nanmean(std_ke)  # Средний std по всем бинам
    
    print(f"{label}: топ-3 коэффициента = {sorted(range(len(dct_model.coefficients)), key=lambda i: abs(dct_model.coefficients[i]), reverse=True)[:3]}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Область стандартного отклонения
    ax.fill_between(omega, median_ke - std_ke, median_ke + std_ke, 
                    color=color, alpha=0.2, label='±1 std')
    
    # Медианные значения
    ax.plot(omega, median_ke, 'o--', color=color, linewidth=1.5, markersize=4, 
            label='Медиана Ke', zorder=3)
    
    ax.plot(x_smooth, model_ke_smooth, '-', color='red', linewidth=2, 
            label='DCT модель (top-3 коэф.)', zorder=4)
    
    ax.set_title(f'Поверхность: {label}', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Угловая скорость робота, рад/с ($\omega$)', fontsize=12)
    ax.set_ylabel('Коэффициент энергопотребления $K_e$', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    filename = f"C:\\UIRS\\surface-classification\\dct2_model_{surface}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Сохранен: dct2_model_{surface}.png")

import pickle

with open(r"C:\UIRS\surface-classification\dct_models.pkl", 'wb') as file:
    pickle.dump((dct_models, dct_models_std), file)

print(f"\nМодели сохранены: dct_models.pkl")
print(f"  - {len(dct_models)} моделей DCT")
print(f"  - Средние std для каждой поверхности")