import pickle
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from scipy.fft import dct

class DCT:
    def __init__(self, data, cutoff_amount=None, range_min=None, range_max=None):
        self.range_min = range_min
        self.range_max = range_max
        self.N = len(data)
        self.coefficients = dct(data, type=2)
        if cutoff_amount is not None:
            indices = sorted(range(len(self.coefficients)), 
                           key=lambda i: abs(self.coefficients[i]), 
                           reverse=True)
            for idx in indices[cutoff_amount:]:
                self.coefficients[idx] = 0

    def numpy_func(self, x, scaled=False):
        x = np.atleast_1d(x).astype(float)
        if scaled:
            x = (x - self.range_min) / (self.range_max - self.range_min) * (self.N - 1)
        result = self.coefficients[0] / 2
        for n in range(1, self.N):
            result += self.coefficients[n] * np.cos((n / self.N) * np.pi * (x + 0.5))
        result *= (1 / self.N)
        return result if len(result) > 1 else result[0]
    
    def __call__(self, x, scaled=False):
        return self.numpy_func(x, scaled)

# Загрузка моделей
MODELS_FILE = Path(r"C:\UIRS\surface-classification\dct_models.pkl")
print(f"Загрузка моделей из {MODELS_FILE}...")
with open(MODELS_FILE, 'rb') as f:
    dct_models, dct_models_std = pickle.load(f)

surfaces = sorted(list(dct_models.keys()))
print(f"Загружено {len(surfaces)} поверхностей.\n")

# Извлечение индексов ненулевых коэффициентов
# Используем порог 1e-9, чтобы исключить ошибки плавающей точки (float64)
nonzero_indices = {}
for surf in surfaces:
    coeffs = dct_models[surf].coefficients
    # Находим индексы, где абсолютное значение больше нуля
    idx = [i for i, c in enumerate(coeffs) if abs(c) > 1e-9]
    nonzero_indices[surf] = sorted(idx)
    
    # Вывод в консоль, как требуется в задании
    print(f"Поверхность: {surf:<25} | Ненулевые коэффициенты (по возрастанию): {nonzero_indices[surf]}")

print("АНАЛИЗ СОВПАДЕНИЙ ")

# Построение матрицы различий 
n_surfs = len(surfaces)
z_matrix = np.zeros((n_surfs, n_surfs), dtype=int)
hover_matrix = [["" for _ in range(n_surfs)] for _ in range(n_surfs)]

identical_pairs = []

for i in range(n_surfs):
    for j in range(n_surfs):
        set_i = set(nonzero_indices[surfaces[i]])
        set_j = set(nonzero_indices[surfaces[j]])
        
        # Симметричная разность (XOR): элементы, которые есть в одном, но не в обоих
        diff_set = set_i ^ set_j
        diff_count = len(diff_set)
        diff_list = sorted(list(diff_set))
        
        z_matrix[i, j] = diff_count
        
        # Формируем текст для всплывающей подсказки (hover)
        if i == j:
            hover_text = (
                f"<b>{surfaces[i]}</b><br>"
                f"Идентична самой себе<br>"
                f"Ненулевые коэф.: {nonzero_indices[surfaces[i]]}"
            )
        else:
            hover_text = (
                f"<b>Сравнение пары:</b><br>"
                f"1: {surfaces[i]}<br>"
                f"2: {surfaces[j]}<br>"
                f"<b>Кол-во отличающихся:</b> {diff_count}<br>"
                f"<b>Номера отличающихся:</b> {diff_list if diff_list else 'Нет (полностью совпадают)'}"
            )
        hover_matrix[i][j] = hover_text
        
        # Собираем пары с полным совпадением
        if i < j and diff_count == 0:
            identical_pairs.append((surfaces[i], surfaces[j]))

# Вывод результатов сравнения в консоль
if identical_pairs:
    print("\n Поверхности с полностью одинаковыми наборами ненулевых коэффициентов:")
    for p1, p2 in identical_pairs:
        print(f"   - {p1}  <-->  {p2}")
else:
    print("\nНет пар поверхностей с полностью одинаковыми наборами ненулевых коэффициентов")

# Визуализация
print("\nПостроение интерактивной матрицы ...")

fig = go.Figure(data=go.Heatmap(
    z=z_matrix,
    x=surfaces,
    y=surfaces,
    hovertext=hover_matrix,
    hoverinfo='text',      
    colorscale='Blues',        # Цветовая шкала (чем темнее, тем больше различий)
    colorbar=dict(
        title=dict(
            text="Кол-во<br>отличающихся<br>коэффициентов",
            side='right'       
        ),
        ticks='outside'
    ),
    showscale=True
))

fig.update_layout(
    title={
        'text': "Матрица различий ненулевых коэффициентов DCT-II ",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18}
    },
    xaxis_title="Поверхность",
    yaxis_title="Поверхность",
    template='plotly_white',   
    height=750,
    width=950,
    xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
    yaxis=dict(tickfont=dict(size=11)),
    margin=dict(l=150, r=50, t=80, b=150)
)

for i in range(n_surfs):
    for j in range(n_surfs):
        fig.add_trace(
            go.Scatter(
                x=[surfaces[j]],
                y=[surfaces[i]],
                mode='text',
                text=[str(z_matrix[i, j])],
                textfont=dict(
                    size=12,
                    color='black' if z_matrix[i, j] < np.max(z_matrix)/2 else 'white'
                ),
                showlegend=False,
                hoverinfo='skip'
            )
        )

output_html = Path(r"C:\UIRS\surface-classification\dct_coefficients_diff_matrix.html")
fig.write_html(str(output_html), include_plotlyjs='cdn')
print(f"\n Интерактивная матрица сохранена в:\n{output_html}")