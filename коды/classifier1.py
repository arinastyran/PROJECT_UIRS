import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.stats import norm
from scipy.fft import dct
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# Класс DCT
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

# 1. Загрузка моделей DCT и данных
MODELS_FILE = r"C:\UIRS\surface-classification\dct_models.pkl"
print(f"\n[1] Загрузка моделей из: {MODELS_FILE}")
with open(MODELS_FILE, 'rb') as file:
    dct_models, dct_models_std = pickle.load(file)

surfaces = sorted(list(dct_models.keys()))
print(f"    Загружено моделей для {len(surfaces)} поверхностей")

DATA_FILE = r"C:\UIRS\surface-classification\data_with_ke_omega.csv"
print(f"\n[2] Загрузка данных для классификации из: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)
print(f"    Загружено {len(df):,} измерений")

# 2. Настройка классификатора
ALPHA = 0.9  # Коэффициент памяти
n_surfaces = len(surfaces)

true_labels = []
predicted_raw = []
predicted_labels = []  

print("\n[3] Классификация измерений ...")

# 3. Основной цикл классификации 
for exp_num, group in tqdm(df.groupby('number_exper'), desc="Эксперименты"):
    
    # СБРОС ПАМЯТИ
    mem_probabilities = np.ones(n_surfaces) / n_surfaces
    
    for idx, row in group.iterrows():
        omega = row['omega_robot']
        ke = row['Ke']
        true_surface = row['surface']
        
        probabilities = np.zeros(n_surfaces)
        
        for i, surface in enumerate(surfaces):
            model = dct_models[surface]
            std = dct_models_std[surface]
            
            expected_ke = model(omega, scaled=True)
            deviation = ke - expected_ke
            probabilities[i] = norm.pdf(deviation, loc=0, scale=std + 1e-3)
        
        if probabilities.sum() > 0:
            probabilities = probabilities / probabilities.sum()
        else:
            probabilities = np.ones(n_surfaces) / n_surfaces
        
        # Предсказание БЕЗ памяти
        pred_idx_raw = np.argmax(probabilities)
        predicted_raw.append(surfaces[pred_idx_raw])
        
        # Обновление памяти
        mem_probabilities = ALPHA * mem_probabilities + (1 - ALPHA) * probabilities
        
        if mem_probabilities.sum() > 0:
            mem_probabilities = mem_probabilities / mem_probabilities.sum()
        
        # Итоговое предсказание
        pred_idx_mem = np.argmax(mem_probabilities)
        predicted_labels.append(surfaces[pred_idx_mem])
        
        true_labels.append(true_surface)

print("Классификация завершена!")

# 4. Оценка качества и генерация Classification Report
print("\n[4] Генерация метрик и таблиц...")

report_dict = classification_report(true_labels, predicted_labels, labels=surfaces, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report_dict).transpose().round(3)

fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('tight')
ax.axis('off')

table = ax.table(
    cellText=df_report.values, 
    colLabels=df_report.columns, 
    rowLabels=df_report.index, 
    cellLoc='center', 
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

plt.title('Classification Report: DCT Probabilistic Classifier (с памятью α=0.9)', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(r"C:\UIRS\surface-classification\classification_report_table.png", dpi=300, bbox_inches='tight')
plt.close()
print("Сохранена таблица: classification_report_table.png")

# 5. Генерация сравнительной матрицы
SURFACE_COLORS = {
    'artificial_grass': '#1f77b4', 'ceramic_tiles': '#ff7f0e',
    'eva_foam_tiles': '#2ca02c', 'foam_underlayment': '#d62728',
    'laminate_flooring': '#9467bd', 'linoleum': '#8c564b',
    'long_carpet': '#e377c2', 'osb': '#7f7f7f',
    'pvc_foamboard': '#bcbd22', 'short_carpet': '#17becf',
}

# Матрица ошибок без памяти
cm_no_memory = confusion_matrix(true_labels, predicted_raw, labels=surfaces)
cm_no_memory_norm = cm_no_memory.astype('float') / cm_no_memory.sum(axis=1)[:, np.newaxis]

# Матрица ошибок с памятью
cm_memory = confusion_matrix(true_labels, predicted_labels, labels=surfaces)
cm_memory_norm = cm_memory.astype('float') / cm_memory.sum(axis=1)[:, np.newaxis]

# Строим матрицы на одном графике для сравнения
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Левая матрица (без памяти)
sns.heatmap(
    cm_no_memory_norm, annot=True, fmt='.2f', cmap='Blues',
    xticklabels=[s.replace('_', '\n') for s in surfaces],
    yticklabels=[s.replace('_', '\n') for s in surfaces],
    ax=axes[0], cbar_kws={'label': 'Доля'}
)
axes[0].set_title('Без памяти', fontsize=12, fontweight='bold', pad=15)
axes[0].set_xlabel('Предсказанный класс', fontsize=10)
axes[0].set_ylabel('Истинный класс', fontsize=10)

# Правая матрица (с памятью)
sns.heatmap(
    cm_memory_norm, annot=True, fmt='.2f', cmap='Greens',
    xticklabels=[s.replace('_', '\n') for s in surfaces],
    yticklabels=[s.replace('_', '\n') for s in surfaces],
    ax=axes[1], cbar_kws={'label': 'Доля'}
)
axes[1].set_title(f'С памятью (α={ALPHA})', fontsize=12, fontweight='bold', pad=15)
axes[1].set_xlabel('Предсказанный класс', fontsize=10)
axes[1].set_ylabel('Истинный класс', fontsize=10)

plt.suptitle('Матрицы ошибок вероятностного классификатора', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(r"C:\UIRS\surface-classification\confusion_matrices_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("Сохранено сравнение матриц: confusion_matrices_comparison.png")

results_df = pd.DataFrame({
    'true_surface': true_labels,
    'predicted_raw': predicted_raw,
    'predicted_labels': predicted_labels
})
results_df.to_csv(r"C:\UIRS\surface-classification\classification_results_dct.csv", index=False)
print("Сохранены результаты: classification_results_dct.csv")

