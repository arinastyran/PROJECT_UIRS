import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.stats import norm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# Функция для вычисления DCT по коэффициентам из словаря (как в Optuna)
def dct_predict(model, omega):
    """Предсказывает Ke для одного значения omega по модели из Optuna"""
    x_scaled = (omega - model['range_min']) / (model['range_max'] - model['range_min']) * (model['N'] - 1)
    
    result = model['coeffs'][0] / 2
    for n in range(1, model['N']):
        result += model['coeffs'][n] * np.cos((n / model['N']) * np.pi * (x_scaled + 0.5))
    return result * (1 / model['N'])

# 1. Загрузка моделей DCT и данных
MODELS_FILE = r"C:\UIRS\surface-classification\dct_models_from_optuna.pkl"
print(f"\n[1] Загрузка моделей из: {MODELS_FILE}")
with open(MODELS_FILE, 'rb') as file:
    dct_models, surfaces = pickle.load(file)

# Явно преобразуем surfaces в список
surfaces = list(surfaces)
print(f"    Загружено моделей для {len(surfaces)} поверхностей")

DATA_FILE = r"C:\UIRS\surface-classification\data_from_optuna.csv"
print(f"\n[2] Загрузка данных для классификации из: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)
print(f"    Загружено {len(df):,} измерений")

# 2. Настройка классификатора
ALPHA =  0.9895765126599654
n_surfaces = len(surfaces)

true_labels = []
predicted_raw = []
predicted_labels = []  

print("\n[3] Классификация измерений ...")

# 3. Основной цикл классификации 
for exp_num, group in tqdm(df.groupby('number_exper'), desc="Эксперименты"):
    # Сортировка по времени (как мы обсуждали)
    group = group.sort_values('Time').reset_index(drop=True)
    
    # СБРОС ПАМЯТИ
    mem_probabilities = np.ones(n_surfaces) / n_surfaces
    
    for idx, row in group.iterrows():
        omega = row['omega_robot']
        ke = row['Ke']
        true_surface = row['surface']
        
        probabilities = np.zeros(n_surfaces)
        
        for i, surface in enumerate(surfaces):
            model = dct_models[surface]
            std = model['std']  # std теперь внутри модели
            
            expected_ke = dct_predict(model, omega)
            deviation = ke - expected_ke
            probabilities[i] = norm.pdf(deviation, loc=0, scale=std + 1e-3)
        
        # Нормализация без ветки else (как в Optuna)
        prob_sum = probabilities.sum()
        if prob_sum > 0:
            probabilities = probabilities / prob_sum
        
        # Предсказание БЕЗ памяти
        pred_idx_raw = int(np.argmax(probabilities))  # Явно преобразуем в int
        predicted_raw.append(surfaces[pred_idx_raw])
        
        # Обновление памяти
        mem_probabilities = ALPHA * mem_probabilities + (1 - ALPHA) * probabilities
        
        # Нормализация памяти без ветки else
        mem_sum = mem_probabilities.sum()
        if mem_sum > 0:
            mem_probabilities = mem_probabilities / mem_sum
        
        # Итоговое предсказание
        pred_idx_mem = int(np.argmax(mem_probabilities))  # Явно преобразуем в int
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

plt.title('Classification Report: DCT Probabilistic Classifier (с памятью α=0.989)', 
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