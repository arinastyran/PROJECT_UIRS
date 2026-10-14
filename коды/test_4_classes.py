import pandas as pd
import numpy as np
from scipy.fft import dct
from scipy.stats import norm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

WINDOW = 29
BINS = 43
DCT_COEFFS = 10
ALPHA = 0.9872495585564633

# МАППИНГ ДЛЯ 4 ГРУПП
surface_mapping = {
    'long_carpet': 'Soft_Rough',
    'short_carpet': 'Soft_Rough',
    'artificial_grass': 'Soft_Rough',
    'osb': 'Soft_Rough',
    'foam_underlayment': 'Soft_Rough',
    
    'eva_foam_tiles': 'EVA_Foam',
    
    'pvc_foamboard': 'Hard_Floors',
    'ceramic_tiles': 'Hard_Floors',
    'linoleum': 'Hard_Floors',
    
    'laminate_flooring': 'Laminate'
}

print(f"Параметры: window={WINDOW}, bins={BINS}, dct={DCT_COEFFS}, alpha={ALPHA:.4f}")
print("Загрузка данных...")
df = pd.read_csv(r"C:\UIRS\surface-classification\data_with_ke_omega.csv")

# Применяем новое деление на классы
df['surface_merged'] = df['surface'].map(surface_mapping)
print(f"Исходных классов: {df['surface'].nunique()}, Новых классов: {df['surface_merged'].nunique()}")

# Биннинг
omega_stats = df.groupby('surface_merged')['omega_robot'].agg(['min', 'max'])
omega_abs_max = min(abs(omega_stats['min'].max()), abs(omega_stats['max'].min()))
df_trimmed = df[(df['omega_robot'] >= -omega_abs_max) & (df['omega_robot'] <= omega_abs_max)].copy()

bin_edges = np.linspace(-omega_abs_max, omega_abs_max, BINS + 1)
df_trimmed['omega_bin'] = pd.cut(df_trimmed['omega_robot'], bins=bin_edges, labels=False, include_lowest=True)

df_binned = df_trimmed.groupby(['surface_merged', 'omega_bin']).agg(
    median_Ke=('Ke', 'median'), std_Ke=('Ke', 'std')
).reset_index()
df_binned['std_Ke'] = df_binned['std_Ke'].fillna(0)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
df_binned['omega_center'] = df_binned['omega_bin'].map(lambda x: bin_centers[int(x)])

# ДКТ
surfaces = sorted(df_binned['surface_merged'].unique())
dct_models = {}

for surface in surfaces:
    surf_data = df_binned[df_binned['surface_merged'] == surface].sort_values('omega_bin')
    median_ke = surf_data['median_Ke'].values
    std_ke = surf_data['std_Ke'].fillna(0).values
    
    coefficients = dct(median_ke, type=2)
    if DCT_COEFFS < len(coefficients):
        indices = sorted(range(len(coefficients)), key=lambda i: abs(coefficients[i]), reverse=True)
        for idx in indices[DCT_COEFFS:]:
            coefficients[idx] = 0
    
    dct_models[surface] = {
        'coeffs': coefficients, 'N': len(coefficients),
        'range_min': surf_data['omega_center'].min(),
        'range_max': surf_data['omega_center'].max(),
        'std': np.nanmean(std_ke)
    }

# Классификация
true_labels = []
predicted_raw = []      # БЕЗ памяти
predicted_labels = []   # С памятью
n_surfaces = len(surfaces)

print("\n[4] Классификация (без памяти и с памятью)...")
for exp_num, group in df_trimmed.groupby('number_exper'):
    mem_prob = np.ones(n_surfaces) / n_surfaces
    
    for idx, row in group.iterrows():
        omega = row['omega_robot']
        ke = row['Ke']
        true_surface = row['surface_merged']
        
        probs = np.zeros(n_surfaces)
        for i, surface in enumerate(surfaces):
            model = dct_models[surface]
            x_scaled = (omega - model['range_min']) / (model['range_max'] - model['range_min']) * (model['N'] - 1)
            result = model['coeffs'][0] / 2
            for n in range(1, model['N']):
                result += model['coeffs'][n] * np.cos((n / model['N']) * np.pi * (x_scaled + 0.5))
            expected_ke = result * (1 / model['N'])
            
            deviation = ke - expected_ke
            probs[i] = norm.pdf(deviation, loc=0, scale=model['std'] + 1e-3)
        
        if probs.sum() > 0:
            probs = probs / probs.sum()
        
        # Предсказание БЕЗ памяти
        pred_idx_raw = np.argmax(probs)
        predicted_raw.append(surfaces[pred_idx_raw])
        
        # Обновление памяти
        mem_prob = ALPHA * mem_prob + (1 - ALPHA) * probs
        if mem_prob.sum() > 0:
            mem_prob = mem_prob / mem_prob.sum()
        
        # Предсказание С памятью
        pred_idx_mem = np.argmax(mem_prob)
        predicted_labels.append(surfaces[pred_idx_mem])
        
        true_labels.append(true_surface)

acc_raw = accuracy_score(true_labels, predicted_raw)
acc_mem = accuracy_score(true_labels, predicted_labels)

# Матрицы ошибок
print("\n[6] Построение матриц ошибок...")
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Левая матрица (без памяти)
cm_raw = confusion_matrix(true_labels, predicted_raw, labels=surfaces)
cm_raw_norm = cm_raw.astype('float') / cm_raw.sum(axis=1)[:, np.newaxis]

sns.heatmap(
    cm_raw_norm, annot=True, fmt='.2f', cmap='Blues',
    xticklabels=surfaces, yticklabels=surfaces,
    ax=axes[0], cbar_kws={'label': 'Доля'}
)
axes[0].set_title(f'Без памяти (Acc={acc_raw*100:.1f}%)', fontsize=12, fontweight='bold', pad=15)
axes[0].set_xlabel('Предсказанный класс', fontsize=10)
axes[0].set_ylabel('Истинный класс', fontsize=10)

# Правая матрица (с памятью)
cm_mem = confusion_matrix(true_labels, predicted_labels, labels=surfaces)
cm_mem_norm = cm_mem.astype('float') / cm_mem.sum(axis=1)[:, np.newaxis]

sns.heatmap(
    cm_mem_norm, annot=True, fmt='.2f', cmap='Greens',
    xticklabels=surfaces, yticklabels=surfaces,
    ax=axes[1], cbar_kws={'label': 'Доля'}
)
axes[1].set_title(f'С памятью α={ALPHA:.3f} (Acc={acc_mem*100:.1f}%)', fontsize=12, fontweight='bold', pad=15)
axes[1].set_xlabel('Предсказанный класс', fontsize=10)
axes[1].set_ylabel('Истинный класс', fontsize=10)

plt.suptitle('Сравнение классификации на 4 укрупненных классах', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(r"C:\UIRS\surface-classification\confusion_matrices_4_classes_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("Сравнительные матрицы ошибок сохранены: confusion_matrices_4_classes_comparison.png")


print("\n[7] Построение таблицы метрик...")
report_dict = classification_report(true_labels, predicted_labels, labels=surfaces, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report_dict).transpose().round(3)

df_report_filtered = df_report[['precision', 'recall']]

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')

table = ax.table(
    cellText=df_report_filtered.values, 
    colLabels=df_report_filtered.columns, 
    rowLabels=df_report_filtered.index, 
    cellLoc='center', 
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

plt.title('Classification Report: 4 укрупненных класса (с памятью)', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(r"C:\UIRS\surface-classification\classification_report_4_classes_table.png", dpi=300, bbox_inches='tight')
plt.close()
print("Таблица метрик сохранена: classification_report_4_classes_table.png")

print("\n Итоговая таблица:")
print(df_report_filtered.to_string())