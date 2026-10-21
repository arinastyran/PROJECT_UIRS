import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Анализ перекрытия распределений...")
df = pd.read_csv(r"C:\UIRS\surface-classification\data_with_ke_omega.csv")

mappings = {
    '10_classes_original': None,
    '6_classes_ke': {
        'long_carpet': 'Soft_Grass', 'short_carpet': 'Soft_Grass', 'artificial_grass': 'Soft_Grass',
        'osb': 'OSB_Foam', 'foam_underlayment': 'OSB_Foam',
        'eva_foam_tiles': 'EVA_Foam',
        'pvc_foamboard': 'Hard_Smooth', 'ceramic_tiles': 'Hard_Smooth',
        'linoleum': 'Linoleum', 'laminate_flooring': 'Laminate'
    },
    '4_classes_ke': {
        'long_carpet': 'Soft_Rough', 'short_carpet': 'Soft_Rough', 'artificial_grass': 'Soft_Rough', 
        'osb': 'Soft_Rough', 'foam_underlayment': 'Soft_Rough',
        'eva_foam_tiles': 'EVA_Foam',
        'pvc_foamboard': 'Hard_Floors', 'ceramic_tiles': 'Hard_Floors', 'linoleum': 'Hard_Floors',
        'laminate_flooring': 'Laminate'
    },
    '3_classes_article': {
        'long_carpet': 'Lateral_Slip', 'artificial_grass': 'Lateral_Slip',
        'laminate_flooring': 'Stable', 'short_carpet': 'Stable', 'linoleum': 'Stable', 'ceramic_tiles': 'Stable',
        'pvc_foamboard': 'Longitudinal_Slip', 'osb': 'Longitudinal_Slip', 
        'foam_underlayment': 'Longitudinal_Slip', 'eva_foam_tiles': 'Longitudinal_Slip'
    }
}

accuracies = {
    '10_classes_original': 0.4885,
    '6_classes_ke': 0.626,
    '4_classes_ke': 0.747,
    '3_classes_article': 0.457
}

output_dir = r"C:\UIRS\surface-classification\eda_plots_all"
os.makedirs(output_dir, exist_ok=True)

results = {}

for group_name, mapping in mappings.items():
    print(f"Анализ: {group_name}")
    
    if mapping is None:
        df['current_class'] = df['surface']
    else:
        df['current_class'] = df['surface'].map(mapping)
    
    classes = sorted(df['current_class'].unique())
    n_classes = len(classes)
    
    # Вычисляем KDE для каждого класса
    kde_dict = {}
    for cls in classes:
        cls_data = df[df['current_class'] == cls]['Ke'].values
        if len(cls_data) > 1:
            kde_dict[cls] = gaussian_kde(cls_data)
    
    # Считаем pairwise overlap
    overlap_matrix = np.zeros((n_classes, n_classes))
    
    for i, cls1 in enumerate(classes):
        for j, cls2 in enumerate(classes):
            if i == j:
                overlap_matrix[i, j] = 1.0
            else:
                # Вычисляем overlap как интеграл от min(KDE1, KDE2)
                x = np.linspace(0, 2.5, 1000)
                kde1 = kde_dict[cls1](x)
                kde2 = kde_dict[cls2](x)
                overlap = np.trapz(np.minimum(kde1, kde2), x)
                overlap_matrix[i, j] = overlap
    
    # Средний overlap (без диагонали)
    mask = ~np.eye(n_classes, dtype=bool)
    avg_overlap = overlap_matrix[mask].mean()
    
    # Максимальный overlap (худшая пара)
    max_overlap = overlap_matrix[mask].max()
    max_pair_idx = np.unravel_index(np.argmax(overlap_matrix * mask), overlap_matrix.shape)
    max_pair = (classes[max_pair_idx[0]], classes[max_pair_idx[1]])
    
    results[group_name] = {
        'avg_overlap': avg_overlap,
        'max_overlap': max_overlap,
        'max_pair': max_pair,
        'accuracy': accuracies[group_name],
        'overlap_matrix': overlap_matrix,
        'classes': classes
    }
    
    print(f"Средний overlap: {avg_overlap:.3f} ({avg_overlap*100:.1f}%)")
    print(f"Максимальный overlap: {max_overlap:.3f} ({max_overlap*100:.1f}%)")
    print(f"Худшая пара: {max_pair[0]} <-> {max_pair[1]}")

# график зависимости точности от overlap
plt.figure(figsize=(12, 8))

variant_names = list(results.keys())
avg_overlaps = [results[v]['avg_overlap'] for v in variant_names]
accuracies_list = [results[v]['accuracy'] for v in variant_names]

plt.subplot(1, 2, 1)
plt.scatter(avg_overlaps, accuracies_list, s=200, c='steelblue', edgecolors='black', zorder=5)
for i, v in enumerate(variant_names):
    plt.annotate(v, (avg_overlaps[i], accuracies_list[i]), 
                 textcoords="offset points", xytext=(10, 5), fontsize=10)
plt.xlabel('Средний overlap распределений', fontsize=12)
plt.ylabel('Точность классификации', fontsize=12)
plt.title('Зависимость точности от перекрытия распределений', fontsize=14)
plt.grid(True, alpha=0.3)

# Bar chart: max overlap для каждого варианта
plt.subplot(1, 2, 2)
max_overlaps = [results[v]['max_overlap'] for v in variant_names]
colors = ['#e67e22', '#3498db', '#2ecc71', '#e74c3c']
bars = plt.bar(variant_names, max_overlaps, color=colors, edgecolor='black')
plt.xlabel('Вариант разбиения', fontsize=12)
plt.ylabel('Максимальный overlap (худшая пара)', fontsize=12)
plt.title('Максимальное перекрытие между классами', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

for bar, overlap in zip(bars, max_overlaps):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{overlap*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'overlap_vs_accuracy.png'), dpi=300, bbox_inches='tight')
plt.close()

# Вывод результатов
print("ИТОГОВАЯ ТАБЛИЦА")
print(f"{'Вариант':<25} {'Точность':<10} {'Средний overlap':<18} {'Макс. overlap':<15} {'Худшая пара'}")

for v in variant_names:
    r = results[v]
    print(f"{v:<25} {r['accuracy']*100:.1f}%{'':<5} {r['avg_overlap']*100:.1f}%{'':<13} {r['max_overlap']*100:.1f}%{'':<10} {r['max_pair'][0]} <-> {r['max_pair'][1]}")
