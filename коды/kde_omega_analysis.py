import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import os

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (14, 8)

print("Загрузка данных для анализа распределения ω...")
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

palettes = {
    '10_classes_original': {
        'ceramic_tiles': '#1f77b4', 'pvc_foamboard': '#ff7f0e', 'linoleum': '#2ca02c',
        'laminate_flooring': '#d62728', 'short_carpet': '#9467bd', 'long_carpet': '#8c564b',
        'eva_foam_tiles': '#e377c2', 'artificial_grass': '#7f7f7f', 'osb': '#bcbd22', 'foam_underlayment': '#17becf'
    },
    '6_classes_ke': {
        'Soft_Grass': '#3498db', 'OSB_Foam': '#e74c3c', 'EVA_Foam': '#2ecc71',
        'Hard_Smooth': '#f39c12', 'Linoleum': '#9b59b6', 'Laminate': '#ff69b4'
    },
    '4_classes_ke': {
        'Soft_Rough': '#e67e22', 'EVA_Foam': '#95a5a6', 'Hard_Floors': '#34495e', 'Laminate': '#f1c40f'
    },
    '3_classes_article': {
        'Lateral_Slip': '#c0392b', 'Stable': '#2980b9', 'Longitudinal_Slip': '#27ae60'
    }
}

output_dir = r"C:\UIRS\surface-classification\eda_plots_all"
os.makedirs(output_dir, exist_ok=True)

for group_name, mapping in mappings.items():
    print(f"Анализ ω для варианта: {group_name}")
    
    if mapping is None:
        df['current_class'] = df['surface']
    else:
        df['current_class'] = df['surface'].map(mapping)
        
    palette = palettes[group_name]
    unique_classes = sorted(df['current_class'].unique())
    color_list = [palette[cls] for cls in unique_classes]
    
    # KDE распределения w
    plt.figure(figsize=(14, 8))
    
    sns.kdeplot(
        data=df, x='omega_robot', hue='current_class', 
        palette=color_list,
        fill=True, alpha=0.3, linewidth=2.5
    )
    
    legend_elements = [Patch(facecolor=color_list[i], label=unique_classes[i], alpha=0.5) 
                       for i in range(len(unique_classes))]
    
    plt.title(f'Плотность распределения угловой скорости ω ({group_name})', fontsize=16)
    plt.xlabel('Угловая скорость ω (рад/с)', fontsize=14)
    plt.ylabel('Плотность', fontsize=14)
    plt.legend(handles=legend_elements, title='Класс', title_fontsize=12, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'kde_omega_{group_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
