import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import os

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (14, 8)

print("Загрузка данных...")
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

output_dir = r"C:\UIRS\surface-classification\eda_plots_all"
os.makedirs(output_dir, exist_ok=True)

original_surface_colors = {
    'ceramic_tiles': '#1f77b4',      
    'pvc_foamboard': '#ff7f0e',      
    'linoleum': '#2ca02c',           
    'laminate_flooring': '#d62728',  
    'short_carpet': '#9467bd',       
    'long_carpet': '#8c564b',        
    'eva_foam_tiles': '#e377c2',     
    'artificial_grass': '#7f7f7f',   
    'osb': '#bcbd22',                
    'foam_underlayment': '#17becf'   
}

palette_6_classes = {
    'Soft_Grass': '#3498db',      
    'OSB_Foam': '#e74c3c',      
    'EVA_Foam': '#2ecc71',       
    'Hard_Smooth': '#f39c12',     
    'Linoleum': '#9b59b6',       
    'Laminate': '#ff69b4'         
}

# 3. Палитра для 4 классов (другие цвета)
palette_4_classes = {
    'Soft_Rough': '#e67e22',    
    'EVA_Foam': '#95a5a6',       
    'Hard_Floors': '#34495e',     
    'Laminate': '#f1c40f'        
}

palette_3_classes = {
    'Lateral_Slip': '#c0392b',           
    'Stable': '#2980b9',                 
    'Longitudinal_Slip': '#27ae60'        
}

palettes = {
    '10_classes_original': original_surface_colors,
    '6_classes_ke': palette_6_classes,
    '4_classes_ke': palette_4_classes,
    '3_classes_article': palette_3_classes
}

for group_name, mapping in mappings.items():
    print(f"Анализ варианта: {group_name}")
    
    # Создаем колонку с классами для текущего варианта
    if mapping is None:
        df['current_class'] = df['surface']
    else:
        df['current_class'] = df['surface'].map(mapping)
        
    palette = palettes[group_name]
    
    # Получаем список уникальных классов в порядке появления
    unique_classes = sorted(df['current_class'].unique())
    
    # Создаем список цветов в правильном порядке
    color_list = [palette[cls] for cls in unique_classes]
    
    # Точечная диаграмма
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        data=df, x='omega_robot', y='Ke', hue='current_class', 
        palette=color_list, alpha=0.5, s=15, edgecolor=None
    )
    plt.title(f'Зависимость Ke от ω ({group_name})', fontsize=16)
    plt.xlabel('Угловая скорость ω (рад/с)', fontsize=14)
    plt.ylabel('Коэффициент Ke', fontsize=14)
    plt.legend(title='Класс', title_fontsize=12, fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'scatter_{group_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Оценка плотности (KDE)
    plt.figure(figsize=(14, 8))
    
    sns.kdeplot(
        data=df, x='Ke', hue='current_class', 
        palette=color_list,
        fill=True, alpha=0.3, linewidth=2
    )
    
    legend_elements = [Patch(facecolor=color_list[i], label=unique_classes[i], alpha=0.5) 
                       for i in range(len(unique_classes))]
    
    plt.title(f'Плотность распределения Ke ({group_name})', fontsize=16)
    plt.xlabel('Коэффициент Ke', fontsize=14)
    plt.ylabel('Плотность', fontsize=14)
    plt.legend(handles=legend_elements, title='Класс', title_fontsize=12, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'kde_{group_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
 
    # Boxplot
    plt.figure(figsize=(14, 8))
    order = df.groupby('current_class')['Ke'].median().sort_values().index.tolist()
    
    # Пересортировываем цвета согласно порядку на графике
    sorted_color_list = [color_list[unique_classes.index(cls)] for cls in order]
    
    sns.boxplot(
        data=df, x='current_class', y='Ke', order=order, palette=sorted_color_list
    )
    
    legend_elements = [Patch(facecolor=sorted_color_list[i], label=order[i]) 
                       for i in range(len(order))]
    plt.legend(handles=legend_elements, title='Класс поверхности',
               bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(f'Статистический разброс Ke ({group_name})', fontsize=16)
    plt.xlabel('Класс поверхности', fontsize=14)
    plt.ylabel('Коэффициент Ke', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'boxplot_{group_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    stats = df.groupby('current_class')['Ke'].agg(['mean', 'median', 'std']).round(3)
    print(stats)
