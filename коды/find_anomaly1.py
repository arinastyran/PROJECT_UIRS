import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (14, 8) # Твой исходный размер фигуры

print(" Загрузка данных и расчет признаков...")
df = pd.read_csv(r"C:\UIRS\surface-classification\combined_processed_4W.csv")

N_nominal, R, L = 0.11, 0.031, 0.159
loads = df[['wheel_load.1', 'wheel_load.2', 'wheel_load.3', 'wheel_load.4']].values
vels = df[['wheel_angular_velocity.1', 'wheel_angular_velocity.2', 
           'wheel_angular_velocity.3', 'wheel_angular_velocity.4']].values

df['Ke'] = np.where(
    (N_nominal * np.abs(vels).sum(axis=1)) > 0, 
    (loads * vels).sum(axis=1) / (N_nominal * np.abs(vels).sum(axis=1)), 
    0
)

v_left = R * (df['wheel_angular_velocity.1'] + df['wheel_angular_velocity.4']) / 2
v_right = R * (df['wheel_angular_velocity.2'] + df['wheel_angular_velocity.3']) / 2
df['omega_robot'] = (v_left - v_right) / L

# Твоя фирменная палитра
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

output_dir = r"C:\UIRS\surface-classification\anomaly_analysis\separate_plots"
os.makedirs(output_dir, exist_ok=True)

surfaces = sorted(df['surface'].unique())
print(f" Построение 10 отдельных графиков в твоем стиле...")

for surf in surfaces:
    print(f"  Обработка: {surf}...")
    
    # Фильтруем данные только для текущей поверхности
    df_surf = df[df['surface'] == surf]
    color = original_surface_colors[surf]
    
    plt.figure(figsize=(14, 8))
    
    # ИСПОЛЬЗУЕМ ТОЧНО ТАКИЕ ЖЕ ПАРАМЕТРЫ, КАК В ТВОЕМ КОДЕ
    sns.scatterplot(
        data=df_surf, 
        x='omega_robot', 
        y='Ke', 
        color=color,          # Один цвет для всей поверхности
        alpha=0.5,            # Твоя прозрачность
        s=15,                 # Твой размер точек
        edgecolor=None        # Твои границы
    )
    
    plt.title(f'Зависимость Ke от ω: {surf}', fontsize=16)
    plt.xlabel('Угловая скорость ω (рад/с)', fontsize=14)
    plt.ylabel('Коэффициент Ke', fontsize=14)
    
    # УБРАЛИ жесткие xlim/ylim, чтобы seaborn сам идеально подогнал масштаб, как в твоем оригинале
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'scatter_{surf}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"     Сохранено: {output_path}")
