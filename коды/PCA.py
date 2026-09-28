import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import os
warnings.filterwarnings('ignore')

# Попытка импорта cuML
try:
    from cuml.manifold import TSNE, UMAP
    USE_CUML = True
except ImportError:
    from sklearn.manifold import TSNE
    import umap
    USE_CUML = False

print("Начат анализ понижения размерности на полных данных.")

# 1. Загрузка данных
print("\n[1/6] Загрузка данных...")
df = pd.read_csv("data_with_ke_omega.csv")
print(f"    Загружено {len(df):,} строк")

# 2. Расчет дополнительных признаков
print("\n[2/6] Расчет дополнительных признаков...")
R = 0.031
df['velocity'] = R * (df['wheel_angular_velocity.1'] + df['wheel_angular_velocity.2'] + 
                      df['wheel_angular_velocity.3'] + df['wheel_angular_velocity.4']) / 4

power_cols = ['estimated_power.1', 'estimated_power.2', 'estimated_power.3', 'estimated_power.4']
load_cols = ['wheel_load.1', 'wheel_load.2', 'wheel_load.3', 'wheel_load.4']

df['total_power'] = df[power_cols].sum(axis=1)
df['total_load'] = df[load_cols].sum(axis=1)

df['More_Power_2_wheels'] = df[['estimated_power.1', 'estimated_power.4']].max(axis=1) + \
                            df[['estimated_power.2', 'estimated_power.3']].max(axis=1)
df['Less_Power_2_wheels'] = df[['estimated_power.1', 'estimated_power.4']].min(axis=1) + \
                            df[['estimated_power.2', 'estimated_power.3']].min(axis=1)
df['Load_2_wheels'] = df[['wheel_load.1', 'wheel_load.4']].min(axis=1) + \
                      df[['wheel_load.2', 'wheel_load.3']].min(axis=1)

# 3. Определение 3-х наборов данных
SET_1 = ['omega_robot', 'velocity', 'linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z',
         'angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z',
         'wheel_load.1', 'wheel_load.2', 'wheel_load.3', 'wheel_load.4',
         'wheel_angular_velocity.1', 'wheel_angular_velocity.2', 'wheel_angular_velocity.3', 'wheel_angular_velocity.4',
         'estimated_power.1', 'estimated_power.2', 'estimated_power.3', 'estimated_power.4',
         'mean_power_left', 'mean_power_right', 'Ke']

SET_2 = ['omega_robot', 'velocity', 'total_power', 'total_load',
         'estimated_power.1', 'estimated_power.2', 'estimated_power.3', 'estimated_power.4',
         'wheel_load.1', 'wheel_load.2', 'wheel_load.3', 'wheel_load.4',
         'More_Power_2_wheels', 'Less_Power_2_wheels', 'Load_2_wheels']

SET_3 = ['total_power', 'estimated_power.1', 'estimated_power.2', 'estimated_power.3', 'estimated_power.4',
         'More_Power_2_wheels', 'Less_Power_2_wheels']

datasets = {1: SET_1, 2: SET_2, 3: SET_3}

SURFACE_COLORS = {
    'artificial_grass': '#1f77b4', 'ceramic_tiles': '#ff7f0e',
    'eva_foam_tiles': '#2ca02c', 'foam_underlayment': '#d62728',
    'laminate_flooring': '#9467bd', 'linoleum': '#8c564b',
    'long_carpet': '#e377c2', 'osb': '#7f7f7f',
    'pvc_foamboard': '#bcbd22', 'short_carpet': '#17becf',
}

os.makedirs("plots", exist_ok=True)

# 4. Основной цикл обработки
for set_num, features in datasets.items():
    print(f"\n[3/6] Обработка набора данных #{set_num} ({len(features)} признаков)...")
    
    df_sub = df[features + ['surface']].dropna().copy()
    X = df_sub[features].values
    y = df_sub['surface']
    
    for scaled_flag in [False, True]:
        scaling_name = "Scaled" if scaled_flag else "No_scaling"
        print(f"  Масштабирование: {'с StandardScaler' if scaled_flag else 'без StandardScaler'}")
        
        if scaled_flag:
            scaler = StandardScaler()
            X_proc = scaler.fit_transform(X)
        else:
            X_proc = X
        
        # PCA (быстро на CPU)
        print("       Вычисляем PCA...")
        pca = PCA(n_components=2, random_state=42)
        pca_res = pca.fit_transform(X_proc)
        var1, var2 = pca.explained_variance_ratio_ * 100
        
        df_pca = pd.DataFrame(pca_res, columns=['PC1', 'PC2'])
        df_pca['surface'] = y
        title_pca = f"PCA, Set #{set_num}, {scaling_name} (PC1: {var1:.1f}%, PC2: {var2:.1f}%)"
        fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='surface', 
                             color_discrete_map=SURFACE_COLORS, title=title_pca)
        fig_pca.write_html(f"plots/PCA_Set{set_num}_{scaling_name}.html", include_plotlyjs='cdn')
        print(f"    PCA завершен. Объясненная дисперсия: {var1:.1f}% + {var2:.1f}% = {var1+var2:.1f}%")
        
        # t-SNE
        print("       Вычисляем t-SNE (все данные)...")
        if USE_CUML:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, 
                      learning_rate=200.0, early_exaggeration=12.0)  
        else:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, n_jobs=-1)
        
        tsne_res = tsne.fit_transform(X_proc)
        
        df_tsne = pd.DataFrame(tsne_res, columns=['tSNE1', 'tSNE2'])
        df_tsne['surface'] = y
        title_tsne = f"t-SNE, Set #{set_num}, {scaling_name}"
        fig_tsne = px.scatter(df_tsne, x='tSNE1', y='tSNE2', color='surface', 
                              color_discrete_map=SURFACE_COLORS, title=title_tsne)
        fig_tsne.write_html(f"plots/tSNE_Set{set_num}_{scaling_name}.html", include_plotlyjs='cdn')
        print(f"    t-SNE завершен.")
        
        # UMAP
        print("       Вычисляем UMAP (все данные)...")
        if USE_CUML:
            umap_model = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        else:
            umap_model = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        
        umap_res = umap_model.fit_transform(X_proc)
        
        df_umap = pd.DataFrame(umap_res, columns=['UMAP1', 'UMAP2'])
        df_umap['surface'] = y
        title_umap = f"UMAP, Set #{set_num}, {scaling_name}"
        fig_umap = px.scatter(df_umap, x='UMAP1', y='UMAP2', color='surface', 
                              color_discrete_map=SURFACE_COLORS, title=title_umap)
        fig_umap.write_html(f"plots/UMAP_Set{set_num}_{scaling_name}.html", include_plotlyjs='cdn')
        print(f"    UMAP завершен.")

print("\n[4/6] Все графики построены!")

# 5. Упаковка результатов
print("[5/6] Упаковка результатов в ZIP-архив...")
!zip -r dim_reduction_results.zip plots/

print("\n[6/6] Архив создан: dim_reduction_results.zip")
print("Готово!")