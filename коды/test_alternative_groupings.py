import pandas as pd
import numpy as np
from scipy.fft import dct
from scipy.stats import norm
from sklearn.metrics import accuracy_score

WINDOW = 29
BINS = 43
DCT_COEFFS = 10
ALPHA = 0.9872495585564633

# Гипотеза 1: 5 классов (разделить Soft_Rough)
mapping_5_v1 = {
    'long_carpet': 'Soft_Carpets', 'short_carpet': 'Soft_Carpets', 'artificial_grass': 'Soft_Carpets',
    'osb': 'Semi_Rigid', 'foam_underlayment': 'Semi_Rigid',
    'eva_foam_tiles': 'EVA_Foam',
    'pvc_foamboard': 'Hard_Floors', 'ceramic_tiles': 'Hard_Floors', 'linoleum': 'Hard_Floors',
    'laminate_flooring': 'Laminate'
}

# Гипотеза 2: 5 классов (выделить Linoleum)
mapping_5_v2 = {
    'long_carpet': 'Soft_Rough', 'short_carpet': 'Soft_Rough', 'artificial_grass': 'Soft_Rough',
    'osb': 'Soft_Rough', 'foam_underlayment': 'Soft_Rough',
    'eva_foam_tiles': 'EVA_Foam',
    'pvc_foamboard': 'Hard_Smooth', 'ceramic_tiles': 'Hard_Smooth',
    'linoleum': 'Linoleum',
    'laminate_flooring': 'Laminate'
}

print("Загрузка данных...")
df = pd.read_csv(r"C:\UIRS\surface-classification\data_with_ke_omega.csv")

def test_classification(df, surface_mapping, name):
    print(f"Тестирование: {name}")
    
    df['surface_merged'] = df['surface'].map(surface_mapping)
    
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
    
    # DCT модели
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
    predicted_labels = []
    n_surfaces = len(surfaces)
    
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
            
            mem_prob = ALPHA * mem_prob + (1 - ALPHA) * probs
            if mem_prob.sum() > 0:
                mem_prob = mem_prob / mem_prob.sum()
            
            pred_idx_mem = np.argmax(mem_prob)
            predicted_labels.append(surfaces[pred_idx_mem])
            true_labels.append(true_surface)
    
    acc = accuracy_score(true_labels, predicted_labels)
    print(f"Точность: {acc*100:.2f}%")
    return acc

acc_5_v1 = test_classification(df, mapping_5_v1, "5 классов (разделить Soft_Rough)")
acc_5_v2 = test_classification(df, mapping_5_v2, "5 классов (выделить Linoleum)")

print(f"4 класса (лучший):  74.7%")
print(f"5 классов (v1):      {acc_5_v1*100:.2f}%")
print(f"5 классов (v2):      {acc_5_v2*100:.2f}%")