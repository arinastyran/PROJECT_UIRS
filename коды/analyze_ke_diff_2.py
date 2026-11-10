import pandas as pd
import numpy as np
import optuna
from scipy.fft import dct
from scipy.stats import norm
from sklearn.metrics import accuracy_score
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

INPUT_FILE = r"C:\UIRS\surface-classification\combined_processed_4W.csv"
OUTPUT_DIR = r"C:\UIRS\surface-classification\ke_diff_2_optuna"
CACHE_FILE = r"C:\UIRS\surface-classification\cached_windows.pkl"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Загрузка сырых данных (для структуры)...")
df_raw = pd.read_csv(INPUT_FILE)
print(f"Загружено: {len(df_raw):,} строк")

# ==========================================
# 1. ЗАГРУЗКА КЭША
# ==========================================
if os.path.exists(CACHE_FILE):
    print(f"\n Загрузка предпосчитанных окон из кэша...")
    with open(CACHE_FILE, 'rb') as f:
        cached_data = pickle.load(f)
    print(f" Загружено {len(cached_data)} вариантов окна.")
else:
    print(" Кэш не найден! Сначала создайте cached_windows.pkl")
    exit()

# ==========================================
# 2. ФУНКЦИЯ РАСЧЕТА НОВОГО ПРИЗНАКА
# ==========================================
def calculate_ke_diff_2(df):
    """Разность Ke между 2 макс и 2 мин нагруженными колесами"""
    N_nominal = 0.11
    loads = df[['wheel_load.1', 'wheel_load.2', 'wheel_load.3', 'wheel_load.4']].values
    vels = df[['wheel_angular_velocity.1', 'wheel_angular_velocity.2', 
               'wheel_angular_velocity.3', 'wheel_angular_velocity.4']].values
    rows = np.arange(len(df))[:, None]
    
    # 2 макс нагруженных
    idx_top = np.argsort(loads, axis=1)[:, -2:]
    p_top = (loads[rows, idx_top] * vels[rows, idx_top]).sum(axis=1)
    d_top = N_nominal * np.abs(vels[rows, idx_top]).sum(axis=1)
    ke_top = np.where(d_top > 0, p_top / d_top, 0)
    
    # 2 мин нагруженных
    idx_bot = np.argsort(loads, axis=1)[:, :2]
    p_bot = (loads[rows, idx_bot] * vels[rows, idx_bot]).sum(axis=1)
    d_bot = N_nominal * np.abs(vels[rows, idx_bot]).sum(axis=1)
    ke_bot = np.where(d_bot > 0, p_bot / d_bot, 0)
    
    return ke_top - ke_bot

# ==========================================
# 3. ФУНКЦИИ КЛАССИФИКАТОРА
# ==========================================
def prepare_binned_data(df_windowed, n_bins):
    df = df_windowed.copy()
    df['Ke_diff_2'] = calculate_ke_diff_2(df)
    
    # Гарантируем наличие omega_robot
    if 'omega_robot' not in df.columns:
        R, L = 0.031, 0.159
        v_left = R * (df['wheel_angular_velocity.1'] + df['wheel_angular_velocity.4']) / 2
        v_right = R * (df['wheel_angular_velocity.2'] + df['wheel_angular_velocity.3']) / 2
        df['omega_robot'] = (v_left - v_right) / L

    omega_stats = df.groupby('surface_merged')['omega_robot'].agg(['min', 'max'])
    omega_abs_max = min(abs(omega_stats['min'].max()), abs(omega_stats['max'].min()))
    
    df_trimmed = df[(df['omega_robot'] >= -omega_abs_max) & (df['omega_robot'] <= omega_abs_max)].copy()
    bin_edges = np.linspace(-omega_abs_max, omega_abs_max, n_bins + 1)
    df_trimmed['omega_bin'] = pd.cut(df_trimmed['omega_robot'], bins=bin_edges, labels=False, include_lowest=True)
    
    df_binned = df_trimmed.groupby(['surface_merged', 'omega_bin']).agg(
        median_feature=('Ke_diff_2', 'median'),
        std_feature=('Ke_diff_2', 'std')
    ).reset_index()
    
    df_binned['std_feature'] = df_binned['std_feature'].fillna(0)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    df_binned['omega_center'] = df_binned['omega_bin'].map(lambda x: bin_centers[int(x)])
    
    return df_binned, df_trimmed

def build_dct_models(df_binned, dct_coeffs):
    classes = sorted(df_binned['surface_merged'].unique())
    dct_models = {}
    for cls in classes:
        cls_data = df_binned[df_binned['surface_merged'] == cls].sort_values('omega_bin')
        if len(cls_data) == 0: continue
        
        median_vals = cls_data['median_feature'].values
        std_vals = cls_data['std_feature'].fillna(0).values
        coefficients = dct(median_vals, type=2)
        
        if dct_coeffs < len(coefficients):
            indices = sorted(range(len(coefficients)), key=lambda i: abs(coefficients[i]), reverse=True)
            for idx in indices[dct_coeffs:]: coefficients[idx] = 0
            
        dct_models[cls] = {
            'coeffs': coefficients, 'N': len(coefficients),
            'range_min': cls_data['omega_center'].min(),
            'range_max': cls_data['omega_center'].max(),
            'std': np.nanmean(std_vals)
        }
    return dct_models, classes

def dct_predict(model, omegas):
    x_scaled = (omegas - model['range_min']) / (model['range_max'] - model['range_min']) * (model['N'] - 1)
    result = np.full_like(omegas, model['coeffs'][0] / 2, dtype=float)
    for n in range(1, model['N']):
        result += model['coeffs'][n] * np.cos((n / model['N']) * np.pi * (x_scaled + 0.5))
    return result * (1 / model['N'])

def classify_with_memory(df_val, dct_models, classes, alpha):
    n_classes = len(classes)
    all_true, all_pred = [], []
    
    for exp_num, group in df_val.groupby('number_exper'):
        omegas = group['omega_robot'].values
        features = group['Ke_diff_2'].values
        true_classes = group['surface_merged'].values
        n_samples = len(omegas)
        
        mem_prob = np.ones(n_classes) / n_classes
        probs_matrix = np.zeros((n_samples, n_classes))
        
        for i, cls in enumerate(classes):
            if cls not in dct_models: continue
            model = dct_models[cls]
            deviation = features - dct_predict(model, omegas)
            probs_matrix[:, i] = norm.pdf(deviation, loc=0, scale=model['std'] + 1e-3)

        row_sums = probs_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        probs_matrix = probs_matrix / row_sums

        predictions = np.zeros(n_samples, dtype=int)
        for t in range(n_samples):
            mem_prob = alpha * mem_prob + (1 - alpha) * probs_matrix[t]
            mem_sum = mem_prob.sum()
            if mem_sum > 0: mem_prob = mem_prob / mem_sum
            predictions[t] = np.argmax(mem_prob)
        
        all_true.extend(true_classes)
        all_pred.extend([classes[idx] for idx in predictions])
    
    return all_true, all_pred

# ==========================================
# 4. ЗАПУСК OPTUNA
# ==========================================
MERGED_MAPPINGS = {
    '10_classes_original': {s: s for s in df_raw['surface'].unique()},
    'article_3_classes': {
        'long_carpet': 'Lateral_Slip', 'artificial_grass': 'Lateral_Slip',
        'laminate_flooring': 'Stable', 'short_carpet': 'Stable', 
        'linoleum': 'Stable', 'ceramic_tiles': 'Stable',
        'pvc_foamboard': 'Longitudinal_Slip', 'osb': 'Longitudinal_Slip', 
        'foam_underlayment': 'Longitudinal_Slip', 'eva_foam_tiles': 'Longitudinal_Slip'
    },
    'manual_4_classes': {
        'long_carpet': 'Soft_Rough', 'short_carpet': 'Soft_Rough', 
        'artificial_grass': 'Soft_Rough', 'osb': 'Soft_Rough', 
        'foam_underlayment': 'Soft_Rough', 'eva_foam_tiles': 'EVA_Foam',
        'pvc_foamboard': 'Hard_Floors', 'ceramic_tiles': 'Hard_Floors', 
        'linoleum': 'Hard_Floors', 'laminate_flooring': 'Laminate'
    },
    'cluster_6_classes': {
        'long_carpet': 'Soft_Grass', 'short_carpet': 'Soft_Grass', 
        'artificial_grass': 'Soft_Grass', 'osb': 'OSB_Foam', 
        'foam_underlayment': 'OSB_Foam', 'eva_foam_tiles': 'EVA_Foam',
        'pvc_foamboard': 'Hard_Smooth', 'ceramic_tiles': 'Hard_Smooth',
        'linoleum': 'Linoleum', 'laminate_flooring': 'Laminate'
    },
    'eva_hard_3_classes': {
        'long_carpet': 'Soft_Rough', 'short_carpet': 'Soft_Rough', 
        'artificial_grass': 'Soft_Rough', 'osb': 'Soft_Rough', 
        'foam_underlayment': 'Soft_Rough', 'eva_foam_tiles': 'Hard_EVA_Merged', 
        'pvc_foamboard': 'Hard_EVA_Merged', 'ceramic_tiles': 'Hard_EVA_Merged', 
        'linoleum': 'Hard_EVA_Merged', 'laminate_flooring': 'Laminate'
    }
}


print("OPTUNA: КЛАССИФИКАЦИЯ НА ПРИЗНАКЕ Ke_diff_2 (Ke_top2 - Ke_bottom2)")
all_results = []
for mapping_name, mapping in MERGED_MAPPINGS.items():
    print(f"\nЗапуск: {mapping_name}")
    
    def objective(trial):
        window_size = trial.suggest_int('window_size', 5, 30)
        n_bins = trial.suggest_int('n_bins', 11, 101, step=2)
        dct_coeffs = trial.suggest_int('dct_coeffs', 1, 10, log=True)
        alpha = trial.suggest_float('alpha', 0.5, 0.99)
        
        df_windowed = cached_data[window_size].copy()
        df_windowed['surface_merged'] = df_windowed['surface'].map(mapping)
        
        df_binned, df_full = prepare_binned_data(df_windowed, n_bins)
        if len(df_binned) == 0 or len(df_full) == 0: return 0.0
        
        dct_models, classes = build_dct_models(df_binned, dct_coeffs)
        if len(classes) == 0: return 0.0
        
        true_labels, predicted_labels = classify_with_memory(df_full, dct_models, classes, alpha)
        return accuracy_score(true_labels, predicted_labels) if len(true_labels) > 0 else 0.0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, gc_after_trial=True, show_progress_bar=True)
    
    all_results.append({
        'variant': mapping_name,
        'accuracy': study.best_value,
        'params': study.best_params
    })
    print(f"   Лучшая точность: {study.best_value*100:.2f}%")

# Итоговая таблица
print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ (Ke_diff_2)")
df_summary = pd.DataFrame(all_results)
print(df_summary[['variant', 'accuracy']].to_string(index=False))

df_summary.to_csv(os.path.join(OUTPUT_DIR, 'results_ke_diff_2.csv'), index=False)