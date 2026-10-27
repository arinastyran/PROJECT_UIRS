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
OUTPUT_DIR = r"C:\UIRS\surface-classification\optuna_ke_top2"
CACHE_FILE = r"C:\UIRS\surface-classification\cached_windows.pkl"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Загрузка сырых данных...")
df_raw = pd.read_csv(INPUT_FILE)
print(f"Загружено: {len(df_raw):,} строк")

print("\nРасчет признака: Ke_top2...")

N_nominal, R, L = 0.11, 0.031, 0.159

loads = df_raw[['wheel_load.1', 'wheel_load.2', 'wheel_load.3', 'wheel_load.4']].values
vels = df_raw[['wheel_angular_velocity.1', 'wheel_angular_velocity.2', 
               'wheel_angular_velocity.3', 'wheel_angular_velocity.4']].values
rows = np.arange(len(df_raw))[:, None]

# Индексы 2 колес с максимальной нагрузкой
idx_top2 = np.argsort(loads, axis=1)[:, -2:]
loads_top2 = loads[rows, idx_top2]
vels_top2 = vels[rows, idx_top2]

# Ke по 2 макс нагруженным колесам
power_top2 = (loads_top2 * vels_top2).sum(axis=1)
denom_top2 = N_nominal * np.abs(vels_top2).sum(axis=1)
df_raw['Ke_top2'] = np.where(denom_top2 > 0, power_top2 / denom_top2, 0)

# Угловая скорость робота (для биннинга)
v_left = R * (df_raw['wheel_angular_velocity.1'] + df_raw['wheel_angular_velocity.4']) / 2
v_right = R * (df_raw['wheel_angular_velocity.2'] + df_raw['wheel_angular_velocity.3']) / 2
df_raw['omega_robot'] = (v_left - v_right) / L

print("Признак Ke_top2 рассчитан!")

if os.path.exists(CACHE_FILE):
    print(f"\nЗагрузка предпосчитанных окон из кэша...")
    with open(CACHE_FILE, 'rb') as f:
        cached_data = pickle.load(f)
    print(f"Загружено {len(cached_data)} вариантов окна из кэша.")
else:
    print("\nПредварительный расчет для всех размеров окна (5-30)...")
    cached_data = {}

    for window_size in range(5, 31):
        print(f"  window_size={window_size}...", end=" ")
        
        exclude_cols = {'Unnamed: 0', 'Time', 'surface', 'wheels', 'number_exper'}
        numeric_cols = [col for col in df_raw.select_dtypes(include=[np.number]).columns if col not in exclude_cols]
        
        df_sorted = df_raw.sort_values(['surface', 'number_exper', 'Time']).copy()
        
        rolled_nums = df_sorted.groupby(['surface', 'number_exper'])[numeric_cols].rolling(
            window=window_size, min_periods=window_size
        ).mean().reset_index(level=[0, 1])
        
        rolled_time = df_sorted.groupby(['surface', 'number_exper'])['Time'].rolling(
            window=window_size, min_periods=window_size
        ).apply(lambda x: x.iloc[-1]).reset_index(level=[0, 1])
        
        rolled_wheels = df_sorted.groupby(['surface', 'number_exper'])['wheels'].rolling(
            window=window_size, min_periods=window_size
        ).apply(lambda x: x.iloc[0]).reset_index(level=[0, 1])
        
        df_windowed = rolled_nums.copy()
        df_windowed['Time'] = rolled_time['Time']
        df_windowed['wheels'] = rolled_wheels['wheels']
        df_windowed = df_windowed.dropna(subset=numeric_cols).reset_index(drop=True)
        
        cached_data[window_size] = df_windowed

    print(f"\nПредпосчитано {len(cached_data)} вариантов окна.")
    
    print(f"💾 Сохранение кэша в {CACHE_FILE}...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cached_data, f)
    print("Кэш успешно сохранен!")


MERGED_MAPPINGS = {
    '10_classes_original': {
        'long_carpet': 'long_carpet', 'short_carpet': 'short_carpet', 
        'artificial_grass': 'artificial_grass', 'osb': 'osb', 
        'foam_underlayment': 'foam_underlayment',
        'eva_foam_tiles': 'eva_foam_tiles',
        'pvc_foamboard': 'pvc_foamboard', 'ceramic_tiles': 'ceramic_tiles', 
        'linoleum': 'linoleum',
        'laminate_flooring': 'laminate_flooring'
    },
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
        'foam_underlayment': 'Soft_Rough',
        'eva_foam_tiles': 'EVA_Foam',
        'pvc_foamboard': 'Hard_Floors', 'ceramic_tiles': 'Hard_Floors', 
        'linoleum': 'Hard_Floors',
        'laminate_flooring': 'Laminate'
    },
    'cluster_6_classes': {
        'long_carpet': 'Soft_Grass', 'short_carpet': 'Soft_Grass', 
        'artificial_grass': 'Soft_Grass',
        'osb': 'OSB_Foam', 'foam_underlayment': 'OSB_Foam',
        'eva_foam_tiles': 'EVA_Foam',
        'pvc_foamboard': 'Hard_Smooth', 'ceramic_tiles': 'Hard_Smooth',
        'linoleum': 'Linoleum', 'laminate_flooring': 'Laminate'
    },
    'eva_hard_3_classes': {
        'long_carpet': 'Soft_Rough', 'short_carpet': 'Soft_Rough', 
        'artificial_grass': 'Soft_Rough', 'osb': 'Soft_Rough', 
        'foam_underlayment': 'Soft_Rough',
        'eva_foam_tiles': 'Hard_EVA_Merged', 
        'pvc_foamboard': 'Hard_EVA_Merged', 'ceramic_tiles': 'Hard_EVA_Merged', 
        'linoleum': 'Hard_EVA_Merged',
        'laminate_flooring': 'Laminate'
    }
}

def prepare_binned_data(df_windowed, n_bins):
    N_nominal = 0.11
    
    loads = df_windowed[['wheel_load.1', 'wheel_load.2', 'wheel_load.3', 'wheel_load.4']].values
    vels = df_windowed[['wheel_angular_velocity.1', 'wheel_angular_velocity.2', 
                        'wheel_angular_velocity.3', 'wheel_angular_velocity.4']].values
    rows = np.arange(len(df_windowed))[:, None]
    
    # Индексы 2 колес с максимальной нагрузкой
    idx_top2 = np.argsort(loads, axis=1)[:, -2:]
    loads_top2 = loads[rows, idx_top2]
    vels_top2 = vels[rows, idx_top2]
    
    power_top2 = (loads_top2 * vels_top2).sum(axis=1)
    denom_top2 = N_nominal * np.abs(vels_top2).sum(axis=1)
    df_windowed['Ke_top2'] = np.where(denom_top2 > 0, power_top2 / denom_top2, 0)
    
    omega_stats = df_windowed.groupby('surface_merged')['omega_robot'].agg(['min', 'max'])
    omega_abs_max = min(abs(omega_stats['min'].max()), abs(omega_stats['max'].min()))
    
    df_trimmed = df_windowed[(df_windowed['omega_robot'] >= -omega_abs_max) & 
                             (df_windowed['omega_robot'] <= omega_abs_max)].copy()
    
    bin_edges = np.linspace(-omega_abs_max, omega_abs_max, n_bins + 1)
    df_trimmed['omega_bin'] = pd.cut(
        df_trimmed['omega_robot'], bins=bin_edges, labels=False, include_lowest=True
    )
    
    df_binned = df_trimmed.groupby(['surface_merged', 'omega_bin']).agg(
        median_feature=('Ke_top2', 'median'),
        std_feature=('Ke_top2', 'std')
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
        if len(cls_data) == 0:
            continue
        
        median_vals = cls_data['median_feature'].values
        std_vals = cls_data['std_feature'].fillna(0).values
        
        coefficients = dct(median_vals, type=2)
        
        if dct_coeffs < len(coefficients):
            indices = sorted(range(len(coefficients)), 
                           key=lambda i: abs(coefficients[i]), 
                           reverse=True)
            for idx in indices[dct_coeffs:]:
                coefficients[idx] = 0
        
        dct_models[cls] = {
            'coeffs': coefficients,
            'N': len(coefficients),
            'range_min': cls_data['omega_center'].min(),
            'range_max': cls_data['omega_center'].max(),
            'std': np.nanmean(std_vals)
        }
    
    return dct_models, classes

def dct_predict_vectorized(model, omegas):
    x_scaled = (omegas - model['range_min']) / (model['range_max'] - model['range_min']) * (model['N'] - 1)
    
    result = np.full_like(omegas, model['coeffs'][0] / 2, dtype=float)
    for n in range(1, model['N']):
        result += model['coeffs'][n] * np.cos((n / model['N']) * np.pi * (x_scaled + 0.5))
    return result * (1 / model['N'])

def classify_with_memory_vectorized(df_validation, dct_models, classes, alpha):
    n_classes = len(classes)
    all_true = []
    all_pred = []
    
    for exp_num, group in df_validation.groupby('number_exper'):
        omegas = group['omega_robot'].values
        features = group['Ke_top2'].values
        true_classes = group['surface_merged'].values
        n_samples = len(omegas)
        
        mem_prob = np.ones(n_classes) / n_classes
        probs_matrix = np.zeros((n_samples, n_classes))
        
        for i, cls in enumerate(classes):
            if cls not in dct_models:
                continue
            model = dct_models[cls]
            expected_val = dct_predict_vectorized(model, omegas)
            deviation = features - expected_val
            probs_matrix[:, i] = norm.pdf(deviation, loc=0, scale=model['std'] + 1e-3)

        row_sums = probs_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        probs_matrix = probs_matrix / row_sums

        predictions = np.zeros(n_samples, dtype=int)
        
        for t in range(n_samples):
            mem_prob = alpha * mem_prob + (1 - alpha) * probs_matrix[t]
            mem_sum = mem_prob.sum()
            if mem_sum > 0:
                mem_prob = mem_prob / mem_sum
            predictions[t] = np.argmax(mem_prob)
        
        pred_classes = [classes[idx] for idx in predictions]
        
        all_true.extend(true_classes)
        all_pred.extend(pred_classes)
    
    return all_true, all_pred


print("OPTUNA: КЛАССИФИКАЦИЯ НА Ke_top2 ")

all_results = []

for mapping_name, mapping in MERGED_MAPPINGS.items():
    print(f"ВАРИАНТ: {mapping_name}")
    
    def objective(trial):
        window_size = trial.suggest_int('window_size', 5, 30)
        n_bins = trial.suggest_int('n_bins', 11, 101, step=2)
        dct_coeffs = trial.suggest_int('dct_coeffs', 1, 10, log=True)
        alpha = trial.suggest_float('alpha', 0.5, 0.99)
        
        df_windowed = cached_data[window_size].copy()
        df_windowed['surface_merged'] = df_windowed['surface'].map(mapping)
        
        df_binned, df_full = prepare_binned_data(df_windowed, n_bins)
        
        if len(df_binned) == 0 or len(df_full) == 0:
            return 0.0
        
        dct_models, classes = build_dct_models(df_binned, dct_coeffs)
        
        if len(classes) == 0:
            return 0.0
        
        true_labels, predicted_labels = classify_with_memory_vectorized(
            df_full, dct_models, classes, alpha
        )
        
        if len(true_labels) == 0:
            return 0.0
        
        return accuracy_score(true_labels, predicted_labels)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, gc_after_trial=True, show_progress_bar=True)
    
    best_params = study.best_params
    best_accuracy = study.best_value
    
    print(f"\nЛучшая точность: {best_accuracy*100:.2f}%")
    print(f"Лучшие параметры:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    all_results.append({
        'variant': mapping_name,
        'accuracy': best_accuracy,
        'params': best_params
    })
    
    # Сохранение моделей
    window_size = int(best_params['window_size'])
    n_bins = int(best_params['n_bins'])
    dct_coeffs = int(best_params['dct_coeffs'])
    alpha = best_params['alpha']
    
    df_windowed = cached_data[window_size].copy()
    df_windowed['surface_merged'] = df_windowed['surface'].map(mapping)
    df_binned, df_trimmed = prepare_binned_data(df_windowed, n_bins)
    dct_models_final, classes_final = build_dct_models(df_binned, dct_coeffs)
    
    with open(f"{OUTPUT_DIR}/dct_models_ke_top2_{mapping_name}.pkl", 'wb') as f:
        pickle.dump((dct_models_final, classes_final, best_params), f)

print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")

df_summary = pd.DataFrame(all_results)
print(df_summary[['variant', 'accuracy']].to_string(index=False))

df_summary.to_csv(f"{OUTPUT_DIR}/results_ke_top2.csv", index=False)
print(f"\nРезультаты сохранены: {OUTPUT_DIR}/results_ke_top2.csv")