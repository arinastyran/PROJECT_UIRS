import pandas as pd
import numpy as np
import optuna
from scipy.fft import dct
from scipy.stats import norm
from sklearn.metrics import accuracy_score
import warnings
import pickle
warnings.filterwarnings('ignore')

INPUT_FILE = r"C:\UIRS\surface-classification\combined_processed_4W.csv"
OUTPUT_DIR = r"C:\UIRS\surface-classification"

print("Загрузка сырых данных...")
df_raw = pd.read_csv(INPUT_FILE)
print(f"Загружено: {len(df_raw):,} строк")

MERGED_MAPPINGS = {
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

# Точности схлопывания
COLLAPSE_ACCURACIES = {
    'article_3_classes': 0.5534,
    'manual_4_classes': 0.7675,
    'cluster_6_classes': 0.6531,
    'eva_hard_3_classes': 0.8096
}

print("\nПредварительный расчет Ke для всех размеров окна (5-30)...")
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
    
    N_nominal, R, L = 0.11, 0.031, 0.159
    numerator = (
        df_windowed['wheel_load.1'] * df_windowed['wheel_angular_velocity.1'] +
        df_windowed['wheel_load.2'] * df_windowed['wheel_angular_velocity.2'] +
        df_windowed['wheel_load.3'] * df_windowed['wheel_angular_velocity.3'] +
        df_windowed['wheel_load.4'] * df_windowed['wheel_angular_velocity.4']
    )
    denominator = N_nominal * (
        df_windowed['wheel_angular_velocity.1'].abs() +
        df_windowed['wheel_angular_velocity.2'].abs() +
        df_windowed['wheel_angular_velocity.3'].abs() +
        df_windowed['wheel_angular_velocity.4'].abs()
    )
    denominator = denominator.replace(0, np.nan)
    df_windowed['Ke'] = (numerator / denominator).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    v_left = R * (df_windowed['wheel_angular_velocity.1'] + df_windowed['wheel_angular_velocity.4']) / 2
    v_right = R * (df_windowed['wheel_angular_velocity.2'] + df_windowed['wheel_angular_velocity.3']) / 2
    df_windowed['omega_robot'] = (v_left - v_right) / L
    
    cached_data[window_size] = df_windowed

print(f"\nПредпосчитано {len(cached_data)} вариантов окна.")


def prepare_binned_data(df_windowed, n_bins):
    """Биннинг данных по omega"""
    omega_stats = df_windowed.groupby('surface')['omega_robot'].agg(['min', 'max'])
    omega_abs_max = min(abs(omega_stats['min'].max()), abs(omega_stats['max'].min()))
    
    df_trimmed = df_windowed[(df_windowed['omega_robot'] >= -omega_abs_max) & 
                             (df_windowed['omega_robot'] <= omega_abs_max)].copy()
    
    bin_edges = np.linspace(-omega_abs_max, omega_abs_max, n_bins + 1)
    df_trimmed['omega_bin'] = pd.cut(
        df_trimmed['omega_robot'], bins=bin_edges, labels=False, include_lowest=True
    )
    
    df_binned = df_trimmed.groupby(['surface', 'omega_bin']).agg(
        median_Ke=('Ke', 'median'),
        std_Ke=('Ke', 'std')
    ).reset_index()
    
    df_binned['std_Ke'] = df_binned['std_Ke'].fillna(0)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    df_binned['omega_center'] = df_binned['omega_bin'].map(lambda x: bin_centers[int(x)])
    
    return df_binned, df_trimmed

def build_dct_models(df_binned, dct_coeffs):
    """Построение DCT-моделей для каждой поверхности"""
    surfaces = sorted(df_binned['surface'].unique())
    dct_models = {}
    
    for surface in surfaces:
        surf_data = df_binned[df_binned['surface'] == surface].sort_values('omega_bin')
        if len(surf_data) == 0:
            continue
        
        median_ke = surf_data['median_Ke'].values
        std_ke = surf_data['std_Ke'].fillna(0).values
        
        coefficients = dct(median_ke, type=2)
        
        if dct_coeffs < len(coefficients):
            indices = sorted(range(len(coefficients)), 
                           key=lambda i: abs(coefficients[i]), 
                           reverse=True)
            for idx in indices[dct_coeffs:]:
                coefficients[idx] = 0
        
        dct_models[surface] = {
            'coeffs': coefficients,
            'N': len(coefficients),
            'range_min': surf_data['omega_center'].min(),
            'range_max': surf_data['omega_center'].max(),
            'std': np.nanmean(std_ke)
        }
    
    return dct_models, surfaces

def dct_predict_vectorized(model, omegas):
    """Векторизованное предсказание Ke для массива omega"""
    x_scaled = (omegas - model['range_min']) / (model['range_max'] - model['range_min']) * (model['N'] - 1)
    
    result = np.full_like(omegas, model['coeffs'][0] / 2, dtype=float)
    for n in range(1, model['N']):
        result += model['coeffs'][n] * np.cos((n / model['N']) * np.pi * (x_scaled + 0.5))
    return result * (1 / model['N'])

def classify_with_memory_vectorized(df_validation, dct_models, surfaces, alpha):
    """Векторизованная классификация с памятью"""
    n_surfaces = len(surfaces)
    all_true = []
    all_pred = []
    
    for exp_num, group in df_validation.groupby('number_exper'):
        omegas = group['omega_robot'].values
        kes = group['Ke'].values
        true_surfs = group['surface'].values
        n_samples = len(omegas)
        
        mem_prob = np.ones(n_surfaces) / n_surfaces
        probs_matrix = np.zeros((n_samples, n_surfaces))
        
        for i, surface in enumerate(surfaces):
            if surface not in dct_models:
                continue
            model = dct_models[surface]
            expected_ke = dct_predict_vectorized(model, omegas)
            deviation = kes - expected_ke
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
        
        pred_surfs = [surfaces[idx] for idx in predictions]
        
        all_true.extend(true_surfs)
        all_pred.extend(pred_surfs)
    
    return all_true, all_pred


def run_optuna_for_mapping(mapping_name, mapping, n_trials=100):
    """Запуск Optuna для конкретного маппинга укрупнения"""
    print(f"OPTUNA ДЛЯ: {mapping_name}")
    
    def objective(trial):
        window_size = trial.suggest_int('window_size', 5, 30)
        n_bins = trial.suggest_int('n_bins', 11, 101, step=2)
        dct_coeffs = trial.suggest_int('dct_coeffs', 1, 10, log=True)
        alpha = trial.suggest_float('alpha', 0.5, 0.99)
        
        df_windowed = cached_data[window_size].copy()
        
        # Применяем маппинг укрупнения
        df_windowed['surface'] = df_windowed['surface'].map(mapping)
        
        df_binned, df_full = prepare_binned_data(df_windowed, n_bins)
        
        if len(df_binned) == 0 or len(df_full) == 0:
            return 0.0
        
        dct_models, surfaces = build_dct_models(df_binned, dct_coeffs)
        
        if len(surfaces) == 0:
            return 0.0
        
        true_labels, predicted_labels = classify_with_memory_vectorized(
            df_full, dct_models, surfaces, alpha
        )
        
        if len(true_labels) == 0:
            return 0.0
        
        return accuracy_score(true_labels, predicted_labels)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)
    
    best_params = study.best_params
    best_accuracy = study.best_value
    
    print(f"\nЛучшая точность: {best_accuracy*100:.2f}%")
    print(f"Лучшие параметры:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Сохраняем результаты
    results = {
        'mapping_name': mapping_name,
        'best_accuracy': best_accuracy,
        'best_params': best_params,
    }
    
    # Сохраняем модели
    window_size = int(best_params['window_size'])
    n_bins = int(best_params['n_bins'])
    dct_coeffs = int(best_params['dct_coeffs'])
    alpha = best_params['alpha']
    
    df_windowed = cached_data[window_size].copy()
    df_windowed['surface'] = df_windowed['surface'].map(mapping)
    df_binned, df_trimmed = prepare_binned_data(df_windowed, n_bins)
    dct_models_opt, surfaces_opt = build_dct_models(df_binned, dct_coeffs)
    
    with open(f"{OUTPUT_DIR}/dct_models_optuna_{mapping_name}.pkl", 'wb') as f:
        pickle.dump((dct_models_opt, surfaces_opt, best_params), f)
    
    # Сохраняем trial results
    trials_df = pd.DataFrame([
        {
            'trial_number': t.number,
            'accuracy': t.value,
            'window_size': int(t.params.get('window_size', 0)),
            'n_bins': int(t.params.get('n_bins', 0)),
            'dct_coeffs': int(t.params.get('dct_coeffs', 0)),
            'alpha': t.params.get('alpha', 0.5)
        }
        for t in study.trials
    ])
    trials_df.to_csv(f"{OUTPUT_DIR}/optuna_trials_{mapping_name}.csv", index=False)
    
    print(f"\nМодели сохранены: dct_models_optuna_{mapping_name}.pkl")
    print(f"Trials сохранены: optuna_trials_{mapping_name}.csv")
    
    return results

print("ЗАПУСК OPTUNA ДЛЯ ВСЕХ ВАРИАНТОВ УКРУПНЕНИЯ")

all_results = []

for mapping_name, mapping in MERGED_MAPPINGS.items():
    result = run_optuna_for_mapping(mapping_name, mapping, n_trials=100)
    all_results.append(result)

print("ИТОГОВАЯ СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
df_summary = pd.DataFrame(all_results)
df_summary['collapse_accuracy'] = df_summary['mapping_name'].map(COLLAPSE_ACCURACIES)
df_summary['difference'] = df_summary['best_accuracy'] - df_summary['collapse_accuracy']
print(df_summary[['mapping_name', 'best_accuracy', 'collapse_accuracy', 'difference']].to_string(index=False))

df_summary.to_csv(f"{OUTPUT_DIR}/optuna_vs_collapse_summary.csv", index=False)
print(f"\nТаблица сохранена: optuna_vs_collapse_summary.csv")
