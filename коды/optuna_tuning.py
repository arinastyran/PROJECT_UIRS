import pandas as pd
import numpy as np
import optuna
from scipy.fft import dct
from scipy.stats import norm
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

INPUT_FILE = r"C:\UIRS\surface-classification\combined_processed_4W.csv"

print("Загрузка сырых данных...")
df_raw = pd.read_csv(INPUT_FILE)
print(f"Загружено: {len(df_raw):,} строк")

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
    print(f"✓")

print(f"\nПредпосчитано {len(cached_data)} вариантов окна.")

# ВЕКТОРИЗОВАННЫЙ КЛАССИФИКАТОР
def prepare_binned_data(df_windowed, n_bins):
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

# ВЕКТОРИЗОВАННОЕ ПРЕДСКАЗАНИЕ DCT
def dct_predict_vectorized(model, omegas):
    """Предсказывает Ke для массива omega сразу (без циклов)"""
    x_scaled = (omegas - model['range_min']) / (model['range_max'] - model['range_min']) * (model['N'] - 1)
    
    result = np.full_like(omegas, model['coeffs'][0] / 2, dtype=float)
    for n in range(1, model['N']):
        result += model['coeffs'][n] * np.cos((n / model['N']) * np.pi * (x_scaled + 0.5))
    return result * (1 / model['N'])

# ВЕКТОРИЗОВАННАЯ КЛАССИФИКАЦИЯ С ПАМЯТЬЮ
def classify_with_memory_vectorized(df_validation, dct_models, surfaces, alpha):
    """ Bекторизованная классификация """
    n_surfaces = len(surfaces)
    all_true = []
    all_pred = []
    
    # Группируем по экспериментам
    for exp_num, group in df_validation.groupby('number_exper'):
        # Извлекаем данные как numpy массивы
        omegas = group['omega_robot'].values
        kes = group['Ke'].values
        true_surfs = group['surface'].values
        n_samples = len(omegas)
        
        # Инициализация памяти 
        mem_prob = np.ones(n_surfaces) / n_surfaces
        
        # Матрица вероятностей: [n_samples, n_surfaces]
        probs_matrix = np.zeros((n_samples, n_surfaces))
        
        # Для каждой поверхности считаем expected_ke для ВСЕХ omega сразу
        for i, surface in enumerate(surfaces):
            if surface not in dct_models:
                continue
            model = dct_models[surface]
            expected_ke = dct_predict_vectorized(model, omegas)  # Вектор
            deviation = kes - expected_ke
            probs_matrix[:, i] = norm.pdf(deviation, loc=0, scale=model['std'] + 1e-3)
        
        # Нормализация вероятностей по строкам
        row_sums = probs_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Избегаем деления на 0
        probs_matrix = probs_matrix / row_sums
        
        # Применяем память через кумулятивное сканирование
        # mem_prob[t] = alpha * mem_prob[t-1] + (1-alpha) * probs[t]
        predictions = np.zeros(n_samples, dtype=int)
        
        for t in range(n_samples):
            mem_prob = alpha * mem_prob + (1 - alpha) * probs_matrix[t]
            mem_sum = mem_prob.sum()
            if mem_sum > 0:
                mem_prob = mem_prob / mem_sum
            predictions[t] = np.argmax(mem_prob)
        
        # Преобразуем индексы в названия поверхностей
        pred_surfs = [surfaces[idx] for idx in predictions]
        
        all_true.extend(true_surfs)
        all_pred.extend(pred_surfs)
    
    return all_true, all_pred

def objective(trial):
    window_size = int(trial.suggest_float('window_size', 5.0, 30.0))
    
    n_bins_float = trial.suggest_float('n_bins', 11.0, 101.0)
    n_bins = int(n_bins_float)
    if n_bins % 2 == 0:
        n_bins += 1
    
    dct_coeffs = int(trial.suggest_float('dct_coeffs', 1.0, 10.0))
    alpha = trial.suggest_float('alpha', 0.5, 0.99)
    
    df_windowed = cached_data[window_size]
    df_binned, df_full = prepare_binned_data(df_windowed, n_bins)
    
    if len(df_binned) == 0 or len(df_full) == 0:
        return 0.0
    
    dct_models, surfaces = build_dct_models(df_binned, dct_coeffs)
    
    if len(surfaces) == 0:
        return 0.0
    
    # ВЕКТОРИЗОВАННАЯ КЛАССИФИКАЦИЯ
    true_labels, predicted_labels = classify_with_memory_vectorized(
        df_full, dct_models, surfaces, alpha
    )
    
    if len(true_labels) == 0:
        return 0.0
    
    return accuracy_score(true_labels, predicted_labels)

print("ЗАПУСК OPTUNA (ПОЛНОСТЬЮ ВЕКТОРИЗОВАННЫЙ)")

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, gc_after_trial=True, show_progress_bar=True)

# После study.optimize(...) вставь этот блок:
best_params = study.best_params
window_size = int(best_params['window_size'])
n_bins = int(best_params['n_bins'])
if n_bins % 2 == 0: n_bins += 1
dct_coeffs = int(best_params['dct_coeffs'])
alpha = best_params['alpha']

print(f"\nЛучшие параметры: window={window_size}, bins={n_bins}, coeffs={dct_coeffs}, alpha={alpha}")

df_windowed = cached_data[window_size]
df_binned, df_trimmed = prepare_binned_data(df_windowed, n_bins)
dct_models_optuna, surfaces_optuna = build_dct_models(df_binned, dct_coeffs)

# Считаем std как в старом коде
dct_models_std_optuna = {}
for surface in surfaces_optuna:
    surf_data = df_binned[df_binned['surface'] == surface].sort_values('omega_bin')
    std_ke = surf_data['std_Ke'].fillna(0).values
    dct_models_std_optuna[surface] = np.nanmean(std_ke)

# Сохраняем в формате, совместимом со старым кодом
import pickle
with open(r"C:\UIRS\surface-classification\dct_models_from_optuna.pkl", 'wb') as f:
    pickle.dump((dct_models_optuna, dct_models_std_optuna), f)

# Сохраняем данные для классификатора
df_trimmed.to_csv(r"C:\UIRS\surface-classification\data_from_optuna.csv", index=False)
print("Модели и данные из Optuna сохранены.")

print("РЕЗУЛЬТАТЫ")
print(f"\nЛучшая точность: {study.best_value:.4f}")
print(f"\nЛучшие гиперпараметры:")
for param, value in study.best_params.items():
    print(f"  {param}: {value}")

results_df = pd.DataFrame([
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
results_df.to_csv(r"C:\UIRS\surface-classification\optuna_results.csv", index=False)
print(f"\nРезультаты сохранены: optuna_results.csv")