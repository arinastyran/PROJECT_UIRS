import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path(r"C:\UIRS\surface-classification\filtered_4W_window10_step1.csv")
print(f"\n[1] Загрузка данных из: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"    Загружено: {len(df):,} строк")

# Расчет Ke и omega
print("\n[2] Расчет Ke и omega...")

N_nominal = 0.11
R = 0.031
L = 0.159

numerator = (
    df['wheel_load.1'] * df['wheel_angular_velocity.1'] +
    df['wheel_load.2'] * df['wheel_angular_velocity.2'] +
    df['wheel_load.3'] * df['wheel_angular_velocity.3'] +
    df['wheel_load.4'] * df['wheel_angular_velocity.4']
)
denominator = N_nominal * (
    df['wheel_angular_velocity.1'].abs() +
    df['wheel_angular_velocity.2'].abs() +
    df['wheel_angular_velocity.3'].abs() +
    df['wheel_angular_velocity.4'].abs()
)
denominator = denominator.replace(0, np.nan)
df['Ke'] = (numerator / denominator).replace([np.inf, -np.inf], np.nan).fillna(0)

v_left = R * (df['wheel_angular_velocity.1'] + df['wheel_angular_velocity.4']) / 2
v_right = R * (df['wheel_angular_velocity.2'] + df['wheel_angular_velocity.3']) / 2
df['omega_robot'] = (v_left - v_right) / L

print(f"    Ke: min={df['Ke'].min():.4f}, max={df['Ke'].max():.4f}, mean={df['Ke'].mean():.4f}")
print(f"    Omega: min={df['omega_robot'].min():.4f}, max={df['omega_robot'].max():.4f} рад/с")

# Обрезка по симметричному диапазону omega 
print("\n[3] Обрезка по симметричному диапазону omega...")

omega_stats = df.groupby('surface')['omega_robot'].agg(['min', 'max'])
omega_min = omega_stats['min'].max()
omega_max = omega_stats['max'].min()

omega_abs_max = min(abs(omega_min), abs(omega_max))
omega_min_sym = -omega_abs_max
omega_max_sym = omega_abs_max

print(f"    Исходный диапазон: [{omega_min:.4f}, {omega_max:.4f}]")
print(f"    Симметричный диапазон: [{omega_min_sym:.4f}, {omega_max_sym:.4f}]")

df_trimmed = df[(df['omega_robot'] >= omega_min_sym) & (df['omega_robot'] <= omega_max_sym)].copy()
print(f"    После обрезки: {len(df_trimmed):,} строк ({100*len(df_trimmed)/len(df):.1f}%)")

# Сохранение ПРОМЕЖУТОЧНОГО файла (ПОСЛЕ обрезки!)
INTERMEDIATE_FILE = Path(r"C:\UIRS\surface-classification\data_with_ke_omega.csv")
print(f"\n[4] Сохранение промежуточного файла (ОБРЕЗАННОГО): {INTERMEDIATE_FILE}")
df_trimmed.to_csv(INTERMEDIATE_FILE, index=False)
print(f"    Сохранено {len(df_trimmed):,} строк")
print(f"    Этот файл будет использоваться для классификатора и ML моделей")

# Второе скользящее окно
print("\n[5] Применение второго скользящего окна (window=5, step=5)...")

WINDOW_SIZE = 5
STEP_SIZE = 5

all_windows = []
groups = df_trimmed.groupby(['surface', 'number_exper'], sort=False)

for (surface, exp_num), group in groups:
    group = group.sort_values('Time').reset_index(drop=True)
    
    if len(group) < WINDOW_SIZE:
        continue
    
    for start_idx in range(0, len(group) - WINDOW_SIZE + 1, STEP_SIZE):
        end_idx = start_idx + WINDOW_SIZE
        window = group.iloc[start_idx:end_idx]
        
        row = {
            'surface': surface,
            'number_exper': exp_num,
            'wheels': window['wheels'].iloc[0],
            'Time': window['Time'].mean(),
            'Ke': window['Ke'].mean(),
            'omega_robot': window['omega_robot'].mean(),
        }
        
        all_windows.append(row)

df_windowed = pd.DataFrame(all_windows)
print(f"    После второго окна: {len(df_windowed):,} строк")

# Группировка по интервалам omega
print("\n[6] Группировка по интервалам omega...")

n_bins = 50
bin_edges = np.linspace(omega_min_sym, omega_max_sym, n_bins + 1)
df_windowed['omega_bin'] = pd.cut(
    df_windowed['omega_robot'], 
    bins=bin_edges, 
    labels=False, 
    include_lowest=True
)

final_data = df_windowed.groupby(['surface', 'omega_bin']).agg(
    median_Ke=('Ke', 'median'),
    std_Ke=('Ke', 'std'),
    count=('Ke', 'count')
).reset_index()

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
final_data['omega_center'] = final_data['omega_bin'].map(lambda x: bin_centers[int(x)])
final_data = final_data.sort_values(['surface', 'omega_bin'])

print(f"    Итого: {len(final_data)} строк (10 поверхностей × {n_bins} интервалов)")

# Сохранение финального файла для DCT-II
OUTPUT_FILE = Path(r"C:\UIRS\surface-classification\dct2_ready_data.csv")
print(f"\n[7] Сохранение финального файла: {OUTPUT_FILE}")
final_data.to_csv(OUTPUT_FILE, index=False)

print(f"\nСозданные файлы:")
print(f"  1. {INTERMEDIATE_FILE.name}")
print(f"     - {len(df_trimmed):,} строк (ОБРЕЗАННЫЕ данные)")
print(f"     - Содержит: все исходные колонки + Ke + omega_robot")
print(f"     - Нужен для: классификатора и ML моделей")
print(f"\n  2. {OUTPUT_FILE.name}")
print(f"     - {len(final_data)} строк")
print(f"     - Содержит: surface, omega_bin, median_Ke, std_Ke, omega_center")
print(f"     - Нужен для: построения моделей DCT-II и графиков")