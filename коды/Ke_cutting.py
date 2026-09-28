import pandas as pd
import numpy as np
from pathlib import Path

WINDOW_SIZE = 10   # Ширина первого скользящего окна
N_BINS = 51        # Количество интервалов для разбивания

INPUT_FILE = Path(r"C:\UIRS\surface-classification\combined_processed_4W.csv")
OUTPUT_INTERMEDIATE = Path(r"C:\UIRS\surface-classification\data_with_ke_omega.csv")
OUTPUT_FINAL = Path(r"C:\UIRS\surface-classification\dct2_ready_data.csv")

print(f"Загрузка данных из {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
print(f"  Загружено: {len(df):,} строк")

# 1. Первое скользящее окно
print(f"\n[1] Применение скользящего окна (size={WINDOW_SIZE}, step=1)...")

# Колонки для усреднения
exclude_cols = {'Unnamed: 0', 'Time', 'surface', 'wheels', 'number_exper'}
numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                if col not in exclude_cols]

print(f"  Усредняем {len(numeric_cols)} числовых колонок (wheel_load, wheel_angular_velocity и др.)")

all_windows = []
groups = df.groupby(['surface', 'number_exper'], sort=False)

for (surface, exp_num), group in groups:
    group = group.sort_values('Time').reset_index(drop=True)
    
    if len(group) < WINDOW_SIZE:
        continue
    
    for start_idx in range(0, len(group) - WINDOW_SIZE + 1, 1):
        end_idx = start_idx + WINDOW_SIZE
        window = group.iloc[start_idx:end_idx]
        
        row = {
            'surface': surface,
            'number_exper': exp_num,
            'wheels': window['wheels'].iloc[0],
            'Time': window['Time'].iloc[-1],  # Последнее время в окне
        }

        for col in numeric_cols:
            row[col] = window[col].mean()
        
        all_windows.append(row)

df_windowed = pd.DataFrame(all_windows)
print(f"  После окна: {len(df_windowed):,} строк")

# 2. Расчет Ke и omega
print("\n[2] Расчет Ke и omega (по усредненным данным после окна)...")
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

print(f"  Ke: min={df_windowed['Ke'].min():.4f}, max={df_windowed['Ke'].max():.4f}, mean={df_windowed['Ke'].mean():.4f}")
print(f"  Omega: min={df_windowed['omega_robot'].min():.4f}, max={df_windowed['omega_robot'].max():.4f} рад/с")

# 3. Симметричная обрезка диапазона omega
print("\n[3] Симметричная обрезка диапазона omega...")
omega_stats = df_windowed.groupby('surface')['omega_robot'].agg(['min', 'max'])
omega_min = omega_stats['min'].max()
omega_max = omega_stats['max'].min()

omega_abs_max = min(abs(omega_min), abs(omega_max))
omega_min_sym = -omega_abs_max
omega_max_sym = omega_abs_max

print(f"  Исходный диапазон: [{omega_min:.4f}, {omega_max:.4f}]")
print(f"  Симметричный диапазон: [{omega_min_sym:.4f}, {omega_max_sym:.4f}]")

df_trimmed = df_windowed[(df_windowed['omega_robot'] >= omega_min_sym) & 
                         (df_windowed['omega_robot'] <= omega_max_sym)].copy()
print(f"  После обрезки: {len(df_trimmed):,} строк ({100*len(df_trimmed)/len(df_windowed):.1f}%)")

# 4. Сохранение промежуточного файла (для классификатора)
print(f"\n[4] Сохранение промежуточного файла: {OUTPUT_INTERMEDIATE}")
df_trimmed.to_csv(OUTPUT_INTERMEDIATE, index=False)
print(f"  Сохранено {len(df_trimmed):,} строк")
print(f"  Этот файл будет использоваться для классификатора и ML моделей")

# 5. Дискретизация по omega
print(f"\n[5] Группировка по {N_BINS} интервалам omega...")

bin_edges = np.linspace(omega_min_sym, omega_max_sym, N_BINS + 1)
df_trimmed['omega_bin'] = pd.cut(
    df_trimmed['omega_robot'], 
    bins=bin_edges, 
    labels=False, 
    include_lowest=True
)

final_data = df_trimmed.groupby(['surface', 'omega_bin']).agg(
    median_Ke=('Ke', 'median'),
    std_Ke=('Ke', 'std'),
    count=('Ke', 'count')
).reset_index()

final_data['std_Ke'] = final_data['std_Ke'].fillna(0)

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
final_data['omega_center'] = final_data['omega_bin'].map(lambda x: bin_centers[int(x)])
final_data = final_data.sort_values(['surface', 'omega_bin'])

print(f"  Итого: {len(final_data)} строк (10 поверхностей × {N_BINS} интервалов)")

# 6. Сохранение финального файла для DCT-II
OUTPUT_FILE = Path(r"C:\UIRS\surface-classification\dct2_ready_data.csv")
print(f"\n[6] Сохранение финального файла: {OUTPUT_FILE}")
final_data.to_csv(OUTPUT_FILE, index=False)

print(f"\nСозданные файлы:")
print(f"  1. {OUTPUT_INTERMEDIATE.name}")
print(f"     - {len(df_trimmed):,} строк (ОБРЕЗАННЫЕ данные после окна)")
print(f"     - Содержит: все исходные колонки + Ke + omega_robot")
print(f"     - Нужен для: классификатора и ML моделей")
print(f"\n  2. {OUTPUT_FILE.name}")
print(f"     - {len(final_data)} строк")
print(f"     - Содержит: surface, omega_bin, median_Ke, std_Ke, omega_center")
print(f"     - Нужен для: построения моделей DCT-II и графиков")
