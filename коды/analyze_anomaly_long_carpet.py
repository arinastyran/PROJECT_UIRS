import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", context="talk")

INPUT_FILE = r"C:\UIRS\surface-classification\combined_processed_4W.csv"
OUTPUT_DIR = r"C:\UIRS\surface-classification\anomaly_analysis\detailed_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(" Загрузка данных...")
df = pd.read_csv(INPUT_FILE)

# Расчет признаков
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

df['total_load'] = df[['wheel_load.1', 'wheel_load.2', 'wheel_load.3', 'wheel_load.4']].sum(axis=1)


TARGET_SURFACE = 'linoleum'

print(f"\n Анализ поверхности: {TARGET_SURFACE} (берем самые контрастные эксперименты)")
df_surf = df[df['surface'] == TARGET_SURFACE].copy()

# Средний Ke для каждого эксперимента
exp_stats = df_surf.groupby('number_exper').agg(
    mean_Ke=('Ke', 'mean'),
    std_Ke=('Ke', 'std'),
    mean_load=('total_load', 'mean'),
    std_load=('total_load', 'std'),
    duration=('Time', lambda x: x.max() - x.min()),
    mean_omega_abs=('omega_robot', lambda x: np.abs(x).mean()),
    max_omega=('omega_robot', lambda x: np.abs(x).max())
).reset_index()

# Делим на группы по медиане
median_ke = exp_stats['mean_Ke'].median()
exp_stats['group'] = exp_stats['mean_Ke'].apply(lambda x: 'LOW_Ke' if x < median_ke else 'HIGH_Ke')

print(f"Медиана Ke: {median_ke:.3f}")
print(f"LOW_Ke экспериментов: {(exp_stats['group'] == 'LOW_Ke').sum()}")
print(f"HIGH_Ke экспериментов: {(exp_stats['group'] == 'HIGH_Ke').sum()}")

# БЕРЕМ САМЫЕ КРАЙНИЕ (контрастные) эксперименты
low_exp = exp_stats[exp_stats['group'] == 'LOW_Ke'].sort_values('mean_Ke').iloc[0]  # Минимальный Ke
high_exp = exp_stats[exp_stats['group'] == 'HIGH_Ke'].sort_values('mean_Ke').iloc[-1]  # Максимальный Ke

low_exp_num = int(low_exp['number_exper'])
high_exp_num = int(high_exp['number_exper'])

print(f"\nВыбранные КОНТРАСТНЫЕ эксперименты:")
print(f"  LOW_Ke:  #{low_exp_num}, mean_Ke={low_exp['mean_Ke']:.3f}, duration={low_exp['duration']:.1f}с")
print(f"  HIGH_Ke: #{high_exp_num}, mean_Ke={high_exp['mean_Ke']:.3f}, duration={high_exp['duration']:.1f}с")

df_low = df_surf[df_surf['number_exper'] == low_exp_num]
df_high = df_surf[df_surf['number_exper'] == high_exp_num]


fig, axes = plt.subplots(4, 2, figsize=(18, 16))

# 1. Ke vs Time
axes[0, 0].plot(df_low['Time'], df_low['Ke'], alpha=0.7, label=f'LOW #{low_exp_num} (Ke={low_exp["mean_Ke"]:.2f})')
axes[0, 0].plot(df_high['Time'], df_high['Ke'], alpha=0.7, label=f'HIGH #{high_exp_num} (Ke={high_exp["mean_Ke"]:.2f})')
axes[0, 0].set_title('Коэффициент Ke во времени', fontsize=14)
axes[0, 0].set_ylabel('Ke')
axes[0, 0].legend()

# 2. Total Load vs Time
axes[1, 0].plot(df_low['Time'], df_low['total_load'], alpha=0.7, label=f'LOW #{low_exp_num}')
axes[1, 0].plot(df_high['Time'], df_high['total_load'], alpha=0.7, label=f'HIGH #{high_exp_num}')
axes[1, 0].set_title('Суммарная нагрузка во времени', fontsize=14)
axes[1, 0].set_ylabel('total_load')
axes[1, 0].legend()

# 3. Omega vs Time
axes[2, 0].plot(df_low['Time'], df_low['omega_robot'], alpha=0.7, label=f'LOW #{low_exp_num}')
axes[2, 0].plot(df_high['Time'], df_high['omega_robot'], alpha=0.7, label=f'HIGH #{high_exp_num}')
axes[2, 0].set_title('Угловая скорость ω во времени', fontsize=14)
axes[2, 0].set_ylabel('omega_robot')
axes[2, 0].set_xlabel('Time (с)')
axes[2, 0].legend()

# 4. Ke vs Omega (scatter)
axes[3, 0].scatter(df_low['omega_robot'], df_low['Ke'], alpha=0.3, s=10, label=f'LOW #{low_exp_num}')
axes[3, 0].scatter(df_high['omega_robot'], df_high['Ke'], alpha=0.3, s=10, label=f'HIGH #{high_exp_num}')
axes[3, 0].set_title('Ke vs omega_robot', fontsize=14)
axes[3, 0].set_ylabel('Ke')
axes[3, 0].set_xlabel('omega_robot')
axes[3, 0].legend()

# 5. Распределение Ke (гистограмма)
axes[0, 1].hist(df_low['Ke'], bins=50, alpha=0.5, label=f'LOW #{low_exp_num}', color='blue')
axes[0, 1].hist(df_high['Ke'], bins=50, alpha=0.5, label=f'HIGH #{high_exp_num}', color='red')
axes[0, 1].set_title('Распределение Ke', fontsize=14)
axes[0, 1].set_xlabel('Ke')
axes[0, 1].legend()

# 6. Распределение omega
axes[1, 1].hist(df_low['omega_robot'], bins=50, alpha=0.5, label=f'LOW #{low_exp_num}', color='blue')
axes[1, 1].hist(df_high['omega_robot'], bins=50, alpha=0.5, label=f'HIGH #{high_exp_num}', color='red')
axes[1, 1].set_title('Распределение omega', fontsize=14)
axes[1, 1].set_xlabel('omega_robot')
axes[1, 1].legend()

# 7. Распределение нагрузки
axes[2, 1].hist(df_low['total_load'], bins=50, alpha=0.5, label=f'LOW #{low_exp_num}', color='blue')
axes[2, 1].hist(df_high['total_load'], bins=50, alpha=0.5, label=f'HIGH #{high_exp_num}', color='red')
axes[2, 1].set_title('Распределение нагрузки', fontsize=14)
axes[2, 1].set_xlabel('total_load')
axes[2, 1].legend()

# 8. Сводная статистика
axes[3, 1].axis('off')
stats_text = f"""
СРАВНЕНИЕ КОНТРАСТНЫХ ЭКСПЕРИМЕНТОВ ({TARGET_SURFACE})

LOW_Ke  эксперимент #{low_exp_num}:
  Средний Ke:      {low_exp['mean_Ke']:.3f}
  Std Ke:          {low_exp['std_Ke']:.3f}
  Средняя нагрузка:{low_exp['mean_load']:.3f}
  Std нагрузки:    {low_exp['std_load']:.3f}
  Длительность:    {low_exp['duration']:.1f} с
  Средний |omega|: {low_exp['mean_omega_abs']:.3f}
  Макс |omega|:    {low_exp['max_omega']:.3f}

HIGH_Ke эксперимент #{high_exp_num}:
  Средний Ke:      {high_exp['mean_Ke']:.3f}
  Std Ke:          {high_exp['std_Ke']:.3f}
  Средняя нагрузка:{high_exp['mean_load']:.3f}
  Std нагрузки:    {high_exp['std_load']:.3f}
  Длительность:    {high_exp['duration']:.1f} с
  Средний |omega|: {high_exp['mean_omega_abs']:.3f}
  Макс |omega|:    {high_exp['max_omega']:.3f}

РАЗЛИЧИЯ:
  Ke:              {((high_exp['mean_Ke'] - low_exp['mean_Ke']) / low_exp['mean_Ke'] * 100):+.1f}%
  Нагрузка:        {((high_exp['mean_load'] - low_exp['mean_load']) / low_exp['mean_load'] * 100):+.1f}%
  Длительность:    {((high_exp['duration'] - low_exp['duration']) / low_exp['duration'] * 100):+.1f}%
  Средний |omega|: {((high_exp['mean_omega_abs'] - low_exp['mean_omega_abs']) / low_exp['mean_omega_abs'] * 100):+.1f}%
"""
axes[3, 1].text(0.05, 0.95, stats_text, fontsize=10, family='monospace', 
                verticalalignment='top', transform=axes[3, 1].transAxes)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'anomaly_contrast_{TARGET_SURFACE}.png'), dpi=300, bbox_inches='tight')
print(f"\n График сохранен: {OUTPUT_DIR}/anomaly_contrast_{TARGET_SURFACE}.png")

# Статистический тест
t_stat, p_value = stats.ttest_ind(df_low['Ke'], df_high['Ke'])
print(f"\n t-тест: t={t_stat:.3f}, p={p_value:.2e}")
print(f"{' Различие статистически значимо' if p_value < 0.05 else ' Различие незначимо'}")
