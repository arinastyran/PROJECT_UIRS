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

# Расчет Ke и omega
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

# Суммарная нагрузка (для анализа)
df['total_load'] = df[['wheel_load.1', 'wheel_load.2', 'wheel_load.3', 'wheel_load.4']].sum(axis=1)


TARGET_SURFACE = 'pvc_foamboard'  # 'eva_foam_tiles', 'long_carpet' linoleum foam_underlayment

print(f"\n Анализ поверхности: {TARGET_SURFACE}")
df_surf = df[df['surface'] == TARGET_SURFACE].copy()

# Считаем средний Ke для каждого эксперимента
exp_stats = df_surf.groupby('number_exper')['Ke'].mean().reset_index()
exp_stats.rename(columns={'Ke': 'mean_Ke'}, inplace=True)

# Находим медиану Ke и делим эксперименты на две группы
median_ke = exp_stats['mean_Ke'].median()
exp_stats['group'] = exp_stats['mean_Ke'].apply(lambda x: 'LOW_Ke' if x < median_ke else 'HIGH_Ke')

print(f"Медиана Ke: {median_ke:.3f}")
print(f"Экспериментов с низким Ke: {(exp_stats['group'] == 'LOW_Ke').sum()}")
print(f"Экспериментов с высоким Ke: {(exp_stats['group'] == 'HIGH_Ke').sum()}")

# ВЫБОР ДВУХ ПРЕДСТАВИТЕЛЬНЫХ ЭКСПЕРИМЕНТОВ
# Берем эксперимент с медианным Ke из каждой группы
low_exp = exp_stats[exp_stats['group'] == 'LOW_Ke'].sort_values('mean_Ke').iloc[len(exp_stats[exp_stats['group'] == 'LOW_Ke'])//2]
high_exp = exp_stats[exp_stats['group'] == 'HIGH_Ke'].sort_values('mean_Ke').iloc[len(exp_stats[exp_stats['group'] == 'HIGH_Ke'])//2]

low_exp_num = int(low_exp['number_exper'])
high_exp_num = int(high_exp['number_exper'])

print(f"\nВыбранные эксперименты:")
print(f"  LOW_Ke:  number_exper={low_exp_num}, mean_Ke={low_exp['mean_Ke']:.3f}")
print(f"  HIGH_Ke: number_exper={high_exp_num}, mean_Ke={high_exp['mean_Ke']:.3f}")

# ПОСТРОЕНИЕ ГРАФИКОВ ВО ВРЕМЕНИ

df_low = df_surf[df_surf['number_exper'] == low_exp_num]
df_high = df_surf[df_surf['number_exper'] == high_exp_num]

fig, axes = plt.subplots(3, 2, figsize=(16, 12))

# Ke vs Time
axes[0, 0].plot(df_low['Time'], df_low['Ke'], alpha=0.7, label=f'LOW_Ke (exp {low_exp_num})')
axes[0, 0].plot(df_high['Time'], df_high['Ke'], alpha=0.7, label=f'HIGH_Ke (exp {high_exp_num})')
axes[0, 0].set_title('Коэффициент Ke во времени', fontsize=14)
axes[0, 0].set_ylabel('Ke')
axes[0, 0].legend()

# Total Load vs Time
axes[1, 0].plot(df_low['Time'], df_low['total_load'], alpha=0.7, label=f'LOW_Ke (exp {low_exp_num})')
axes[1, 0].plot(df_high['Time'], df_high['total_load'], alpha=0.7, label=f'HIGH_Ke (exp {high_exp_num})')
axes[1, 0].set_title('Суммарная нагрузка во времени', fontsize=14)
axes[1, 0].set_ylabel('total_load')
axes[1, 0].legend()

# Omega vs Time
axes[2, 0].plot(df_low['Time'], df_low['omega_robot'], alpha=0.7, label=f'LOW_Ke (exp {low_exp_num})')
axes[2, 0].plot(df_high['Time'], df_high['omega_robot'], alpha=0.7, label=f'HIGH_Ke (exp {high_exp_num})')
axes[2, 0].set_title('Угловая скорость робота во времени', fontsize=14)
axes[2, 0].set_ylabel('omega_robot')
axes[2, 0].set_xlabel('Time (с)')
axes[2, 0].legend()

# Ke vs Omega (scatter)
axes[0, 1].scatter(df_low['omega_robot'], df_low['Ke'], alpha=0.3, s=10, label=f'LOW_Ke (exp {low_exp_num})')
axes[0, 1].scatter(df_high['omega_robot'], df_high['Ke'], alpha=0.3, s=10, label=f'HIGH_Ke (exp {high_exp_num})')
axes[0, 1].set_title('Ke vs omega_robot', fontsize=14)
axes[0, 1].set_ylabel('Ke')
axes[0, 1].legend()

# Total Load vs Omega
axes[1, 1].scatter(df_low['omega_robot'], df_low['total_load'], alpha=0.3, s=10, label=f'LOW_Ke (exp {low_exp_num})')
axes[1, 1].scatter(df_high['omega_robot'], df_high['total_load'], alpha=0.3, s=10, label=f'HIGH_Ke (exp {high_exp_num})')
axes[1, 1].set_title('Total Load vs omega_robot', fontsize=14)
axes[1, 1].set_ylabel('total_load')
axes[1, 1].legend()

# Статистика
axes[2, 1].axis('off')
stats_text = f"""
СТАТИСТИКА:

LOW_Ke  эксперимент #{low_exp_num}:
  Средний Ke: {df_low['Ke'].mean():.3f}
  Средняя нагрузка: {df_low['total_load'].mean():.3f}
  Диапазон omega: [{df_low['omega_robot'].min():.2f}, {df_low['omega_robot'].max():.2f}]

HIGH_Ke эксперимент #{high_exp_num}:
  Средний Ke: {df_high['Ke'].mean():.3f}
  Средняя нагрузка: {df_high['total_load'].mean():.3f}
  Диапазон omega: [{df_high['omega_robot'].min():.2f}, {df_high['omega_robot'].max():.2f}]

Различия:
  Ke: {((df_high['Ke'].mean() - df_low['Ke'].mean()) / df_low['Ke'].mean() * 100):.1f}%
  Нагрузка: {((df_high['total_load'].mean() - df_low['total_load'].mean()) / df_low['total_load'].mean() * 100):.1f}%
"""
axes[2, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace', verticalalignment='center')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'anomaly_comparison_{TARGET_SURFACE}.png'), dpi=300, bbox_inches='tight')
print(f"\n График сравнения сохранен: {OUTPUT_DIR}/anomaly_comparison_{TARGET_SURFACE}.png")
print("СТАТИСТИЧЕСКИЙ ТЕСТ (t-тест Стьюдента)")

t_stat, p_value = stats.ttest_ind(df_low['Ke'], df_high['Ke'])
print(f"t-статистика: {t_stat:.3f}")
print(f"p-value: {p_value:.2e}")

if p_value < 0.05:
    print(f" РАЗЛИЧИЕ СТАТИСТИЧЕСКИ ЗНАЧИМО (p < 0.05)")
    print(f"   Две группы экспериментов действительно имеют разные распределения Ke!")
else:
    print(f"  Различие НЕ статистически значимо (p >= 0.05)")
