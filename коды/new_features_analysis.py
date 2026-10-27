import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import os
import warnings
warnings.filterwarnings('ignore')

# Настройки стиля
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (16, 9)

INPUT_FILE = r"C:\UIRS\surface-classification\combined_processed_4W.csv"
OUTPUT_DIR = r"C:\UIRS\surface-classification\new_features_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Загрузка сырых данных...")
df_raw = pd.read_csv(INPUT_FILE)
print(f"Загружено: {len(df_raw):,} строк")

print("\nРасчет новых признаков...")

N_nominal, R, L = 0.11, 0.031, 0.159

# Базовые массивы 
loads = df_raw[['wheel_load.1', 'wheel_load.2', 'wheel_load.3', 'wheel_load.4']].values
vels = df_raw[['wheel_angular_velocity.1', 'wheel_angular_velocity.2', 
               'wheel_angular_velocity.3', 'wheel_angular_velocity.4']].values
rows = np.arange(len(df_raw))[:, None]

# Суммарная нагрузка и мощность (все 4 колеса)
df_raw['total_load'] = loads.sum(axis=1)
df_raw['total_power'] = (loads * vels).sum(axis=1)

# Угловая скорость робота (все 4 колеса)
v_left_all = R * (vels[:, 0] + vels[:, 3]) / 2
v_right_all = R * (vels[:, 1] + vels[:, 2]) / 2
df_raw['omega_robot'] = (v_left_all - v_right_all) / L

# Функция для расчета угловой скорости по выбранным 2 колесам
def calc_omega_from_2_wheels(velocities, indices):
    selected_vels = velocities[rows, indices] # (N, 2)
    is_left = (indices == 0) | (indices == 3)
    is_right = (indices == 1) | (indices == 2)
    
    n_left = is_left.sum(axis=1)
    n_right = is_right.sum(axis=1)
    
    vels_left = np.where(is_left, selected_vels, 0).sum(axis=1)
    vels_right = np.where(is_right, selected_vels, 0).sum(axis=1)
    
    avg_v_left = np.where(n_left > 0, vels_left / n_left, 0)
    avg_v_right = np.where(n_right > 0, vels_right / n_right, 0)
    
    return (R * avg_v_left - R * avg_v_right) / L

# Признаки для 2 колес с макс нагрузкой
idx_top2 = np.argsort(loads, axis=1)[:, -2:] # Индексы 2 макс нагрузок
loads_top2 = loads[rows, idx_top2]
vels_top2 = vels[rows, idx_top2]

df_raw['power_top2'] = (loads_top2 * vels_top2).sum(axis=1)
denom_top2 = N_nominal * np.abs(vels_top2).sum(axis=1)
df_raw['Ke_top2'] = np.where(denom_top2 > 0, df_raw['power_top2'] / denom_top2, 0)
df_raw['omega_top2'] = calc_omega_from_2_wheels(vels, idx_top2)

# Признаки для 2 колес с мин нагрузкой
idx_bottom2 = np.argsort(loads, axis=1)[:, :2] # Индексы 2 мин нагрузок
loads_bottom2 = loads[rows, idx_bottom2]
vels_bottom2 = vels[rows, idx_bottom2]

df_raw['power_bottom2'] = (loads_bottom2 * vels_bottom2).sum(axis=1)
denom_bottom2 = N_nominal * np.abs(vels_bottom2).sum(axis=1)
df_raw['Ke_bottom2'] = np.where(denom_bottom2 > 0, df_raw['power_bottom2'] / denom_bottom2, 0)
df_raw['omega_bottom2'] = calc_omega_from_2_wheels(vels, idx_bottom2)

# Базовый Ke для сравнения
numerator_all = (loads * vels).sum(axis=1)
denominator_all = N_nominal * np.abs(vels).sum(axis=1)
df_raw['Ke_all'] = np.where(denominator_all > 0, numerator_all / denominator_all, 0)

print("Все признаки рассчитаны")

features = {
    'Ke_all': 'Базовый Ke (все колеса) - для сравнения',
    'total_load': 'Суммарная нагрузка (все колеса)',
    'total_power': 'Суммарная мощность (все колеса)',
    'power_top2': 'Мощность по 2 макс нагруженным колесам',
    'power_bottom2': 'Мощность по 2 мин нагруженным колесам',
    'Ke_top2': 'Ke по 2 макс нагруженным колесам',
    'Ke_bottom2': 'Ke по 2 мин нагруженным колесам',
    'omega_robot': 'Угловая скорость робота (все колеса)',
    'omega_top2': 'Угловая скорость (по 2 макс колесам)',
    'omega_bottom2': 'Угловая скорость (по 2 мин колесам)'
}

print("\nПостроение boxplotов...")

palette_10 = {
    'ceramic_tiles': '#1f77b4', 'pvc_foamboard': '#ff7f0e', 'linoleum': '#2ca02c',
    'laminate_flooring': '#d62728', 'short_carpet': '#9467bd', 'long_carpet': '#8c564b',
    'eva_foam_tiles': '#e377c2', 'artificial_grass': '#7f7f7f', 'osb': '#bcbd22', 'foam_underlayment': '#17becf'
}

for feature_name, feature_label in features.items():
    print(f"  {feature_label}...")
    
    plt.figure(figsize=(16, 9))
    
    sns.boxplot(
        data=df_raw, 
        x='surface', 
        y=feature_name,
        showfliers=False,
        palette=palette_10,
        order=sorted(df_raw['surface'].unique())
    )
    
    plt.title(f'Статистический разброс: {feature_label}', fontsize=16)
    plt.xlabel('Поверхность', fontsize=14)
    plt.ylabel(feature_label, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, f'boxplot_10classes_{feature_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

print("Все boxplotы сохранены")

print("\nРасчет overlap для каждого признака...")

def calculate_overlap_10(df, feature):
    classes = sorted(df['surface'].unique())
    n_classes = len(classes)
    
    kde_dict = {}
    for cls in classes:
        cls_data = df[df['surface'] == cls][feature].dropna().values
        if len(cls_data) > 1:
            kde_dict[cls] = gaussian_kde(cls_data)
    
    overlap_matrix = np.zeros((n_classes, n_classes))
    
    # Берем диапазон от минимума до максимума по всему признаку
    x_min, x_max = df[feature].min(), df[feature].max()
    x = np.linspace(x_min, x_max, 1000)
    
    for i, cls1 in enumerate(classes):
        for j, cls2 in enumerate(classes):
            if i == j:
                overlap_matrix[i, j] = 1.0
            else:
                if cls1 in kde_dict and cls2 in kde_dict:
                    kde1 = kde_dict[cls1](x)
                    kde2 = kde_dict[cls2](x)
                    overlap_matrix[i, j] = np.trapezoid(np.minimum(kde1, kde2), x)
    
    mask = ~np.eye(n_classes, dtype=bool)
    avg_overlap = overlap_matrix[mask].mean()
    max_overlap = overlap_matrix[mask].max()
    
    # Находим худшую пару
    max_idx = np.unravel_index(np.argmax(overlap_matrix * mask), overlap_matrix.shape)
    worst_pair = f"{classes[max_idx[0]]} <-> {classes[max_idx[1]]}"
    
    return avg_overlap, max_overlap, worst_pair

overlap_results = []

for feature_name, feature_label in features.items():
    avg_ov, max_ov, worst = calculate_overlap_10(df_raw, feature_name)
    overlap_results.append({
        'Признак': feature_label,
        'Средний overlap': f"{avg_ov*100:.2f}%",
        'Максимальный overlap': f"{max_ov*100:.2f}%",
        'Худшая пара': worst
    })
    print(f"  {feature_label}: avg={avg_ov*100:.2f}%, max={max_ov*100:.2f}% ({worst})")

print("ИТОГОВАЯ ТАБЛИЦА")

df_overlap = pd.DataFrame(overlap_results)
print(df_overlap.to_string(index=False))

df_overlap.to_csv(os.path.join(OUTPUT_DIR, 'features_overlap_10classes.csv'), index=False)
print(f"\nТаблица сохранена: features_overlap_10classes.csv")

# Находим самый перспективный признак (минимальный max overlap, исключая базовый Ke)
df_overlap['max_overlap_num'] = df_overlap['Максимальный overlap'].str.replace('%', '').astype(float)
# Исключаем базовый Ke из конкуренции, чтобы найти лучший новый признак
candidates = df_overlap[df_overlap['Признак'] != 'Базовый Ke (все колеса) - для сравнения']
best_feature = candidates.loc[candidates['max_overlap_num'].idxmin()]

print(f"САМЫЙ ПЕРСПЕКТИВНЫЙ НОВЫЙ ПРИЗНАК: {best_feature['Признак']}")
print(f"Максимальный overlap: {best_feature['Максимальный overlap']} (пара: {best_feature['Худшая пара']})")
