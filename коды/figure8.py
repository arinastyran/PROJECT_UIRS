import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = r"C:\UIRS\surface-classification\dct2_ready_data.csv"
df = pd.read_csv(INPUT_FILE)

# ФИКСИРОВАННАЯ ЦВЕТОВАЯ ПАЛИТРА ДЛЯ 10 ПОВЕРХНОСТЕЙ
SURFACE_COLORS = {
    'artificial_grass':     '#1f77b4',  # синий
    'ceramic_tiles':        '#ff7f0e',  # оранжевый
    'eva_foam_tiles':       '#2ca02c',  # зелёный
    'foam_underlayment':    '#d62728',  # красный
    'laminate_flooring':    '#9467bd',  # фиолетовый
    'linoleum':             '#8c564b',  # коричневый
    'long_carpet':          '#e377c2',  # розовый
    'osb':                  '#7f7f7f',  # серый
    'pvc_foamboard':        '#bcbd22',  # оливковый
    'short_carpet':         '#17becf',  # бирюзовый
}

# Сортируем поверхности для стабильного порядка в легенде
surfaces = sorted(df['surface'].unique())

# ВАРИАНТ 1: График с ±std 
fig, ax = plt.subplots(figsize=(14, 7))
for surface in surfaces:
    surf_data = df[df['surface'] == surface].sort_values('omega_bin')
    color = SURFACE_COLORS[surface]
    label = surface.replace('_', ' ') 
    # Закрашенная область стандартного отклонения
    ax.fill_between(
        surf_data['omega_center'],
        surf_data['median_Ke'] - surf_data['std_Ke'],
        surf_data['median_Ke'] + surf_data['std_Ke'],
        color=color, alpha=0.15, zorder=1
    )
    # Линия медианы
    ax.plot(
        surf_data['omega_center'],
        surf_data['median_Ke'],
        color=color, linewidth=2, linestyle='--',
        label=label, zorder=2
    )
ax.set_xlabel('Угловая скорость робота, рад/с ($\omega$)', fontsize=12)
ax.set_ylabel('Коэффициент энергопотребления $K_e$', fontsize=12)
ax.set_title('Зависимость медианного $K_e$ от угловой скорости (с ±std)', fontsize=14, pad=15)
ax.grid(True, alpha=0.3, linestyle='--')

ax.legend(
    title='Surface type',
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    fontsize=10,
    frameon=True
)
plt.tight_layout()
plt.savefig(r"C:\UIRS\surface-classification\ke_vs_omega_with_std.png", dpi=300, bbox_inches='tight')
plt.show()

# График без std (только линии)
fig, ax = plt.subplots(figsize=(14, 7))
for surface in surfaces:
    surf_data = df[df['surface'] == surface].sort_values('omega_bin')
    color = SURFACE_COLORS[surface]
    label = surface.replace('_', ' ')
    
    ax.plot(
        surf_data['omega_center'],
        surf_data['median_Ke'],
        color=color, linewidth=2.5, linestyle='-',
        label=label
    )
ax.set_xlabel('Угловая скорость робота, рад/с ($\omega$)', fontsize=12)
ax.set_ylabel('Коэффициент энергопотребления $K_e$', fontsize=12)
ax.set_title('Зависимость медианного $K_e$ от угловой скорости (без std)', fontsize=14, pad=15)
ax.grid(True, alpha=0.3, linestyle='--')

ax.legend(
    title='Surface type',
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    fontsize=10,
    frameon=True
)

plt.tight_layout()
plt.savefig(r"C:\UIRS\surface-classification\ke_vs_omega_no_std.png", dpi=300, bbox_inches='tight')
plt.show()
