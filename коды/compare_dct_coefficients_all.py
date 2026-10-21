import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy.fft import dct

WINDOW = 29
BINS = 43
DCT_COEFFS = 10
ALPHA = 0.9872495585564633

# Все варианты разбиения
mappings = {
    '10_classes_original': None,  
    
    '6_classes_ke': {
        'long_carpet': 'Soft_Grass', 'short_carpet': 'Soft_Grass', 'artificial_grass': 'Soft_Grass',
        'osb': 'OSB_Foam', 'foam_underlayment': 'OSB_Foam',
        'eva_foam_tiles': 'EVA_Foam',
        'pvc_foamboard': 'Hard_Smooth', 'ceramic_tiles': 'Hard_Smooth',
        'linoleum': 'Linoleum', 'laminate_flooring': 'Laminate'
    },
    
    '4_classes_ke': {
        'long_carpet': 'Soft_Rough', 'short_carpet': 'Soft_Rough', 'artificial_grass': 'Soft_Rough', 
        'osb': 'Soft_Rough', 'foam_underlayment': 'Soft_Rough',
        'eva_foam_tiles': 'EVA_Foam',
        'pvc_foamboard': 'Hard_Floors', 'ceramic_tiles': 'Hard_Floors', 'linoleum': 'Hard_Floors',
        'laminate_flooring': 'Laminate'
    },
    
    '3_classes_article': {
        'long_carpet': 'Lateral_Slip', 'artificial_grass': 'Lateral_Slip',
        'laminate_flooring': 'Stable', 'short_carpet': 'Stable', 'linoleum': 'Stable', 'ceramic_tiles': 'Stable',
        'pvc_foamboard': 'Longitudinal_Slip', 'osb': 'Longitudinal_Slip', 
        'foam_underlayment': 'Longitudinal_Slip', 'eva_foam_tiles': 'Longitudinal_Slip'
    }
}

print("Загрузка данных...")
df = pd.read_csv(r"C:\UIRS\surface-classification\data_with_ke_omega.csv")
print(f"Загружено: {len(df):,} строк")

# Биннинг (общий для всех вариантов)
omega_stats = df.groupby('surface')['omega_robot'].agg(['min', 'max'])
omega_abs_max = min(abs(omega_stats['min'].max()), abs(omega_stats['max'].min()))
df_trimmed = df[(df['omega_robot'] >= -omega_abs_max) & (df['omega_robot'] <= omega_abs_max)].copy()

bin_edges = np.linspace(-omega_abs_max, omega_abs_max, BINS + 1)
df_trimmed['omega_bin'] = pd.cut(df_trimmed['omega_robot'], bins=bin_edges, labels=False, include_lowest=True)


def build_dct_models(df_binned, dct_coeffs, surface_col='surface'):
    surfaces = sorted(df_binned[surface_col].unique())
    dct_models = {}
    
    for surface in surfaces:
        surf_data = df_binned[df_binned[surface_col] == surface].sort_values('omega_bin')
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
# Алализ коэффициентов
output_dir = Path(r"C:\UIRS\surface-classification\coefficient_analysis")
output_dir.mkdir(exist_ok=True)

all_results = {}

for variant_name, surface_mapping in mappings.items():
    print(f"АНАЛИЗ: {variant_name}")

    # Применяем маппинг если есть
    df_work = df_trimmed.copy()
    if surface_mapping is not None:
        df_work['surface_merged'] = df_work['surface'].map(surface_mapping)
        surface_col = 'surface_merged'
    else:
        surface_col = 'surface'
    
    # Биннинг по укрупнённым классам
    df_binned = df_work.groupby([surface_col, 'omega_bin']).agg(
        median_Ke=('Ke', 'median'), std_Ke=('Ke', 'std')
    ).reset_index()
    df_binned['std_Ke'] = df_binned['std_Ke'].fillna(0)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    df_binned['omega_center'] = df_binned['omega_bin'].map(lambda x: bin_centers[int(x)])
    
    # Строим модели
    dct_models, surfaces = build_dct_models(df_binned, DCT_COEFFS, surface_col)
    print(f"Построено моделей: {len(surfaces)}")
    
    # Извлечение индексов ненулевых коэффициентов
    nonzero_indices = {}
    for surf in surfaces:
        coeffs = dct_models[surf]['coeffs']
        idx = [i for i, c in enumerate(coeffs) if abs(c) > 1e-9]
        nonzero_indices[surf] = sorted(idx)
        print(f"  {surf:<25} | Ненулевые коэф.: {nonzero_indices[surf]}")
    
    # Матрица различий
    n_surfs = len(surfaces)
    z_matrix = np.zeros((n_surfs, n_surfs), dtype=int)
    hover_matrix = [["" for _ in range(n_surfs)] for _ in range(n_surfs)]
    identical_pairs = []
    
    for i in range(n_surfs):
        for j in range(n_surfs):
            set_i = set(nonzero_indices[surfaces[i]])
            set_j = set(nonzero_indices[surfaces[j]])
            diff_set = set_i ^ set_j
            diff_count = len(diff_set)
            diff_list = sorted(list(diff_set))
            
            z_matrix[i, j] = diff_count
            
            if i == j:
                hover_text = f"<b>{surfaces[i]}</b><br>Идентична самой себе<br>Ненулевые коэф.: {nonzero_indices[surfaces[i]]}"
            else:
                hover_text = (
                    f"<b>Сравнение пары:</b><br>"
                    f"1: {surfaces[i]}<br>"
                    f"2: {surfaces[j]}<br>"
                    f"<b>Кол-во отличающихся:</b> {diff_count}<br>"
                    f"<b>Номера отличающихся:</b> {diff_list if diff_list else 'Нет (полностью совпадают)'}"
                )
            hover_matrix[i][j] = hover_text
            
            if i < j and diff_count == 0:
                identical_pairs.append((surfaces[i], surfaces[j]))
    
    # Вывод совпадающих пар
    if identical_pairs:
        print(f"\n  Поверхности с полностью одинаковыми наборами ненулевых коэффициентов:")
        for p1, p2 in identical_pairs:
            print(f"     - {p1}  <-->  {p2}")
    else:
        print(f"\n  Нет пар с полностью одинаковыми наборами ненулевых коэффициентов")
    
    all_results[variant_name] = {
        'surfaces': surfaces,
        'nonzero_indices': nonzero_indices,
        'z_matrix': z_matrix,
        'hover_matrix': hover_matrix,
        'identical_pairs': identical_pairs,
        'dct_models': dct_models
    }
    
    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=surfaces,
        y=surfaces,
        hovertext=hover_matrix,
        hoverinfo='text',
        colorscale='Blues',
        colorbar=dict(
            title=dict(text="Кол-во<br>отличающихся<br>коэффициентов", side='right'),
            ticks='outside'
        ),
        showscale=True
    ))
    
    fig.update_layout(
        title={
            'text': f"Матрица различий ненулевых коэффициентов DCT-II ({variant_name})",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title="Поверхность",
        yaxis_title="Поверхность",
        template='plotly_white',
        height=700,
        width=900,
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
        margin=dict(l=150, r=50, t=80, b=150)
    )
    
    for i in range(n_surfs):
        for j in range(n_surfs):
            fig.add_trace(
                go.Scatter(
                    x=[surfaces[j]],
                    y=[surfaces[i]],
                    mode='text',
                    text=[str(z_matrix[i, j])],
                    textfont=dict(
                        size=11,
                        color='black' if z_matrix[i, j] < np.max(z_matrix)/2 else 'white'
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
    
    output_html = output_dir / f"dct_diff_matrix_{variant_name}.html"
    fig.write_html(str(output_html), include_plotlyjs='cdn')
    print(f"\n  Матрица сохранена: {output_html}")

    # Графики совпадения коэффициентов
    n_surfaces = len(surfaces)
    n_cols = 3
    n_rows = (n_surfaces + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=surfaces,
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    for idx, surface in enumerate(surfaces):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        coeffs = dct_models[surface]['coeffs']
        nonzero_idx = nonzero_indices[surface]
        
        # Создаём массив для графика: все коэффициенты, но ненулевые выделяем цветом
        colors = ['red' if i in nonzero_idx else '#333333' for i in range(len(coeffs))]
        
        fig.add_trace(
            go.Bar(
                x=list(range(len(coeffs))),
                y=coeffs,
                marker_color=colors,
                name=surface,
                showlegend=False
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Индекс коэффициента", row=row, col=col)
        fig.update_yaxes(title_text="Значение", row=row, col=col)
    
    fig.update_layout(
        title={
            'text': f"Совпадение ненулевых коэффициентов DCT ({variant_name})\nКрасный = ненулевой",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=300 * n_rows,
        width=1200,
        template='plotly_white',
        showlegend=False
    )
    
    output_bars = output_dir / f"dct_coefficients_bars_{variant_name}.html"
    fig.write_html(str(output_bars), include_plotlyjs='cdn')
    print(f"  Графики коэффициентов сохранены: {output_bars}")