import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Настройки
INPUT_FILE = Path(r"C:\UIRS\surface-classification\combined_processed_4W.csv")
OUTPUT_FILE = Path(r"C:\UIRS\surface-classification\filtered_4W_window10_step1.csv")
WINDOW_SIZE = 10   # Размер окна (10 измерений)
STEP_SIZE = 1      

def apply_moving_window_filter(df, window_size, step_size):

    
    # Числовые колонки для усреднения (исключаем метаданные)
    exclude_cols = {'Unnamed: 0', 'Time', 'surface', 'wheels', 'number_exper'}
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                    if col not in exclude_cols]
    
    print(f"Колонки для фильтрации: {len(numeric_cols)}")
    print(f"Параметры: окно={window_size}, шаг={step_size} (перекрытие={100*(1-step_size/window_size):.0f}%)")
    
    all_windows = []
    
    # Группируем по surface и number_exper
    groups = df.groupby(['surface', 'number_exper'], sort=False)
    total_groups = len(groups)
    
    print(f"\nОбрабатываю {total_groups} групп (surface × experiment)...")
    
    for (surface, exp_num), group in tqdm(groups, desc="Фильтрация", unit="группа"):
        # Сортируем по времени
        group = group.sort_values('Time').reset_index(drop=True)
        
        if len(group) < window_size:
            continue
        
        # Применяем скользящее окно
        for start_idx in range(0, len(group) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window = group.iloc[start_idx:end_idx]
            
            # Создаем усредненную строку
            row = {
                'surface': surface,
                'number_exper': exp_num,
                'wheels': window['wheels'].iloc[0],
                'Time': window['Time'].mean(),  # Среднее время окна
                'window_start_idx': start_idx,
                'window_size': window_size,
            }
            
            # Усредняем все числовые колонки
            for col in numeric_cols:
                row[col] = window[col].mean()
            
            all_windows.append(row)
    
    # Создаем DataFrame
    result_df = pd.DataFrame(all_windows)
    
    # Переупорядочиваем колонки
    first_cols = ['surface', 'number_exper', 'wheels', 'Time', 'window_start_idx', 'window_size']
    other_cols = [col for col in result_df.columns if col not in first_cols]
    result_df = result_df[first_cols + other_cols]
    
    return result_df


def main():
    print("ФИЛЬТРАЦИЯ СКОЛЬЗЯЩИМ ОКНОМ (90% перекрытие)")
    
    # Чтение данных
    print(f"\n1. Чтение файла: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"   Загружено: {len(df):,} строк, {len(df.columns)} колонок")
    
    # Применение фильтра
    print(f"\n2. Применение скользящего окна...")
    filtered_df = apply_moving_window_filter(df, WINDOW_SIZE, STEP_SIZE)
    
    # Статистика
    print(f"\n3. Результаты:")
    print(f"   Строк до фильтрации: {len(df):,}")
    print(f"   Строк после фильтрации: {len(filtered_df):,}")
    print(f"   Сокращение: {100*(1-len(filtered_df)/len(df)):.1f}%")
    print(f"   Фактическая частота: {100/STEP_SIZE:.1f} Гц (было 100 Гц)")
    
    # Статистика по поверхностям
    print(f"\n4. Распределение по поверхностям:")
    for surface in sorted(filtered_df['surface'].unique()):
        count = len(filtered_df[filtered_df['surface'] == surface])
        orig_count = len(df[df['surface'] == surface])
        print(f"   {surface:20s}: {count:,} окон (было {orig_count:,})")
    
    # Сохранение
    print(f"\n5. Сохранение в: {OUTPUT_FILE}")
    filtered_df.to_csv(OUTPUT_FILE, index=False)
    
    print("ГОТОВО!")


if __name__ == "__main__":
    main()