# Объединение всех данных в одну таблицу (отдельная таблица для каждого из роботов 4 и 6 колес)
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Настройки
WHEELS = 4  # число колёс: 4 или 6

#DATA_DIR = Path("data/train_set/processed")
DATA_DIR = Path(r"C:\UIRS\processed")
OUTPUT = DATA_DIR / f"combined_processed_{WHEELS}W.csv"


# Регулировка для имени файла
PATTERN = re.compile(
    r'^(?P<surface>.+?)_(?P<wheels>\d+)W_(?P<number>\d+)$'
)


def parse_filename(stem):
    """Разбор имени файла surface_4W_1.csv"""
    match = PATTERN.match(stem)
    if not match:
        return None, None, None

    surface = match.group("surface")
    wheels = int(match.group("wheels"))
    number = int(match.group("number"))

    return surface, wheels, number


def process_files(data_dir, output_file, wheels_filter):
    # Все CSV-файлы
    csv_files = list(data_dir.glob("*.csv"))

    # Хранилище всех таблиц
    all_rows = []

    # Статистика
    stats = {}

    for file_path in tqdm(csv_files, desc="Чтение файлов", unit="файл"):
        stem = file_path.stem
        surface, wheels, number = parse_filename(stem)

        # Пропускаем файлы других типов
        if wheels is None or wheels != wheels_filter:
            continue

        # Загрузка CSV
        df = pd.read_csv(file_path)

        # СОРТИРОВКА ВНУТРИ ФАЙЛА ПО TIME
        # Берём первый столбец как Time, если пользователь захочет изменить мы легко подстроим
        time_col = df.columns[0]   # первый столбец
        df = df.sort_values(by=time_col, ascending=True)

        # Добавляем новые столбцы
        df["surface"] = surface
        df["wheels"] = wheels
        df["number_exper"] = number

        # Добавляем в список
        all_rows.append(df)

        # Обновляем статистику
        stats[surface] = stats.get(surface, 0) + 1

    # Склеиваем все таблицы
    full_table = pd.concat(all_rows, ignore_index=True)

    # ГЛОБАЛЬНАЯ СОРТИРОВКА
    # Сначала по surface → number_exper → Time
    full_table = full_table.sort_values(
        by=["surface", "number_exper", full_table.columns[0]],
        ascending=[True, True, True]
    )

    # Запись результата
    full_table.to_csv(output_file, index=False)

    print(f"\nГотово! Итог сохранён в: {output_file}")

    # СТАТИСТИКА
    print("\nКоличество обработанных экспериментов по поверхностям:")
    for surface, count in sorted(stats.items()):
        print(f"  • {surface}: {count}")

    print("\nДанные отсортированы по: surface → number_exper → Time")


# Запуск
process_files(DATA_DIR, OUTPUT, WHEELS)
