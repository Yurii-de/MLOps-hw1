#!/usr/bin/env python3
"""
Скрипт для пересоздания шаблонных датасетов как общих (owner=None).
Удаляет старые датасеты iris и adult и создает новые как общие.
"""

import shutil
import sys
from pathlib import Path

# Добавляем путь к src в PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd

from src.models.dataset_storage import DatasetStorage


def main():
    """Пересоздать шаблонные датасеты как общие."""
    print("=" * 60)
    print("Пересоздание шаблонных датасетов как общих")
    print("=" * 60)

    # Инициализация storage
    storage_dir = BASE_DIR / "datasets"
    dataset_storage = DatasetStorage(storage_dir)

    # Датасеты для пересоздания
    datasets_to_recreate = {
        "iris": {
            "file": "test_data/iris.csv",
            "target": "species",
            "description": "Iris classification dataset (общий)"
        },
        "adult": {
            "file": "test_data/adult.csv",
            "target": "income",
            "description": "Adult income classification dataset (общий)"
        }
    }

    for dataset_id, info in datasets_to_recreate.items():
        print(f"\n🔄 Обработка датасета: {dataset_id}")

        # Удаляем старый датасет если существует
        old_dataset_path = storage_dir / f"{dataset_id}.pkl"
        old_encoders_dir = storage_dir / f"{dataset_id}_encoders"
        old_target_encoder = storage_dir / f"{dataset_id}_target_encoder.json"

        if old_dataset_path.exists():
            print(f"  ❌ Удаление старого датасета: {old_dataset_path}")
            old_dataset_path.unlink()

        if old_encoders_dir.exists():
            print(f"  ❌ Удаление старых энкодеров: {old_encoders_dir}")
            shutil.rmtree(old_encoders_dir)

        if old_target_encoder.exists():
            print(f"  ❌ Удаление старого target энкодера: {old_target_encoder}")
            old_target_encoder.unlink()

        # Загружаем CSV
        csv_path = BASE_DIR / info["file"]
        if not csv_path.exists():
            print(f"  ⚠️ Файл не найден: {csv_path}, пропускаем")
            continue

        print(f"  📂 Загрузка из: {csv_path}")
        df = pd.read_csv(csv_path)

        # Создаем новый датасет как общий (owner=None)
        print("  💾 Создание общего датасета...")
        dataset_info = dataset_storage.save_dataset(
            dataset_id=dataset_id,
            df=df,
            target_column=info["target"],
            preprocess_categorical=True,
            owner=None  # Делаем общим!
        )

        print(f"  ✅ Датасет '{dataset_id}' создан как общий")
        print(f"     Строк: {dataset_info['rows']}")
        print(f"     Колонок: {dataset_info['columns']}")
        print(f"     Target: {dataset_info['target_column']}")
        print(f"     Владелец: {dataset_info.get('owner', 'Общий')}")

        if dataset_info.get('categorical_columns_processed'):
            print(f"     Категориальные колонки обработаны: {len(dataset_info['categorical_columns_processed'])}")

    print("\n" + "=" * 60)
    print("✅ Пересоздание завершено!")
    print("=" * 60)
    print("\nТеперь датасеты 'iris' и 'adult' являются общими и:")
    print("  🌐 Доступны всем пользователям")
    print("  🔒 Защищены от удаления")
    print("  📊 Отображаются с меткой 'Владелец: Общий'")

if __name__ == "__main__":
    main()
