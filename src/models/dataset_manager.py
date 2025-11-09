"""Модуль для управления датасетами."""

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DatasetManager:
    """Менеджер для работы с датасетами."""

    def __init__(self, storage_dir: str = "datasets"):
        """Инициализация менеджера датасетов.
        
        Args:
            storage_dir: Директория для хранения датасетов
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_dir / "datasets_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Загрузить метаданные датасетов."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Сохранить метаданные датасетов."""
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def _preprocess_categorical(
        self, df: pd.DataFrame, categorical_columns: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
        """Предобработка категориальных переменных.
        
        Args:
            df: DataFrame с данными
            categorical_columns: Список категориальных колонок
            
        Returns:
            Tuple из обработанного DataFrame и словаря с энкодерами
        """
        encoders = {}
        df_copy = df.copy()

        for col in categorical_columns:
            if col in df_copy.columns:
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                encoders[col] = le

        return df_copy, encoders

    def _detect_categorical_columns(
        self, df: pd.DataFrame, max_unique_ratio: float = 0.05
    ) -> List[str]:
        """Автоматическое определение категориальных колонок.
        
        Args:
            df: DataFrame с данными
            max_unique_ratio: Максимальное отношение уникальных значений к общему числу строк
            
        Returns:
            Список категориальных колонок
        """
        categorical_cols = []

        for col in df.columns:
            # Если колонка типа object или category
            if df[col].dtype in ['object', 'category']:
                categorical_cols.append(col)
            # Если численная колонка с малым количеством уникальных значений
            elif df[col].dtype in ['int64', 'float64']:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < max_unique_ratio:
                    categorical_cols.append(col)

        return categorical_cols

    def upload_dataset(
        self,
        file_content: bytes,
        dataset_id: str,
        target_column: str,
        preprocess_categorical: bool = True,
        categorical_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Загрузить и обработать датасет.
        
        Args:
            file_content: Содержимое CSV файла
            dataset_id: Уникальный ID датасета
            target_column: Название колонки с целевой переменной
            preprocess_categorical: Применять ли предобработку категориальных переменных
            categorical_columns: Список категориальных колонок (если None, определяются автоматически)
            
        Returns:
            Словарь с информацией о датасете
        """
        # Загрузка CSV
        df = pd.read_csv(io.BytesIO(file_content))

        if target_column not in df.columns:
            raise ValueError(f"Колонка '{target_column}' не найдена в датасете. Доступные колонки: {list(df.columns)}")

        # Определение категориальных колонок
        if preprocess_categorical:
            if categorical_columns is None:
                categorical_columns = self._detect_categorical_columns(df)

            # Исключаем target из категориальных, если он там есть
            categorical_columns = [col for col in categorical_columns if col != target_column]

            # Предобработка категориальных переменных
            if categorical_columns:
                df, encoders = self._preprocess_categorical(df, categorical_columns)

                # Сохраняем энкодеры
                encoders_dir = self.storage_dir / f"{dataset_id}_encoders"
                encoders_dir.mkdir(exist_ok=True)

                for col, encoder in encoders.items():
                    encoder_file = encoders_dir / f"{col}_encoder.json"
                    with open(encoder_file, "w", encoding="utf-8") as f:
                        json.dump({
                            "classes": encoder.classes_.tolist(),
                        }, f, indent=2, ensure_ascii=False)
        else:
            categorical_columns = []

        # Разделение на признаки и целевую переменную
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns]
        y = df[target_column]

        # Если target категориальный, кодируем его
        target_encoder = None
        if y.dtype == 'object' or y.dtype.name == 'category':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)

            # Сохраняем энкодер target
            target_encoder_file = self.storage_dir / f"{dataset_id}_target_encoder.json"
            with open(target_encoder_file, "w", encoding="utf-8") as f:
                json.dump({
                    "classes": target_encoder.classes_.tolist(),
                }, f, indent=2, ensure_ascii=False)

        # Сохранение обработанного датасета
        processed_df = X.copy()
        processed_df[target_column] = y

        dataset_file = self.storage_dir / f"{dataset_id}.csv"
        processed_df.to_csv(dataset_file, index=False)

        # Сохранение метаданных
        metadata = {
            "dataset_id": dataset_id,
            "rows": len(df),
            "columns": len(df.columns),
            "target_column": target_column,
            "feature_columns": feature_columns,
            "categorical_columns_processed": categorical_columns,
            "target_is_categorical": target_encoder is not None,
            "created_at": datetime.now().isoformat(),
        }

        self.metadata[dataset_id] = metadata
        self._save_metadata()

        return metadata

    def get_dataset(self, dataset_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Получить датасет для обучения.
        
        Args:
            dataset_id: ID датасета
            
        Returns:
            Tuple из признаков (X) и целевой переменной (y)
        """
        if dataset_id not in self.metadata:
            raise ValueError(f"Датасет с ID '{dataset_id}' не найден")

        dataset_file = self.storage_dir / f"{dataset_id}.csv"
        if not dataset_file.exists():
            raise FileNotFoundError(f"Файл датасета '{dataset_file}' не найден")

        df = pd.read_csv(dataset_file)
        metadata = self.metadata[dataset_id]
        target_column = metadata["target_column"]

        X = df.drop(columns=[target_column]).values
        y = df[target_column].values

        return X, y

    def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """Получить информацию о датасете.
        
        Args:
            dataset_id: ID датасета
            
        Returns:
            Словарь с метаданными датасета
        """
        if dataset_id not in self.metadata:
            raise ValueError(f"Датасет с ID '{dataset_id}' не найден")

        return self.metadata[dataset_id]

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Получить список всех датасетов.
        
        Returns:
            Список метаданных всех датасетов
        """
        return list(self.metadata.values())

    def delete_dataset(self, dataset_id: str):
        """Удалить датасет.
        
        Args:
            dataset_id: ID датасета
        """
        if dataset_id not in self.metadata:
            raise ValueError(f"Датасет с ID '{dataset_id}' не найден")

        # Удаление файла датасета
        dataset_file = self.storage_dir / f"{dataset_id}.csv"
        if dataset_file.exists():
            dataset_file.unlink()

        # Удаление энкодеров
        encoders_dir = self.storage_dir / f"{dataset_id}_encoders"
        if encoders_dir.exists():
            for file in encoders_dir.iterdir():
                file.unlink()
            encoders_dir.rmdir()

        target_encoder_file = self.storage_dir / f"{dataset_id}_target_encoder.json"
        if target_encoder_file.exists():
            target_encoder_file.unlink()

        # Удаление метаданных
        del self.metadata[dataset_id]
        self._save_metadata()
