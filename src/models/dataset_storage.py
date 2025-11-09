"""Хранилище загруженных датасетов."""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils.logger import setup_logger

logger = setup_logger()


class DatasetStorage:
    """Хранилище для сохранения и загрузки датасетов."""

    def __init__(self, storage_dir: str = "datasets"):
        """
        Инициализация хранилища.

        Args:
            storage_dir: Директория для хранения датасетов
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._datasets: Dict[str, dict] = {}
        logger.info(f"Dataset storage initialized at {self.storage_dir.absolute()}")

    def _detect_categorical_columns(
        self, df: pd.DataFrame, max_unique_ratio: float = 0.05
    ) -> List[str]:
        """
        Автоматическое определение категориальных колонок.

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
                if unique_ratio < max_unique_ratio and df[col].nunique() < 20:
                    categorical_cols.append(col)

        return categorical_cols

    def _preprocess_categorical(
        self, df: pd.DataFrame, categorical_columns: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
        """
        Предобработка категориальных переменных.

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

    def save_dataset(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        target_column: str,
        preprocess_categorical: bool = True,
        categorical_columns: Optional[List[str]] = None,
        owner: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Сохранить датасет в хранилище.

        Args:
            dataset_id: Уникальный ID датасета
            df: DataFrame с данными
            target_column: Название колонки с целевой переменной
            preprocess_categorical: Применять ли предобработку категориальных переменных
            categorical_columns: Список категориальных колонок (если None, определяются автоматически)

        Returns:
            Информация о сохраненном датасете

        Raises:
            ValueError: Если target_column не найден в датасете
        """
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )

        # Предобработка категориальных переменных
        categorical_cols_processed = []
        encoders = {}

        if preprocess_categorical:
            if categorical_columns is None:
                categorical_columns = self._detect_categorical_columns(df)

            # Исключаем target из категориальных, если он там есть
            categorical_columns = [col for col in categorical_columns if col != target_column]

            # Предобработка категориальных переменных
            if categorical_columns:
                df, encoders = self._preprocess_categorical(df, categorical_columns)
                categorical_cols_processed = categorical_columns

                # Сохраняем энкодеры
                encoders_dir = self.storage_dir / f"{dataset_id}_encoders"
                encoders_dir.mkdir(exist_ok=True)

                for col, encoder in encoders.items():
                    encoder_file = encoders_dir / f"{col}_encoder.json"
                    with open(encoder_file, "w", encoding="utf-8") as f:
                        json.dump({
                            "classes": encoder.classes_.tolist(),
                        }, f, indent=2, ensure_ascii=False)

        # Обработка целевой переменной, если она категориальная
        target_encoder = None
        y = df[target_column]

        if y.dtype == 'object' or y.dtype.name == 'category':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)

            # Сохраняем энкодер target
            target_encoder_file = self.storage_dir / f"{dataset_id}_target_encoder.json"
            with open(target_encoder_file, "w", encoding="utf-8") as f:
                json.dump({
                    "classes": target_encoder.classes_.tolist(),
                }, f, indent=2, ensure_ascii=False)
        else:
            # Преобразуем в numpy array
            y = y.values

        # Разделяем на признаки и target
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns].values

        # Метаданные
        dataset_info = {
            "dataset_id": dataset_id,
            "X": X,
            "y": y,  # y теперь всегда numpy array
            "feature_columns": feature_columns,
            "target_column": target_column,
            "rows": len(df),
            "columns": len(df.columns),
            "categorical_columns_processed": categorical_cols_processed,
            "target_is_categorical": target_encoder is not None,
            "created_at": datetime.now().isoformat(),
            "owner": owner,
        }

        # Сохраняем в памяти
        self._datasets[dataset_id] = dataset_info

        # Сохраняем на диск
        dataset_path = self.storage_dir / f"{dataset_id}.pkl"

        try:
            with open(dataset_path, "wb") as f:
                pickle.dump(dataset_info, f)
            logger.info(f"Dataset '{dataset_id}' saved to {dataset_path}")
        except Exception as e:
            logger.error(f"Failed to save dataset '{dataset_id}': {e}")
            raise

        return dataset_info

    def load_dataset(self, dataset_id: str) -> Dict[str, any]:
        """
        Загрузить датасет из хранилища.

        Args:
            dataset_id: ID датасета

        Returns:
            Информация о датасете с данными

        Raises:
            FileNotFoundError: Если датасет не найден
        """
        # Проверяем в памяти
        if dataset_id in self._datasets:
            logger.info(f"Dataset '{dataset_id}' loaded from memory")
            return self._datasets[dataset_id]

        # Загружаем с диска
        dataset_path = self.storage_dir / f"{dataset_id}.pkl"

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset '{dataset_id}' not found")

        try:
            with open(dataset_path, "rb") as f:
                dataset_info = pickle.load(f)

            self._datasets[dataset_id] = dataset_info
            logger.info(f"Dataset '{dataset_id}' loaded from {dataset_path}")
            return dataset_info
        except Exception as e:
            logger.error(f"Failed to load dataset '{dataset_id}': {e}")
            raise

    def get_training_data(self, dataset_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Получить данные для обучения.

        Args:
            dataset_id: ID датасета

        Returns:
            Кортеж (X, y) с признаками и целевой переменной
        """
        dataset_info = self.load_dataset(dataset_id)
        return dataset_info["X"], dataset_info["y"]

    def encode_features(self, dataset_id: str, df: pd.DataFrame) -> np.ndarray:
        """
        Применить энкодеры датасета к новым данным для предсказания.
        
        Args:
            dataset_id: ID датасета (для получения сохраненных энкодеров)
            df: DataFrame с сырыми данными
            
        Returns:
            Закодированный numpy array
            
        Raises:
            FileNotFoundError: Если энкодеры не найдены
            ValueError: Если данные не соответствуют ожидаемому формату
        """
        # Загружаем энкодеры признаков
        encoders_dir = self.storage_dir / f"{dataset_id}_encoders"

        if not encoders_dir.exists():
            logger.warning(f"No encoders found for dataset '{dataset_id}', returning data as-is")
            # Преобразуем все колонки в числовой формат
            return pd.DataFrame(df).apply(pd.to_numeric, errors='coerce').fillna(0).values

        df_encoded = df.copy()

        # Применяем энкодеры к каждой категориальной колонке
        for encoder_file in encoders_dir.glob("*.json"):
            col_name = encoder_file.stem.replace("_encoder", "")

            if col_name in df_encoded.columns:
                with open(encoder_file, 'r') as f:
                    encoder_data = json.load(f)

                # Создаем маппинг из класса в код
                class_to_code = {cls: idx for idx, cls in enumerate(encoder_data['classes'])}

                # Применяем кодирование
                df_encoded[col_name] = df_encoded[col_name].astype(str).map(class_to_code)

                # Обрабатываем неизвестные значения
                if df_encoded[col_name].isna().any():
                    unknown_values = df[col_name][df_encoded[col_name].isna()].unique()
                    logger.warning(
                        f"Unknown values in column '{col_name}': {unknown_values}. "
                        f"Replacing with -1"
                    )
                    df_encoded[col_name] = df_encoded[col_name].fillna(-1)

                # Преобразуем в int
                df_encoded[col_name] = df_encoded[col_name].astype(int)

        # Преобразуем все остальные колонки в числовой формат
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object':
                df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').fillna(0)

        logger.info(f"Encoded features for dataset '{dataset_id}': {df_encoded.shape}")

        # Возвращаем numpy array с float типом
        return df_encoded.values.astype(float)

    def list_datasets(self) -> List[str]:
        """
        Получить список всех датасетов.

        Returns:
            Список ID датасетов
        """
        # Собираем из памяти и с диска
        dataset_ids = set(self._datasets.keys())

        for path in self.storage_dir.glob("*.pkl"):
            dataset_id = path.stem
            dataset_ids.add(dataset_id)

        logger.info(f"Found {len(dataset_ids)} datasets")
        return sorted(list(dataset_ids))

    def delete_dataset(self, dataset_id: str) -> None:
        """
        Удалить датасет из хранилища.

        Args:
            dataset_id: ID датасета

        Raises:
            FileNotFoundError: Если датасет не найден
        """
        # Удаляем из памяти
        if dataset_id in self._datasets:
            del self._datasets[dataset_id]

        # Удаляем с диска
        dataset_path = self.storage_dir / f"{dataset_id}.pkl"

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset '{dataset_id}' not found")

        try:
            dataset_path.unlink()

            # Удаление энкодеров
            encoders_dir = self.storage_dir / f"{dataset_id}_encoders"
            if encoders_dir.exists():
                for file in encoders_dir.iterdir():
                    file.unlink()
                encoders_dir.rmdir()

            target_encoder_file = self.storage_dir / f"{dataset_id}_target_encoder.json"
            if target_encoder_file.exists():
                target_encoder_file.unlink()

            logger.info(f"Dataset '{dataset_id}' deleted")
        except Exception as e:
            logger.error(f"Failed to delete dataset '{dataset_id}': {e}")
            raise

    def get_dataset_info(self, dataset_id: str) -> Dict[str, any]:
        """
        Получить информацию о датасете без данных.

        Args:
            dataset_id: ID датасета

        Returns:
            Информация о датасете (без X и y)
        """
        dataset_info = self.load_dataset(dataset_id)

        return {
            "dataset_id": dataset_info["dataset_id"],
            "rows": dataset_info["rows"],
            "columns": dataset_info["columns"],
            "feature_columns": dataset_info["feature_columns"],
            "target_column": dataset_info["target_column"],
            "created_at": dataset_info["created_at"],
            "owner": dataset_info.get("owner"),
        }
