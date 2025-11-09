"""Хранилище обученных моделей."""

import pickle
from pathlib import Path
from typing import Dict, List

from src.utils.logger import setup_logger

from .base_model import BaseMLModel

logger = setup_logger()


class ModelStorage:
    """Хранилище для сохранения и загрузки обученных моделей."""

    def __init__(self, storage_dir: str = "models"):
        """
        Инициализация хранилища.

        Args:
            storage_dir: Директория для хранения моделей
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, BaseMLModel] = {}
        logger.info(f"Model storage initialized at {self.storage_dir.absolute()}")

    def save_model(self, model: BaseMLModel) -> None:
        """
        Сохранить модель в хранилище.

        Args:
            model: Модель для сохранения

        Raises:
            ValueError: Если модель не обучена
        """
        if not model.is_trained:
            raise ValueError(f"Cannot save untrained model '{model.model_id}'")

        # Сохраняем в памяти
        self._models[model.model_id] = model

        # Сохраняем на диск
        model_path = self.storage_dir / f"{model.model_id}.pkl"

        try:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"Model '{model.model_id}' saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model '{model.model_id}': {e}")
            raise

    def load_model(self, model_id: str) -> BaseMLModel:
        """
        Загрузить модель из хранилища.

        Args:
            model_id: ID модели

        Returns:
            Загруженная модель

        Raises:
            FileNotFoundError: Если модель не найдена
        """
        # Проверяем в памяти
        if model_id in self._models:
            logger.info(f"Model '{model_id}' loaded from memory")
            return self._models[model_id]

        # Загружаем с диска
        model_path = self.storage_dir / f"{model_id}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model '{model_id}' not found")

        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            self._models[model_id] = model
            logger.info(f"Model '{model_id}' loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model '{model_id}': {e}")
            raise

    def delete_model(self, model_id: str) -> None:
        """
        Удалить модель из хранилища.

        Args:
            model_id: ID модели

        Raises:
            FileNotFoundError: Если модель не найдена
        """
        # Удаляем из памяти
        if model_id in self._models:
            del self._models[model_id]

        # Удаляем с диска
        model_path = self.storage_dir / f"{model_id}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model '{model_id}' not found")

        try:
            model_path.unlink()
            logger.info(f"Model '{model_id}' deleted from storage")
        except Exception as e:
            logger.error(f"Failed to delete model '{model_id}': {e}")
            raise

    def list_models(self) -> List[str]:
        """
        Получить список всех сохраненных моделей.

        Returns:
            Список ID моделей
        """
        # Сканируем директорию
        model_files = list(self.storage_dir.glob("*.pkl"))
        model_ids = [f.stem for f in model_files]

        logger.info(f"Found {len(model_ids)} models in storage")
        return model_ids

    def get_model_info(self, model_id: str) -> Dict:
        """
        Получить информацию о модели.

        Args:
            model_id: ID модели

        Returns:
            Словарь с информацией о модели

        Raises:
            FileNotFoundError: Если модель не найдена
        """
        model = self.load_model(model_id)
        return model.get_info()

    def model_exists(self, model_id: str) -> bool:
        """
        Проверить существование модели.

        Args:
            model_id: ID модели

        Returns:
            True если модель существует
        """
        return model_id in self._models or (self.storage_dir / f"{model_id}.pkl").exists()
