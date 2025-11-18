"""Базовый класс для ML моделей."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


class BaseMLModel(ABC):
    """Абстрактный базовый класс для всех ML моделей."""

    def __init__(self, model_id: str, hyperparameters: Optional[Dict[str, Any]] = None):
        """
        Инициализация модели.

        Args:
            model_id: Уникальный идентификатор модели
            hyperparameters: Гиперпараметры модели
        """
        self.model_id = model_id
        self.hyperparameters = hyperparameters or self.get_default_hyperparameters()
        self.model = None
        self.created_at = datetime.now().isoformat()
        self.is_trained = False
        self.owner = None  # Будет установлен при сохранении

    @abstractmethod
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """
        Получить гиперпараметры по умолчанию.

        Returns:
            Словарь с гиперпараметрами по умолчанию
        """
        pass

    @abstractmethod
    def get_model_description(self) -> str:
        """
        Получить описание модели.

        Returns:
            Описание модели
        """
        pass

    @abstractmethod
    def _create_model(self) -> Any:
        """
        Создать экземпляр модели с заданными гиперпараметрами.

        Returns:
            Объект модели
        """
        pass

    def train(self, train_features: List[List[float]], target: List[int]) -> Dict[str, float]:
        """
        Обучить модель на данных.

        Args:
            train_features: Матрица признаков
            target: Вектор меток

        Returns:
            Словарь с метриками обучения

        Raises:
            ValueError: Если данные невалидны
        """
        # Преобразуем в numpy arrays если ещё не преобразованы
        train_array = np.array(train_features) if not isinstance(train_features, np.ndarray) else train_features
        target_array = np.array(target) if not isinstance(target, np.ndarray) else target

        if train_array.size == 0 or target_array.size == 0:
            raise ValueError("Training data cannot be empty")

        if len(train_array) != len(target_array):
            raise ValueError("Number of samples and labels must match")

        self.model = self._create_model()
        self.model.fit(train_array, target_array)
        self.is_trained = True

        # Вычисляем базовые метрики
        train_score = self.model.score(train_array, target_array)

        return {"train_accuracy": float(train_score)}

    def predict(self, train_features: List[List[float]]) -> List[int]:
        """
        Получить предсказания модели.

        Args:
            train_features: Матрица признаков (list of lists или numpy array)

        Returns:
            Список предсказанных классов

        Raises:
            RuntimeError: Если модель не обучена
            ValueError: Если данные невалидны
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before making predictions")

        # Преобразуем в numpy array если еще не преобразовано
        train_array = np.array(train_features)

        # Проверка на пустой массив
        if train_array.size == 0:
            raise ValueError("Input data cannot be empty")

        predictions = self.model.predict(train_array)

        return predictions.tolist()

    def predict_proba(self, train_features: List[List[float]]) -> Optional[List[List[float]]]:
        """
        Получить вероятности классов.

        Args:
            train_features: Матрица признаков (list of lists или numpy array)

        Returns:
            Матрица вероятностей или None если модель не поддерживает

        Raises:
            RuntimeError: Если модель не обучена
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before making predictions")

        # Преобразуем в numpy array если еще не преобразовано
        train_array = np.array(train_features)

        # Проверка на пустой массив
        if train_array.size == 0:
            raise ValueError("Input data cannot be empty")

        if not hasattr(self.model, "predict_proba"):
            return None

        probabilities = self.model.predict_proba(train_array)

        return probabilities.tolist()

    def get_info(self) -> Dict[str, Any]:
        """
        Получить информацию о модели.

        Returns:
            Словарь с информацией о модели
        """
        info = {
            "model_id": self.model_id,
            "model_type": self.__class__.__name__,
            "created_at": self.created_at,
            "hyperparameters": self.hyperparameters,
            "is_trained": self.is_trained,
            "owner": self.owner,
        }

        # Добавляем количество признаков, если модель обучена
        if self.is_trained and self.model is not None and hasattr(self.model, 'n_features_in_'):
            info["n_features"] = self.model.n_features_in_

        return info
