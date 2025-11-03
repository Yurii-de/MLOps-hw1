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

    def train(self, X: List[List[float]], y: List[int]) -> Dict[str, float]:
        """
        Обучить модель на данных.

        Args:
            X: Матрица признаков
            y: Вектор меток

        Returns:
            Словарь с метриками обучения

        Raises:
            ValueError: Если данные невалидны
        """
        if not X or not y:
            raise ValueError("Training data cannot be empty")

        if len(X) != len(y):
            raise ValueError("Number of samples and labels must match")

        X_array = np.array(X)
        y_array = np.array(y)

        self.model = self._create_model()
        self.model.fit(X_array, y_array)
        self.is_trained = True

        # Вычисляем базовые метрики
        train_score = self.model.score(X_array, y_array)

        return {"train_accuracy": float(train_score)}

    def predict(self, X: List[List[float]]) -> List[int]:
        """
        Получить предсказания модели.

        Args:
            X: Матрица признаков

        Returns:
            Список предсказанных классов

        Raises:
            RuntimeError: Если модель не обучена
            ValueError: Если данные невалидны
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before making predictions")

        if not X:
            raise ValueError("Input data cannot be empty")

        X_array = np.array(X)
        predictions = self.model.predict(X_array)

        return predictions.tolist()

    def predict_proba(self, X: List[List[float]]) -> Optional[List[List[float]]]:
        """
        Получить вероятности классов.

        Args:
            X: Матрица признаков

        Returns:
            Матрица вероятностей или None если модель не поддерживает

        Raises:
            RuntimeError: Если модель не обучена
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before making predictions")

        if not X:
            raise ValueError("Input data cannot be empty")

        if not hasattr(self.model, "predict_proba"):
            return None

        X_array = np.array(X)
        probabilities = self.model.predict_proba(X_array)

        return probabilities.tolist()

    def get_info(self) -> Dict[str, Any]:
        """
        Получить информацию о модели.

        Returns:
            Словарь с информацией о модели
        """
        return {
            "model_id": self.model_id,
            "model_type": self.__class__.__name__,
            "created_at": self.created_at,
            "hyperparameters": self.hyperparameters,
            "is_trained": self.is_trained,
        }
