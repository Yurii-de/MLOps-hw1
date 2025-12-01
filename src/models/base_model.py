"""Базовый класс для ML моделей."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from src.config import MLFLOW_TRACKING_URI
from src.logger import setup_logger

logger = setup_logger()

# Настройка MLFlow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


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

        # Начинаем MLFlow run
        with mlflow.start_run(run_name=f"{self.__class__.__name__}_{self.model_id}"):
            # Логируем гиперпараметры
            mlflow.log_params(self.hyperparameters)
            mlflow.log_param("model_type", self.__class__.__name__)
            mlflow.log_param("model_id", self.model_id)
            mlflow.log_param("created_at", self.created_at)

            # Логируем информацию о данных
            mlflow.log_param("n_samples", len(train_array))
            mlflow.log_param("n_features", train_array.shape[1])

            self.model = self._create_model()
            self.model.fit(train_array, target_array)
            self.is_trained = True

            # Вычисляем метрики
            train_predictions = self.model.predict(train_array)
            train_accuracy = accuracy_score(target_array, train_predictions)

            # Логируем метрики
            mlflow.log_metric("train_accuracy", train_accuracy)

            # Логируем модель
            mlflow.sklearn.log_model(self.model, "model")

            # Логируем classification report как artifact
            report = classification_report(target_array, train_predictions, output_dict=True)
            mlflow.log_dict(report, "classification_report.json")

            logger.info(f"Model {self.model_id} trained with accuracy: {train_accuracy:.4f}")

            return {"train_accuracy": float(train_accuracy)}

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
