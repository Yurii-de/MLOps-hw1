"""Фабрика для создания моделей."""

from typing import Any, Dict, List, Optional

from .base_model import BaseMLModel
from .logistic_regression import LogisticRegressionModel
from .random_forest import RandomForestModel


class ModelFactory:
    """Фабрика для создания ML моделей."""

    _model_registry: Dict[str, type[BaseMLModel]] = {
        "RandomForest": RandomForestModel,
        "LogisticRegression": LogisticRegressionModel,
    }

    @classmethod
    def create_model(
        cls, model_type: str, model_id: str, hyperparameters: Optional[Dict[str, Any]] = None
    ) -> BaseMLModel:
        """
        Создать модель заданного типа.

        Args:
            model_type: Тип модели (RandomForest, LogisticRegression)
            model_id: Уникальный идентификатор модели
            hyperparameters: Гиперпараметры модели

        Returns:
            Экземпляр модели

        Raises:
            ValueError: Если тип модели не поддерживается
        """
        if model_type not in cls._model_registry:
            available_models = ", ".join(cls._model_registry.keys())
            raise ValueError(
                f"Model type '{model_type}' is not supported. "
                f"Available models: {available_models}"
            )

        model_class = cls._model_registry[model_type]
        return model_class(model_id=model_id, hyperparameters=hyperparameters)

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """
        Получить список доступных типов моделей.

        Returns:
            Список словарей с информацией о доступных моделях
        """
        available_models = []

        for model_name, model_class in cls._model_registry.items():
            # Создаем временный экземпляр для получения информации
            temp_model = model_class(model_id="temp")

            available_models.append(
                {
                    "name": model_name,
                    "description": temp_model.get_model_description(),
                    "default_hyperparameters": temp_model.get_default_hyperparameters(),
                }
            )

        return available_models

    @classmethod
    def register_model(cls, name: str, model_class: type[BaseMLModel]) -> None:
        """
        Зарегистрировать новый тип модели.

        Args:
            name: Название типа модели
            model_class: Класс модели
        """
        cls._model_registry[name] = model_class
