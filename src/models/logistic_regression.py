"""Logistic Regression модель."""

from typing import Any, Dict

from sklearn.linear_model import LogisticRegression

from .base_model import BaseMLModel


class LogisticRegressionModel(BaseMLModel):
    """Logistic Regression классификатор."""

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """
        Получить гиперпараметры по умолчанию для Logistic Regression.

        Returns:
            Словарь с гиперпараметрами по умолчанию
        """
        return {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": 42,
        }

    def get_model_description(self) -> str:
        """
        Получить описание модели.

        Returns:
            Описание Logistic Regression модели
        """
        return (
            "Logistic Regression - линейная модель для классификации. "
            "Быстрая и интерпретируемая. "
            "Гиперпараметры: C (регуляризация), max_iter, solver, random_state"
        )

    def _create_model(self) -> LogisticRegression:
        """
        Создать экземпляр Logistic Regression модели.

        Returns:
            Объект LogisticRegression
        """
        return LogisticRegression(**self.hyperparameters)
