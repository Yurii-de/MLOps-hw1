"""Random Forest модель."""

from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseMLModel


class RandomForestModel(BaseMLModel):
    """Random Forest классификатор."""

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """
        Получить гиперпараметры по умолчанию для Random Forest.

        Returns:
            Словарь с гиперпараметрами по умолчанию
        """
        return {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
        }

    def get_model_description(self) -> str:
        """
        Получить описание модели.

        Returns:
            Описание Random Forest модели
        """
        return (
            "Random Forest Classifier - ансамбль деревьев решений. "
            "Подходит для классификации с высокой точностью. "
            "Гиперпараметры: n_estimators, max_depth, min_samples_split, "
            "min_samples_leaf, random_state"
        )

    def _create_model(self) -> RandomForestClassifier:
        """
        Создать экземпляр Random Forest модели.

        Returns:
            Объект RandomForestClassifier
        """
        return RandomForestClassifier(**self.hyperparameters)
