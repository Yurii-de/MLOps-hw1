"""Pydantic модели для валидации данных."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TrainingData(BaseModel):
    """Данные для обучения модели."""

    features: List[List[float]] = Field(..., description="Матрица признаков для обучения")
    labels: List[int] = Field(..., description="Метки классов")

    class Config:
        json_schema_extra = {
            "example": {
                "features": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                "labels": [0, 1, 0],
            }
        }


class PredictionData(BaseModel):
    """Данные для предсказания."""

    features: List[List[float]] = Field(..., description="Матрица признаков для предсказания")

    class Config:
        json_schema_extra = {
            "example": {
                "features": [[2.0, 3.0], [4.0, 5.0]],
            }
        }


class TrainModelRequest(BaseModel):
    """Запрос на обучение модели."""

    model_type: str = Field(..., description="Тип модели (RandomForest, LogisticRegression)")
    model_name: str = Field(..., description="Уникальное имя для сохранения модели")
    hyperparameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Гиперпараметры модели"
    )
    train_data: TrainingData = Field(..., description="Данные для обучения")

    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "RandomForest",
                "model_name": "my_rf_model",
                "hyperparameters": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
                "train_data": {
                    "features": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                    "labels": [0, 1, 0, 1],
                },
            }
        }


class TrainModelResponse(BaseModel):
    """Ответ на запрос обучения модели."""

    model_id: str = Field(..., description="ID обученной модели")
    model_type: str = Field(..., description="Тип модели")
    message: str = Field(..., description="Сообщение о статусе")
    metrics: Optional[Dict[str, float]] = Field(default=None, description="Метрики модели")


class PredictRequest(BaseModel):
    """Запрос на получение предсказания."""

    features: List[List[float]] = Field(..., description="Матрица признаков")

    class Config:
        json_schema_extra = {
            "example": {
                "features": [[2.0, 3.0], [4.0, 5.0]],
            }
        }


class PredictResponse(BaseModel):
    """Ответ с предсказаниями."""

    model_id: str = Field(..., description="ID модели")
    predictions: List[int] = Field(..., description="Предсказанные классы")
    probabilities: Optional[List[List[float]]] = Field(
        default=None, description="Вероятности классов"
    )


class ModelInfo(BaseModel):
    """Информация о модели."""

    model_id: str = Field(..., description="ID модели")
    model_type: str = Field(..., description="Тип модели")
    created_at: str = Field(..., description="Дата создания")
    hyperparameters: Dict[str, Any] = Field(..., description="Гиперпараметры модели")


class AvailableModel(BaseModel):
    """Информация о доступном типе модели."""

    name: str = Field(..., description="Название типа модели")
    description: str = Field(..., description="Описание модели")
    default_hyperparameters: Dict[str, Any] = Field(
        ..., description="Гиперпараметры по умолчанию"
    )


class HealthResponse(BaseModel):
    """Ответ health check."""

    status: str = Field(..., description="Статус сервиса")
    version: str = Field(..., description="Версия API")
    models_count: int = Field(..., description="Количество обученных моделей")


class ErrorResponse(BaseModel):
    """Ответ с ошибкой."""

    error: str = Field(..., description="Описание ошибки")
    detail: Optional[str] = Field(default=None, description="Детали ошибки")


# Схемы для аутентификации


class Token(BaseModel):
    """JWT токен."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Тип токена")


class TokenData(BaseModel):
    """Данные из токена."""

    username: Optional[str] = None


class UserLogin(BaseModel):
    """Данные для входа."""

    username: str = Field(..., description="Имя пользователя")
    password: str = Field(..., description="Пароль")

    class Config:
        json_schema_extra = {"example": {"username": "admin", "password": "admin123"}}


class UserCreate(BaseModel):
    """Данные для регистрации."""

    username: str = Field(..., min_length=3, max_length=50, description="Имя пользователя")
    email: str = Field(..., description="Email")
    password: str = Field(..., min_length=6, description="Пароль")

    class Config:
        json_schema_extra = {
            "example": {"username": "newuser", "email": "newuser@example.com", "password": "password123"}
        }


class User(BaseModel):
    """Информация о пользователе."""

    username: str = Field(..., description="Имя пользователя")
    email: str = Field(..., description="Email")
    disabled: bool = Field(default=False, description="Пользователь отключен")
