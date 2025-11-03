"""Тесты для REST API."""

import pytest
from fastapi.testclient import TestClient

from src.api.rest_api import app

client = TestClient(app)


def test_health_check():
    """Тест health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "models_count" in data


def test_get_available_models():
    """Тест получения списка доступных моделей."""
    response = client.get("/models")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2  # Минимум 2 типа моделей

    # Проверяем структуру
    for model in data:
        assert "name" in model
        assert "description" in model
        assert "default_hyperparameters" in model


def test_train_model():
    """Тест обучения модели."""
    payload = {
        "model_type": "RandomForest",
        "model_name": "test_rf_model",
        "hyperparameters": {"n_estimators": 50, "random_state": 42},
        "train_data": {
            "features": [[1, 2], [3, 4], [5, 6], [7, 8]],
            "labels": [0, 1, 0, 1],
        },
    }

    response = client.post("/models/train", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["model_id"] == "test_rf_model"
    assert data["model_type"] == "RandomForest"
    assert "metrics" in data


def test_predict():
    """Тест получения предсказания."""
    # Сначала обучаем модель
    train_payload = {
        "model_type": "LogisticRegression",
        "model_name": "test_lr_model",
        "hyperparameters": {"C": 1.0, "random_state": 42},
        "train_data": {
            "features": [[1, 2], [3, 4], [5, 6], [7, 8]],
            "labels": [0, 1, 0, 1],
        },
    }

    train_response = client.post("/models/train", json=train_payload)
    assert train_response.status_code == 200

    # Получаем предсказание
    pred_payload = {"features": [[2, 3], [4, 5]]}

    pred_response = client.post("/models/test_lr_model/predict", json=pred_payload)
    assert pred_response.status_code == 200

    data = pred_response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2


def test_delete_model():
    """Тест удаления модели."""
    # Обучаем модель
    train_payload = {
        "model_type": "RandomForest",
        "model_name": "test_delete_model",
        "train_data": {
            "features": [[1, 2], [3, 4]],
            "labels": [0, 1],
        },
    }

    client.post("/models/train", json=train_payload)

    # Удаляем модель
    response = client.delete("/models/test_delete_model")
    assert response.status_code == 200


def test_model_not_found():
    """Тест ошибки - модель не найдена."""
    response = client.post("/models/nonexistent_model/predict", json={"features": [[1, 2]]})
    assert response.status_code == 404
