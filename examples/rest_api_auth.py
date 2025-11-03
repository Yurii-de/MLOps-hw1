"""Примеры использования REST API с аутентификацией."""

import requests

# Базовый URL API
BASE_URL = "http://localhost:8000"


def example_authentication():
    """Пример аутентификации."""
    print("=" * 80)
    print("REST API Authentication Examples")
    print("=" * 80)

    # 1. Регистрация нового пользователя (опционально)
    print("\n1. Register new user (optional)")
    print("-" * 40)

    register_data = {"username": "testuser", "email": "test@example.com", "password": "testpass123"}

    try:
        response = requests.post(f"{BASE_URL}/auth/register", json=register_data)

        if response.status_code == 200:
            print(f"✅ User registered: {response.json()}")
        else:
            print(f"User already exists or error: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

    # 2. Вход и получение токена
    print("\n2. Login and get access token")
    print("-" * 40)

    login_data = {"username": "admin", "password": "admin123"}  # Или testuser/testpass123

    response = requests.post(f"{BASE_URL}/auth/login", data=login_data)

    if response.status_code == 200:
        token_data = response.json()
        access_token = token_data["access_token"]
        print(f"✅ Access token obtained: {access_token[:50]}...")

        # Headers для авторизованных запросов
        headers = {"Authorization": f"Bearer {access_token}"}

        # 3. Получить информацию о себе
        print("\n3. Get current user info")
        print("-" * 40)

        response = requests.get(f"{BASE_URL}/auth/me", headers=headers)
        print(f"User info: {response.json()}")

        # 4. Получить список доступных моделей
        print("\n4. Get available models")
        print("-" * 40)

        response = requests.get(f"{BASE_URL}/models", headers=headers)
        models = response.json()
        print(f"Available models: {len(models)}")
        for model in models:
            print(f"  - {model['name']}: {model['description']}")

        # 5. Обучить модель
        print("\n5. Train a model")
        print("-" * 40)

        train_data = {
            "model_type": "RandomForest",
            "model_name": "auth_test_model",
            "hyperparameters": {"n_estimators": 50, "max_depth": 5, "random_state": 42},
            "train_data": {
                "features": [[1, 2], [3, 4], [5, 6], [7, 8]],
                "labels": [0, 1, 0, 1],
            },
        }

        response = requests.post(f"{BASE_URL}/models/train", json=train_data, headers=headers)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Model trained: {result['model_id']}")
            print(f"   Metrics: {result['metrics']}")
        else:
            print(f"Error or model exists: {response.json()}")

        # 6. Получить предсказание
        print("\n6. Get prediction")
        print("-" * 40)

        predict_data = {"features": [[2, 3], [4, 5]]}

        response = requests.post(
            f"{BASE_URL}/models/auth_test_model/predict", json=predict_data, headers=headers
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Predictions: {result['predictions']}")
            if result.get("probabilities"):
                print(f"   Probabilities: {result['probabilities']}")
        else:
            print(f"Error: {response.json()}")

        # 7. Попытка без токена (должна провалиться)
        print("\n7. Try request without token (should fail)")
        print("-" * 40)

        response = requests.get(f"{BASE_URL}/models")

        if response.status_code == 401:
            print("✅ Correctly rejected: Unauthorized")
        else:
            print(f"Unexpected: {response.status_code}")

        # 8. Попытка с неверным токеном
        print("\n8. Try request with invalid token (should fail)")
        print("-" * 40)

        bad_headers = {"Authorization": "Bearer invalid_token"}
        response = requests.get(f"{BASE_URL}/models", headers=bad_headers)

        if response.status_code == 401:
            print("✅ Correctly rejected: Invalid token")
        else:
            print(f"Unexpected: {response.status_code}")

        print("\n" + "=" * 80)
        print("Authentication examples completed!")
        print("=" * 80)

    else:
        print(f"❌ Login failed: {response.json()}")
        print("\nMake sure:")
        print("1. REST API is running: uvicorn src.api.rest_api:app --reload")
        print("2. Using correct credentials (admin/admin123 or user/user123)")


if __name__ == "__main__":
    print("Starting REST API authentication examples...")
    print("Make sure the REST API is running on localhost:8000\n")

    try:
        example_authentication()
    except Exception as e:
        print(f"\nFatal error: {e}")
        print("\nMake sure the REST API server is running!")
