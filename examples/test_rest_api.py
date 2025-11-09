#!/usr/bin/env python3
"""
Скрипт для тестирования REST API.

Выполняет основные операции:
1. Health check
2. Аутентификация
3. Список доступных моделей
4. Список датасетов
5. Обучение модели на датасете
6. Получение предсказания
7. Удаление модели

Требования:
- REST API должен быть запущен (python -m uvicorn src.api.rest_api:app --host 0.0.0.0 --port 8000)
- Датасет 'iris' должен существовать (python examples/recreate_shared_datasets.py)
"""

import sys
from pathlib import Path

# Добавляем родительскую директорию в путь для импортов
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import requests
from typing import Optional


API_URL = "http://localhost:8000"
TOKEN: Optional[str] = None


def get_headers():
    """Получить заголовки с токеном."""
    if TOKEN:
        return {"Authorization": f"Bearer {TOKEN}"}
    return {}


def print_section(title: str):
    """Печать заголовка секции."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_subsection(title: str):
    """Печать подзаголовка."""
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


def test_health_check():
    """Тест 1: Health check."""
    print_subsection("1️⃣  HEALTH CHECK")
    
    try:
        response = requests.get(f"{API_URL}/health")
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Status: {data['status']}")
        print(f"📦 Version: {data['version']}")
        print(f"🎯 Models count: {data['models_count']}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_login():
    """Тест 2: Аутентификация."""
    global TOKEN
    
    print_subsection("2️⃣  AUTHENTICATION")
    
    try:
        response = requests.post(
            f"{API_URL}/auth/login",
            data={"username": "admin", "password": "admin123"}
        )
        response.raise_for_status()
        data = response.json()
        
        TOKEN = data["access_token"]
        print(f"✅ Authenticated as: admin")
        print(f"🎫 Token type: {data['token_type']}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_available_models():
    """Тест 3: Список доступных типов моделей."""
    print_subsection("3️⃣  AVAILABLE MODEL TYPES")
    
    try:
        response = requests.get(f"{API_URL}/models", headers=get_headers())
        response.raise_for_status()
        data = response.json()
        
        for model in data:
            print(f"\n📊 {model['name']}")
            print(f"   Description: {model['description'][:80]}...")
            print(f"   Hyperparameters: {list(model['default_hyperparameters'].keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_list_datasets():
    """Тест 4: Список датасетов."""
    print_subsection("4️⃣  DATASETS")
    
    try:
        response = requests.get(f"{API_URL}/datasets", headers=get_headers())
        response.raise_for_status()
        data = response.json()
        
        if not data:
            print("⚠️  No datasets found!")
            print("   Run: python examples/recreate_shared_datasets.py")
            return False
        
        for dataset in data:
            print(f"\n📁 {dataset['dataset_id']}")
            print(f"   Rows: {dataset['rows']}, Columns: {dataset['columns']}")
            print(f"   Target: {dataset['target_column']}")
            features = dataset['feature_columns']
            print(f"   Features: {', '.join(features[:3])}{'...' if len(features) > 3 else ''}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_train_model():
    """Тест 5: Обучение модели на датасете."""
    print_subsection("5️⃣  TRAIN MODEL FROM DATASET")
    
    print("🚀 Training RandomForest on 'iris' dataset...")
    
    try:
        response = requests.post(
            f"{API_URL}/models/train-from-dataset",
            headers=get_headers(),
            json={
                "model_type": "RandomForest",
                "model_name": "rest_demo_model",
                "dataset_id": "iris",
                "hyperparameters": {
                    "n_estimators": 50,
                    "max_depth": 5,
                    "random_state": 42
                }
            }
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ {data['message']}")
        print("📈 Metrics:")
        for key, value in data['metrics'].items():
            print(f"   {key}: {value:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"   Details: {error_detail}")
            except:
                pass
        return False


def test_list_trained_models():
    """Тест 6: Список обученных моделей."""
    print_subsection("6️⃣  TRAINED MODELS")
    
    try:
        response = requests.get(f"{API_URL}/models/trained", headers=get_headers())
        response.raise_for_status()
        data = response.json()
        
        if not data:
            print("⚠️  No trained models found")
            return True
        
        for model in data:
            print(f"\n🎯 {model['model_id']}")
            print(f"   Type: {model['model_type']}")
            print(f"   Owner: {model.get('owner', 'N/A')}")
            print(f"   Created: {model['created_at']}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_predict():
    """Тест 7: Получение предсказания."""
    print_subsection("7️⃣  PREDICTION")
    
    # Iris sample: sepal_length, sepal_width, petal_length, petal_width
    sample = [5.1, 3.5, 1.4, 0.2]  # Should predict class 0 (setosa)
    print(f"🔍 Input features: {sample}")
    
    try:
        response = requests.post(
            f"{API_URL}/models/rest_demo_model/predict",
            headers=get_headers(),
            json={"features": [sample]}
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"📊 Prediction: {data['predictions'][0]}")
        
        if data.get('probabilities'):
            probs = data['probabilities'][0]
            print("📈 Probabilities:")
            for i, prob in enumerate(probs):
                print(f"   Class {i}: {prob:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"   Details: {error_detail}")
            except:
                pass
        return False


def test_delete_model():
    """Тест 8: Удаление модели."""
    print_subsection("8️⃣  CLEANUP (Delete demo model)")
    
    try:
        response = requests.delete(
            f"{API_URL}/models/rest_demo_model",
            headers=get_headers()
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ {data['message']}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Запуск всех тестов."""
    print_section("REST API Demo Client")
    print()
    print("📡 Connecting to REST API at http://localhost:8000...")
    
    tests = [
        ("Health Check", test_health_check),
        ("Authentication", test_login),
        ("Available Models", test_available_models),
        ("List Datasets", test_list_datasets),
        ("Train Model", test_train_model),
        ("List Trained Models", test_list_trained_models),
        ("Prediction", test_predict),
        ("Delete Model", test_delete_model),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            results.append((name, False))
    
    # Сводка результатов
    print_section("SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    print()
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
