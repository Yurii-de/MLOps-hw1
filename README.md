# ML API Service - MLOps Homework 1

Machine Learning API service with REST endpoints, JWT authentication, and model management.

## 🎯 Features

- ✅ **ML Models**: RandomForest, LogisticRegression
- ✅ **REST API**: 10 endpoints (FastAPI)
- ✅ **JWT Authentication**: Secure API access
- ✅ **Model Management**: Train, predict, retrain, delete
- ✅ **Tests**: pytest coverage
- ✅ **Swagger UI**: Interactive API docs

---

## 🔐 Authentication

Project uses **JWT Bearer tokens** for API protection.

### Pre-configured users:

| Username | Password |
|----------|----------|
| `admin` | `admin123` |
| `user` | `user123` |

### Quick example:

```bash
# 1. Get token
curl -X POST http://localhost:8000/auth/login \
  -d "username=admin&password=admin123"

# 2. Use token in requests
curl http://localhost:8000/models \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## Описание проекта

Сервис для обучения и использования ML-моделей с поддержкой REST API и gRPC. Проект позволяет:
- Обучать различные типы ML-моделей с настраиваемыми гиперпараметрами
- Получать предсказания от обученных моделей
- Управлять моделями (переобучение, удаление)
- Взаимодействовать через REST API, gRPC и веб-интерфейс

## Поддерживаемые модели
---

## 📦 Installation

1. Clone repository:
```bash
git clone https://github.com/Yurii-de/MLOps-hw1.git
cd mo
```

2. Install dependencies with Poetry:
```bash
poetry install
poetry shell
```

---

## 🚀 Running

### REST API

```bash
poetry run uvicorn src.api.rest_api:app --reload
```

- API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`

---

## 📋 REST API Endpoints

### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - Login and get JWT token
- `GET /auth/me` - Get current user info

### Health Check
- `GET /health` - Service status (no auth required)

### Models Management (authentication required)
- `GET /models` - Получить список доступных типов моделей
- `GET /models/trained` - Получить список обученных моделей
- `POST /models/train` - Обучить новую модель
- `POST /models/{model_id}/retrain` - Переобучить существующую модель
- `DELETE /models/{model_id}` - Удалить модель
- `POST /models/{model_id}/predict` - Получить предсказание

## gRPC API

Для работы с gRPC используйте клиент из `examples/grpc_client.py` или `examples/grpc_client.ipynb`.

Доступные методы:
- `ListAvailableModels` - Список доступных типов моделей
- `TrainModel` - Обучение модели
- `Predict` - Получение предсказаний
- `RetrainModel` - Переобучение модели
- `DeleteModel` - Удаление модели
- `HealthCheck` - Проверка статуса

### Пример использования gRPC клиента:

```bash
poetry run python examples/grpc_client.py
```

Или откройте ноутбук:
```bash
poetry run jupyter notebook examples/grpc_client.ipynb
- `GET /models` - List available model types
- `GET /models/trained` - List all trained models
- `POST /models/train` - Train new model
- `POST /models/{model_name}/predict` - Get predictions
- `POST /models/{model_name}/retrain` - Retrain existing model
- `DELETE /models/{model_name}` - Delete model

---

## 💡 Usage Examples

### Train a model:
```bash
# 1. Login and get token
TOKEN=$(curl -X POST http://localhost:8000/auth/login \
  -d "username=admin&password=admin123" | jq -r '.access_token')

# 2. Train RandomForest
curl -X POST http://localhost:8000/models/train \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "RandomForest",
    "model_name": "my_model",
    "hyperparameters": {"n_estimators": 100, "max_depth": 10},
    "train_data": {
      "features": [[1,2], [3,4], [5,6]],
      "labels": [0, 1, 0]
    }
  }'
```

### Get predictions:
```bash
curl -X POST http://localhost:8000/models/my_model/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"features": [[2,3], [4,5]]}'
```

---

## 🧪 Testing

Run tests:
```bash
poetry run pytest tests/ -v
```

---

## 📁 Project Structure

```
mo/
├── src/
│   ├── api/
│   │   └── rest_api.py       # FastAPI application
│   ├── auth/
│   │   ├── jwt_handler.py    # JWT token management
│   │   └── user_manager.py   # User authentication
│   ├── models/
│   │   ├── base_model.py     # Base ML model class
│   │   ├── random_forest.py  # RandomForest implementation
│   │   ├── logistic_regression.py
│   │   ├── model_factory.py  # Model factory
│   │   └── model_storage.py  # Model persistence
│   ├── schemas/
│   │   └── models.py         # Pydantic schemas
│   └── utils/
│       └── logger.py         # Logging configuration
├── examples/
│   └── rest_api_auth.py      # Authentication examples
├── tests/
│   └── test_api.py           # API tests
│   └── TASK_DISTRIBUTION.md  # Распределение задач
├── pyproject.toml
├── README.md
└── .gitignore
```

├── pyproject.toml            # Poetry dependencies
├── .gitignore
└── README.md

```

---

## 📝 License

MIT
