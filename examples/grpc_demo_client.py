"""
Простой клиент для демонстрации работы с gRPC API.

Этот скрипт показывает основные операции:
1. Health Check
2. Список датасетов
3. Обучение модели на датасете
4. Предсказание

Требования:
- gRPC сервер должен быть запущен (python src/api/grpc_server.py)
- Датасет 'iris' должен существовать (python examples/recreate_shared_datasets.py)
"""

import sys
from pathlib import Path

# Добавляем родительскую директорию в путь для импортов
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import grpc

try:
    from src.proto import ml_service_pb2, ml_service_pb2_grpc
except ImportError:
    print("❌ ERROR: Proto files not generated!")
    print("Run: python generate_proto.py")
    exit(1)


def get_auth_token(stub):
    """
    Аутентификация и получение JWT токена.
    
    Args:
        stub: gRPC stub
        
    Returns:
        str: JWT токен
    """
    print("🔐 Authenticating...")
    login_request = ml_service_pb2.LoginRequest(
        username="admin",
        password="admin"
    )

    try:
        response = stub.Login(login_request)
        print("✅ Authenticated as: admin")
        print(f"🎫 Token type: {response.token_type}")
        return response.access_token
    except grpc.RpcError as e:
        print(f"❌ Authentication failed: {e.details()}")
        exit(1)


def main():
    """Запуск примеров gRPC API."""
    print("=" * 70)
    print("gRPC ML Service - Demo Client")
    print("=" * 70)
    print()

    # Подключение к серверу
    print("📡 Connecting to gRPC server at localhost:50051...")
    channel = grpc.insecure_channel("localhost:50051")
    stub = ml_service_pb2_grpc.MLServiceStub(channel)

    # Получение JWT токена
    token = get_auth_token(stub)

    # Метаданные с токеном для защищенных запросов
    metadata = [('authorization', f'Bearer {token}')]

    try:
        # =====================================================================
        # 1. Health Check (без авторизации)
        # =====================================================================
        print("\n" + "─" * 70)
        print("1️⃣  HEALTH CHECK")
        print("─" * 70)

        response = stub.HealthCheck(ml_service_pb2.Empty())
        print(f"✅ Status: {response.status}")
        print(f"📦 Version: {response.version}")
        print(f"🎯 Models count: {response.models_count}")

        # =====================================================================
        # 2. Список доступных типов моделей
        # =====================================================================
        print("\n" + "─" * 70)
        print("2️⃣  AVAILABLE MODEL TYPES")
        print("─" * 70)

        response = stub.ListAvailableModels(ml_service_pb2.Empty(), metadata=metadata)
        for model in response.models:
            print(f"\n📊 {model.name}")
            print(f"   Description: {model.description}")
            print(f"   Hyperparameters: {dict(model.default_hyperparameters)}")

        # =====================================================================
        # 3. Список датасетов
        # =====================================================================
        print("\n" + "─" * 70)
        print("3️⃣  DATASETS")
        print("─" * 70)

        response = stub.ListDatasets(ml_service_pb2.Empty(), metadata=metadata)
        if not response.datasets:
            print("⚠️  No datasets found!")
            print("   Run: python recreate_shared_datasets.py")
            return

        for dataset in response.datasets:
            print(f"\n📁 {dataset.dataset_id}")
            print(f"   Rows: {dataset.rows}, Columns: {dataset.columns}")
            print(f"   Target: {dataset.target_column}")
            print(f"   Features: {', '.join(dataset.feature_columns[:3])}{'...' if len(dataset.feature_columns) > 3 else ''}")

        # =====================================================================
        # 4. Обучение модели на датасете
        # =====================================================================
        print("\n" + "─" * 70)
        print("4️⃣  TRAIN MODEL FROM DATASET")
        print("─" * 70)

        print("🚀 Training RandomForest on 'iris' dataset...")

        response = stub.TrainModelFromDataset(
            ml_service_pb2.TrainFromDatasetRequest(
                model_type="RandomForest",
                model_name="grpc_demo_model",
                dataset_id="iris",
                hyperparameters={
                    "n_estimators": "50",
                    "max_depth": "5",
                    "random_state": "42"
                }
            ),
            metadata=metadata
        )

        print(f"✅ {response.message}")
        print("📈 Metrics:")
        for key, value in response.metrics.items():
            print(f"   {key}: {value:.4f}")

        # =====================================================================
        # 5. Список обученных моделей
        # =====================================================================
        print("\n" + "─" * 70)
        print("5️⃣  TRAINED MODELS")
        print("─" * 70)

        response = stub.ListTrainedModels(ml_service_pb2.Empty(), metadata=metadata)
        for model in response.models:
            print(f"\n🎯 {model.model_id}")
            print(f"   Type: {model.model_type}")
            print(f"   Created: {model.created_at}")

        # =====================================================================
        # 6. Предсказание (пример с Iris)
        # =====================================================================
        print("\n" + "─" * 70)
        print("6️⃣  PREDICTION")
        print("─" * 70)

        # Iris sample: sepal_length, sepal_width, petal_length, petal_width
        sample = [5.1, 3.5, 1.4, 0.2]  # Should predict class 0 (setosa)
        print(f"🔍 Input features: {sample}")

        response = stub.Predict(
            ml_service_pb2.PredictRequest(
                model_id="grpc_demo_model",
                features=[ml_service_pb2.FloatArray(values=sample)]
            ),
            metadata=metadata
        )

        print(f"📊 Prediction: {response.predictions[0]}")

        if response.probabilities:
            probs = response.probabilities[0].values
            print("📈 Probabilities:")
            for i, prob in enumerate(probs):
                print(f"   Class {i}: {prob:.4f}")

        # =====================================================================
        # 7. Удаление модели (cleanup)
        # =====================================================================
        print("\n" + "─" * 70)
        print("7️⃣  CLEANUP (Delete demo model)")
        print("─" * 70)

        response = stub.DeleteModel(
            ml_service_pb2.DeleteRequest(model_id="grpc_demo_model"),
            metadata=metadata
        )
        print(f"✅ {response.message}")

    except grpc.RpcError as e:
        print(f"\n❌ gRPC Error: {e.code()}")
        print(f"   Details: {e.details()}")
        print("\n💡 Make sure the gRPC server is running:")
        print("   python src/api/grpc_server.py")

    except Exception as e:
        print(f"\n❌ Error: {e}")

    finally:
        channel.close()

    print("\n" + "=" * 70)
    print("✅ Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
