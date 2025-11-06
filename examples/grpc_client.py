"""Пример использования gRPC клиента."""

import json

import grpc

# Импорт сгенерированных proto файлов
# Сначала нужно сгенерировать файлы командой:
# python -m grpc_tools.protoc -I src/proto --python_out=src/proto --grpc_python_out=src/proto src/proto/ml_service.proto
try:
    from src.proto import ml_service_pb2, ml_service_pb2_grpc
except ImportError:
    print(
        "ERROR: Proto files not generated. Run:\n"
        "python -m grpc_tools.protoc -I src/proto --python_out=src/proto "
        "--grpc_python_out=src/proto src/proto/ml_service.proto"
    )
    exit(1)


def run_examples():
    """Запустить примеры использования gRPC API."""
    # Подключение к серверу
    channel = grpc.insecure_channel("localhost:50051")
    stub = ml_service_pb2_grpc.MLServiceStub(channel)

    print("=" * 80)
    print("gRPC Client Examples - ML API Service")
    print("=" * 80)

    # 1. Health Check
    print("\n1. Health Check")
    print("-" * 40)
    try:
        response = stub.HealthCheck(ml_service_pb2.Empty())
        print(f"Status: {response.status}")
        print(f"Version: {response.version}")
        print(f"Models count: {response.models_count}")
    except grpc.RpcError as e:
        print(f"Error: {e.details()}")

    # 2. Получить список доступных моделей
    print("\n2. List Available Models")
    print("-" * 40)
    try:
        response = stub.ListAvailableModels(ml_service_pb2.Empty())
        for model in response.models:
            print(f"\nModel: {model.name}")
            print(f"Description: {model.description}")
            print("Default hyperparameters:")
            for key, value in model.default_hyperparameters.items():
                print(f"  {key}: {value}")
    except grpc.RpcError as e:
        print(f"Error: {e.details()}")

    # 3. Обучить Random Forest модель
    print("\n3. Train Random Forest Model")
    print("-" * 40)

    # Подготовка данных
    features_data = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [2.0, 3.0],
        [4.0, 5.0],
    ]
    labels_data = [0, 1, 0, 1, 0, 1]

    # Создаем массивы признаков
    feature_arrays = [ml_service_pb2.FloatArray(values=row) for row in features_data]

    # Создаем запрос
    train_data = ml_service_pb2.TrainingData(features=feature_arrays, labels=labels_data)

    hyperparameters = {
        "n_estimators": json.dumps(100),
        "max_depth": json.dumps(10),
        "random_state": json.dumps(42),
    }

    train_request = ml_service_pb2.TrainRequest(
        model_type="RandomForest",
        model_name="grpc_rf_model",
        hyperparameters=hyperparameters,
        train_data=train_data,
    )

    try:
        response = stub.TrainModel(train_request)
        print(f"Model ID: {response.model_id}")
        print(f"Model Type: {response.model_type}")
        print(f"Message: {response.message}")
        print("Metrics:")
        for key, value in response.metrics.items():
            print(f"  {key}: {value}")
    except grpc.RpcError as e:
        print(f"Error: {e.details()}")

    # 4. Получить список обученных моделей
    print("\n4. List Trained Models")
    print("-" * 40)
    try:
        response = stub.ListTrainedModels(ml_service_pb2.Empty())
        for model in response.models:
            print(f"\nModel ID: {model.model_id}")
            print(f"Type: {model.model_type}")
            print(f"Created: {model.created_at}")
    except grpc.RpcError as e:
        print(f"Error: {e.details()}")

    # 5. Получить предсказание
    print("\n5. Get Prediction")
    print("-" * 40)

    # Данные для предсказания
    pred_features = [[2.5, 3.5], [4.5, 5.5], [6.5, 7.5]]

    pred_arrays = [ml_service_pb2.FloatArray(values=row) for row in pred_features]

    predict_request = ml_service_pb2.PredictRequest(
        model_id="grpc_rf_model", features=pred_arrays
    )

    try:
        response = stub.Predict(predict_request)
        print(f"Model ID: {response.model_id}")
        print(f"Predictions: {list(response.predictions)}")

        if response.probabilities:
            print("\nProbabilities:")
            for i, prob_array in enumerate(response.probabilities):
                print(f"  Sample {i + 1}: {list(prob_array.values)}")
    except grpc.RpcError as e:
        print(f"Error: {e.details()}")

    # 6. Обучить Logistic Regression модель
    print("\n6. Train Logistic Regression Model")
    print("-" * 40)

    hyperparameters_lr = {
        "C": json.dumps(1.0),
        "max_iter": json.dumps(1000),
        "random_state": json.dumps(42),
    }

    train_request_lr = ml_service_pb2.TrainRequest(
        model_type="LogisticRegression",
        model_name="grpc_lr_model",
        hyperparameters=hyperparameters_lr,
        train_data=train_data,
    )

    try:
        response = stub.TrainModel(train_request_lr)
        print(f"Model ID: {response.model_id}")
        print(f"Message: {response.message}")
    except grpc.RpcError as e:
        print(f"Error: {e.details()}")

    # 7. Переобучить модель
    print("\n7. Retrain Model")
    print("-" * 40)

    # Новые данные
    new_features = [[0.5, 1.5], [2.5, 3.5], [4.5, 5.5], [6.5, 7.5]]
    new_labels = [0, 0, 1, 1]

    new_feature_arrays = [ml_service_pb2.FloatArray(values=row) for row in new_features]
    new_train_data = ml_service_pb2.TrainingData(features=new_feature_arrays, labels=new_labels)

    retrain_request = ml_service_pb2.RetrainRequest(
        model_id="grpc_rf_model",
        model_type="RandomForest",
        hyperparameters=hyperparameters,
        train_data=new_train_data,
    )

    try:
        response = stub.RetrainModel(retrain_request)
        print(f"Model ID: {response.model_id}")
        print(f"Message: {response.message}")
    except grpc.RpcError as e:
        print(f"Error: {e.details()}")

    # 8. Удалить модель
    print("\n8. Delete Model")
    print("-" * 40)

    delete_request = ml_service_pb2.DeleteRequest(model_id="grpc_lr_model")

    try:
        response = stub.DeleteModel(delete_request)
        print(f"Message: {response.message}")
    except grpc.RpcError as e:
        print(f"Error: {e.details()}")

    # Закрываем соединение
    channel.close()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    print("Starting gRPC client examples...")
    print("Make sure the gRPC server is running on localhost:50051\n")

    try:
        run_examples()
    except Exception as e:
        print(f"\nFatal error: {e}")
        print("\nMake sure:")
        print("1. gRPC server is running: python src/api/grpc_server.py")
        print("2. Proto files are generated")
