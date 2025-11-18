"""gRPC сервер для работы с ML моделями."""

import json
import sys
from concurrent import futures
from pathlib import Path

import grpc

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

# Импорты будут работать после генерации proto файлов
try:
    from src.proto import ml_service_pb2, ml_service_pb2_grpc
except ImportError:
    print("WARNING: Proto files not generated. Run: python generate_proto.py")
    ml_service_pb2 = None
    ml_service_pb2_grpc = None

from src import __version__
from src.auth.jwt_handler import create_access_token, decode_access_token, verify_password
from src.logger import setup_logger
from src.models.dataset_storage import DatasetStorage
from src.models.model_manager import ModelFactory, ModelStorage

logger = setup_logger()

# Простая БД пользователей
USERS_DB = {
    "admin": {"username": "admin", "hashed_password": "$2b$12$AZe3X7DXDd9lljvdmL3gmeJ0pPqtW4pBcV8WqaqTg8J7.2uTMnHri"},  # admin
}

# Методы, не требующие аутентификации
UNPROTECTED_METHODS = [
    "/ml_service.MLService/Login",
    "/ml_service.MLService/HealthCheck",
]


def _get_user_from_metadata(context):
    """
    Извлечь пользователя из metadata токена.
    
    Args:
        context: gRPC контекст
        
    Returns:
        dict: Информация о пользователе или None
    """
    metadata = dict(context.invocation_metadata())
    token = metadata.get("authorization", "").replace("Bearer ", "")

    if not token:
        return None

    try:
        payload = decode_access_token(token)
        return {"username": payload.get("sub")}
    except Exception as e:
        logger.warning(f"Invalid token in gRPC: {e}")
        return None


class AuthInterceptor(grpc.ServerInterceptor):
    """Interceptor для JWT аутентификации в gRPC."""

    def intercept_service(self, continuation, handler_call_details):
        """Проверка JWT токена перед вызовом метода."""
        method = handler_call_details.method

        # Пропускаем методы без аутентификации
        if method in UNPROTECTED_METHODS:
            return continuation(handler_call_details)

        # Проверяем токен
        metadata = dict(handler_call_details.invocation_metadata)
        token = metadata.get("authorization", "").replace("Bearer ", "")

        if not token:
            logger.warning(f"gRPC: No token provided for {method}")
            return grpc.unary_unary_rpc_method_handler(
                lambda request, context: context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    "Authentication required. Please provide JWT token in metadata."
                )
            )

        try:
            decode_access_token(token)
            return continuation(handler_call_details)
        except Exception as e:
            error_msg = f"Invalid token: {str(e)}"
            logger.warning(f"gRPC: Invalid token for {method}: {e}")
            return grpc.unary_unary_rpc_method_handler(
                lambda request, context: context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    error_msg
                )
            )


class MLServiceServicer:
    """Реализация gRPC сервиса для ML моделей."""

    def __init__(self):
        """Инициализация сервиса."""
        self.model_storage = ModelStorage(storage_dir=str(BASE_DIR / "models"))
        self.dataset_storage = DatasetStorage(storage_dir=str(BASE_DIR / "datasets"))
        logger.info("gRPC MLService initialized")

    def Login(self, request, context):
        """
        Аутентификация пользователя.
        
        Args:
            request: LoginRequest с username и password
            context: gRPC контекст
            
        Returns:
            LoginResponse с access_token
        """
        logger.info(f"gRPC: Login attempt for user '{request.username}'")

        try:
            # Проверяем пользователя
            user = USERS_DB.get(request.username)
            if not user:
                logger.warning(f"gRPC: User '{request.username}' not found")
                context.set_code(grpc.StatusCode.UNAUTHENTICATED)
                context.set_details("Invalid username or password")
                return ml_service_pb2.LoginResponse()

            # Проверяем пароль
            if not verify_password(request.password, user["hashed_password"]):
                logger.warning(f"gRPC: Invalid password for user '{request.username}'")
                context.set_code(grpc.StatusCode.UNAUTHENTICATED)
                context.set_details("Invalid username or password")
                return ml_service_pb2.LoginResponse()

            # Создаем токен
            token = create_access_token(data={"sub": request.username})
            logger.info(f"gRPC: User '{request.username}' logged in successfully")

            return ml_service_pb2.LoginResponse(
                access_token=token,
                token_type="bearer"
            )

        except Exception as e:
            logger.error(f"gRPC error in Login: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.LoginResponse()

    def ListAvailableModels(self, request, context):
        """
        Получить список доступных типов моделей.

        Args:
            request: Пустой запрос
            context: gRPC контекст

        Returns:
            Список доступных моделей
        """
        logger.info("gRPC: ListAvailableModels called")

        try:
            available_models = ModelFactory.get_available_models()
            response_models = []

            for model_info in available_models:
                # Конвертируем гиперпараметры в строки
                hyperparams = {
                    k: json.dumps(v) for k, v in model_info["default_hyperparameters"].items()
                }

                model_msg = ml_service_pb2.AvailableModelInfo(
                    name=model_info["name"],
                    description=model_info["description"],
                    default_hyperparameters=hyperparams,
                )
                response_models.append(model_msg)

            return ml_service_pb2.AvailableModelsResponse(models=response_models)

        except Exception as e:
            logger.error(f"gRPC error in ListAvailableModels: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.AvailableModelsResponse()

    def TrainModel(self, request, context):
        """
        Обучить новую модель.

        Args:
            request: Запрос с параметрами обучения
            context: gRPC контекст

        Returns:
            Ответ с информацией об обученной модели
        """
        logger.info(f"gRPC: TrainModel called for '{request.model_name}'")

        try:
            # Проверяем существование модели
            if self.model_storage.model_exists(request.model_name):
                context.set_code(grpc.StatusCode.ALREADY_EXISTS)
                context.set_details(f"Model '{request.model_name}' already exists")
                return ml_service_pb2.TrainResponse()

            # Парсим гиперпараметры
            hyperparameters = None
            if request.hyperparameters:
                hyperparameters = {k: json.loads(v) for k, v in request.hyperparameters.items()}

            # Создаем модель
            model = ModelFactory.create_model(
                model_type=request.model_type,
                model_id=request.model_name,
                hyperparameters=hyperparameters,
            )

            # Подготавливаем данные
            features = [[val for val in row.values] for row in request.train_data.features]
            labels = list(request.train_data.labels)

            # Обучаем модель
            metrics = model.train(train_features=features, target=labels)

            # Сохраняем модель
            self.model_storage.save_model(model)

            logger.info(f"gRPC: Model '{request.model_name}' trained successfully")

            return ml_service_pb2.TrainResponse(
                model_id=request.model_name,
                model_type=request.model_type,
                message="Model trained successfully",
                metrics=metrics,
            )

        except ValueError as e:
            logger.error(f"gRPC validation error: {e}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return ml_service_pb2.TrainResponse()
        except Exception as e:
            logger.error(f"gRPC error in TrainModel: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.TrainResponse()

    def Predict(self, request, context):
        """
        Получить предсказание модели.

        Args:
            request: Запрос с данными для предсказания
            context: gRPC контекст

        Returns:
            Ответ с предсказаниями
        """
        logger.info(f"gRPC: Predict called for model '{request.model_id}'")

        try:
            # Загружаем модель
            model = self.model_storage.load_model(request.model_id)

            # Подготавливаем данные
            features = [[val for val in row.values] for row in request.features]

            # Получаем предсказания
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)

            # Формируем ответ
            prob_arrays = []
            if probabilities:
                prob_arrays = [
                    ml_service_pb2.FloatArray(values=prob) for prob in probabilities
                ]

            logger.info(f"gRPC: Predictions generated for model '{request.model_id}'")

            return ml_service_pb2.PredictResponse(
                model_id=request.model_id, predictions=predictions, probabilities=prob_arrays
            )

        except FileNotFoundError as e:
            logger.error(f"gRPC: Model not found: {e}")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return ml_service_pb2.PredictResponse()
        except Exception as e:
            logger.error(f"gRPC error in Predict: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.PredictResponse()

    def RetrainModel(self, request, context):
        """
        Переобучить существующую модель.

        Args:
            request: Запрос с параметрами переобучения
            context: gRPC контекст

        Returns:
            Ответ с информацией о переобученной модели
        """
        logger.info(f"gRPC: RetrainModel called for '{request.model_id}'")

        try:
            # Проверяем существование модели
            if not self.model_storage.model_exists(request.model_id):
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Model '{request.model_id}' not found")
                return ml_service_pb2.TrainResponse()

            # Парсим гиперпараметры
            hyperparameters = None
            if request.hyperparameters:
                hyperparameters = {k: json.loads(v) for k, v in request.hyperparameters.items()}

            # Создаем новую модель
            model = ModelFactory.create_model(
                model_type=request.model_type,
                model_id=request.model_id,
                hyperparameters=hyperparameters,
            )

            # Подготавливаем данные
            features = [[val for val in row.values] for row in request.train_data.features]
            labels = list(request.train_data.labels)

            # Обучаем модель
            metrics = model.train(train_features=features, target=labels)

            # Сохраняем модель
            self.model_storage.save_model(model)

            logger.info(f"gRPC: Model '{request.model_id}' retrained successfully")

            return ml_service_pb2.TrainResponse(
                model_id=request.model_id,
                model_type=request.model_type,
                message="Model retrained successfully",
                metrics=metrics,
            )

        except Exception as e:
            logger.error(f"gRPC error in RetrainModel: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.TrainResponse()

    def DeleteModel(self, request, context):
        """
        Удалить модель.

        Args:
            request: Запрос с ID модели
            context: gRPC контекст

        Returns:
            Ответ об удалении
        """
        logger.info(f"gRPC: DeleteModel called for '{request.model_id}'")

        try:
            self.model_storage.delete_model(request.model_id)
            logger.info(f"gRPC: Model '{request.model_id}' deleted successfully")

            return ml_service_pb2.DeleteResponse(
                message=f"Model '{request.model_id}' deleted successfully"
            )

        except FileNotFoundError as e:
            logger.error(f"gRPC: Model not found: {e}")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return ml_service_pb2.DeleteResponse()
        except Exception as e:
            logger.error(f"gRPC error in DeleteModel: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.DeleteResponse()

    def HealthCheck(self, request, context):
        """
        Проверка статуса сервиса.

        Args:
            request: Пустой запрос
            context: gRPC контекст

        Returns:
            Статус сервиса
        """
        logger.info("gRPC: HealthCheck called")

        try:
            models_count = len(self.model_storage.list_models())

            return ml_service_pb2.HealthCheckResponse(
                status="healthy", version=__version__, models_count=models_count
            )

        except Exception as e:
            logger.error(f"gRPC error in HealthCheck: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.HealthCheckResponse()

    def ListTrainedModels(self, request, context):
        """
        Получить список обученных моделей.

        Args:
            request: Пустой запрос
            context: gRPC контекст

        Returns:
            Список обученных моделей
        """
        logger.info("gRPC: ListTrainedModels called")

        try:
            model_ids = self.model_storage.list_models()
            trained_models = []

            for model_id in model_ids:
                try:
                    info = self.model_storage.get_model_info(model_id)

                    # Конвертируем гиперпараметры
                    hyperparams = {k: json.dumps(v) for k, v in info["hyperparameters"].items()}

                    model_msg = ml_service_pb2.TrainedModelInfo(
                        model_id=info["model_id"],
                        model_type=info["model_type"],
                        created_at=info["created_at"],
                        hyperparameters=hyperparams,
                    )
                    trained_models.append(model_msg)
                except Exception as e:
                    logger.warning(f"Failed to get info for model '{model_id}': {e}")
                    continue

            return ml_service_pb2.TrainedModelsResponse(models=trained_models)

        except Exception as e:
            logger.error(f"gRPC error in ListTrainedModels: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.TrainedModelsResponse()

    def UploadDataset(self, request, context):
        """
        Загрузить датасет.

        Args:
            request: Запрос с данными датасета
            context: gRPC контекст

        Returns:
            Информация о загруженном датасете
        """
        logger.info(f"gRPC: UploadDataset called for target='{request.target_column}'")

        try:
            import io

            import pandas as pd

            # Читаем CSV из bytes
            df = pd.read_csv(io.BytesIO(request.file_content))

            # Генерируем ID если не указан
            import uuid
            dataset_id = request.dataset_name if request.dataset_name else f"dataset_{uuid.uuid4().hex[:8]}"

            # Сохраняем датасет
            dataset_info = self.dataset_storage.save_dataset(
                dataset_id=dataset_id,
                df=df,
                target_column=request.target_column,
                preprocess_categorical=request.preprocess_categorical,
            )

            logger.info(f"Dataset '{dataset_id}' uploaded successfully")

            return ml_service_pb2.UploadDatasetResponse(
                dataset_id=dataset_info["dataset_id"],
                rows=dataset_info["rows"],
                columns=dataset_info["columns"],
                target_column=dataset_info["target_column"],
                feature_columns=dataset_info["feature_columns"],
                message=f"Dataset uploaded successfully with {dataset_info['rows']} rows",
            )

        except Exception as e:
            logger.error(f"gRPC error in UploadDataset: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.UploadDatasetResponse()

    def ListDatasets(self, request, context):
        """
        Получить список датасетов.

        Args:
            request: Пустой запрос
            context: gRPC контекст

        Returns:
            Список датасетов
        """
        logger.info("gRPC: ListDatasets called")

        try:
            dataset_ids = self.dataset_storage.list_datasets()
            datasets = []

            for dataset_id in dataset_ids:
                try:
                    info = self.dataset_storage.get_dataset_info(dataset_id)

                    dataset_msg = ml_service_pb2.DatasetInfoResponse(
                        dataset_id=info["dataset_id"],
                        rows=info["rows"],
                        columns=info["columns"],
                        target_column=info["target_column"],
                        feature_columns=info["feature_columns"],
                        created_at=info["created_at"],
                    )
                    datasets.append(dataset_msg)
                except Exception as e:
                    logger.warning(f"Failed to get info for dataset '{dataset_id}': {e}")
                    continue

            return ml_service_pb2.ListDatasetsResponse(datasets=datasets)

        except Exception as e:
            logger.error(f"gRPC error in ListDatasets: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.ListDatasetsResponse()

    def GetDatasetInfo(self, request, context):
        """
        Получить информацию о датасете.

        Args:
            request: Запрос с ID датасета
            context: gRPC контекст

        Returns:
            Информация о датасете
        """
        logger.info(f"gRPC: GetDatasetInfo called for dataset_id='{request.dataset_id}'")

        try:
            info = self.dataset_storage.get_dataset_info(request.dataset_id)

            return ml_service_pb2.DatasetInfoResponse(
                dataset_id=info["dataset_id"],
                rows=info["rows"],
                columns=info["columns"],
                target_column=info["target_column"],
                feature_columns=info["feature_columns"],
                created_at=info["created_at"],
            )

        except Exception as e:
            logger.error(f"gRPC error in GetDatasetInfo: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.DatasetInfoResponse()

    def TrainModelFromDataset(self, request, context):
        """
        Обучить модель на датасете.

        Args:
            request: Запрос с параметрами обучения
            context: gRPC контекст

        Returns:
            Результаты обучения
        """
        logger.info(
            f"gRPC: TrainModelFromDataset called: model='{request.model_name}', "
            f"dataset='{request.dataset_id}'"
        )

        try:
            # Загружаем данные из датасета
            train_features, target = self.dataset_storage.get_training_data(request.dataset_id)

            logger.info(f"Training data loaded: train_features shape {train_features.shape}, target shape {target.shape}")

            # Конвертируем гиперпараметры
            hyperparameters = {}
            for key, value in request.hyperparameters.items():
                try:
                    hyperparameters[key] = json.loads(value)
                except json.JSONDecodeError:
                    hyperparameters[key] = value

            # Создаем модель
            model = ModelFactory.create_model(
                model_type=request.model_type,
                model_id=request.model_name,
                hyperparameters=hyperparameters,
            )

            # Обучаем модель
            metrics = model.train(train_features.tolist(), target.tolist())

            # Сохраняем модель
            self.model_storage.save_model(model)

            logger.info(f"Model '{request.model_name}' trained successfully on dataset '{request.dataset_id}'")

            # Конвертируем метрики для protobuf
            metrics_dict = {k: float(v) for k, v in metrics.items()}

            return ml_service_pb2.TrainResponse(
                model_id=request.model_name,
                model_type=request.model_type,
                message=f"Model trained successfully on dataset '{request.dataset_id}'",
                metrics=metrics_dict,
            )

        except Exception as e:
            logger.error(f"gRPC error in TrainModelFromDataset: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ml_service_pb2.TrainResponse()


def serve(port: int = 50051):
    """
    Запустить gRPC сервер.

    Args:
        port: Порт для запуска сервера
    """
    if ml_service_pb2_grpc is None:
        print("ERROR: Proto files not generated. Cannot start server.")
        print("Run: python -m grpc_tools.protoc -I src/proto --python_out=src/proto --grpc_python_out=src/proto src/proto/ml_service.proto")
        return

    # Создаем сервер с AuthInterceptor
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=[AuthInterceptor()]
    )
    ml_service_pb2_grpc.add_MLServiceServicer_to_server(MLServiceServicer(), server)
    server.add_insecure_port(f"[::]:{port}")

    logger.info(f"Starting gRPC server on port {port}")
    server.start()
    print(f"gRPC server started on port {port}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server")
        server.stop(0)


if __name__ == "__main__":
    serve()
