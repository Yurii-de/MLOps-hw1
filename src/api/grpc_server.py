"""gRPC сервер для работы с ML моделями."""

import json
from concurrent import futures

import grpc

# Импорты будут работать после генерации proto файлов
# python -m grpc_tools.protoc -I src/proto --python_out=src/proto --grpc_python_out=src/proto src/proto/ml_service.proto
try:
    from src.proto import ml_service_pb2, ml_service_pb2_grpc
except ImportError:
    print("WARNING: Proto files not generated. Run: python -m grpc_tools.protoc -I src/proto --python_out=src/proto --grpc_python_out=src/proto src/proto/ml_service.proto")
    ml_service_pb2 = None
    ml_service_pb2_grpc = None

from src import __version__
from src.models.model_factory import ModelFactory
from src.models.model_storage import ModelStorage
from src.utils.logger import setup_logger

logger = setup_logger()


class MLServiceServicer:
    """Реализация gRPC сервиса для ML моделей."""

    def __init__(self):
        """Инициализация сервиса."""
        self.model_storage = ModelStorage()
        logger.info("gRPC MLService initialized")

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
            metrics = model.train(X=features, y=labels)

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
            metrics = model.train(X=features, y=labels)

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

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
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
