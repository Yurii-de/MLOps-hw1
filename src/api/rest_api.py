"""REST API для работы с ML моделями."""

from datetime import timedelta
from typing import List

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from src import __version__
from src.auth.jwt_handler import create_access_token, verify_token, ACCESS_TOKEN_EXPIRE_MINUTES
from src.auth.user_manager import authenticate_user, create_user, get_user
from src.models.model_factory import ModelFactory
from src.models.model_storage import ModelStorage
from src.schemas.models import (
    AvailableModel,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
    Token,
    TrainModelRequest,
    TrainModelResponse,
    User,
    UserCreate,
)
from src.utils.logger import setup_logger

# Настройка логирования
logger = setup_logger()

# Создание приложения
app = FastAPI(
    title="ML API Service",
    description="API для обучения и использования ML моделей",
    version=__version__,
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация хранилища моделей
model_storage = ModelStorage()

# OAuth2 схема для токенов
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Получить текущего пользователя из токена.

    Args:
        token: JWT токен

    Returns:
        Данные пользователя

    Raises:
        HTTPException: Если токен невалиден
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = verify_token(token)

    if payload is None:
        logger.warning("Invalid token provided")
        raise credentials_exception

    username: str = payload.get("sub")

    if username is None:
        raise credentials_exception

    user = get_user(username)

    if user is None:
        raise credentials_exception

    if user.get("disabled"):
        raise HTTPException(status_code=400, detail="Inactive user")

    return user


# Эндпоинты аутентификации


@app.post("/auth/register", response_model=User, tags=["Authentication"])
async def register(user_data: UserCreate):
    """
    Регистрация нового пользователя.

    Args:
        user_data: Данные нового пользователя

    Returns:
        Информация о созданном пользователе

    Raises:
        HTTPException: Если пользователь уже существует
    """
    logger.info(f"Registration attempt for username: {user_data.username}")

    try:
        user = create_user(
            username=user_data.username, email=user_data.email, password=user_data.password
        )

        logger.info(f"User '{user_data.username}' registered successfully")

        return User(
            username=user["username"], email=user["email"], disabled=user["disabled"]
        )
    except ValueError as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/login", response_model=Token, tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Вход пользователя и получение JWT токена.

    Args:
        form_data: Данные формы (username, password)

    Returns:
        JWT токен

    Raises:
        HTTPException: Если аутентификация не удалась
    """
    logger.info(f"Login attempt for username: {form_data.username}")

    user = authenticate_user(form_data.username, form_data.password)

    if not user:
        logger.warning(f"Failed login attempt for username: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )

    logger.info(f"User '{form_data.username}' logged in successfully")

    return Token(access_token=access_token, token_type="bearer")


@app.get("/auth/me", response_model=User, tags=["Authentication"])
async def get_me(current_user: dict = Depends(get_current_user)):
    """
    Получить информацию о текущем пользователе.

    Args:
        current_user: Текущий пользователь из токена

    Returns:
        Информация о пользователе
    """
    return User(
        username=current_user["username"],
        email=current_user["email"],
        disabled=current_user["disabled"],
    )


# Основные эндпоинты (теперь с аутентификацией)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Проверка статуса сервиса.

    Returns:
        Информация о статусе сервиса
    """
    logger.info("Health check requested")

    models_count = len(model_storage.list_models())

    return HealthResponse(status="healthy", version=__version__, models_count=models_count)


@app.get(
    "/models",
    response_model=List[AvailableModel],
    tags=["Models"],
    summary="Получить список доступных типов моделей",
)
async def get_available_models(current_user: dict = Depends(get_current_user)):
    """
    Получить список всех доступных для обучения типов моделей.

    Args:
        current_user: Текущий пользователь (требуется аутентификация)

    Returns:
        Список доступных типов моделей с описанием и гиперпараметрами
    """
    logger.info(f"User '{current_user['username']}' fetching available model types")

    try:
        available_models = ModelFactory.get_available_models()
        return available_models
    except Exception as e:
        logger.error(f"Error fetching available models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/models/trained",
    response_model=List[ModelInfo],
    tags=["Models"],
    summary="Получить список обученных моделей",
)
async def get_trained_models(current_user: dict = Depends(get_current_user)):
    """
    Получить список всех обученных моделей.

    Args:
        current_user: Текущий пользователь (требуется аутентификация)

    Returns:
        Список обученных моделей с их параметрами
    """
    logger.info(f"User '{current_user['username']}' fetching trained models")

    try:
        model_ids = model_storage.list_models()
        models_info = []

        for model_id in model_ids:
            try:
                info = model_storage.get_model_info(model_id)
                models_info.append(ModelInfo(**info))
            except Exception as e:
                logger.warning(f"Failed to get info for model '{model_id}': {e}")
                continue

        return models_info
    except Exception as e:
        logger.error(f"Error fetching trained models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/models/train",
    response_model=TrainModelResponse,
    tags=["Models"],
    summary="Обучить новую модель",
)
async def train_model(request: TrainModelRequest, current_user: dict = Depends(get_current_user)):
    """
    Обучить новую ML модель.

    Args:
        request: Параметры обучения модели
        current_user: Текущий пользователь (требуется аутентификация)

    Returns:
        Информация об обученной модели

    Raises:
        HTTPException: Если модель с таким именем уже существует или произошла ошибка
    """
    logger.info(
        f"User '{current_user['username']}' training model '{request.model_name}' "
        f"of type '{request.model_type}' with hyperparameters: {request.hyperparameters}"
    )

    # Проверяем существование модели
    if model_storage.model_exists(request.model_name):
        logger.warning(f"Model '{request.model_name}' already exists")
        raise HTTPException(
            status_code=400, detail=f"Model '{request.model_name}' already exists. Use retrain endpoint."
        )

    try:
        # Создаем модель
        model = ModelFactory.create_model(
            model_type=request.model_type,
            model_id=request.model_name,
            hyperparameters=request.hyperparameters,
        )

        # Обучаем модель
        metrics = model.train(
            X=request.train_data.features, y=request.train_data.labels
        )

        # Сохраняем модель
        model_storage.save_model(model)

        logger.info(f"Model '{request.model_name}' trained successfully. Metrics: {metrics}")

        return TrainModelResponse(
            model_id=request.model_name,
            model_type=request.model_type,
            message="Model trained successfully",
            metrics=metrics,
        )

    except ValueError as e:
        logger.error(f"Validation error during training: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/models/{model_id}/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Получить предсказание модели",
)
async def predict(model_id: str, request: PredictRequest, current_user: dict = Depends(get_current_user)):
    """
    Получить предсказание от обученной модели.

    Args:
        model_id: ID модели
        request: Данные для предсказания
        current_user: Текущий пользователь (требуется аутентификация)

    Returns:
        Предсказания модели

    Raises:
        HTTPException: Если модель не найдена или произошла ошибка
    """
    logger.info(f"User '{current_user['username']}' prediction requested for model '{model_id}'")

    try:
        # Загружаем модель
        model = model_storage.load_model(model_id)

        # Получаем предсказания
        predictions = model.predict(request.features)

        # Получаем вероятности если доступно
        probabilities = model.predict_proba(request.features)

        logger.info(f"Predictions generated for model '{model_id}'")

        return PredictResponse(
            model_id=model_id, predictions=predictions, probabilities=probabilities
        )

    except FileNotFoundError as e:
        logger.error(f"Model '{model_id}' not found")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Validation error during prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/models/{model_id}/retrain",
    response_model=TrainModelResponse,
    tags=["Models"],
    summary="Переобучить существующую модель",
)
async def retrain_model(model_id: str, request: TrainModelRequest, current_user: dict = Depends(get_current_user)):
    """
    Переобучить существующую модель на новых данных.

    Args:
        model_id: ID модели для переобучения
        request: Параметры обучения
        current_user: Текущий пользователь (требуется аутентификация)

    Returns:
        Информация о переобученной модели

    Raises:
        HTTPException: Если модель не найдена или произошла ошибка
    """
    logger.info(f"User '{current_user['username']}' retraining model '{model_id}'")

    try:
        # Проверяем существование модели
        if not model_storage.model_exists(model_id):
            logger.error(f"Model '{model_id}' not found for retraining")
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

        # Создаем новую модель с теми же параметрами
        model = ModelFactory.create_model(
            model_type=request.model_type,
            model_id=model_id,
            hyperparameters=request.hyperparameters,
        )

        # Обучаем модель
        metrics = model.train(
            X=request.train_data.features, y=request.train_data.labels
        )

        # Сохраняем модель (перезаписываем)
        model_storage.save_model(model)

        logger.info(f"Model '{model_id}' retrained successfully. Metrics: {metrics}")

        return TrainModelResponse(
            model_id=model_id,
            model_type=request.model_type,
            message="Model retrained successfully",
            metrics=metrics,
        )

    except ValueError as e:
        logger.error(f"Validation error during retraining: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete(
    "/models/{model_id}",
    tags=["Models"],
    summary="Удалить модель",
)
async def delete_model(model_id: str, current_user: dict = Depends(get_current_user)):
    """
    Удалить обученную модель.

    Args:
        model_id: ID модели для удаления
        current_user: Текущий пользователь (требуется аутентификация)

    Returns:
        Сообщение об успешном удалении

    Raises:
        HTTPException: Если модель не найдена или произошла ошибка
    """
    logger.info(f"User '{current_user['username']}' deleting model '{model_id}'")

    try:
        model_storage.delete_model(model_id)
        logger.info(f"Model '{model_id}' deleted successfully")

        return {"message": f"Model '{model_id}' deleted successfully"}

    except FileNotFoundError as e:
        logger.error(f"Model '{model_id}' not found for deletion")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
