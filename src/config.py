"""Конфигурация приложения."""

from pathlib import Path

# Базовая директория проекта (корень проекта)
BASE_DIR = Path(__file__).resolve().parent.parent

# Директории для хранения данных
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets"
LOGS_DIR = BASE_DIR / "logs"

# Настройки логирования
LOG_FILE = LOGS_DIR / "app.log"
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Настройки аутентификации JWT
SECRET_KEY = "your-secret-key-here"  # В продакшене должен быть в переменных окружения
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Настройки API
API_TITLE = "ML API Service"
API_DESCRIPTION = "API для обучения и использования ML моделей"

# Создание директорий при импорте
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
