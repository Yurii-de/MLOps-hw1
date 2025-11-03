"""Утилиты для логирования."""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "ml_api_service", log_file: str = "logs/app.log") -> logging.Logger:
    """
    Настройка логгера для приложения.

    Args:
        name: Имя логгера
        log_file: Путь к файлу логов

    Returns:
        Настроенный логгер
    """
    # Создаем директорию для логов если её нет
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Если хендлеры уже добавлены, не добавляем повторно
    if logger.handlers:
        return logger

    # Формат логов
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Хендлер для файла
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Хендлер для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Добавляем хендлеры
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
