"""Утилиты для логирования."""

import logging
import sys

from src.config import LOG_DATE_FORMAT, LOG_FILE, LOG_FORMAT, LOG_LEVEL


def setup_logger(name: str = "ml_api_service", log_file: str = None) -> logging.Logger:
    """
    Настройка логгера для приложения.

    Args:
        name: Имя логгера
        log_file: Путь к файлу логов (если None, используется из config)

    Returns:
        Настроенный логгер
    """
    if log_file is None:
        log_file = str(LOG_FILE)

    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))

    # Если хендлеры уже добавлены, не добавляем повторно
    if logger.handlers:
        return logger

    # Формат логов
    formatter = logging.Formatter(
        LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
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
