"""Управление пользователями."""

from typing import Optional

from src.auth.jwt_handler import get_password_hash, verify_password

# Простое хранилище пользователей (в продакшене использовать БД)
# Пароль: admin
USERS_DB = {
    "admin": {
        "username": "admin",
        "email": "admin@example.com",
        "hashed_password": "$2b$12$AZe3X7DXDd9lljvdmL3gmeJ0pPqtW4pBcV8WqaqTg8J7.2uTMnHri",
        "disabled": False,
    },
}


def get_user(username: str) -> Optional[dict]:
    """
    Получить пользователя по username.

    Args:
        username: Имя пользователя

    Returns:
        Данные пользователя или None
    """
    if username in USERS_DB:
        return USERS_DB[username]
    return None


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """
    Аутентифицировать пользователя.

    Args:
        username: Имя пользователя
        password: Пароль

    Returns:
        Данные пользователя или None если аутентификация не удалась
    """
    user = get_user(username)

    if not user:
        return None

    if not verify_password(password, user["hashed_password"]):
        return None

    return user


def create_user(username: str, email: str, password: str) -> dict:
    """
    Создать нового пользователя.

    Args:
        username: Имя пользователя
        email: Email
        password: Пароль

    Returns:
        Данные созданного пользователя

    Raises:
        ValueError: Если пользователь уже существует
    """
    if username in USERS_DB:
        raise ValueError(f"User '{username}' already exists")

    user_data = {
        "username": username,
        "email": email,
        "hashed_password": get_password_hash(password),
        "disabled": False,
    }

    USERS_DB[username] = user_data

    return user_data
