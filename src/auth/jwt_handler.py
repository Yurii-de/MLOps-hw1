"""Модуль аутентификации с JWT токенами."""

from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

# Конфигурация
SECRET_KEY = "your-secret-key-change-in-production-use-env-variable"  # В продакшене из .env!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Контекст для хеширования паролей
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Проверить пароль.

    Args:
        plain_password: Пароль в открытом виде
        hashed_password: Хешированный пароль

    Returns:
        True если пароль совпадает
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Получить хеш пароля.

    Args:
        password: Пароль в открытом виде

    Returns:
        Хешированный пароль
    """
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Создать JWT токен.

    Args:
        data: Данные для включения в токен
        expires_delta: Время жизни токена

    Returns:
        JWT токен
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


def verify_token(token: str) -> Optional[dict]:
    """
    Проверить и декодировать JWT токен.

    Args:
        token: JWT токен

    Returns:
        Данные из токена или None если токен невалиден
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


# Алиас для обратной совместимости
decode_access_token = verify_token
