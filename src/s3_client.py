"""Клиент для работы с MinIO/S3."""

from pathlib import Path

from minio import Minio
from minio.error import S3Error

from src.config import MINIO_ACCESS_KEY, MINIO_BUCKET_NAME, MINIO_ENDPOINT, MINIO_SECRET_KEY
from src.logger import setup_logger

logger = setup_logger()


class S3Client:
    """Клиент для работы с MinIO/S3 хранилищем."""

    def __init__(
        self,
        endpoint: str = MINIO_ENDPOINT,
        access_key: str = MINIO_ACCESS_KEY,
        secret_key: str = MINIO_SECRET_KEY,
        bucket_name: str = MINIO_BUCKET_NAME,
    ):
        """
        Инициализация S3 клиента.

        Args:
            endpoint: URL MinIO сервера
            access_key: Ключ доступа
            secret_key: Секретный ключ
            bucket_name: Имя бакета
        """
        self.endpoint = endpoint.replace("http://", "").replace("https://", "")
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name

        self.client = Minio(
            self.endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False,  # Для локального MinIO
        )

        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """Создает бакет, если он не существует."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created bucket: {self.bucket_name}")
            else:
                logger.info(f"Bucket {self.bucket_name} already exists")
        except S3Error as e:
            logger.error(f"Error creating bucket: {e}")

    def upload_file(self, local_path: str | Path, s3_key: str) -> bool:
        """
        Загружает файл в S3.

        Args:
            local_path: Локальный путь к файлу
            s3_key: Ключ в S3

        Returns:
            True если успешно
        """
        try:
            local_path = Path(local_path)
            self.client.fput_object(
                self.bucket_name,
                s3_key,
                str(local_path),
                content_type="application/octet-stream",
            )
            logger.info(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except S3Error as e:
            logger.error(f"Error uploading file: {e}")
            return False

    def download_file(self, s3_key: str, local_path: str | Path) -> bool:
        """
        Скачивает файл из S3.

        Args:
            s3_key: Ключ в S3
            local_path: Локальный путь для сохранения

        Returns:
            True если успешно
        """
        try:
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.client.fget_object(self.bucket_name, s3_key, str(local_path))
            logger.info(f"Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
        except S3Error as e:
            logger.error(f"Error downloading file: {e}")
            return False

    def list_objects(self, prefix: str = "") -> list[str]:
        """
        Список объектов в бакете.

        Args:
            prefix: Префикс для фильтрации

        Returns:
            Список ключей объектов
        """
        try:
            objects = self.client.list_objects(self.bucket_name, prefix=prefix)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            logger.error(f"Error listing objects: {e}")
            return []

    def delete_object(self, s3_key: str) -> bool:
        """
        Удаляет объект из S3.

        Args:
            s3_key: Ключ объекта

        Returns:
            True если успешно
        """
        try:
            self.client.remove_object(self.bucket_name, s3_key)
            logger.info(f"Deleted s3://{self.bucket_name}/{s3_key}")
            return True
        except S3Error as e:
            logger.error(f"Error deleting object: {e}")
            return False


# Глобальный экземпляр клиента
s3_client = S3Client()
