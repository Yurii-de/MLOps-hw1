"""Менеджер для работы с DVC."""

import subprocess
from pathlib import Path
from typing import Optional

from src.logger import setup_logger

logger = setup_logger()


class DVCManager:
    """Менеджер для операций с DVC."""

    def __init__(self, repo_path: Optional[str] = None):
        """
        Инициализация DVC менеджера.

        Args:
            repo_path: Путь к репозиторию (по умолчанию текущая директория)
        """
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()

    def add_file(self, file_path: str | Path) -> bool:
        """
        Добавить файл под версионирование DVC.

        Args:
            file_path: Путь к файлу

        Returns:
            True если успешно
        """
        try:
            file_path = Path(file_path)
            cmd = ["dvc", "add", str(file_path)]
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"DVC added file: {file_path}")
            logger.debug(f"DVC output: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to add file to DVC: {e}")
            logger.error(f"DVC stderr: {e.stderr}")
            return False

    def push(self, remote: str = "myremote") -> bool:
        """
        Отправить изменения в remote storage.

        Args:
            remote: Имя remote

        Returns:
            True если успешно
        """
        try:
            cmd = ["dvc", "push", "-r", remote]
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("DVC push completed")
            logger.debug(f"DVC output: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to push to DVC remote: {e}")
            logger.error(f"DVC stderr: {e.stderr}")
            return False

    def pull(self, remote: str = "myremote") -> bool:
        """
        Скачать изменения из remote storage.

        Args:
            remote: Имя remote

        Returns:
            True если успешно
        """
        try:
            cmd = ["dvc", "pull", "-r", remote]
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("DVC pull completed")
            logger.debug(f"DVC output: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pull from DVC remote: {e}")
            logger.error(f"DVC stderr: {e.stderr}")
            return False


# Глобальный экземпляр
dvc_manager = DVCManager()
