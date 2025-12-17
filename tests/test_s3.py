import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock minio before importing src.s3_client to avoid connection attempts during import
mock_minio_module = MagicMock()
sys.modules["minio"] = mock_minio_module
sys.modules["minio.error"] = MagicMock()

from src.s3_client import S3Client


@pytest.fixture
def mock_minio():
    # Since we mocked the module, S3Client uses the mock class.
    # We can configure the mock instance here.
    return mock_minio_module.Minio

def test_init_creates_bucket(mock_minio):
    client_instance = mock_minio.return_value
    client_instance.bucket_exists.return_value = False

    S3Client()

    client_instance.make_bucket.assert_called_once()

def test_upload_file(mock_minio):
    client_instance = mock_minio.return_value
    s3 = S3Client()
    result = s3.upload_file("test.txt", "test.txt")

    assert result is True
    client_instance.fput_object.assert_called_once()

def test_download_file(mock_minio):
    client_instance = mock_minio.return_value
    s3 = S3Client()

    with patch("src.s3_client.Path") as mock_path:
        result = s3.download_file("test.txt", "test.txt")

    assert result is True
    client_instance.fget_object.assert_called_once()
