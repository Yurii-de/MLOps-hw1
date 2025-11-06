"""Скрипт для генерации gRPC кода из proto файлов (Windows)."""

import subprocess
import sys
from pathlib import Path


def generate_proto():
    """Генерация Python кода из proto файлов."""
    print("Generating gRPC code from proto files...")

    proto_dir = Path("src/proto")
    proto_file = proto_dir / "ml_service.proto"

    if not proto_file.exists():
        print(f"❌ Proto file not found: {proto_file}")
        return False

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={proto_dir}",
        f"--grpc_python_out={proto_dir}",
        str(proto_file),
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        print("✅ gRPC code generated successfully!")
        print("\nGenerated files:")
        print("  - src/proto/ml_service_pb2.py")
        print("  - src/proto/ml_service_pb2_grpc.py")
        print("\nYou can now run the gRPC server:")
        print("  python src/api/grpc_server.py")

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating gRPC code: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


if __name__ == "__main__":
    success = generate_proto()
    sys.exit(0 if success else 1)
