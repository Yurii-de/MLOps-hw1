#!/bin/bash

# Скрипт для генерации gRPC кода из proto файлов

echo "Generating gRPC code from proto files..."

# Генерация Python кода из proto файла
python -m grpc_tools.protoc \
    -I src/proto \
    --python_out=src/proto \
    --grpc_python_out=src/proto \
    src/proto/ml_service.proto

# Проверка успешности
if [ $? -eq 0 ]; then
    echo "✅ gRPC code generated successfully!"
    echo ""
    echo "Generated files:"
    echo "  - src/proto/ml_service_pb2.py"
    echo "  - src/proto/ml_service_pb2_grpc.py"
    echo ""
    echo "You can now run the gRPC server:"
    echo "  python src/api/grpc_server.py"
else
    echo "❌ Error generating gRPC code"
    exit 1
fi
