import os
import subprocess
import sys
from pathlib import Path

# Устанавливаем правильный путь (родительская директория от examples/)
project_path = Path(__file__).resolve().parent.parent
os.chdir(str(project_path))
sys.path.insert(0, str(project_path))

print("=" * 60)
print("Starting Streamlit Dashboard...")
print("=" * 60)
print()
print("Dashboard будет доступен по адресу: http://localhost:8501")
print()

# Запускаем streamlit
subprocess.run(["streamlit", "run", "src/dashboard/app.py"])
