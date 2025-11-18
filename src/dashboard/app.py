"""Streamlit дашборд для работы с ML API."""

from typing import Dict

import pandas as pd
import requests
import streamlit as st

# Конфигурация API
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="ML API Dashboard", page_icon="🤖", layout="wide", initial_sidebar_state="expanded"
)


# Управление токеном в session state
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "username" not in st.session_state:
    st.session_state.username = None


def get_headers():
    """Получить headers с токеном."""
    if st.session_state.access_token:
        return {"Authorization": f"Bearer {st.session_state.access_token}"}
    return {}


def login_user(username: str, password: str):
    """Войти в систему."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/login",
            data={"username": username, "password": password},
        )
        response.raise_for_status()
        data = response.json()
        st.session_state.access_token = data["access_token"]
        st.session_state.username = username
        return True, "Успешный вход!"
    except Exception as e:
        return False, f"Ошибка входа: {e}"


def logout_user():
    """Выйти из системы."""
    st.session_state.access_token = None
    st.session_state.username = None


def register_user(username: str, password: str):
    """Зарегистрировать пользователя."""
    try:
        # Генерируем email автоматически
        email = f"{username}@mlapi.local"

        response = requests.post(
            f"{API_BASE_URL}/auth/register",
            json={"username": username, "email": email, "password": password},
        )
        response.raise_for_status()
        return True, "Регистрация успешна! Теперь можете войти."
    except Exception as e:
        return False, f"Ошибка регистрации: {e}"


# Функции для работы с API
def get_available_models():
    """Получить список доступных типов моделей."""
    try:
        response = requests.get(f"{API_BASE_URL}/models", headers=get_headers())
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка при получении списка моделей: {e}")
        return []


def get_trained_models():
    """Получить список обученных моделей."""
    try:
        response = requests.get(f"{API_BASE_URL}/models/trained", headers=get_headers())
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка при получении обученных моделей: {e}")
        return []


def train_model(model_type: str, model_name: str, hyperparameters: Dict, features, labels):
    """Обучить модель."""
    try:
        payload = {
            "model_type": model_type,
            "model_name": model_name,
            "hyperparameters": hyperparameters,
            "train_data": {"features": features, "labels": labels},
        }

        response = requests.post(f"{API_BASE_URL}/models/train", json=payload, headers=get_headers())
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка при обучении модели: {e}")
        return None


def predict(model_id: str, features):
    """Получить предсказание."""
    try:
        payload = {"features": features}
        response = requests.post(
            f"{API_BASE_URL}/models/{model_id}/predict", json=payload, headers=get_headers()
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        # Показываем детальную ошибку от сервера
        try:
            error_detail = e.response.json().get("detail", str(e))
            st.error(f"Ошибка сервера: {error_detail}")
        except Exception:
            st.error(f"Ошибка при получении предсказания: {e}")
        return None
    except Exception as e:
        st.error(f"Ошибка при получении предсказания: {e}")
        return None


def predict_csv_from_dataset(model_id: str, dataset_id: str, csv_file):
    """Получить предсказание из CSV с автоматическим кодированием."""
    try:
        files = {'file': ('data.csv', csv_file, 'text/csv')}
        data = {'dataset_id': dataset_id}

        response = requests.post(
            f"{API_BASE_URL}/models/{model_id}/predict-csv",
            files=files,
            data=data,
            headers=get_headers()
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        try:
            error_detail = e.response.json().get("detail", str(e))
            st.error(f"Ошибка сервера: {error_detail}")
        except Exception:
            st.error(f"Ошибка при получении предсказания: {e}")
        return None
    except Exception as e:
        st.error(f"Ошибка при получении предсказания: {e}")
        return None


# ─────────────────────────────────────────────────────────────────

def has_feature_encoders(dataset_id: str) -> bool:
    """Проверить, есть ли у датасета энкодеры признаков (не только таргета)."""
    from pathlib import Path

    # Проверяем наличие папки с энкодерами
    encoders_dir = Path("datasets") / f"{dataset_id}_encoders"

    if not encoders_dir.exists():
        return False

    # Проверяем, что есть хотя бы один энкодер (не считая target_encoder)
    encoder_files = list(encoders_dir.glob("*.json"))

    # У iris только target_encoder, у adult - много энкодеров для признаков
    return len(encoder_files) > 0


def delete_model(model_id: str):
    """Удалить модель."""
    try:
        response = requests.delete(f"{API_BASE_URL}/models/{model_id}", headers=get_headers())
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка при удалении модели: {e}")
        return None


def delete_dataset(dataset_id: str):
    """Удалить датасет."""
    try:
        response = requests.delete(f"{API_BASE_URL}/datasets/{dataset_id}", headers=get_headers())
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка при удалении датасета: {e}")
        return None


def health_check():
    """Проверить статус API."""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}


def upload_dataset(file, target_column: str, dataset_name: str = None, preprocess_categorical: bool = True, make_shared: bool = False):
    """Загрузить датасет."""
    try:
        files = {"file": (file.name, file, "text/csv")}
        data = {
            "target_column": target_column,
            "preprocess_categorical": str(preprocess_categorical).lower(),
            "make_shared": str(make_shared).lower()
        }
        if dataset_name:
            data["dataset_name"] = dataset_name

        response = requests.post(
            f"{API_BASE_URL}/datasets/upload",
            headers=get_headers(),
            files=files,
            data=data
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        # Извлекаем детальное сообщение об ошибке из ответа API
        try:
            error_detail = e.response.json().get("detail", str(e))
        except Exception:
            error_detail = str(e)
        st.error(f"❌ {error_detail}")
        return None
    except Exception as e:
        st.error(f"Ошибка при загрузке датасета: {e}")
        return None


def get_datasets():
    """Получить список датасетов."""
    try:
        response = requests.get(f"{API_BASE_URL}/datasets", headers=get_headers())
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка при получении списка датасетов: {e}")
        return []


def get_dataset_info(dataset_id: str):
    """Получить информацию о датасете."""
    try:
        response = requests.get(f"{API_BASE_URL}/datasets/{dataset_id}", headers=get_headers())
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка при получении информации о датасете: {e}")
        return None


def train_model_from_dataset(model_type: str, model_name: str, dataset_id: str, hyperparameters: Dict):
    """Обучить модель на датасете."""
    try:
        payload = {
            "model_type": model_type,
            "model_name": model_name,
            "dataset_id": dataset_id,
            "hyperparameters": hyperparameters,
        }
        response = requests.post(
            f"{API_BASE_URL}/models/train-from-dataset",
            json=payload,
            headers=get_headers()
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка при обучении модели на датасете: {e}")
        return None


# Проверка авторизации
if not st.session_state.access_token:
    st.title("🔐 Вход в систему")

    tab1, tab2 = st.tabs(["Вход", "Регистрация"])

    with tab1:
        st.header("Войти")

        with st.form("login_form"):
            username = st.text_input("Имя пользователя")
            password = st.text_input("Пароль", type="password")
            submit = st.form_submit_button("Войти")

            if submit:
                if username and password:
                    success, message = login_user(username, password)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Введите имя пользователя и пароль")

        st.info("**Тестовый пользователь:**\n- admin / admin")

    with tab2:
        st.header("Регистрация")

        with st.form("register_form"):
            new_username = st.text_input("Имя пользователя", key="reg_username")
            new_password = st.text_input("Пароль", type="password", key="reg_password")
            new_password_confirm = st.text_input("Подтвердите пароль", type="password", key="reg_password_confirm")
            register_submit = st.form_submit_button("Зарегистрироваться")

            if register_submit:
                if new_username and new_password and new_password_confirm:
                    if new_password != new_password_confirm:
                        st.error("Пароли не совпадают")
                    elif len(new_password) < 6:
                        st.error("Пароль должен быть не менее 6 символов")
                    else:
                        success, message = register_user(new_username, new_password)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                else:
                    st.error("Заполните все поля")

    st.stop()


# Основной интерфейс (после авторизации)
st.title("🤖 ML API Service Dashboard")
st.markdown(f"Добро пожаловать, **{st.session_state.username}**!")

# Sidebar - Статус API
with st.sidebar:
    st.header("👤 Пользователь")
    st.write(f"**{st.session_state.username}**")

    if st.button("🚪 Выйти"):
        logout_user()
        st.rerun()

    st.divider()

    st.header("📊 Статус API")

    if st.button("🔄 Обновить статус"):
        st.rerun()

    health = health_check()

    if health.get("status") == "healthy":
        st.success("✅ API работает")
        st.metric("Версия", health.get("version", "N/A"))
        st.metric("Обученных моделей", health.get("models_count", 0))
    else:
        st.error("❌ API недоступен")
        st.write(health.get("error", ""))

    st.divider()

    st.header("📚 Навигация")
    page = st.radio(
        "Выберите действие:",
        ["📁 Управление датасетами", "🎯 Обучить модель", "🔮 Получить предсказание", "📋 Управление моделями"],
    )


# Основное содержимое
if page == "📁 Управление датасетами":
    st.header("📁 Управление датасетами")

    tab1, tab2 = st.tabs(["Загрузить датасет", "Мои датасеты"])

    with tab1:
        st.subheader("📤 Загрузка датасета")

        upload_method = st.radio(
            "Выберите способ загрузки:",
            ["📂 Выбрать готовый датасет", "📁 Загрузить свой CSV"],
            help="Выберите один из предоставленных датасетов или загрузите свой собственный"
        )

        if upload_method == "📂 Выбрать готовый датасет":
            st.write("**Доступные готовые датасеты:**")

            # Проверяем наличие файлов
            import os
            test_data_path = "test_data"
            available_datasets = {}

            if os.path.exists(os.path.join(test_data_path, "iris.csv")):
                available_datasets["Iris Dataset"] = {
                    "file": "iris.csv",
                    "description": "Классический датасет классификации ирисов (150 строк, 5 колонок)",
                    "target": "species"
                }

            if os.path.exists(os.path.join(test_data_path, "adult.csv")):
                available_datasets["Adult Income Dataset"] = {
                    "file": "adult.csv",
                    "description": "Датасет для предсказания уровня дохода (32561 строк, 15 колонок)",
                    "target": "income"
                }

            if not available_datasets:
                st.error("❌ Готовые датасеты не найдены в папке test_data/")
                st.info("Убедитесь, что файлы iris.csv и adult.csv находятся в папке test_data/")
            else:
                selected_dataset_name = st.selectbox(
                    "Выберите датасет:",
                    list(available_datasets.keys())
                )

                dataset_info = available_datasets[selected_dataset_name]

                # Показываем информацию о датасете
                st.info(f"ℹ️ {dataset_info['description']}")

                # Загружаем и показываем preview
                dataset_path = os.path.join(test_data_path, dataset_info['file'])
                df_preview = pd.read_csv(dataset_path)

                st.write("**Предпросмотр данных (первые 5 строк):**")
                st.dataframe(df_preview.head())

                st.write(f"**Размер датасета:** {len(df_preview)} строк, {len(df_preview.columns)} колонок")

                # Название для загрузки
                default_name = dataset_info['file'].replace('.csv', '')
                dataset_name = st.text_input(
                    "Название датасета в системе:",
                    value=default_name,
                    help="Имя под которым датасет будет сохранен в системе"
                )

                target_column = st.selectbox(
                    "Целевая переменная (target):",
                    options=df_preview.columns.tolist(),
                    index=df_preview.columns.tolist().index(dataset_info['target']) if dataset_info['target'] in df_preview.columns.tolist() else 0
                )

                preprocess_categorical = st.checkbox(
                    "Автоматическая предобработка категориальных переменных",
                    value=True,
                    help="Автоматически определит и закодирует категориальные признаки"
                )

                make_shared = st.checkbox(
                    "🌐 Сделать общим (доступен всем)",
                    value=True,  # По умолчанию шаблонные датасеты общие
                    help="Общие датасеты доступны всем пользователям и не могут быть удалены"
                )

                if st.button("🚀 Загрузить выбранный датасет", type="primary"):
                    with st.spinner(f"Загрузка датасета {selected_dataset_name}..."):
                        with open(dataset_path, 'rb') as f:
                            result = upload_dataset(
                                f,
                                target_column,
                                dataset_name,
                                preprocess_categorical,
                                make_shared
                            )

                        if result:
                            st.success(f"✅ {result['message']}")
                            st.write("**Информация о датасете:**")
                            st.json(result)

                            if preprocess_categorical and result.get('message'):
                                st.info("ℹ️ Категориальные переменные были автоматически закодированы")

        else:  # Загрузить свой CSV
            uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"])

            if uploaded_file is not None:
                # Предпросмотр датасета
                df = pd.read_csv(uploaded_file)
                st.write("**Предпросмотр данных (первые 5 строк):**")
                st.dataframe(df.head())

                st.write(f"**Размер датасета:** {len(df)} строк, {len(df.columns)} колонок")

                # Настройки загрузки
                col1, col2 = st.columns(2)

                with col1:
                    dataset_name = st.text_input(
                        "Название датасета (опционально)",
                        placeholder="Оставьте пустым для автогенерации"
                    )

                    target_column = st.selectbox(
                        "Выберите целевую переменную (target):",
                        options=df.columns.tolist()
                    )

                with col2:
                    preprocess_categorical = st.checkbox(
                        "Автоматическая предобработка категориальных переменных",
                        value=True,
                        help="Автоматически определит и закодирует категориальные признаки"
                    )

                    make_shared = st.checkbox(
                        "🌐 Сделать общим (доступен всем)",
                        value=False,
                        help="Общие датасеты доступны всем пользователям и не могут быть удалены"
                    )

                    st.write("**Колонки датасета:**")
                    for col in df.columns:
                        dtype_icon = "🔢" if df[col].dtype in ['int64', 'float64'] else "📝"
                        st.text(f"{dtype_icon} {col}")

                if st.button("🚀 Загрузить датасет", type="primary"):
                    with st.spinner("Загрузка датасета..."):
                        # Сброс файла в начало
                        uploaded_file.seek(0)

                        result = upload_dataset(
                            uploaded_file,
                            target_column,
                            dataset_name if dataset_name else None,
                            preprocess_categorical,
                            make_shared
                        )

                        if result:
                            st.success(f"✅ {result['message']}")
                            st.write("**Информация о датасете:**")
                            st.json(result)

                            if preprocess_categorical and result.get('message'):
                                st.info("ℹ️ Категориальные переменные были автоматически закодированы")

    with tab2:
        st.subheader("📊 Загруженные датасеты")

        if st.button("🔄 Обновить список"):
            st.rerun()

        datasets = get_datasets()

        if not datasets:
            st.info("Нет загруженных датасетов. Загрузите датасет во вкладке 'Загрузить датасет'.")
        else:
            st.write(f"**Всего датасетов:** {len(datasets)}")

            # Получаем текущего пользователя
            current_username = st.session_state.get("username", "unknown")

            for dataset in datasets:
                # Проверяем владельца
                owner = dataset.get('owner')
                display_owner = owner if owner else "Общий"
                is_owner = owner == current_username
                icon = "📁" if is_owner else "🌐"

                with st.expander(f"{icon} {dataset['dataset_id']}", expanded=False):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Строк", dataset['rows'])
                    with col2:
                        st.metric("Колонок", dataset['columns'])
                    with col3:
                        st.metric("Target", dataset['target_column'])

                    st.write("**Признаки (features):**")
                    st.write(", ".join(dataset['feature_columns']))

                    st.write(f"**Создан:** {dataset['created_at']}")

                    # Владелец
                    if is_owner:
                        st.info(f"👤 Владелец: **{display_owner}** (вы)")
                    elif not owner:
                        st.info(f"👤 Владелец: **{display_owner}**")
                    else:
                        st.warning(f"👤 Владелец: **{display_owner}**")

                    # Проверка наличия энкодеров
                    has_encoders = has_feature_encoders(dataset['dataset_id'])
                    if has_encoders:
                        st.success("🔐 Датасет содержит энкодеры для категориальных признаков")

                    # Кнопка удаления с подтверждением
                    st.write("")  # Отступ

                    # Проверяем, можно ли удалить (только владелец или не общий датасет)
                    if not owner:
                        st.info("🔒 Общие датасеты нельзя удалить")
                    elif not is_owner:
                        st.warning("🔒 Только владелец может удалить этот датасет")
                    else:
                        # Проверяем, ожидается ли подтверждение
                        confirm_key = f"confirm_delete_dataset_{dataset['dataset_id']}"
                        if st.session_state.get(confirm_key, False):
                            st.warning("⚠️ Вы уверены? Это действие нельзя отменить!")

                            col_btn1, col_btn2 = st.columns(2)
                            with col_btn1:
                                if st.button("✅ Да, удалить", key=f"confirm_yes_{dataset['dataset_id']}", type="primary"):
                                    with st.spinner(f"Удаление датасета {dataset['dataset_id']}..."):
                                        result = delete_dataset(dataset['dataset_id'])
                                        if result:
                                            st.success(f"✅ Датасет {dataset['dataset_id']} удален")
                                            st.session_state[confirm_key] = False
                                            st.rerun()
                            with col_btn2:
                                if st.button("❌ Отмена", key=f"confirm_no_{dataset['dataset_id']}", type="secondary"):
                                    st.session_state[confirm_key] = False
                                    st.rerun()
                        else:
                            if st.button("🗑️ Удалить датасет", key=f"delete_dataset_{dataset['dataset_id']}", type="secondary"):
                                st.session_state[confirm_key] = True
                                st.rerun()

elif page == "🎯 Обучить модель":
    st.header("🎯 Обучение модели на датасете")

    # Получаем список датасетов
    datasets = get_datasets()

    if not datasets:
        st.warning("⚠️ Нет загруженных датасетов. Сначала загрузите датасет в разделе 'Управление датасетами'.")
    else:
        # Выбор датасета
        dataset_names = [ds['dataset_id'] for ds in datasets]
        selected_dataset_id = st.selectbox("Выберите датасет:", dataset_names)

        # Показываем информацию о датасете
        selected_dataset = next(ds for ds in datasets if ds['dataset_id'] == selected_dataset_id)

        # Подсчитываем количество признаков
        num_features = len(selected_dataset['feature_columns'])

        with st.expander("ℹ️ Информация о датасете", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Строк", selected_dataset['rows'])
            with col2:
                st.metric("Колонок", selected_dataset['columns'])
            with col3:
                st.metric("Признаков", num_features)
            with col4:
                st.metric("Target", selected_dataset['target_column'])

            st.write("**Признаки (features):**")
            st.write(", ".join(selected_dataset['feature_columns']))

        # Получаем доступные типы моделей
        available_models = get_available_models()

        if not available_models:
            st.warning("Нет доступных типов моделей")
        else:
            # Выбор типа модели
            model_names = [m["name"] for m in available_models]
            selected_model_type = st.selectbox("Выберите тип модели:", model_names, key="dataset_model_select")

            # Находим выбранную модель
            selected_model = next(m for m in available_models if m["name"] == selected_model_type)

            # Показываем описание
            with st.expander("ℹ️ Описание модели"):
                st.write(selected_model["description"])

            # Имя модели
            model_name = st.text_input(
                "Введите имя для модели:",
                value=f"model_{selected_dataset_id}"
            )

            # Гиперпараметры
            st.subheader("⚙️ Гиперпараметры")

            hyperparameters = {}
            default_hyperparams = selected_model["default_hyperparameters"]

            cols = st.columns(2)

            for idx, (key, default_value) in enumerate(default_hyperparams.items()):
                col = cols[idx % 2]

                with col:
                    if isinstance(default_value, int):
                        if key == "random_state":
                            hyperparameters[key] = st.number_input(
                                key, value=default_value, min_value=0, max_value=9999, key=f"dataset_{key}"
                            )
                        else:
                            hyperparameters[key] = st.number_input(
                                key, value=default_value, min_value=1, key=f"dataset_{key}"
                            )
                    elif isinstance(default_value, float):
                        hyperparameters[key] = st.number_input(
                            key, value=default_value, min_value=0.0, format="%.4f", key=f"dataset_{key}"
                        )
                    elif isinstance(default_value, str):
                        hyperparameters[key] = st.text_input(key, value=default_value, key=f"dataset_{key}")
                    elif default_value is None:
                        use_none = st.checkbox(f"Использовать None для {key}", value=True, key=f"dataset_none_{key}")
                        if not use_none:
                            hyperparameters[key] = st.number_input(f"{key} (значение)", value=10, key=f"dataset_val_{key}")
                        else:
                            hyperparameters[key] = None

            # Кнопка обучения
            if st.button("🚀 Обучить модель на датасете", type="primary"):
                with st.spinner("Обучение модели..."):
                    result = train_model_from_dataset(
                        selected_model_type,
                        model_name,
                        selected_dataset_id,
                        hyperparameters
                    )

                    if result:
                        st.success(f"✅ {result['message']}")
                        st.write("**Результаты обучения:**")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("ID модели", result['model_id'])
                        with col2:
                            st.metric("Тип модели", result['model_type'])

                        if result.get('metrics'):
                            st.write("**Метрики:**")
                            st.json(result['metrics'])


elif page == "🔮 Получить предсказание":
    st.header("🔮 Получение предсказаний")

    # Получаем обученные модели
    trained_models = get_trained_models()

    if not trained_models:
        st.warning("Нет обученных моделей. Сначала обучите модель.")
    else:
        # Выбор модели
        model_ids = [m["model_id"] for m in trained_models]
        selected_model_id = st.selectbox("Выберите модель:", model_ids)

        # Показываем информацию о модели
        selected_model_info = next(m for m in trained_models if m["model_id"] == selected_model_id)

        with st.expander("ℹ️ Информация о модели", expanded=True):
            # Показываем ключевую информацию метриками
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Тип модели", selected_model_info.get('model_type', 'N/A'))
            with col2:
                st.metric("Статус", "Обучена ✓" if selected_model_info.get('is_trained') else "Не обучена")
            with col3:
                n_features = selected_model_info.get('n_features', 'N/A')
                st.metric("Признаков", n_features)

            # Показываем полную информацию
            st.write("**Полная информация:**")
            st.json(selected_model_info)

        # Показываем важное предупреждение о количестве признаков
        n_features = selected_model_info.get('n_features', None)
        if n_features:
            st.warning(f"""
            ⚠️ **Важно:** Эта модель ожидает **{n_features} признаков** для предсказания. 
            Убедитесь, что вы предоставляете именно столько значений.
            """)
        else:
            st.info("""
            ℹ️ Информация о количестве признаков недоступна. 
            Убедитесь, что данные для предсказания соответствуют тем, на которых модель была обучена.
            """)


        # Данные для предсказания
        st.subheader("📊 Данные для предсказания")

        # Выбор датасета для энкодеров (если модель обучена на датасете с категориальными признаками)
        datasets = get_datasets()
        # Фильтруем только датасеты с энкодерами признаков (не только target)
        datasets_with_encoders = [d for d in datasets if has_feature_encoders(d["dataset_id"])]

        if datasets_with_encoders:
            dataset_options = ["Без энкодинга (числовые данные)"] + [d["dataset_id"] for d in datasets_with_encoders]
            selected_dataset = st.selectbox(
                "Датасет для энкодеров (выберите, если есть категориальные признаки):",
                dataset_options,
                help="Выберите датасет, на котором обучалась модель, для автоматического кодирования категориальных признаков"
            )
        else:
            st.info("ℹ️ Нет датасетов с категориальными признаками. Используется режим без энкодинга.")
            selected_dataset = "Без энкодинга (числовые данные)"

        use_encoding = selected_dataset != "Без энкодинга (числовые данные)"

        uploaded_file = st.file_uploader("Загрузите CSV файл с признаками", type="csv", key="pred_csv")

        pred_features = None
        num_features_uploaded = 0
        features_match = True
        uploaded_csv_bytes = None

        if uploaded_file:
            # Сохраняем содержимое файла для отправки на сервер
            uploaded_csv_bytes = uploaded_file.getvalue()

            df = pd.read_csv(uploaded_file)
            st.write("**Предпросмотр данных (первые 5 строк):**")
            st.dataframe(df.head())

            # Убираем target колонку если она есть
            feature_columns = [col for col in df.columns if col.lower() not in ['target', 'label', 'class', 'species', 'income']]

            if len(feature_columns) < len(df.columns):
                st.info(f"ℹ️ Обнаружена и удалена целевая колонка. Используются {len(feature_columns)} признаков.")
                df = df[feature_columns]

            num_features_uploaded = len(feature_columns)
            pred_features = df.values.tolist()
            st.success(f"✅ Загружено {len(pred_features)} примеров для предсказания")

            # Проверка соответствия количества признаков
            if use_encoding and selected_dataset:
                # Для режима с энкодингом проверяем соответствие датасету
                dataset_info = get_dataset_info(selected_dataset)
                if dataset_info:
                    expected_features = dataset_info['feature_columns']
                    expected_count = len(expected_features)
                    if num_features_uploaded != expected_count:
                        features_match = False
                        st.error(f"⚠️ Предупреждение: Датасет '{selected_dataset}' ожидает {expected_count} признаков, а загружено {num_features_uploaded}!")
                        st.info(f"📋 Ожидаемые колонки: {', '.join(expected_features[:5])}{'...' if len(expected_features) > 5 else ''}")
            elif not use_encoding and selected_model_info.get("n_features") is not None:
                # Для режима без энкодинга проверяем соответствие модели
                if num_features_uploaded != selected_model_info["n_features"]:
                    features_match = False
                    st.error(f"⚠️ Предупреждение: Модель ожидает {selected_model_info['n_features']} признаков, а загружено {num_features_uploaded}!")

        # Кнопка предсказания с динамическим цветом
        if not pred_features:
            st.warning("⚠️ Пожалуйста, загрузите CSV файл с данными для предсказания")
        elif not features_match:
            # Красная кнопка (secondary) при несовпадении признаков
            if st.button("⚠️ Получить предсказание (несовпадение признаков!)", type="secondary"):
                if use_encoding:
                    st.error("❌ Невозможно выполнить предсказание: количество признаков не совпадает с датасетом!")
                else:
                    st.error("❌ Невозможно выполнить предсказание: количество признаков не совпадает с моделью!")
        else:
            # Синяя кнопка (primary) при совпадении признаков
            if st.button("🔮 Получить предсказание", type="primary"):
                if pred_features:
                    with st.spinner("Получение предсказаний..."):
                        if use_encoding:
                            # Используем новый метод с автоматическим кодированием
                            import io
                            csv_file = io.BytesIO(uploaded_csv_bytes)
                            result = predict_from_csv(selected_model_id, selected_dataset, csv_file)
                        else:
                            # Используем старый метод для числовых данных
                            result = predict(selected_model_id, pred_features)

                    if result:
                        st.success("✅ Предсказания получены")

                        # Показываем результаты
                        results_df = pd.DataFrame(
                            {
                                "Sample": range(1, len(result["predictions"]) + 1),
                                "Prediction": result["predictions"],
                            }
                        )

                        if result.get("probabilities"):
                            for i, probs in enumerate(result["probabilities"]):
                                for j, prob in enumerate(probs):
                                    results_df[f"Class_{j}_Prob"] = [
                                        p[j] for p in result["probabilities"]
                                    ]

                        st.dataframe(results_df)

                        st.json(result)
                else:
                    st.error("Пожалуйста, предоставьте данные для предсказания")


else:  # Управление моделями
    st.header("📋 Управление моделями")

    # Получаем обученные модели
    trained_models = get_trained_models()

    if not trained_models:
        st.info("Нет обученных моделей")
    else:
        st.subheader(f"Всего моделей: {len(trained_models)}")

        # Группируем модели по владельцу
        current_username = st.session_state.get("username", "unknown")
        my_models = [m for m in trained_models if m.get("owner") == current_username]
        other_models = [m for m in trained_models if m.get("owner") != current_username]

        # Показываем свои модели
        if my_models:
            st.write(f"**👤 Ваши модели ({len(my_models)}):**")
            for model in my_models:
                owner = model.get('owner')
                display_owner = owner if owner else "Общий"

                with st.expander(f"🎯 {model['model_id']}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Тип", model.get('model_type', 'N/A'))
                    with col2:
                        st.metric("Статус", "✅ Обучена" if model.get('is_trained') else "❌ Не обучена")
                    with col3:
                        st.metric("Признаков", model.get('n_features', 'N/A'))

                    st.info(f"👤 Владелец: **{display_owner}** (вы)")
                    st.write(f"📅 Создана: {model.get('created_at', 'N/A')}")

        # Показываем чужие модели
        if other_models:
            st.write(f"**👥 Модели других пользователей ({len(other_models)}):**")
            for model in other_models:
                owner = model.get('owner')
                display_owner = owner if owner else "Общий"

                with st.expander(f"🌐 {model['model_id']}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Тип", model.get('model_type', 'N/A'))
                    with col2:
                        st.metric("Статус", "✅ Обучена" if model.get('is_trained') else "❌ Не обучена")
                    with col3:
                        st.metric("Признаков", model.get('n_features', 'N/A'))

                    if owner:
                        st.warning(f"👤 Владелец: **{display_owner}**")
                    else:
                        st.info(f"👤 Владелец: **{display_owner}**")
                    st.write(f"📅 Создана: {model.get('created_at', 'N/A')}")

        # Удаление модели
        st.divider()
        st.subheader("🗑️ Удаление модели")

        model_ids = [m["model_id"] for m in trained_models]
        model_to_delete = st.selectbox("Выберите модель для удаления:", model_ids, key="delete")

        # Проверяем владельца
        selected_model = next((m for m in trained_models if m["model_id"] == model_to_delete), None)
        model_owner = selected_model.get("owner") if selected_model else None
        is_owner = model_owner == current_username

        col1, col2 = st.columns([1, 4])

        with col1:
            if st.button("🗑️ Удалить", type="secondary"):
                if not model_owner:
                    st.error("❌ Общие модели нельзя удалить")
                elif not is_owner:
                    st.warning("⚠️ Вы не являетесь владельцем этой модели")
                else:
                    with st.spinner(f"Удаление модели '{model_to_delete}'..."):
                        result = delete_model(model_to_delete)

                    if result:
                        st.success(f"✅ {result['message']}")
                        st.rerun()


# Footer
st.divider()
st.markdown("---")
st.markdown("**ML API Service Dashboard** | Built with Streamlit")
