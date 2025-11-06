"""Streamlit дашборд для работы с ML API."""

import json
from typing import Any, Dict

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


def register_user(username: str, email: str, password: str):
    """Зарегистрировать пользователя."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/register",
            json={"username": username, "email": email, "password": password},
        )
        response.raise_for_status()
        return True, "Регистрация успешна! Теперь можете войти."
    except Exception as e:
        return False, f"Ошибка регистрации: {e}"

st.title("🤖 ML API Service Dashboard")
st.markdown("Интерактивный интерфейс для обучения и использования ML моделей")


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
    except Exception as e:
        st.error(f"Ошибка при получении предсказания: {e}")
        return None


def delete_model(model_id: str):
    """Удалить модель."""
    try:
        response = requests.delete(f"{API_BASE_URL}/models/{model_id}", headers=get_headers())
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка при удалении модели: {e}")
        return None


def health_check():
    """Проверить статус API."""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}


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

        st.info("**Тестовые пользователи:**\n- admin / admin123\n- user / user123")

    with tab2:
        st.header("Регистрация")

        with st.form("register_form"):
            new_username = st.text_input("Имя пользователя", key="reg_username")
            new_email = st.text_input("Email", key="reg_email")
            new_password = st.text_input("Пароль", type="password", key="reg_password")
            register_submit = st.form_submit_button("Зарегистрироваться")

            if register_submit:
                if new_username and new_email and new_password:
                    success, message = register_user(new_username, new_email, new_password)
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
        ["🎯 Обучить модель", "🔮 Получить предсказание", "📋 Управление моделями"],
    )


# Основное содержимое
if page == "🎯 Обучить модель":
    st.header("🎯 Обучение новой модели")

    # Получаем доступные типы моделей
    available_models = get_available_models()

    if not available_models:
        st.warning("Нет доступных типов моделей")
    else:
        # Выбор типа модели
        model_names = [m["name"] for m in available_models]
        selected_model_type = st.selectbox("Выберите тип модели:", model_names)

        # Находим выбранную модель
        selected_model = next(m for m in available_models if m["name"] == selected_model_type)

        # Показываем описание
        with st.expander("ℹ️ Описание модели"):
            st.write(selected_model["description"])

        # Имя модели
        model_name = st.text_input("Введите имя для модели:", value="my_model")

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
                            key, value=default_value, min_value=0, max_value=9999
                        )
                    else:
                        hyperparameters[key] = st.number_input(
                            key, value=default_value, min_value=1
                        )
                elif isinstance(default_value, float):
                    hyperparameters[key] = st.number_input(
                        key, value=default_value, min_value=0.0, format="%.4f"
                    )
                elif isinstance(default_value, str):
                    hyperparameters[key] = st.text_input(key, value=default_value)
                elif default_value is None:
                    use_none = st.checkbox(f"Использовать None для {key}", value=True)
                    if not use_none:
                        hyperparameters[key] = st.number_input(f"{key} (значение)", value=10)
                    else:
                        hyperparameters[key] = None

        # Данные для обучения
        st.subheader("📊 Данные для обучения")

        data_input_method = st.radio(
            "Способ ввода данных:", ["Ручной ввод (JSON)", "Загрузить CSV", "Пример данных"]
        )

        features = None
        labels = None

        if data_input_method == "Ручной ввод (JSON)":
            col1, col2 = st.columns(2)

            with col1:
                features_input = st.text_area(
                    "Признаки (features):",
                    value='[[1, 2], [3, 4], [5, 6], [7, 8]]',
                    height=150,
                )

            with col2:
                labels_input = st.text_area("Метки (labels):", value="[0, 1, 0, 1]", height=150)

            try:
                features = json.loads(features_input)
                labels = json.loads(labels_input)

                st.success(f"✅ Данные: {len(features)} примеров, {len(features[0])} признаков")
            except Exception as e:
                st.error(f"Ошибка парсинга JSON: {e}")

        elif data_input_method == "Загрузить CSV":
            uploaded_file = st.file_uploader("Загрузите CSV файл", type="csv")

            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.write("Предпросмотр данных:")
                st.dataframe(df.head())

                label_column = st.selectbox("Выберите колонку с метками:", df.columns)

                if label_column:
                    feature_columns = [col for col in df.columns if col != label_column]
                    features = df[feature_columns].values.tolist()
                    labels = df[label_column].values.tolist()

                    st.success(
                        f"✅ Загружено: {len(features)} примеров, {len(feature_columns)} признаков"
                    )

        else:  # Пример данных
            st.info("Используются примерные данные для демонстрации")
            features = [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]]
            labels = [0, 1, 0, 1, 0, 1]

            df = pd.DataFrame(features, columns=["feature_1", "feature_2"])
            df["label"] = labels
            st.dataframe(df)

        # Кнопка обучения
        if st.button("🚀 Обучить модель", type="primary"):
            if features and labels:
                with st.spinner("Обучение модели..."):
                    result = train_model(
                        selected_model_type, model_name, hyperparameters, features, labels
                    )

                if result:
                    st.success(f"✅ {result['message']}")
                    st.json(result)
            else:
                st.error("Пожалуйста, предоставьте данные для обучения")


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

        with st.expander("ℹ️ Информация о модели"):
            st.json(selected_model_info)

        # Ввод данных для предсказания
        st.subheader("📊 Данные для предсказания")

        pred_input_method = st.radio(
            "Способ ввода:", ["Ручной ввод (JSON)", "Загрузить CSV", "Пример данных"], key="pred"
        )

        pred_features = None

        if pred_input_method == "Ручной ввод (JSON)":
            features_input = st.text_area(
                "Признаки (features):", value="[[2, 3], [4, 5]]", height=150
            )

            try:
                pred_features = json.loads(features_input)
                st.success(f"✅ {len(pred_features)} примеров для предсказания")
            except Exception as e:
                st.error(f"Ошибка парсинга JSON: {e}")

        elif pred_input_method == "Загрузить CSV":
            uploaded_file = st.file_uploader("Загрузите CSV файл", type="csv", key="pred_csv")

            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.write("Предпросмотр данных:")
                st.dataframe(df.head())

                pred_features = df.values.tolist()
                st.success(f"✅ {len(pred_features)} примеров загружено")

        else:  # Пример данных
            pred_features = [[2, 3], [4, 5], [6, 7]]
            df = pd.DataFrame(pred_features, columns=["feature_1", "feature_2"])
            st.dataframe(df)

        # Кнопка предсказания
        if st.button("🔮 Получить предсказание", type="primary"):
            if pred_features:
                with st.spinner("Получение предсказаний..."):
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

        # Показываем таблицу с моделями
        models_df = pd.DataFrame(trained_models)
        st.dataframe(models_df, use_container_width=True)

        # Удаление модели
        st.divider()
        st.subheader("🗑️ Удаление модели")

        model_ids = [m["model_id"] for m in trained_models]
        model_to_delete = st.selectbox("Выберите модель для удаления:", model_ids, key="delete")

        col1, col2 = st.columns([1, 4])

        with col1:
            if st.button("🗑️ Удалить", type="secondary"):
                with st.spinner(f"Удаление модели '{model_to_delete}'..."):
                    result = delete_model(model_to_delete)

                if result:
                    st.success(f"✅ {result['message']}")
                    st.rerun()


# Footer
st.divider()
st.markdown("---")
st.markdown("**ML API Service Dashboard** | Built with Streamlit")
