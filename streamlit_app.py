import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Настройки страницы
st.set_page_config(page_title="Parkinson's Prediction", layout="centered")

# Заголовок приложения
st.title("🧠 Parkinson's Disease Prediction")
st.write("Введите параметры голоса, и модель предскажет вероятность заболевания.")

# Загружаем данные
file_path = "parkinsons.data"  # Локальный файл
df = pd.read_csv(file_path)

# Удаляем ненужный столбец "name"
df = df.drop(columns=["name"])

# Разделяем данные на признаки и целевую переменную
X = df.drop(columns=["status"])
y = df["status"]

# Разделяем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Стандартизация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучаем логистическую регрессию
model = LogisticRegression(max_iter=565, random_state=42)
model.fit(X_train_scaled, y_train)

# Оценка модели
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# Отображаем accuracy
st.write(f"📊 **Точность модели:** {accuracy:.2%}")

# Создаём боковую панель для ввода данных
st.sidebar.header("Введите признаки:")

# Поля для ввода параметров
user_input = {}
for col in X.columns:
    user_input[col] = st.sidebar.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

# Преобразуем ввод пользователя в DataFrame
input_data = pd.DataFrame([user_input])

# Нормализуем ввод
input_scaled = scaler.transform(input_data)

# Делаем предсказание
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# Отображаем результат
st.subheader("🔍 Результаты предсказания:")
if prediction[0] == 1:
    st.error("⚠️ Высокая вероятность болезни Паркинсона!")
else:
    st.success("✅ Низкая вероятность болезни Паркинсона.")

# Показываем вероятность классов
df_prediction_proba = pd.DataFrame(prediction_proba, columns=["Healthy", "Parkinson"])
st.dataframe(df_prediction_proba)

# Визуализация данных
st.subheader("📊 Визуализация данных")

fig = px.scatter(df, x="MDVP:Fo(Hz)", y="MDVP:Jitter(%)", color="status", title="Голосовые характеристики")
st.plotly_chart(fig)

fig2 = px.histogram(df, x="PPE", color="status", nbins=30, title="Распределение PPE (Pitch Period Entropy)")
st.plotly_chart(fig2)
