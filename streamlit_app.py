import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Настройки страницы
st.set_page_config(page_title="Parkinson's Prediction", layout="centered")

# Заголовок приложения
st.title("🧠 Прогнозирование болезни Паркинсона")
st.write("Введите параметры, и модель предскажет вероятность заболевания.")

# Загружаем данные
file_path = "https://raw.githubusercontent.com/AlexxxAI/HomeWorks/main/parkinsons.data"
df = pd.read_csv(file_path)

# Удаляем ненужный столбец "name"
df = df.drop(columns=["name"])

# Вычисляем корреляцию с целевой переменной
correlations = df.corr()["status"].abs().sort_values(ascending=False)

# Выбираем три наиболее коррелирующих признака
valid_features = [col for col in correlations.index if df[col].nunique() > 10]
top_features = valid_features[0:3]

# Разделяем данные на признаки и целевую переменную
X = df[top_features[:2]]  # Два самых коррелирующих признака
y = df["status"]

# Разделяем на обучающую (70%) и тестовую (30%) выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Выполняем стандартизацию признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучаем логистическую регрессию
logreg_model = LogisticRegression(max_iter=565)
logreg_model.fit(X_train_scaled, y_train)
y_pred = logreg_model.predict(X_test_scaled)

# Отображаем accuracy
st.write(f"📊 **Точность модели:** {logreg_model.score(X_test_scaled, y_test):.2%}")

# Создаём боковую панель для ввода данных
st.sidebar.header("Введите признаки:")

# Отображаемые названия для удобства
feature_names = {
    "spread1": "Разброс частот в голосе (spread1)",
    "PPE": "Дрожание голоса (PPE)"
}

# Изменяем подписи в боковой панели
user_input = {}
for col in X.columns:
    user_input[col] = st.sidebar.slider(feature_names[col], float(df[col].min()), float(df[col].max()), float(df[col].mean()))

# Преобразуем ввод пользователя в DataFrame
input_data = pd.DataFrame([user_input])

# Нормализуем ввод
input_scaled = scaler.transform(input_data)

# Делаем предсказание
prediction = logreg_model.predict(input_scaled)
prediction_proba = logreg_model.predict_proba(input_scaled)

# Отображаем результат
st.subheader("🔍 Результаты предсказания:")
if prediction[0] == 1:
    st.error("⚠️ Высокая вероятность болезни Паркинсона!")
else:
    st.success("✅ Низкая вероятность болезни Паркинсона.")

# Показываем вероятность классов
df_prediction_proba = pd.DataFrame(prediction_proba, columns=["Здоров", "Паркинсон"])
st.dataframe(df_prediction_proba)

# Визуализация данных
st.subheader("📊 Визуализация данных")

fig = px.scatter(df, x=top_features[0], y=top_features[1], color="status", title="Два наиболее коррелирующих признака")
st.plotly_chart(fig)

# 3D Визуализация
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(df[top_features[0]], df[top_features[1]], df[top_features[2]], c=df["status"], cmap="coolwarm")
ax.set_xlabel(top_features[0])
ax.set_ylabel(top_features[1])
ax.set_zlabel(top_features[2])
plt.title("3D Визуализация трех наиболее коррелирующих признаков")
st.pyplot(fig)
