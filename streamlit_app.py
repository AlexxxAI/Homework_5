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

df = df.drop(columns=["name"])

# Вычисляем корреляцию с целевой переменной
correlations = df.corr()["status"].abs().sort_values(ascending=False)
valid_features = [col for col in correlations.index if df[col].nunique() > 10]
top_features = valid_features[:3]

# Разделяем данные
X = df[top_features[:2]]  # Два самых коррелирующих признака
y = df["status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Стандартизация
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение модели
logreg_model = LogisticRegression(max_iter=565)
logreg_model.fit(X_train_scaled, y_train)

# Интерфейс боковой панели
st.sidebar.header("Введите признаки:")
feature_names = {"spread1": "Разброс частот (spread1)", "PPE": "Дрожание голоса (PPE)"}

user_input = {}
for col in X.columns:
    user_input[col] = st.sidebar.slider(feature_names[col], float(df[col].min()), float(df[col].max()), float(df[col].mean()))

input_data = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_data)

# Кнопка предсказания
if st.sidebar.button("Сделать предсказание"):
    prediction = logreg_model.predict(input_scaled)
    prediction_proba = logreg_model.predict_proba(input_scaled)
    df_prediction_proba = pd.DataFrame(prediction_proba, columns=["Здоров", "Паркинсон"])
    
    st.subheader("🔍 Результаты предсказания:")
    if prediction[0] == 1:
        st.error("⚠️ Высокая вероятность болезни Паркинсона!")
    else:
        st.success("✅ Низкая вероятность болезни Паркинсона.")
    
    # График вероятностей
    fig_prob = px.bar(df_prediction_proba.T, title="Распределение вероятностей предсказания")
    st.plotly_chart(fig_prob)
    
    # Поясняющий текст
    if prediction_proba[0][1] > 0.8:
        st.warning("⚠️ Важно! Вероятность болезни выше 80%. Рекомендуется обратиться к специалисту.")

# Загрузка CSV-файла
st.subheader("📂 Загрузите CSV-файл для пакетного анализа")
uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])
if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)
    df_uploaded_scaled = scaler.transform(df_uploaded[top_features[:2]])
    batch_predictions = logreg_model.predict(df_uploaded_scaled)
    df_uploaded["prediction"] = batch_predictions
    st.write("🔍 Результаты пакетного предсказания:")
    st.dataframe(df_uploaded)

# Визуализации
st.subheader("📊 Визуализация данных")
fig = px.scatter(df, x=top_features[0], y=top_features[1], color="status", title="Два наиболее коррелирующих признака")
st.plotly_chart(fig)

fig_density = px.density_contour(df, x=top_features[0], y=top_features[1], color="status", title="Плотность распределения данных")
st.plotly_chart(fig_density)

fig3d = plt.figure(figsize=(12, 12))
ax = fig3d.add_subplot(111, projection="3d")
scatter = ax.scatter(df[top_features[0]], df[top_features[1]], df[top_features[2]], c=df["status"], cmap="coolwarm")
ax.set_xlabel(top_features[0])
ax.set_ylabel(top_features[1])
ax.set_zlabel(top_features[2])
plt.title("3D Визуализация трёх признаков")
st.pyplot(fig3d)
