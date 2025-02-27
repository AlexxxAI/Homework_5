# Добавленный функционал:
# 1) Кнопка "Сделать предсказание" вместо мгновенного обновления при изменении ползунков.
# 2) График распределения вероятностей предсказания в виде столбцов.
# 3) Диаграмма плотности распределения данных
# 4) Если вероятность болезни >80%, выходит предупреждающий текст
# 5) Feature Importance – визуализация важности признаков.
# 6) Тепловая карта корреляций – интерактивный анализ данных.
# 7) Сохранение предсказаний – возможность скачать CSV

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

st.subheader("📌 Важность признаков")
shap_values = shap.Explainer(logreg_model, X_train_scaled)(X_test_scaled)
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test, feature_names=top_features[:2], show=False)
st.pyplot(fig)

st.subheader("🔥 Тепловая карта корреляций")
st.write(px.imshow(df.corr(), title="Матрица корреляций"))

st.subheader("📥 Сохранение предсказаний")
if st.button("💾 Скачать CSV с результатами"):
    result_df = pd.DataFrame(user_input, index=[0])
    result_df["Предсказание"] = "Паркинсон" if prediction[0] == 1 else "Здоров"
    result_df.to_csv("prediction_results.csv", index=False)
    st.download_button("📥 Скачать", "prediction_results.csv")
