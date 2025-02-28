# Добавленный функционал:
# 1) Кнопка "Сделать предсказание" вместо мгновенного обновления при изменении ползунков.
# 2) Диаграмма плотности распределения данных
# 3) Если вероятность болезни >80%, выходит предупреждающий текст
# 4) Feature Importance – визуализация важности признаков.
# 5) Сохранение и загрузка пользовательского ввода – возможность сохранять и загружать введенные параметры.
# 6) Режимы интерфейса – "Простой" и "Расширенный" режимы для удобства.


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import shap
import io

# Настройки страницы
st.set_page_config(page_title="Parkinson's Prediction", layout="centered")

# Заголовок приложения
st.title("🧠 Прогнозирование болезни Паркинсона")
st.write("Введите параметры, и модель предскажет вероятность заболевания.")

# Добавляем режимы использования
mode = st.sidebar.radio("Выберите режим:", ["Простой", "Расширенный"])

# Загружаем данные
file_path = "https://raw.githubusercontent.com/AlexxxAI/HomeWorks/main/parkinsons.data"
df = pd.read_csv(file_path)

df = df.drop(columns=["name"])

st.write("Данные загружены!")

# Вычисляем корреляцию с целевой переменной
correlations = df.corr()["status"].abs().sort_values(ascending=False)
valid_features = [col for col in correlations.index if df[col].nunique() > 10]
top_features = valid_features[:2]  # Оставляем только два признака

# Разделяем данные
X = df[top_features]
y = df["status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Вывод исходных данных, если выбран расширенный режим
if mode == "Расширенный":
    with st.expander("Исходные данные"):
        st.write("**Признаки (X)**")
        st.dataframe(X)
        st.write("**Целевая переменная (y)**")
        st.dataframe(y)

# Стандартизация
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение модели логистической регрессии
logreg_model = LogisticRegression(max_iter=565)
logreg_model.fit(X_train_scaled, y_train)

# Интерфейс боковой панели
st.sidebar.header("Введите признаки:")
feature_names = {"spread1": "Разброс частот (spread1)", "PPE": "Дрожание голоса (PPE)"}

user_input = {}
for col in X.columns:
    user_input[col] = st.sidebar.slider(feature_names[col], float(df[col].min()), float(df[col].max()), float(df[col].mean()))

# Сохранение пользовательского ввода
if st.sidebar.button("💾 Сохранить ввод"):
    st.session_state["saved_input"] = user_input
    st.success("Ввод сохранен!")

if "saved_input" in st.session_state and st.sidebar.button("🔄 Загрузить сохраненный ввод"):
    user_input = st.session_state["saved_input"]
    st.success("Сохраненные параметры загружены!")

input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

# Вывод блока подготовки данных, если выбран расширенный режим
if mode == "Расширенный":
    with st.expander("Data Preparation"):
        st.write("**Input Data**")
        st.dataframe(input_df)

# Инициализация переменной предсказания как None
prediction = None

# Кнопка предсказания
if st.sidebar.button("Сделать предсказание"):
    prediction = logreg_model.predict(input_scaled)
    prediction_proba = logreg_model.predict_proba(input_scaled)
    df_prediction_proba = pd.DataFrame(prediction_proba, columns=["Здоров", "Паркинсон"])
    df_prediction_proba = df_prediction_proba.round(2)  # Округляем до двух знаков
    
    st.subheader("🔍 Результаты предсказания:")
    
    if prediction[0] == 1:
        st.markdown(
            "<div style='background-color: #ffcccc; padding: 10px; border-radius: 5px;'><strong>⚠️ Высокая вероятность болезни Паркинсона!</strong></div>",
            unsafe_allow_html=True
        )
    else:
        st.success("✅ Низкая вероятность болезни Паркинсона.")
    
    # Вывод вероятностей через ProgressColumn
    st.dataframe(
        df_prediction_proba,
        column_config={
            "Здоров": st.column_config.ProgressColumn(
                "Здоров",
                format="%.2f",
                width="medium",
                min_value=0,
                max_value=1
            ),
            "Паркинсон": st.column_config.ProgressColumn(
                "Паркинсон",
                format="%.2f",
                width="medium",
                min_value=0,
                max_value=1
            ),
        },
        hide_index=True
    )
    
    # Вывод предсказанного класса
    parkinson_labels = np.array(["Здоров", "Паркинсон"])
    st.success(f"Predicted status: **{parkinson_labels[prediction][0]}**")
    
    # Кнопка для скачивания предсказаний
    csv = df_prediction_proba.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Скачать предсказание", data=csv, file_name="prediction.csv", mime="text/csv")

    # SHAP Force Plot
    shap.initjs()
    explainer = shap.Explainer(logreg_model, X_train_scaled)
    shap_values = explainer(input_scaled)
    st.subheader("📊 Интерпретация предсказания (SHAP)")
    shap_html = shap.force_plot(shap_values.base_values[0], shap_values.values[0], input_df.iloc[0], feature_names=top_features, matplotlib=False)
    st.components.v1.html(shap_html, height=300)

# Визуализации
st.subheader("📊 Визуализация данных")
fig = px.scatter(df, x=top_features[0], y=top_features[1], color="status", title="Два наиболее коррелирующих признака")
st.plotly_chart(fig)

fig_density = px.density_contour(df, x=top_features[0], y=top_features[1], color="status", title="Плотность распределения данных")
st.plotly_chart(fig_density)

fig_hist1 = px.histogram(df, x=top_features[0], nbins=30, title=f"Распределение {feature_names[top_features[0]]}")
st.plotly_chart(fig_hist1)
fig_hist2 = px.histogram(df, x=top_features[1], nbins=30, title=f"Распределение {feature_names[top_features[1]]}")
st.plotly_chart(fig_hist2)
