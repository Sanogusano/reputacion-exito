
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

st.set_page_config(page_title="Reputación Grupo Éxito", layout="wide")
st.title("🔎 Reputación Grupo Éxito - Dashboard RepTrak + BETO")

# Subida de archivo
uploaded_file = st.file_uploader("📄 Sube el archivo Excel", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="Sheet0")
    df = df[["Date", "Title", "Snippet", "Sentiment", "Page Type", "Domain"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["text"] = df["Title"].fillna("") + " " + df["Snippet"].fillna("")
    df = df.dropna(subset=["text"])

    # Clasificador BETO
    with st.spinner("Cargando modelo BETO..."):
        model_name = "dccuchile/bert-base-spanish-wwm-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7)
        labels = [
            "Productos/Servicios", "Innovación", "Lugar de trabajo",
            "Gobernanza", "Ciudadanía", "Liderazgo", "Resultados financieros"
        ]
        model.config.id2label = dict(enumerate(labels))

    def classify_text(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
        return model.config.id2label[pred]

    st.info("Clasificando textos...")
    df["Dimensión"] = df["text"].apply(classify_text)

    # Cálculo del impacto
    def impacto(sentiment, page_type):
        mult = {"twitter": 1, "news": 3, "blog": 2}
        base = mult.get(str(page_type).lower(), 1)
        if sentiment == "positive":
            return 1 * base
        elif sentiment == "negative":
            return -1 * base
        return 0

    df["Impacto"] = df.apply(lambda x: impacto(x["Sentiment"], x["Page Type"]), axis=1)
    pesos = {
        "Productos/Servicios": 0.19,
        "Innovación": 0.14,
        "Lugar de trabajo": 0.12,
        "Gobernanza": 0.14,
        "Ciudadanía": 0.14,
        "Liderazgo": 0.14,
        "Resultados financieros": 0.13
    }
    df["Score"] = df.apply(lambda x: x["Impacto"] * pesos.get(x["Dimensión"], 0), axis=1)

    # Radar por dimensión
    radar_df = df.groupby("Dimensión")["Score"].sum().reset_index()
    fig_radar = px.line_polar(radar_df, r="Score", theta="Dimensión", line_close=True,
                              title="Círculo de Reputación")
    st.plotly_chart(fig_radar, use_container_width=True)

    # Velas japonesas
    df_day = df.groupby("Date").agg(
        Open=("Score", lambda x: x.iloc[0]),
        High=("Score", max),
        Low=("Score", min),
        Close=("Score", lambda x: x.iloc[-1])
    ).reset_index()

    fig_candle = go.Figure(data=[go.Candlestick(
        x=df_day["Date"],
        open=df_day["Open"],
        high=df_day["High"],
        low=df_day["Low"],
        close=df_day["Close"]
    )])
    fig_candle.update_layout(title="Evolución diaria de la Reputación", xaxis_title="Fecha", yaxis_title="Score")
    st.plotly_chart(fig_candle, use_container_width=True)

    # Mostrar tabla
    with st.expander("🗂 Ver datos procesados"):
        st.dataframe(df)
