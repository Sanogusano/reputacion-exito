
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Configuración de la página
st.set_page_config(page_title="Monitor Reputacional 360°", layout="wide")

# Estilo personalizado inspirado en UI 1 con paleta UI 2
st.markdown("""
    <style>
    html, body, .stApp {
        background-color: #f7f9fc;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        color: #172b4d;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .css-1cpxqw2, .stButton>button {
        background-color: #ffffff;
        color: #333;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Encabezado
st.title("🧭 Monitor Reputacional 360°")
st.markdown("Visualización en tiempo real basada en dimensiones RepTrak y clasificación automática con BETO")

# Carga de archivo
uploaded_file = st.file_uploader("📄 Sube tu archivo Excel con menciones", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="Sheet0")
    df = df[["Date", "Title", "Snippet", "Sentiment", "Page Type", "Domain"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["text"] = df["Title"].fillna("") + " " + df["Snippet"].fillna("")
    df = df.dropna(subset=["text"])

    # Modelo
    with st.spinner("🔍 Clasificando menciones con BETO..."):
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
        pred = torch.argmax(outputs.logits, dim=1).item()
        return model.config.id2label[pred]

    df["Dimensión"] = df["text"].apply(classify_text)

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
        "Productos/Servicios": 0.19, "Innovación": 0.14, "Lugar de trabajo": 0.12,
        "Gobernanza": 0.14, "Ciudadanía": 0.14, "Liderazgo": 0.14, "Resultados financieros": 0.13
    }
    df["Score"] = df.apply(lambda x: x["Impacto"] * pesos.get(x["Dimensión"], 0), axis=1)

    # Visualización - radar
    radar_df = df.groupby("Dimensión")["Score"].sum().reset_index()
    fig_radar = px.line_polar(
        radar_df,
        r="Score",
        theta="Dimensión",
        line_close=True,
        title="🌀 Círculo Reputacional",
        template="plotly_white",
        color_discrete_sequence=["#4CAF50"]
    )
    fig_radar.update_traces(fill='toself', line_color="#00C49F")
    st.plotly_chart(fig_radar, use_container_width=True)

    # Visualización - velas japonesas
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
        close=df_day["Close"],
        increasing_line_color='#00C49F',
        decreasing_line_color='#FF6B6B'
    )])
    fig_candle.update_layout(
        title="📈 Evolución diaria reputacional",
        xaxis_title="Fecha", yaxis_title="Score",
        template="plotly_white",
        margin=dict(t=50, b=20)
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    # Tabla de comentarios
    st.markdown("### 💬 Últimos Comentarios Clasificados")
    st.dataframe(df[["Date", "text", "Dimensión", "Impacto", "Score"]].sort_values(by="Date", ascending=False).head(10))
