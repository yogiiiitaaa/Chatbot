import streamlit as st
import requests
import os
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="AI Multi Chatbot", layout="centered")
st.title("🚀 AI Multi-Functional Chatbot")

# ----------------------------
# Load Sentiment Model
# ----------------------------
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

sentiment_model = load_sentiment_model()

# ----------------------------
# DeepSeek API Function
# ----------------------------
def ask_deepseek(question):
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        return "DeepSeek API Key not found."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": question}]
    }

    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=data
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.text}"

# ----------------------------
# Prediction Function
# ----------------------------
def predict_value():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])

    model = LinearRegression()
    model.fit(X, y)

    prediction = model.predict([[6]])
    return prediction[0]

# ----------------------------
# Visualization Function
# ----------------------------
def visualize_sentiment(label):
    labels = ["Positive", "Negative"]
    values = [1 if label == "POSITIVE" else 0,
              1 if label == "NEGATIVE" else 0]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title("Sentiment Result")
    st.pyplot(fig)

# ----------------------------
# User Input
# ----------------------------
user_input = st.text_input("Ask something:")

if st.button("Submit"):

    # 1️⃣ Chatbot Answer
    answer = ask_deepseek(user_input)
    st.subheader("🤖 Chatbot Answer")
    st.write(answer)

    # 2️⃣ Sentiment Analysis
    sentiment = sentiment_model(user_input)[0]
    st.subheader("📊 Sentiment Analysis")
    st.write(f"Label: {sentiment['label']}")
    st.write(f"Confidence: {round(sentiment['score'],2)}")

    # 3️⃣ Visualization Trigger
    if "visualize" in user_input.lower():
        st.subheader("📈 Visualization")
        visualize_sentiment(sentiment["label"])

    # 4️⃣ Prediction Trigger
    if "predict" in user_input.lower():
        st.subheader("🔮 Prediction")
        prediction = predict_value()
        st.write(f"Predicted Value: {prediction}")
