import streamlit as st
import requests

st.set_page_config(
    page_title="AI Screenwriter",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 AI Screenwriter Assistant")
st.markdown("Ваш интеллектуальный помощник по драматургии и сценарному мастерству.")

with st.sidebar:
    st.header("Настройки")
    mode = st.radio(
        "Выберите режим:",
        ["Консультация (RAG)", "Генерация сцены"]
    )
    st.info("**Консультация:** Ответы на основе Кэмпбелла, Берна и классики.\n\n**Генерация:** Создание черновика сцены.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Задайте вопрос или опишите сцену..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    BACKEND_URL = "http://screenwriter_api:8000"
    
    payload = {}
    api_endpoint = ""

    if mode == "Консультация (RAG)":
        api_endpoint = f"{BACKEND_URL}/consult"
        payload = {"question": prompt}
    else:
        api_endpoint = f"{BACKEND_URL}/generate/scene"

        payload = {
            "genre": "Drama", 
            "characters": "Hero, Antagonist", 
            "plot_outline": prompt, 
            "tone": "Serious"
        }

    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            try:
                response = requests.post(api_endpoint, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("response", data.get("scene_script", "Нет ответа"))
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Ошибка сервера: {response.status_code}")
            except Exception as e:
                st.error(f"Не удалось связаться с сервером: {e}")