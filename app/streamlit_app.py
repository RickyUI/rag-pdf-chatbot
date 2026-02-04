import streamlit as st
from functions import process_pdf, get_rag_chain
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="PDF Chatbot RAG", page_icon="🤖", layout="centered")

# --- STYLING ---
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("🤖 Chat con tu PDF (RAG)")
st.caption("Sube un archivo PDF y hazle preguntas basadas en su contenido.")

# --- SIDEBAR (Upload) ---
with st.sidebar:
    st.header("Configuración")
    uploaded_file = st.file_uploader("Sube tu archivo PDF", type="pdf")
    
    if uploaded_file:
        if "vectorstore" not in st.session_state or st.session_state.get("last_uploaded") != uploaded_file.name:
            with st.spinner("Procesando PDF..."):
                st.session_state.vectorstore = process_pdf(uploaded_file)
                st.session_state.last_uploaded = uploaded_file.name
                st.success("¡PDF procesado con éxito!")

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("¿Qué quieres saber del PDF?"):
    if "vectorstore" not in st.session_state:
        st.error("Por favor, sube un PDF primero en la barra lateral.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                chain = get_rag_chain(st.session_state.vectorstore)
                response = chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
