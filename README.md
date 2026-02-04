# PDF Chatbot - RAG System

This project is an interactive application that allows users to upload PDF files and ask questions based on their content using Retrieval-Augmented Generation (RAG) techniques. It combines the power of OpenAI, LangChain, and ChromaDB with a user-friendly interface built using Streamlit.

## Features

- **PDF Upload**: Upload any PDF file for analysis.
- **Intelligent Processing**: Text is split into optimized chunks to preserve context and meaning.
- **Semantic Search**: Uses ChromaDB and OpenAI Embeddings to find the most relevant information.
- **Chat Interface**: Streamlined and modern user experience similar to ChatGPT.
- **Session Memory**: Maintains conversation history throughout the session.

## Technologies Used

- **Frontend**: [Streamlit](https://streamlit.io/)
- **LLM Framework**: [LangChain](https://www.langchain.com/)
- **AI Models**: OpenAI (GPT-4o mini and Text Embedding 3 Small)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Data Loading**: PyPDFLoader

## Prerequisites

- Python 3.10 or higher.
- An OpenAI API Key (`OPENAI_API_KEY`).

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-username/rag-pdf-chatbot.git
   cd rag-pdf-chatbot
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the project root and add your OpenAI key:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

To start the application, run the following command from the project root:

```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```text
rag_pdf_chatbot/
├── app/
│   ├── streamlit_app.py   # Main UI application entry point
│   └── functions.py       # Core RAG system logic
├── data/                  # Directory for sample documents
├── .env                   # Environment variables (API Keys)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## How It Works

1. **Ingestion**: The PDF is loaded and divided into small fragments (chunks).
2. **Embedding**: Each fragment is converted into a numerical vector.
3. **Storage**: Vectors are stored in ChromaDB.
4. **Retrieval**: When a user asks a question, the system searches for the 5 most semantically similar fragments.
5. **Generation**: The LLM receives the question and the retrieved fragments as context to generate a precise response.

---
Developed for AI and RAG learning purposes.
