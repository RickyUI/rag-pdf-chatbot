import os
import uuid
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def process_pdf(uploaded_file):
    """Processes an uploaded PDF: loads, chunks, and creates a vector store."""
    # 1. Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # 2. Load PDF
        loader = PyPDFLoader(tmp_path)
        data = loader.load()

        # 3. Chunk text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", " "]
        )
        chunks = text_splitter.split_documents(data)

        # 4. Create unique IDs and filter duplicates
        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]
        unique_ids = set()
        unique_chunks = []
        for id, chunk in zip(ids, chunks):
            if id not in unique_ids:
                unique_ids.add(id)
                unique_chunks.append(chunk)

        # 5. Create VectorStore
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma.from_documents(
            documents=unique_chunks,
            ids=list(unique_ids),
            embedding=embeddings
        )
        
        return vectorstore
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def get_rag_chain(vectorstore):
    """Creates a RAG chain for answering questions."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    template = """Eres un asistente experto en analizar documentos. Utiliza el siguiente contexto para responder la pregunta del usuario de forma detallada y precisa. 
    
    Si la información no está en el contexto, indícalo amablemente.
    
    Contexto:
    {context}
    
    Pregunta: {question}
    
    Respuesta detallada:"""
    prompt = ChatPromptTemplate.from_template(template)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain
