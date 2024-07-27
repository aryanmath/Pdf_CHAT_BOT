import os
import io
from fastapi import FastAPI, UploadFile, File
import uvicorn
from typing import List
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

app = FastAPI()

model = genai.GenerativeModel("gemini-pro")
prompt_template = """
Answer the questions in as much detail as possible based on the provided context. Ensure that your answers align with the given context. If the context is unclear or insufficient, do not provide incorrect or assumed answers. Instead, specify the exact information you need to answer the queries accurately. If necessary, respond with thanks.
Additionally, if definitions, key terms, or examples related to the context are requested, please provide them.
\n\n 
Context:\n{context}?\n
Question:\n{question}\n
Answer:
"""

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

async def get_pdf_text(file_content):
    text = ""
    file_like = io.BytesIO(file_content)
    pdf_reader = PdfReader(file_like)
    if pdf_reader.pages:
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    else:
        print("Warning: PDF file is empty or doesn't have any pages.")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def load_faiss_index(pickle_file):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_index = FAISS.load_local(pickle_file, allow_dangerous_deserialization=True, embeddings=embeddings)
    return faiss_index

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

@app.post("/upload/")
async def upload_pdf(files: List[UploadFile] = File(...)):
    texts = []
    for file in files:
        content = await file.read()
        text = await get_pdf_text(content)
        texts.append(text)
    all_text = " ".join(texts)
    text_chunks = get_text_chunks(all_text)
    vector_store = get_vector_store(text_chunks)
    return {"message": "Text processed and vector store created successfully"}

@app.get("/ask/")
async def ask_question(question: str):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embeddings-001")
    new_db = load_faiss_index("faiss_index")
    docs = new_db.similarity_search(question)
    if docs:
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": question})
        return {"question": question, "answer": response["output_text"]}
    else:
        return {"question": question, "answer": "No relevant documents found."}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)