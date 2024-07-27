import streamlit as st
import requests

st.title('PDF Processor and Question Answering System')

st.header('Upload PDF')
uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type='pdf')
if uploaded_files:
    files = [('files', (file.name, file, 'application/pdf')) for file in uploaded_files]
    response = requests.post('http://127.0.0.1:8001/upload/', files=files)
    if response.status_code == 200:
        st.success("Files successfully uploaded and processed.")
        st.json(response.json())
    else:
        st.error("Failed to upload files.")

st.header('Ask a Question')
question = st.text_input("Enter your question:")
if st.button('Submit'):
    response = requests.get(f'http://127.0.0.1:8001/ask/?question={question}')
    if response.status_code == 200:
        answer = response.json().get('answer', 'No response')
        st.text_area("Answer:", value=answer, height=150)
    else:
        st.error("Failed to get an answer.")


# uvicorn main:app --host 127.0.0.1 --port 8001