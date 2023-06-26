from flask import Flask, request, jsonify, render_template
# from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import importlib
import os
# load_dotenv()
app = Flask(__name__)
@app.route('/api/ask_pdf', methods=['POST'])
def ask_pdf():
    # upload file
    pdf = request.files['pdf']

    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        # get user question
        user_question = request.form.get('question')
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)

            return jsonify(response)

    return jsonify({'error': 'PDF file not provided.'})

def main():
    os.environ["OPENAI_API_KEY"] = "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"
    llm = OpenAI(temperature=0.9)
    app.run(debug=True)
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    main()
