from langchain import HuggingFaceHub
from langchain.llms import huggingface_hub
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import SystemMessage
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS, Chroma
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import RetrievalQA, ConversationalRetrievalChain

import pandas as pd

from huggingface_hub import notebook_login

from datasets import load_dataset

import faiss

import os

from transformers import AutoTokenizer

from fastapi import FastAPI


os.environ["HUGGINGFACE_API_KEY"] = "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"

notebook_login()

HUGGINGFACE_API_KEY = "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"

model = HuggingFaceHub(repo_id="google/flan-t5-large",
                       model_kwargs={"temperature": 0.5, "max_length": 64},
                       huggingfacehub_api_token=HUGGINGFACE_API_KEY)

dir = "scraped_data.csv"

df = pd.read_csv(dir)

df['Data'] = df['Title'] + ', ' + df['Text']
df.drop(['Title', 'Text'], axis=1, inplace=True)
df.to_csv('output.csv', index=False)

df = pd.read_csv("output.csv")

# Extract the first column data
first_column_data = df.iloc[:, 0]

# Write the first column data to the text file
with open("scraped_data.txt", 'w') as txt_file:
    for item in first_column_data:
        txt_file.write(str(item) + '\n')

updated_dir = ""  # "/content"

loader = DirectoryLoader(f"{updated_dir}", glob='**/*.txt')
docs = loader.load()
text_split = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=0)
text = text_split.split_documents(docs)

text[0]

len(text)

"""Embeddings"""

embeddings = HuggingFaceEmbeddings(model_name="google/flan-t5-large")

vectorStore = FAISS.from_documents(docs, embeddings)

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

retriever = vectorStore.as_retriever(
    search_type="similarity", search_kwargs={"k": 2})

chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

result

app = FastAPI()


@app.post("/process_query/")
async def process_query(request_data: dict):

    query = request_data.get("query")

    chain = load_qa_with_sources_chain(model, chain_type="refine")

    documents = vectorStore.similarity_search(query)
    result = chain.run({"input_documents": documents, "question": query})

    processed_query = "Query Answer: " + result
    return {"result": processed_query}
