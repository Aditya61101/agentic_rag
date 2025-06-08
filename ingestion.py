import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
# above will provide a list of list, so we flatten it
docs_list = [doc for sublist in docs for doc in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs_list)

if not os.path.exists("./.chroma"):
    vector_store=Chroma.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings(),
        collection_name="rag-chroma",
        persist_directory="./.chroma"
    )

# retrieval part
retriever = Chroma(
    embedding=OpenAIEmbeddings(),
    collection_name="rag-chroma",
    persist_directory="./.chroma"
).as_retriever()