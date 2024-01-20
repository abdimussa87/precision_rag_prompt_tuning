from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(text_chunks, embeddings)

    return vectorstore


def get_retriever(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    return retriever
