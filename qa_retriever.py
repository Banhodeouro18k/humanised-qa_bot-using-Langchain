from typing import List

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


class QARetriver:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()

        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

    def retrieve(self, query: str) -> str:
        try:
            retriever = FAISS.load_local(
                folder_path="embeddings",
                index_name="qa_index",
                embeddings=self.embeddings,
            )

            qa = ConversationalRetrievalChain.from_llm(
                OpenAI(temperature=0.2), retriever.as_retriever(), memory=self.memory
            )

            result = qa({"question": query})

            return result["answer"]

        except Exception as e:
            raise e

    def generate_embeddings(self, src_file_path: str, embeddings_index: str):
        try:
            loader = CSVLoader(file_path=src_file_path, encoding="utf-8")

            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=100, chunk_overlap=20, length_function=len,
            )

            documents: List[Document] = text_splitter.split_documents(documents)

            retriever = FAISS.from_documents(documents, self.embeddings)

            retriever.save_local(folder_path="embeddings", index_name=embeddings_index)

        except Exception as e:
            raise e
