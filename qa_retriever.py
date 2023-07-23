from typing import List

from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


class QARetriver:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()

    def retrieve(self, query: str) -> str:
        try:
            prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
            {context}
            Question: {question}
            Answer in English:"""

            prompt = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            vectorstore = FAISS.load_local(
                folder_path="embeddings",
                index_name="qa_index",
                embeddings=self.embeddings,
            )

            qa = RetrievalQA.from_chain_type(
                llm=OpenAI(temperature=0),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": prompt},
            )

            result = qa.run(query)

            return result

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
