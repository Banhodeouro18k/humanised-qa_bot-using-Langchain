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
            prompt_template = """
            You are an event manager,you need to check the events nearby of the user's place. You have certain patterned questions. 
            Your task is to convience them and get the user's information which includes name, email and phone number.
            
            Bot : Hello, how are you ?
            User : 
            
            Bot : Are you looking for any events ?
            User : Maybe yes
            
            Bot : May I know your place ?
            User : <user place>
            
            Bot : Please select a event from this place             
            {context}
            
            Question: {question}
            Answer:
            """
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

            response = qa({"query": query})

            return response["result"]

        except Exception as e:
            raise e

    def generate_embeddings(self, src_file_path: str, embeddings_index: str) -> None:
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
