from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate

embeddings = OpenAIEmbeddings()

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Answer in English:"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

loader = CSVLoader(file_path="data.csv", encoding="utf-8")

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=20, length_function=len,
)

documents = text_splitter.split_documents(documents)

vectorstore = FAISS.from_documents(documents, embeddings)

llm = OpenAI(temperature=0)

chain_type_kwargs = {"prompt": prompt}

qa = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=vectorstore.as_retriever(),chain_type_kwargs=chain_type_kwargs)

query = "Heyyyooo u like to party?"

ans = qa.run(query)

print(ans)