import chainlit as cl
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationKGMemory
from langchain.vectorstores import FAISS

embeddings = OpenAIEmbeddings()

template = """
You are replication of human. Now you are task is to have a friendly conversation and recommend him events nearby based on the conversation
In the friendly conversation, you might ask the user's hobbies and make sure you engage with him until you understand the user's behaviour. 
By analyzing his behaviour try to bring the user to our objective of going to events by your friendly conversation. Use the conversation style from the data source provided.
Do not ask repeated questions. After 3 questions try to recommend some nearby events based on the user's information previously gathered.
The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.
{context}
Question : {question}
Answer:  
"""


@cl.on_chat_start
async def main():
    prompt = PromptTemplate(template=template, input_variables=["question", "context"])

    llm = ChatOpenAI(temperature=0.2)

    vectorstore = FAISS.load_local(
        folder_path="embeddings", index_name="qa_index", embeddings=embeddings
    )

    memory = ConversationKGMemory(llm=llm)

    qa_chain = RetrievalQA.from_llm(
        llm=llm, prompt=prompt, memory=memory, retriever=vectorstore.as_retriever()
    )

    await cl.AskUserMessage(content="Hello, how are you?", timeout=10).send()

    cl.user_session.set("llm_chain", qa_chain)


@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")

    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    await cl.Message(content=res["result"]).send()
