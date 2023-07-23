import chainlit as cl
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS

embeddings = OpenAIEmbeddings()

template = """
You have certain patterned questions.Your job is to ask questions related to events only, do not ask any other questions apart from that .
Your task is to convience the user to look for an event and get thier information which includes name, email and phone number. At the first question you should initiate a conversation asking are you looking for any 
events and then based on the user response, you should ask the user information one after other and store the user information.
{context}

Question: {question}
Answer:
"""


@cl.on_chat_start
def main():
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    vectorstore = FAISS.load_local(
        folder_path="embeddings", index_name="qa_index", embeddings=embeddings,
    )

    memory = ConversationBufferWindowMemory(
        k=7, return_messages=True, memory_key="chat_history"
    )

    qa_chain = RetrievalQA.from_llm(
        llm=ChatOpenAI(temperature=0.2),
        prompt=prompt,
        memory=memory,
        retriever=vectorstore.as_retriever(),
    )

    cl.user_session.set("llm_chain", qa_chain)


@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")

    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    await cl.Message(content=res["result"]).send()
