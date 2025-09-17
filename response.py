import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_store")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

def get_conversational_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    llm = Ollama(model="llama3.2:1b", temperature=0.1, base_url=OLLAMA_HOST)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return qa_chain

def query_documents(question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = Ollama(model="llama3.2:1b", temperature=0.1, base_url=OLLAMA_HOST)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa(question)
    print(f"\nAnswer:\n{result['result']}\n")
    print("Sources:")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata.get('source')}")


def query_with_history(chain, question):
    result = chain({"question": question})
    
    answer = result['answer']
    sources = [doc.metadata.get('source', 'N/A') for doc in result['source_documents']]
    
    return answer, sources