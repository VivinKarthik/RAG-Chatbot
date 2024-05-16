#import Essential dependencies
import streamlit as sl
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings import HuggingFaceEmbeddings
import requests
from langchain.document_loaders import TextLoader
from langchain_openai import ChatOpenAI

def load_vectordb():
        model = 'sentence-transformers/all-MiniLM-l6-v2'
        model_kwargs = {'device':'cpu'}
        encode_kwargs = {'normalize_embeddings': False}

        embeddings = HuggingFaceEmbeddings(model_name=model,
                                        model_kwargs = model_kwargs,
                                        encode_kwargs = encode_kwargs)
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
        
#function to load the OPENAI LLM
def load_model():     
        llm = ChatOpenAI(api_key="")
        return llm

#creating prompt template using langchain
def load_prompt():
        prompt =  """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer:
        """
        prompt = ChatPromptTemplate.from_template(prompt)
        return prompt


def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


if __name__=='__main__':
        sl.header("Hi I am Sherlock, a question answering bot")
        sl.write(" You can chat by Entering your queries ")
        knowledgeBase=load_vectordb()
        llm=load_model()
        prompt=load_prompt()
        
        query=sl.text_input('Enter your query')

        model = 'sentence-transformers/all-MiniLM-l6-v2'
        model_kwargs = {'device':'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        
        
        if(query):
                #getting only the chunks that are similar to the query for llm to produce the output
                similar_embeddings=knowledgeBase.similarity_search(query)
                similar_embeddings=FAISS.from_documents(documents=similar_embeddings, embedding=HuggingFaceEmbeddings(model_name=model,
                                                                           model_kwargs = model_kwargs,
                                                                           encode_kwargs = encode_kwargs))
                
                #creating the chain for integrating llm,prompt,stroutputparser
                retriever = similar_embeddings.as_retriever()
                rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                
                response=rag_chain.invoke(query)
                sl.write(response)
                
        
        
        
        
