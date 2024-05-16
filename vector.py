from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
import requests
from langchain.document_loaders import TextLoader
from langchain.document_loaders import HuggingFaceDatasetLoader



#create a new file named vectorstore in your current directory.
if __name__=="__main__":
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        dataset_name = "databricks/databricks-dolly-15k"
        page_content_column = "context"  # or any other column you're interested in

        model = 'sentence-transformers/all-MiniLM-l6-v2'
        model_kwargs = {'device':'cpu'}
        encode_kwargs = {'normalize_embeddings': False}


        # Create a loader instance
        loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(documents=splits, 
                                           embedding=HuggingFaceEmbeddings(model_name=model,
                                                                           model_kwargs = model_kwargs,
                                                                           encode_kwargs = encode_kwargs)
                                           )
        vectorstore.save_local(DB_FAISS_PATH)
