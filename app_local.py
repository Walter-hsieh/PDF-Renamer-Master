# main.py

import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
# DOC_PATH = "C:\\Users\\walte\\Desktop\\rename\\pdf\\test.pdf"
# MODEL_NAME = "llama3.2"
MODEL_NAME = "deepseek-r1:1.5b"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"


def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = PyPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        return None


def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


def create_vector_db(chunks):
    """Create a vector database from document chunks."""
    # Pull the embedding model if not already available
    ollama.pull(EMBEDDING_MODEL)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_name=VECTOR_STORE_NAME,
    )
    logging.info("Vector database created.")
    return vector_db


def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever


def create_chain(retriever, llm):
    """Create the chain"""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created successfully.")
    return chain


# for deepseek-r model
def extract_filename(response_text):
    """
    Extracts the filename that appears after </think> tag in the response.
    
    Args:
        response_text (str): The full response from DeepSeek
        
    Returns:
        str: The extracted filename, or None if no filename is found
    """
    # Split the text at </think>
    parts = response_text.split('</think>')
    
    if len(parts) < 2:
        return None
    
    # Get the text after </think>
    after_think = parts[1].strip()
    
    # Split by newlines and get the first non-empty line
    lines = [line.strip() for line in after_think.split('\n') if line.strip()]
    
    if not lines:
        return None
        
    return lines[0]


def main(DOC_PATH):
    # Load and process the PDF document
    data = ingest_pdf(DOC_PATH)
    if data is None:
        return

    # Split the documents into chunks
    chunks = split_documents(data)

    # Create the vector database
    vector_db = create_vector_db(chunks)

    # Initialize the language model
    llm = ChatOllama(model=MODEL_NAME)

    # Create the retriever
    retriever = create_retriever(vector_db, llm)

    # Create the chain with preserved syntax
    chain = create_chain(retriever, llm)

    # Example query
    # question = '''what is the title of this file? and what about it?'''

    # question = '''Based on file provided, generate a file name in this format:[year published]_[aspect of the technology]_[main topic]_[primary application].pdf.
    # Please do not give any response except for the file name. Do not include symbol like /, \, ~, !, @, #, or $ in the file name.'''

    question = '''Generate a file name in this format: [YYYY]_[technology_aspect]_[main_topic]_[application].pdf
    Rules:
    - Use only letters, numbers, and underscores
    - Keep each segment concise (1-3 words)
    - Focus on the most prominent themes
    - Lowercase letters only
    - Do not give any response other than the final file name.
    '''

    # Get the response
    res = chain.invoke(input=question)
    # print("Response:")
    # print(res)

    res = extract_filename(res)

    return (res)



from chromadb import Client


if __name__ == "__main__":
    folder_path = "C:\\Users\\walte\\Desktop\\renamer\\pdf"
    files = os.listdir(folder_path)
    print(f"files: {files}")
    new_names = []
    for file in files:
        DOC_PATH = os.path.join(folder_path, file)
        print(DOC_PATH)
        new_name = main(DOC_PATH)
        print(new_name)
        new_names.append(new_name)
        client = Client()
        existing_collections = client.list_collections()
        print(existing_collections)
        client.delete_collection("simple-rag")
    print(new_names)

    for old_name, new_name in zip(files, new_names):
        old_file_path = os.path.join(folder_path, old_name)
        new_file_path = os.path.join(folder_path, new_name)
        os.rename(old_file_path, new_file_path)
    print("=== rename completed! ===")