import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Function to get HTML files only
def get_html_files(directory):
    html_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".html"):  # Filter only HTML files
                html_files.append(os.path.join(root, file))
    return html_files

# Ingest documents
def ingest_docs():
    # Specify the root directory of your downloaded files
    html_files = get_html_files("HTTYD-docs")
    print(f"Found {len(html_files)} HTML files")

    raw_documents = []

    # Load each HTML file using UnstructuredHTMLLoader
    for html_file in html_files:
        try:
            loader = UnstructuredHTMLLoader(html_file)
            docs = loader.load()
            raw_documents.extend(docs)
        except Exception as e:
            print(f"Error loading {html_file}: {e}")

    print(f"Loaded {len(raw_documents)} raw documents")

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(documents)} documents")

    # Add source metadata to each document
    for doc in documents:
        new_url = doc.metadata.get("source", "")
        if new_url:
            new_url = new_url.replace("HTTYD-docs", "https://howtotrainyourdragon.fandom.com/wiki/How_to_Train_Your_Dragon_Wiki")
            doc.metadata.update({"source": new_url})

    # Now add to Pinecone (ensure Pinecone is initialized elsewhere)
    print(f"Going to add {len(documents)} documents to Pinecone")
    Pinecone.from_documents(documents, embeddings, index_name="httyd-docs-index")

# Call the function to ingest documents
if __name__ == "__main__":
    ingest_docs()