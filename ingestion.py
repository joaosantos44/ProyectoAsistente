import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


def get_html_files(directory):
    html_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".html"): 
                html_files.append(os.path.join(root, file))
    return html_files


def ingest_docs():
    html_files = get_html_files("HTTYD-docs")
    print(f"Found {len(html_files)} HTML files")
    raw_documents = []

    for html_file in html_files:
        try:
            loader = UnstructuredHTMLLoader(html_file)
            docs = loader.load()
            raw_documents.extend(docs)
        except Exception as e:
            print(f"Error loading {html_file}: {e}")
    print(f"Loaded {len(raw_documents)} raw documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(documents)} documents")

    for doc in documents:
        new_url = doc.metadata.get("source", "")
        if new_url:
            new_url = new_url.replace("HTTYD-docs", "https://howtotrainyourdragon.fandom.com/wiki/How_to_Train_Your_Dragon_Wiki")
            doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} documents to Pinecone")
    Pinecone.from_documents(documents, embeddings, index_name="httyd-docs-index")

if __name__ == "__main__":
    ingest_docs()
