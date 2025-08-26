
import requests
from chromadb import HttpClient
from backend.app.utils.pdf_loader import load_and_split_pdf


def create_embeddings_for_file(file_path: str, vector_db_host, vector_db_port, collection_name, embedding_api_url):
    def get_embedding(text: str) -> list:
        response = requests.post(embedding_api_url, json={"inputs": text})
        response.raise_for_status()
        return response.json()

    # Load and split document
    docs = load_and_split_pdf(file_path)
    print(f"Split into {len(docs)} chunks")

    # Extract texts and generate embeddings
    texts = [doc.page_content for doc in docs]
    embeddings = [get_embedding(text)[0] for text in texts] 

    # Connect to Chroma
    client = HttpClient(host=vector_db_host, port=vector_db_port)

    # Now use Chroma client to add with embeddings
    collection = client.get_or_create_collection(name=collection_name)

    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=[doc.metadata for doc in docs],
        ids=[f"id{i}" for i in range(len(texts))]
    )

    # Add documents
    print("Documents added to Chroma.")