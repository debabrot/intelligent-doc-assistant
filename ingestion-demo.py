import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# 1. Load the PDF
pdf_path = "2410.09062v1.pdf"  # Replace with your PDF file path
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

print(f"Split into {len(docs)} chunks")

# 3. Create embeddings using all-MiniLM-L6-v2
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Store in persistent Chroma DB
persist_directory = "chroma_db"  # Directory to persist the database
os.makedirs(persist_directory, exist_ok=True)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_function,
    persist_directory=persist_directory
)

print("Documents stored in Chroma DB successfully.")

# Optional: Load and query to verify
# vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
# results = vectorstore.similarity_search("your query here", k=2)
# for res in results:
#     print(res.page_content)