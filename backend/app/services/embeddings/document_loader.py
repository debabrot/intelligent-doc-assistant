"""Loads document"""

import uuid
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.app.utils.identifiers import generate_deterministic_id
from backend.app.domain.protocols import (
    DocumentChunk,
    DocumentLoaderProtocol,
    TokenizerProtocol
)


class PDFDocumentLoader(DocumentLoaderProtocol):
    def __init__(self, tokenizer: TokenizerProtocol):
        self.tokenizer = tokenizer

    def load_and_split(self, file_path: str, chunk_size: int = 256, chunk_overlap: int = 50) -> List[DocumentChunk]:
        if chunk_size <= chunk_overlap:
            raise ValueError("chunk_size must be greater than chunk_overlap")

        loader = PyPDFLoader(file_path, extract_images=False)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.tokenizer.count_tokens,
            separators=["\n\n", "\n", " ", ""]
        )

        split_docs = text_splitter.split_documents(documents)

        return [
            DocumentChunk(
                content=doc.page_content,
                metadata={
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", -1),
                    "chunk_index": i
                },
                id=generate_deterministic_id(
                    doc.page_content,
                    {
                        "source": doc.metadata.get("source", "unknown"),
                        "page": doc.metadata.get("page", -1),
                        "chunk_index": i
                    }
                )
            )
            for i, doc in enumerate(split_docs)
        ]
