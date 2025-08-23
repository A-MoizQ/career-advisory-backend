import pymupdf4llm
import fitz
from io import BytesIO
import pdfplumber

# LangChain imports for robust PDF parsing
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders.blob_loaders import BlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.pdf import PyPDFParser

# for logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")


# Custom in-memory blob loader for LangChain
class InMemoryBlobLoader(BlobLoader):
    def __init__(self, file_bytes: bytes, filename: str):
        self.file_bytes = file_bytes
        self.filename = filename
    
    def yield_blobs(self):
        from langchain_core.document_loaders.blob_loaders import Blob
        yield Blob.from_data(self.file_bytes, path=self.filename)


async def extract_pdf_text_with_fallbacks(pdf_bytes: bytes, filename: str) -> str:
    """
    Extract text from PDF using multiple methods with fallbacks:
    1. LangChain UnstructuredPDFLoader
    2. LangChain PyPDFParser  
    3. PDFPlumber
    """
    # Method 0: Try PyMuPDF4LLM Markdown extraction
    try:
        # Open PDF bytes as a PyMuPDF Document
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        # Convert to Markdown via PyMuPDF4LLM
        md = pymupdf4llm.to_markdown(doc=doc, page_chunks=False)
        if md and len(md.strip()) > 50:
            logger.info("✓ pymupdf4llm succeeded")
            return md
    except Exception as e:
        logger.error(f"pymupdf4llm failed: {e}")
    
    # Method 1: Try LangChain UnstructuredPDFLoader
    try:
        logger.info("Attempting extraction with LangChain UnstructuredPDFLoader...")
        # Save bytes to a temporary file-like object
        temp_file = BytesIO(pdf_bytes)
        temp_file.name = filename  # UnstructuredPDFLoader needs a name attribute
        
        loader = UnstructuredPDFLoader(temp_file, mode="elements")
        docs = loader.load()
        
        if docs:
            text_content = "\n".join([doc.page_content for doc in docs])
            if len(text_content.strip()) > 50:  # Reasonable threshold
                logger.info("✓ LangChain UnstructuredPDFLoader succeeded")
                return text_content
    except Exception as e:
        logger.error(f"✗ LangChain UnstructuredPDFLoader failed: {e}")

    # Method 2: Try LangChain PyPDFParser with custom blob loader
    try:
        logger.info("Attempting extraction with LangChain PyPDFParser...")
        blob_loader = InMemoryBlobLoader(pdf_bytes, filename)
        parser = PyPDFParser()
        loader = GenericLoader(blob_loader, parser)
        
        docs = loader.load()
        if docs:
            text_content = "\n".join([doc.page_content for doc in docs])
            if len(text_content.strip()) > 50:
                logger.info("✓ LangChain PyPDFParser succeeded")
                return text_content
    except Exception as e:
        logger.error(f"✗ LangChain PyPDFParser failed: {e}")

    # Method 3: Try PDFPlumber (original method)
    try:
        logger.info("Attempting extraction with PDFPlumber...")
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            pages_text = []
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if page_text:
                    pages_text.append(page_text)
            
            text_content = "\n".join(pages_text)
            if len(text_content.strip()) > 50:
                logger.info("✓ PDFPlumber succeeded")
                return text_content
    except Exception as e:
        logger.error(f"✗ PDFPlumber failed: {e}")

    # If all methods fail
    logger.error("✗ All extraction methods failed")
    return ""