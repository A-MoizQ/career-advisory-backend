from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import httpx
import pdfplumber
import io
import json
from io import BytesIO
import pymupdf4llm
import fitz

# LangChain imports for robust PDF parsing
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders.blob_loaders import BlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.pdf import PyPDFParser

# for logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

app = FastAPI()

# Custom in-memory blob loader for LangChain
class InMemoryBlobLoader(BlobLoader):
    def __init__(self, file_bytes: bytes, filename: str):
        self.file_bytes = file_bytes
        self.filename = filename
    
    def yield_blobs(self):
        from langchain_core.document_loaders.blob_loaders import Blob
        yield Blob.from_data(self.file_bytes, path=self.filename)

# Mode-specific system prompts
MODE_PROMPTS = {
    "career_advice": "You're a helpful AI career advisor.",
    "resume_review": "You are an expert resume reviewer. Analyze the provided resume text, identify its structure (sections like Summary, Experience, Education, Skills), and provide specific, actionable feedback for improvement. Maintain the original formatting as much as possible in your analysis.",
    "job_hunt": "You suggest job hunting strategies and tips.",
    "learning_roadmap": "You recommend learning paths based on goals.",
    "mock_interview": "You act as a mock interviewer and give feedback."
}

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

@app.post("/chat")
async def chat(
    api_key: str = Form(...),
    mode: str = Form(...),
    messages: str = Form(...),
    file: UploadFile = File(None)
):
    logger.info(f"/chat called — file={file!r}, content_type={getattr(file,'content_type',None)}")
    
    try:
        message_list = json.loads(messages)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid 'messages' JSON format.")

    # If a file is uploaded, process it with robust extraction
    if file:
        if file.content_type == 'application/pdf':
            try:
                logger.info(f"Processing uploaded PDF: {file.filename}")
                
                # Read PDF content into memory
                pdf_bytes = await file.read()
                
                # Use robust extraction with fallbacks
                file_text = await extract_pdf_text_with_fallbacks(pdf_bytes, file.filename)
                
                if not file_text.strip():
                    raise HTTPException(
                        status_code=422, 
                        detail="Unable to extract text from PDF. The file might be a scanned document (image-based PDF) or password-protected. Please try uploading a text-based PDF or convert your document to text format."
                    )
                
                # Create a message with the extracted text
                file_message_content = f'The user has uploaded the file "{file.filename}". Its content is:\n\n---START OF FILE---\n{file_text}\n---END OF FILE---\n\nNow, please review it.'
                
                # Debug: Print extracted text length
                logger.info(f"✓ Successfully extracted {len(file_text)} characters from PDF")

                # Add this as the most recent user message
                message_list.append({"role": "user", "content": file_message_content})

            except HTTPException:
                raise  # Re-raise HTTP exceptions
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to process PDF file: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF is allowed.")

    system_prompt = MODE_PROMPTS.get(mode, "You are a helpful assistant.")
    full_messages = [{"role": "system", "content": system_prompt}] + message_list
    
    # Debug: Print messages being sent (but truncate long content)
    print("--- MESSAGES SENT TO OPENAI API ---")
    debug_messages = []
    for msg in full_messages:
        if len(msg["content"]) > 500:
            debug_msg = msg.copy()
            debug_msg["content"] = msg["content"][:500] + "... [CONTENT TRUNCATED]"
            debug_messages.append(debug_msg)
        else:
            debug_messages.append(msg)
    print(json.dumps(debug_messages, indent=2))
    print("------------------------------------")
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": full_messages,
        "max_tokens": 2048,
        "temperature": 0.5,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post("https://api.openai.com/v1/chat/completions",
                                         headers=headers, json=payload)
        
        response.raise_for_status()
        data = response.json()
        return {"reply": data["choices"][0]["message"]["content"]}

    except httpx.HTTPStatusError as e:
        error_details = e.response.json()
        raise HTTPException(status_code=e.response.status_code, detail=error_details)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))