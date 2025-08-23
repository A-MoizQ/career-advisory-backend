from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import httpx
import json

from pdf_parser import extract_pdf_text_with_fallbacks
from prompts import build_system_prompt
from response_parser import extract_json_from_text, structured_json_to_markdown, normalize_markdown, sanitize_raw_text

# for logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

app = FastAPI()


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

    system_prompt = build_system_prompt(mode)
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
        "temperature": 0.0,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post("https://api.openai.com/v1/chat/completions",
                                         headers=headers, json=payload)
        
        response.raise_for_status()
        data = response.json()
        reply_text = data["choices"][0]["message"]["content"]

        # 1) Sanitize possible box-drawing and artifacts
        reply_text = sanitize_raw_text(reply_text)

        # 2) Try to parse structured JSON first (most reliable)
        structured = extract_json_from_text(reply_text)
        if structured:
            try:
                md = structured_json_to_markdown(structured)
                if md and len(md.strip()) > 10:
                    reply_text = md
                else:
                    # fallback to normalized text
                    reply_text = normalize_markdown(reply_text)
            except Exception:
                reply_text = normalize_markdown(reply_text)
        else:
            # 3) No JSON: run conservative normalization
            reply_text = normalize_markdown(reply_text)

        return {"reply": reply_text}

    except httpx.HTTPStatusError as e:
        error_details = e.response.json()
        raise HTTPException(status_code=e.response.status_code, detail=error_details)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))