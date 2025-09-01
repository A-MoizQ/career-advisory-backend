# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import httpx
import json
import uuid
import os

from pdf_parser import extract_pdf_text_with_fallbacks
from prompts import build_system_prompt
from response_parser import (
    extract_json_from_text,
    sanitize_raw_text,
    convert_structured_or_fix,  # new helper that converts structured JSON or repairs raw markdown
)
from mode_handlers import MODE_HANDLERS
from session_store import load_session, save_session, delete_session

# logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

app = FastAPI()


@app.post("/chat")
async def chat(
    api_key: str = Form(...),
    mode: str = Form(...),
    messages: str = Form(...),
    file: UploadFile = File(None),
    session_id: str = Form(None),
    answers: str = Form(None),  # JSON string mapping qid -> answer
):
    logger.info(f"/chat called — mode={mode}, file={file!r}, session_id={session_id!r}")

    # Validate messages JSON
    try:
        message_list = json.loads(messages)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid 'messages' JSON format.")

    # File handling (unchanged)
    if file:
        if file.content_type == 'application/pdf':
            try:
                logger.info(f"Processing uploaded PDF: {file.filename}")
                pdf_bytes = await file.read()
                file_text = await extract_pdf_text_with_fallbacks(pdf_bytes, file.filename)
                if not file_text.strip():
                    raise HTTPException(
                        status_code=422,
                        detail="Unable to extract text from PDF. The file might be a scanned document or password-protected."
                    )
                file_message_content = (
                    f'The user has uploaded the file "{file.filename}". Its content is:\n\n'
                    f'---START OF FILE---\n{file_text}\n---END OF FILE---\n\nNow, please review it.'
                )
                logger.info(f"✓ Extracted {len(file_text)} characters from PDF")
                message_list.append({"role": "user", "content": file_message_content})
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to process PDF file: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF is allowed.")

    # Validate mode handler
    handler = MODE_HANDLERS.get(mode)
    if not handler:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")

    session = None
    if session_id:
        session = load_session(session_id)
    
    # Always create a session if one doesn't exist for this conversation
    if not session:
        session = {
            "session_id": str(uuid.uuid4()),
            "mode": mode,
            "state": {"answers": {}}
        }
        logger.info(f"Created new session: {session['session_id']}")

    if answers:
        try:
            answers_obj = json.loads(answers)
            session.setdefault("state", {}).setdefault("answers", {}).update(answers_obj)
            logger.info(f"Updated session with answers: {answers_obj}")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid 'answers' JSON format.")

    # The core logic is now here. The handler's "brain" makes the decision.
    if handler.requires_session():
        clar_qs = await handler.needs_clarification(
            session=session,
            incoming_messages=message_list,
            api_key=api_key
        )
        
        # Save the session state after the handler has potentially modified it
        save_session(session)

        if clar_qs:
            # The LLM decided to ask questions.
            intro_message = "To provide the best guidance, I have a few questions for you:"
            if answers: # This means it's a follow-up
                intro_message = "Thanks for the information. I have a few more follow-up questions to ensure I understand correctly:"

            return {
                "reply": intro_message,
                "clarifying_questions": clar_qs,
                "session_id": session["session_id"]
            }

    # If clar_qs is None, the LLM decided to create a plan.
    logger.info("Proceeding to generate career plan.")
    
    system_prompt = build_system_prompt(mode)
    full_messages = [{"role": "system", "content": system_prompt}]

    extra_system_messages = handler.prepare_system_messages(session=session, incoming_messages=message_list, answers=None)
    full_messages.extend([{"role": "system", "content": m} for m in extra_system_messages])

    # Add only the most recent user message to avoid cluttering the final prompt
    full_messages.append(message_list[-1])


    # Debug print (truncate long)
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

    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": full_messages,
        "max_tokens": 2048,
        "temperature": 0.0,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        response.raise_for_status()
        data = response.json()
        reply_text_raw = data["choices"][0]["message"]["content"]

        # Sanitize and try parse
        reply_text = sanitize_raw_text(reply_text_raw)
        structured = extract_json_from_text(reply_text)

        # Delegate to handler to decide persistence and returned payload
        result = handler.handle_llm_response(session=session, structured=structured, raw_text=reply_text)

        # If the handler wants to preserve JSON (raw), return a fenced JSON block (preferred by your prompt).
        preserve_json = bool(result.get("preserve_json", False))

        if preserve_json and structured:
            # Return a single fenced JSON block as required by your prompt rules
            try:
                fenced = "```json\n" + json.dumps(structured, ensure_ascii=False, indent=2) + "\n```"
                reply_md = fenced
            except Exception:
                # fallback to repair/conversion
                reply_md = convert_structured_or_fix(reply_text, structured)
        else:
            # Convert structured JSON to beautiful markdown OR repair raw markdown
            # convert_structured_or_fix handles both structured JSON and raw markdown repair
            reply_md = convert_structured_or_fix(reply_text, structured)

        # Debug logs for troubleshooting formatting issues (optional)
        logger.debug(f"Structured extracted: {json.dumps(structured) if structured else 'None'}")
        logger.debug(f"Final reply_md (first 800 chars): {reply_md[:800]}")

        # Prepare return payload
        return_payload = {"reply": reply_md}
        
        # Handle session persistence
        if result.get("persist") and result.get("session"):
            save_session(result["session"])
            return_payload["session_id"] = result["session"]["session_id"]
        else:
            # if handler explicitly cleared session, delete it
            if session and result.get("session") is None:
                delete_session(session["session_id"])
            # if handler returned updated session, save it
            if result.get("session"):
                save_session(result["session"])
                return_payload["session_id"] = result["session"]["session_id"]

        # include clarifying questions if provided
        if result.get("clarifying_questions"):
            return_payload["clarifying_questions"] = result["clarifying_questions"]
        if result.get("pending_questions"):
            return_payload["pending_questions"] = result["pending_questions"]

        return return_payload

    except httpx.HTTPStatusError as e:
        error_details = e.response.json()
        raise HTTPException(status_code=e.response.status_code, detail=error_details)
    except Exception as e:
        logger.exception("Unhandled error in /chat")
        raise HTTPException(status_code=500, detail=str(e))
