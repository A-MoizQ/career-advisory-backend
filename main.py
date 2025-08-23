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
    structured_json_to_markdown,
    normalize_markdown,
    sanitize_raw_text,
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

    # Load session if provided and ensure it's for this mode
    session = None
    if session_id:
        session = load_session(session_id)
        if session and session.get("mode") != mode:
            # mismatch: ignore session to avoid cross-mode pollution
            logger.info(f"Session {session_id} present but mode mismatch (expected {mode}). Ignoring session.")
            session = None

    # Parse answers if provided
    answers_obj = None
    if answers:
        try:
            answers_obj = json.loads(answers)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid 'answers' JSON format.")

    # CRITICAL: For session-based handlers, always check for clarification first
    if handler.requires_session():
        # Check if we need clarifying questions (this is the key enforcement point)
        clar_qs = await handler.needs_clarification(session=session, incoming_messages=message_list, api_key=api_key)
        
        if clar_qs:
            # We need clarifying questions - ensure session exists
            if not session:
                session = {
                    "session_id": session_id or str(uuid.uuid4()),
                    "mode": mode,
                    "state": {"answers": {}}
                }
            
            # If user provided answers, merge them into session
            if answers_obj:
                session.setdefault("state", {}).setdefault("answers", {}).update(answers_obj)
                save_session(session)
                
                # Re-check if we still need more clarification after these answers
                clar_qs_recheck = await handler.needs_clarification(session=session, incoming_messages=message_list, api_key=api_key)
                if clar_qs_recheck:
                    # Still need more questions
                    return {
                        "reply": "Thank you for your answers. Please answer these additional questions to help me provide better guidance.",
                        "clarifying_questions": clar_qs_recheck,
                        "session_id": session["session_id"]
                    }
                # else: we have enough answers, continue to planning below
            else:
                # First time asking questions
                session.setdefault("state", {}).setdefault("pending_questions", [q["id"] for q in clar_qs])
                save_session(session)
                
                return {
                    "reply": "To provide you with the most effective career guidance, I need to understand your specific situation better. Please answer these questions:",
                    "clarifying_questions": clar_qs,
                    "session_id": session["session_id"]
                }

    # If we reach here, either:
    # 1. Handler doesn't require session (resume_review, etc.)
    # 2. Handler requires session but we have sufficient answers to proceed
    
    # Build system prompt and messages
    system_prompt = build_system_prompt(mode)
    full_messages = [{"role": "system", "content": system_prompt}]

    # Ask handler for extra system messages (includes collected answers for career advice)
    extra_system_messages = handler.prepare_system_messages(session=session, incoming_messages=message_list, answers=answers_obj)
    for m in extra_system_messages:
        full_messages.append({"role": "system", "content": m})

    # Add conversation messages
    full_messages += message_list

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

        # Convert structured JSON to markdown unless handler preserves JSON
        preserve_json = bool(result.get("preserve_json", False))

        if structured and not preserve_json:
            try:
                md_from_struct = structured_json_to_markdown(structured)
                if md_from_struct and len(md_from_struct.strip()) > 5:
                    reply_md = md_from_struct
                else:
                    reply_md = normalize_markdown(reply_text)
            except Exception:
                reply_md = normalize_markdown(reply_text)
        else:
            reply_candidate = result.get("reply")
            if reply_candidate:
                reply_md = reply_candidate
            else:
                reply_md = normalize_markdown(reply_text)

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