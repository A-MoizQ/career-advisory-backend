# main.py
import json
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import File, Form, HTTPException, UploadFile, FastAPI

from pdf_parser import extract_pdf_text_with_fallbacks
from prompts import build_system_prompt
from response_parser import (
    convert_structured_or_fix,
    extract_json_from_text,
    sanitize_raw_text,
)
from mode_handlers import MODE_HANDLERS
from session_store import delete_session, load_session, save_session

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

app = FastAPI()


def _parse_messages(messages_raw: str) -> List[Dict[str, Any]]:
    try:
        parsed = json.loads(messages_raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - FastAPI handles validation
        raise HTTPException(status_code=400, detail="Invalid 'messages' JSON format.") from exc
    if not isinstance(parsed, list):
        raise HTTPException(status_code=400, detail="'messages' must be a JSON array.")
    return parsed


async def _append_file_payload(file: Optional[UploadFile], messages: List[Dict[str, Any]]) -> None:
    if not file:
        return
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF is allowed.")

    try:
        logger.info("Processing uploaded PDF: %s", file.filename)
        pdf_bytes = await file.read()
        file_text = await extract_pdf_text_with_fallbacks(pdf_bytes, file.filename)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to process PDF file: {exc}") from exc

    if not file_text.strip():
        raise HTTPException(
            status_code=422,
            detail="Unable to extract text from PDF. The file might be a scanned document or password-protected.",
        )

    logger.info("✓ Extracted %s characters from PDF", len(file_text))
    messages.append(
        {
            "role": "user",
            "content": (
                f'The user has uploaded the file "{file.filename}". Its content is:\n\n'
                f"---START OF FILE---\n{file_text}\n---END OF FILE---\n\nNow, please review it."
            ),
        }
    )


def _get_handler(mode: str):
    handler = MODE_HANDLERS.get(mode)
    if not handler:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")
    return handler


def _load_existing_session(session_id: Optional[str]) -> Optional[Dict[str, Any]]:
    return load_session(session_id) if session_id else None


def _create_session(mode: str) -> Dict[str, Any]:
    session = {
        "session_id": str(uuid.uuid4()),
        "mode": mode,
        "state": {"answers": {}},
    }
    logger.info("Created new session: %s", session["session_id"])
    return session


def _ingest_answers(session: Dict[str, Any], answers_raw: Optional[str]) -> bool:
    if not answers_raw:
        return False
    try:
        answers_obj = json.loads(answers_raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid 'answers' JSON format.") from exc

    session.setdefault("state", {}).setdefault("answers", {}).update(answers_obj)
    logger.info("Updated session with answers: %s", answers_obj)
    return True


def _collect_warnings(state: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    if not state.get("plan_created", False):
        warnings.append("The plan could not be automatically validated. Please review it carefully.")
    if state.get("assumptions_mode"):
        warnings.append("Some assumptions were made due to missing details.")
    return warnings


async def _handle_session_flow(
    handler,
    session: Dict[str, Any],
    messages: List[Dict[str, Any]],
    api_key: str,
    answers_supplied: bool,
) -> Dict[str, Any]:
    clarifying_questions = await handler.needs_clarification(
        session=session,
        incoming_messages=messages,
        api_key=api_key,
    )

    save_session(session)

    if clarifying_questions:
        intro = (
            "Thanks for the information. I have a few more follow-up questions to ensure I understand correctly:"
            if answers_supplied
            else "To provide the best guidance, I have a few questions for you:"
        )
        return {
            "reply": intro,
            "clarifying_questions": clarifying_questions,
            "session_id": session["session_id"],
        }

    state = session.get("state", {})
    plan_text = state.get("plan_text")
    if not isinstance(plan_text, str) or not plan_text.strip():
        logger.error("Career advice graph completed without generating a plan.")
        raise HTTPException(status_code=500, detail="Plan generation failed. Please try again.")

    response_payload: Dict[str, Any] = {
        "reply": sanitize_raw_text(plan_text),
        "session_id": session["session_id"],
    }

    warnings = _collect_warnings(state)
    if warnings:
        response_payload["warnings"] = warnings

    return response_payload


def _truncate_messages_for_log(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not logger.isEnabledFor(logging.DEBUG):
        return []
    truncated: List[Dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str) and len(content) > 500:
            clone = dict(message)
            clone["content"] = content[:500] + "... [CONTENT TRUNCATED]"
            truncated.append(clone)
        else:
            truncated.append(message)
    return truncated


async def _invoke_completion_api(payload: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


def _build_reply_markdown(
    reply_text: str,
    structured: Optional[Dict[str, Any]],
    preserve_json: bool,
) -> str:
    if preserve_json and structured:
        try:
            return "```json\n" + json.dumps(structured, ensure_ascii=False, indent=2) + "\n```"
        except Exception:  # noqa: BLE001 - fallback handled below
            return convert_structured_or_fix(reply_text, structured)
    return convert_structured_or_fix(reply_text, structured)


def _sync_session_side_effects(
    original_session: Optional[Dict[str, Any]],
    handler_result: Dict[str, Any],
    payload: Dict[str, Any],
) -> None:
    new_session = handler_result.get("session")
    if handler_result.get("persist") and new_session:
        save_session(new_session)
        payload["session_id"] = new_session["session_id"]
        return

    if original_session and new_session is None:
        delete_session(original_session["session_id"])

    if new_session:
        save_session(new_session)
        payload["session_id"] = new_session["session_id"]


async def _handle_non_session_flow(
    handler,
    session: Optional[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    api_key: str,
    mode: str,
) -> Dict[str, Any]:
    system_prompt = build_system_prompt(mode)
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "system", "content": system_prompt}],
        "max_tokens": 2048,
        "temperature": 0.0,
    }

    extra_system_messages = handler.prepare_system_messages(session=session, incoming_messages=messages, answers=None)
    payload["messages"].extend({"role": "system", "content": message} for message in extra_system_messages)
    payload["messages"].append(messages[-1])

    truncated = _truncate_messages_for_log(payload["messages"])
    if truncated:
        logger.debug("Messages sent to completion API: %s", json.dumps(truncated, indent=2))

    data = await _invoke_completion_api(payload, api_key)

    reply_text_raw = data["choices"][0]["message"]["content"]
    reply_text = sanitize_raw_text(reply_text_raw)
    structured = extract_json_from_text(reply_text)
    handler_result = handler.handle_llm_response(session=session, structured=structured, raw_text=reply_text)

    reply_md = _build_reply_markdown(reply_text, structured, bool(handler_result.get("preserve_json")))

    logger.debug("Structured extracted: %s", json.dumps(structured) if structured else "None")
    logger.debug("Final reply_md (first 800 chars): %s", reply_md[:800])

    response_payload = {"reply": reply_md}
    _sync_session_side_effects(session, handler_result, response_payload)

    if handler_result.get("clarifying_questions"):
        response_payload["clarifying_questions"] = handler_result["clarifying_questions"]
    if handler_result.get("pending_questions"):
        response_payload["pending_questions"] = handler_result["pending_questions"]

    return response_payload


@app.post("/chat")
async def chat(
    api_key: str = Form(...),
    mode: str = Form(...),
    messages: str = Form(...),
    file: UploadFile = File(None),
    session_id: str = Form(None),
    answers: str = Form(None),
):
    logger.info("/chat called — mode=%s, file=%s, session_id=%s", mode, file, session_id)

    message_list = _parse_messages(messages)
    await _append_file_payload(file, message_list)

    handler = _get_handler(mode)
    session = _load_existing_session(session_id)

    if handler.requires_session():
        if not session:
            session = _create_session(mode)
        answers_supplied = _ingest_answers(session, answers)
        return await _handle_session_flow(handler, session, message_list, api_key, answers_supplied)

    return await _handle_non_session_flow(handler, session, message_list, api_key, mode)

