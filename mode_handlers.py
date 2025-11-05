"""Mode handlers orchestrating backend behavior using LangGraph and LangChain."""
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from orchestration.career_graph import CareerAdviceGraph, CareerGraphResult

logger = logging.getLogger("uvicorn.error")

MessagePayload = Dict[str, Any]
SessionDict = Dict[str, Any]
QuestionList = List[Dict[str, Any]]


def _convert_new_messages(message_payloads: Iterable[MessagePayload]) -> List[BaseMessage]:
    """Convert raw message dictionaries into LangChain message objects."""
    converted: List[BaseMessage] = []
    for payload in message_payloads:
        role = payload.get("role")
        content = payload.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        if role == "assistant":
            converted.append(AIMessage(content=content))
        elif role == "system":
            converted.append(SystemMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    return converted


def _extract_new_user_messages(state: SessionDict, incoming: List[MessagePayload]) -> List[HumanMessage]:
    processed = max(int(state.get("processed_messages", 0) or 0), 0)
    new_payloads = incoming[processed:]
    user_messages = [msg for msg in _convert_new_messages(new_payloads) if isinstance(msg, HumanMessage)]
    if not user_messages and processed == len(incoming):
        last_user = state.get("last_user_message")
        if isinstance(last_user, str) and last_user.strip():
            user_messages = [HumanMessage(content=last_user)]
    return user_messages


def _merge_graph_result(
    session: SessionDict,
    result: CareerGraphResult,
    total_messages: int,
) -> None:
    updated_session = deepcopy(result.session) if result.session else {}
    prior_session_id = session.get("session_id")
    if prior_session_id and "session_id" not in updated_session:
        updated_session["session_id"] = prior_session_id
    state = updated_session.setdefault("state", {})
    state["processed_messages"] = total_messages
    state["last_decision"] = result.decision
    if result.plan_text is not None:
        state["plan_text"] = result.plan_text
    if result.plan_valid is not None:
        state["plan_valid"] = result.plan_valid
        state["plan_created"] = result.plan_valid
    state["assumptions_mode"] = result.assumptions_mode or state.get("assumptions_mode", False)
    state["fallback_used"] = result.fallback_used
    state["question_rounds"] = result.question_rounds

    session.clear()
    session.update(updated_session)


@dataclass
class CareerAdviceHandler:
    """Career advice mode orchestrated via LangGraph."""

    _graph: CareerAdviceGraph = CareerAdviceGraph()

    def requires_session(self) -> bool:
        return True

    async def needs_clarification(
        self,
        session: Optional[SessionDict],
        incoming_messages: List[MessagePayload],
        api_key: str,
    ) -> Optional[QuestionList]:
        if not session:
            return None

        state = session.setdefault("state", {})
        new_messages = _extract_new_user_messages(state, incoming_messages)

        result: CareerGraphResult = await self._graph.run(
            session=session,
            new_messages=new_messages,
            api_key=api_key,
        )

        _merge_graph_result(session, result, len(incoming_messages))

        if result.decision == "ASK_QUESTIONS":
            return result.questions

        return None

    def prepare_system_messages(
        self,
        session: Optional[SessionDict],
        incoming_messages: List[MessagePayload],
        answers: Optional[Dict[str, str]],
    ) -> List[str]:
        return []

    def handle_llm_response(
        self,
        session: Optional[SessionDict],
        structured: Optional[Dict[str, Any]],
        raw_text: str,
    ) -> Dict[str, Any]:
        return {"persist": True, "session": session}


class _NoOpHandler:
    """Placeholder handlers for modes not yet implemented with LangGraph."""

    def requires_session(self) -> bool:
        return False

    async def needs_clarification(
        self,
        session: Optional[SessionDict],
        incoming_messages: List[MessagePayload],
        api_key: str,
    ) -> Optional[QuestionList]:  # pragma: no cover - trivial pass-through
        return None

    def prepare_system_messages(
        self,
        session: Optional[SessionDict],
        incoming_messages: List[MessagePayload],
        answers: Optional[Dict[str, str]],
    ) -> List[str]:  # pragma: no cover - trivial pass-through
        return []

    def handle_llm_response(
        self,
        session: Optional[SessionDict],
        structured: Optional[Dict[str, Any]],
        raw_text: str,
    ) -> Dict[str, Any]:  # pragma: no cover - trivial pass-through
        return {"persist": False, "session": session, "reply": None}


MODE_HANDLERS = {
    "career_advice": CareerAdviceHandler(),
    "resume_review": _NoOpHandler(),
    "job_hunt": _NoOpHandler(),
    "learning_roadmap": _NoOpHandler(),
    "mock_interview": _NoOpHandler(),
}
