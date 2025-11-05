"""LangGraph orchestration for the career-advice workflow."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Annotated, Any, Dict, Iterable, List, Literal, Optional, Tuple, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .career_logic import (
    evaluate_missing_fields,
    extract_answer_text,
    filter_and_validate_questions,
    generate_confirmation_questions,
    generate_deterministic_questions,
    merge_extracted_into_state,
    safe_parse_json,
    validate_plan_output,
)

logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_api_key(config: RunnableConfig) -> str:
    api_key = (config.get("configurable", {}) or {}).get("api_key")
    if not api_key:
        raise ValueError("API key required for LLM invocation")
    return api_key


def _extract_question_ids(questions: Iterable[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for question in questions:
        qid = question.get("id")
        if isinstance(qid, str) and qid:
            ids.append(qid)
    return ids


def _condense_answers(session_state: Dict[str, Any], limit: int = 10) -> str:
    answers = session_state.get("answers", {})
    if not isinstance(answers, dict) or not answers:
        return "<none>"
    recent_items = list(answers.items())[-limit:]
    condensed = [f"{key}: {str(value)[:240]}" for key, value in recent_items]
    return "\n".join(condensed) if condensed else "<none>"

# ---------------------------------------------------------------------------
# Environment configuration and defaults
# ---------------------------------------------------------------------------

EXTRACTOR_MODEL = os.environ.get("EXTRACTOR_MODEL", os.environ.get("CLARIFIER_MODEL", "gpt-4o-mini"))
CLARIFIER_MODEL = os.environ.get("CLARIFIER_MODEL", "gpt-4o-mini")
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "gpt-4o-mini")

EXTRACTOR_MAX_TOKENS = int(os.environ.get("EXTRACTOR_MAX_TOKENS", "800"))
CLARIFIER_MAX_TOKENS = int(os.environ.get("CLARIFIER_MAX_TOKENS", "900"))
PLANNER_MAX_TOKENS = int(os.environ.get("PLANNER_MAX_TOKENS", "2048"))

CONFIDENCE_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.70"))
MAX_CLARIFY_ROUNDS = int(os.environ.get("MAX_CLARIFY_ROUNDS", "4"))
MAX_QUESTIONS_PER_ROUND = int(os.environ.get("MAX_QUESTIONS_PER_ROUND", "5"))
CONFIRMATION_QUESTION_CAP = 3


# ---------------------------------------------------------------------------
# LangGraph state + result dataclasses
# ---------------------------------------------------------------------------


class CareerAdviceState(TypedDict, total=False):
    """State schema persisted by LangGraph."""

    messages: Annotated[List[BaseMessage], add_messages]
    session: Dict[str, Any]
    missing: Dict[str, List[str]]
    confirmations: List[Tuple[str, str]]
    questions: List[Dict[str, Any]]
    decision: str
    plan_text: str
    plan_valid: bool
    fallback_used: bool
    assumptions_mode: bool


@dataclass
class CareerGraphResult:
    decision: Literal["ASK_QUESTIONS", "CREATE_PLAN", "PLAN_EXISTS", "ERROR"]
    questions: List[Dict[str, Any]]
    session: Dict[str, Any]
    plan_text: Optional[str]
    plan_valid: bool
    question_rounds: int
    assumptions_mode: bool
    fallback_used: bool


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _message_content_to_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        rendered: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                rendered.append(str(block.get("text", "")))
            elif isinstance(block, str):
                rendered.append(block)
        return "\n".join(rendered)
    if content is None:
        return ""
    return str(content)


async def _call_chat_model(
    model_name: str,
    messages: List[BaseMessage],
    api_key: str,
    *,
    temperature: float = 0.0,
    max_tokens: int,
) -> AIMessage:
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        max_tokens=max_tokens,
    )
    return await llm.ainvoke(messages)


# ---------------------------------------------------------------------------
# Career advice graph
# ---------------------------------------------------------------------------


class CareerAdviceGraph:
    def __init__(self) -> None:
        self._checkpointer = InMemorySaver()
        builder: StateGraph = StateGraph(CareerAdviceState)

        builder.add_node("bootstrap", self._bootstrap)
        builder.add_node("extract", self._run_extractor)
        builder.add_node("analyze", self._analyze)
        builder.add_node("decide", self._decide)
        builder.add_node("questions", self._generate_questions)
        builder.add_node("plan", self._generate_plan)
        builder.add_node("finalize", self._finalize)

        builder.add_edge(START, "bootstrap")
        builder.add_edge("bootstrap", "extract")
        builder.add_edge("extract", "analyze")
        builder.add_edge("analyze", "decide")
        builder.add_conditional_edges(
            "decide",
            self._branch,
            {
                "ask": "questions",
                "plan": "plan",
                "plan_exists": "finalize",
                "error": "finalize",
            },
        )
        builder.add_edge("questions", "finalize")
        builder.add_edge("plan", "finalize")
        builder.add_edge("finalize", END)

        self._graph = builder.compile(checkpointer=self._checkpointer)

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    def _bootstrap(
        self,
        state: CareerAdviceState,
        config: Optional[RunnableConfig] = None,
    ) -> CareerAdviceState:
        session = state.get("session") or {}
        session.setdefault("mode", "career_advice")
        session_state = session.setdefault("state", {})
        session_state.setdefault("answers", {})
        session_state.setdefault("question_rounds", 0)
        session_state.setdefault("last_questions", [])
        session_state.setdefault("assumptions_mode", False)
        state["session"] = session
        state.setdefault("missing", {})
        state.setdefault("confirmations", [])
        state.setdefault("questions", [])
        state.setdefault("fallback_used", False)
        state.setdefault("plan_valid", False)
        return state

    async def _run_extractor(self, state: CareerAdviceState, config: RunnableConfig) -> CareerAdviceState:
        session = state.get("session", {})
        session_state = session.get("state", {})
        api_key = _require_api_key(config)

        last_user_message: Optional[HumanMessage] = None
        for message in reversed(state.get("messages", [])):
            if isinstance(message, HumanMessage):
                last_user_message = message
                break

        user_text = _message_content_to_text(last_user_message) if last_user_message else ""
        session_state["last_user_message"] = user_text
        session["state"] = session_state
        answers_text = extract_answer_text(session_state.get("answers", {}))

        system_prompt = (
            "You are a strict JSON extractor for a career-advisory system. "
            "Output ONLY valid JSON that adheres to the schema described below. Do NOT output any prose.\n\n"
            "SCHEMA:\n"
            "{\n"
            "  \"extracted\": {\n"
            "    \"background\": {\n"
            "      \"credentials\": [\"...\"],\n"
            "      \"hard_skills\": [\"...\"],\n"
            "      \"soft_skills\": [\"...\"],\n"
            "      \"action_verbs\": [\"...\"],\n"
            "      \"timeline\": \"...\"\n"
            "    },\n"
            "    \"goal\": {\n"
            "      \"role_keywords\": [\"...\"],\n"
            "      \"long_term_goals\": \"...\"\n"
            "    },\n"
            "    \"logistics\": {\n"
            "      \"location\": \"...\",\n"
            "      \"availability\": \"...\",\n"
            "      \"relocation\": \"...\"\n"
            "    },\n"
            "    \"resources\": {\n"
            "      \"budget\": \"...\",\n"
            "      \"time_commitment\": \"...\"\n"
            "    }\n"
            "  },\n"
            "  \"meta\": {\"confidence\": 0.0, \"notes\": \"...\"}\n"
            "}\n\n"
            "Rules:\n"
            "- Prefer lists for skills and role_keywords. Use empty lists or empty strings when unknown, not null.\n"
            "- Confidence is a number between 0 and 1 estimating how confident you are in the extraction; be conservative.\n"
            "- Do NOT invent specifics that aren't present. If unsure, leave fields empty and set confidence lower."
        )
        user_prompt = (
            "User raw message:\n" + (user_text or "<none>") +
            "\n\nPreviously collected answers (raw):\n" + (answers_text or "<none>") +
            "\n\nReturn the JSON now."
        )

        messages: List[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = await _call_chat_model(
                model_name=EXTRACTOR_MODEL,
                messages=messages,
                api_key=api_key,
                temperature=0.0,
                max_tokens=EXTRACTOR_MAX_TOKENS,
            )
            raw = response.text
            parsed = safe_parse_json(raw)
            if parsed:
                merge_extracted_into_state(session, parsed)
            else:
                logger.warning("Extractor LLM returned invalid JSON.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Extractor LLM call failed: %s", exc)

        state["session"] = session
        return state

    def _analyze(
        self,
        state: CareerAdviceState,
        config: Optional[RunnableConfig] = None,
    ) -> CareerAdviceState:
        session = state.get("session", {})
        missing, confirmations = evaluate_missing_fields(session, CONFIDENCE_THRESHOLD)
        state["missing"] = missing
        state["confirmations"] = confirmations
        return state

    def _decide(
        self,
        state: CareerAdviceState,
        config: Optional[RunnableConfig] = None,
    ) -> CareerAdviceState:
        session = state.get("session", {})
        session_state = session.get("state", {})
        if session_state.get("plan_created"):
            state["decision"] = "plan_exists"
            return state

        rounds = session_state.get("question_rounds", 0)
        if rounds >= MAX_CLARIFY_ROUNDS:
            state["decision"] = "plan"
            state["assumptions_mode"] = True
            session_state["assumptions_mode"] = True
            return state

        if state.get("confirmations"):
            state["decision"] = "ask"
            return state

        if state.get("missing"):
            state["decision"] = "ask"
            return state

        state["decision"] = "plan"
        return state

    def _branch(self, state: CareerAdviceState) -> str:
        return state.get("decision") or "error"

    async def _generate_questions(self, state: CareerAdviceState, config: RunnableConfig) -> CareerAdviceState:
        session = state.get("session", {})
        session_state = session.get("state", {})
        answers = session_state.get("answers", {})
        prev_questions = session_state.get("last_questions", [])
        prev_ids = _extract_question_ids(prev_questions)

        questions: List[Dict[str, Any]] = []
        fallback_used = False

        confirmations: List[Tuple[str, str]] = state.get("confirmations", [])
        missing: Dict[str, List[str]] = state.get("missing", {})

        if confirmations:
            confirmation_candidates = generate_confirmation_questions(
                confirmations,
                answers,
                max_questions=CONFIRMATION_QUESTION_CAP,
            )
            questions = filter_and_validate_questions(
                confirmation_candidates,
                missing,
                prev_ids,
                answers,
            )

        if not questions and missing:
            clarifier_questions = await self._call_clarifier_llm(
                missing=missing,
                session=session,
                prev_ids=prev_ids,
                last_user_text=session_state.get("last_user_message", ""),
                config=config,
            )
            questions = filter_and_validate_questions(
                clarifier_questions,
                missing,
                prev_ids,
                answers,
            )

        if not questions and missing:
            questions = generate_deterministic_questions(missing, max_questions=3)
            fallback_used = True

        session_state["last_questions"] = questions
        session_state["question_rounds"] = session_state.get("question_rounds", 0) + 1
        session["state"] = session_state

        state["questions"] = questions
        state["fallback_used"] = fallback_used
        state["session"] = session
        return state

    async def _generate_plan(self, state: CareerAdviceState, config: RunnableConfig) -> CareerAdviceState:
        session = state.get("session", {})
        session_state = session.get("state", {})
        api_key = _require_api_key(config)

        extracted = session_state.get("extracted", {})
        answers = session_state.get("answers", {})

        planner_system = (
            "You are an expert, evidence-based, and brutally honest AI career advisor."
            " You have been given the user's collected information below."
            " Produce a prioritized, actionable plan following this structure exactly:\n"
            "1. Honest assessment\n2. Feasibility & fit\n3. Concrete, prioritized roadmap (30d,90d,6m,12m) with KPIs and resources\n"
            "4. Role & job-targeting recommendations (3-5 roles)\n5. Resume/LinkedIn guidance (5 bullets, 3 headline variants)\n"
            "6. Interview & hiring strategy\n7. Risks, trade-offs & fallback options\n8. Checkpoint schedule & metrics\n9. One-line next step\n\n"
            "CRITICAL: You are in the PLANNING PHASE: Do NOT ask any clarifying questions."
        )

        if session_state.get("assumptions_mode"):
            planner_system += "\nIf information is missing, state your assumptions explicitly before providing guidance."

        lines = ["=== USER EXTRACTED ===", json.dumps(extracted, ensure_ascii=False, indent=2), "", "=== USER REPORTED ANSWERS ==="]
        for key, value in answers.items():
            lines.append(f"- {key}: {value}")
        lines.append("=== END ===")
        content = "\n".join(lines)

        messages = [
            SystemMessage(content=planner_system),
            HumanMessage(content=content),
        ]

        plan_valid = False
        plan_text = ""

        try:
            response = await _call_chat_model(
                model_name=PLANNER_MODEL,
                messages=messages,
                api_key=api_key,
                temperature=0.0,
                max_tokens=PLANNER_MAX_TOKENS,
            )
            plan_text = response.text
            plan_valid = validate_plan_output(plan_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Plan generation failed: %s", exc)
            plan_text = ""
            plan_valid = False

        session_state["plan_text"] = plan_text
        session_state["plan_created"] = bool(plan_valid)
        session["state"] = session_state

        state["plan_text"] = plan_text
        state["plan_valid"] = plan_valid
        state["session"] = session
        state["decision"] = "plan"
        return state

    def _finalize(
        self,
        state: CareerAdviceState,
        config: Optional[RunnableConfig] = None,
    ) -> CareerAdviceState:
        # ensure confirmation artifacts don't accumulate excessively
        state["confirmations"] = []
        return state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        *,
        session: Dict[str, Any],
        new_messages: List[BaseMessage],
        api_key: str,
    ) -> CareerGraphResult:
        initial_state: CareerAdviceState = {
            "session": session,
            "messages": new_messages,
        }
        config_payload: Dict[str, Any] = {"configurable": {"api_key": api_key}}
        session_id = session.get("session_id")
        if session_id:
            config_payload["configurable"]["thread_id"] = session_id

        result_state: CareerAdviceState = await self._graph.ainvoke(
            initial_state,
            config=config_payload,
        )

        updated_session: Dict[str, Any] = result_state.get("session", session)
        decision_internal = result_state.get("decision", "error")
        if decision_internal == "ask":
            decision = "ASK_QUESTIONS"
        elif decision_internal == "plan":
            decision = "CREATE_PLAN"
        elif decision_internal == "plan_exists":
            decision = "PLAN_EXISTS"
        else:
            decision = "ERROR"

        session_state = updated_session.get("state", {})
        plan_text = session_state.get("plan_text")
        plan_valid = bool(session_state.get("plan_created")) if decision != "ERROR" else False
        questions = result_state.get("questions", []) or []
        question_rounds = session_state.get("question_rounds", 0)
        assumptions_mode = bool(session_state.get("assumptions_mode"))
        fallback_used = bool(result_state.get("fallback_used"))

        return CareerGraphResult(
            decision=decision,
            questions=questions,
            session=updated_session,
            plan_text=plan_text,
            plan_valid=plan_valid,
            question_rounds=question_rounds,
            assumptions_mode=assumptions_mode,
            fallback_used=fallback_used,
        )

    # ------------------------------------------------------------------
    # Clarifier helper
    # ------------------------------------------------------------------

    async def _call_clarifier_llm(
        self,
        *,
        missing: Dict[str, List[str]],
        session: Dict[str, Any],
        prev_ids: List[str],
        last_user_text: str,
        config: RunnableConfig,
    ) -> List[Dict[str, Any]]:
        api_key = _require_api_key(config)

        session_state = session.get("state", {})
        condensed_block = _condense_answers(session_state)

        missing_desc = "; ".join(f"{cat}: {', '.join(vals)}" for cat, vals in missing.items())
        prev_ids_json = json.dumps(prev_ids)

        system_prompt = f"""
You are a clarification generator for a career-advisory system. Your job: produce a compact, unambiguous JSON object listing exactly what is missing and up to {MAX_QUESTIONS_PER_ROUND} clarifying questions that map 1:1 to missing required subfields.

Context (do NOT include in your output except where specified):
- Missing required subfields: {missing_desc}
- Previously collected answers (truncated):\n{condensed_block}
- Previously asked question ids (do NOT repeat): {prev_ids_json}

OUTPUT RULES (MANDATORY):
- Output only valid JSON. Do NOT include any explanation or plaintext before/after the JSON.
- JSON schema MUST be:
{{
  "missing": ["category_subfield"],
  "strategy": "short explanation why these questions",
  "questions": [{{"id":"category_subfield","question":"...","type":"text","required":true}}],
  "self_review": {{"notes":"","duplicates":[],"issues":[]}}
}}
- Ask at most {MAX_QUESTIONS_PER_ROUND} questions. Prefer fewer when missing subfields are closely related.
- Each question id MUST be exactly category_subfield (e.g. background_hard_skills).
- Questions MUST NOT request information already provided in 'Previously collected answers'.
- Do NOT rephrase a previously asked question id from the provided list.
- Questions should be direct and answerable in one sentence or short list.

If you cannot comply, return an empty questions array.
""".strip()

        last_user = last_user_text or "<none>"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User query/context:\n{last_user}\n\nWhen you output JSON, follow the schema exactly."),
        ]

        try:
            response = await _call_chat_model(
                model_name=CLARIFIER_MODEL,
                messages=messages,
                api_key=api_key,
                temperature=0.0,
                max_tokens=CLARIFIER_MAX_TOKENS,
            )
            parsed = safe_parse_json(response.text)
            if parsed and isinstance(parsed.get("questions"), list):
                return parsed.get("questions", [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Clarifier LLM call failed: %s", exc)
        return []