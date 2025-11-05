"""Deterministic utilities and heuristics for the LangGraph-based career advisor pipeline."""
from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Tuple

from response_parser import extract_json_from_text

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Domain constants reused from the legacy implementation
# ---------------------------------------------------------------------------

REQUIRED_SUBFIELDS: Dict[str, List[str]] = {
    "background": [
        "credentials",
        "hard_skills",
        "soft_skills",
        "action_verbs",
        "timeline",
    ],
    "goal": [
        "role_keywords",
        "long_term_goals",
    ],
    "logistics": [
        "location",
        "availability",
        "relocation",
    ],
    "resources": [
        "budget",
        "time_commitment",
    ],
}

HEURISTICS: Dict[str, List[str]] = {
    "credentials": [
        "degree",
        "certification",
        "license",
        "credential",
        "diploma",
        "certified",
        "bsc",
        "bs",
        "ms",
        "mba",
        "phd",
        "md",
    ],
    "hard_skills": [
        "python",
        "java",
        "sql",
        "cloud",
        "data",
        "analytics",
        "ml",
        "ai",
        "devops",
    ],
    "soft_skills": [
        "communication",
        "leadership",
        "teamwork",
        "adaptability",
        "problem-solving",
        "critical thinking",
        "collaboration",
    ],
    "action_verbs": [
        "developed",
        "led",
        "implemented",
        "optimized",
        "managed",
        "designed",
        "automated",
    ],
    "role_keywords": [
        "engineer",
        "developer",
        "analyst",
        "scientist",
        "manager",
        "specialist",
    ],
    "timeline": ["months", "years", "recent", "since", "duration", "available", "availability"],
    "location": ["remote", "onsite", "hybrid", "international", "local", "relocate"],
    "availability": ["hours", "per week", "full-time", "part-time", "contract", "freelance"],
    "budget": ["budget", "compensation", "paid", "salary", "stipend", "unpaid"],
    "long_term_goals": ["long term", "future", "eventually", "growth", "evolve", "aspire"],
}


# ---------------------------------------------------------------------------
# Shared text utilities
# ---------------------------------------------------------------------------


def extract_answer_text(answers: Dict[str, str]) -> str:
    return " \n".join(v.lower() for v in answers.values() if isinstance(v, str))


def extract_candidate_tokens(text: str) -> List[str]:
    """Extract tokens while preserving dots, hashes, pluses and dashes."""
    if not text:
        return []
    tokens = re.findall(r"[A-Za-z0-9#\+\.\-/&]+", text.lower())
    return [t for t in tokens if len(t) > 1]


def normalize_skill_token(token: str) -> str:
    t = token.lower()
    t = t.replace(".js", "js").replace(".", "").replace(" ", "").replace("_", "")
    if t in ("reactjs", "reactnative"):
        return "react"
    if t.startswith("node"):
        return "nodejs"
    return t


def is_text_duplicate(a: str, b: str, threshold: float = 0.78) -> bool:
    if not a or not b:
        return False
    a_norm = re.sub(r"\s+", " ", a.strip().lower())
    b_norm = re.sub(r"\s+", " ", b.strip().lower())
    if a_norm in b_norm or b_norm in a_norm:
        return True
    return SequenceMatcher(None, a_norm, b_norm).ratio() >= threshold


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def merge_extracted_into_state(session_state: Dict[str, Any], parsed: Dict[str, Any]) -> None:
    """Persist structured extractor output and derived tokens into the session state."""
    extracted = parsed.get("extracted", {}) if isinstance(parsed, dict) else {}
    meta = parsed.get("meta", {}) if isinstance(parsed, dict) else {}

    state = session_state.setdefault("state", {})
    state["extracted"] = extracted
    state["extracted_meta"] = meta

    combined: List[str] = []
    for category in ("background", "goal", "logistics", "resources"):
        cdict = extracted.get(category, {}) if isinstance(extracted.get(category, {}), dict) else {}
        for sub, value in cdict.items():
            if isinstance(value, list):
                combined.extend(normalize_skill_token(str(x)) for x in value if isinstance(x, str))
            elif isinstance(value, str) and value.strip():
                combined.extend(normalize_skill_token(tok) for tok in extract_candidate_tokens(value))

    answers = deepcopy(state.get("answers", {}))
    answers_text = extract_answer_text(answers)
    combined.extend(normalize_skill_token(tok) for tok in extract_candidate_tokens(answers_text))

    # deduplicate tokens preserving order
    seen: Dict[str, bool] = {}
    deduped: List[str] = []
    for token in combined:
        if token and token not in seen:
            seen[token] = True
            deduped.append(token)

    state["extracted_tokens"] = deduped


# ---------------------------------------------------------------------------
# Validation and question planning
# ---------------------------------------------------------------------------


def _satisfied_by_extractor(
    category: str,
    subfield: str,
    session_state: Dict[str, Any],
    conf_threshold: float,
) -> Tuple[bool, bool]:
    """Return tuple (satisfied, needs_confirmation)"""
    state = session_state.get("state", {})
    answers = deepcopy(state.get("answers", {}))
    extracted = deepcopy(state.get("extracted", {}))
    meta = deepcopy(state.get("extracted_meta", {}))
    tokens: List[str] = state.get("extracted_tokens", []) or []
    confidence = float(meta.get("confidence", 0.0) or 0.0)

    # explicit answer present
    for key in answers.keys():
        if key == f"{category}_{subfield}" or key.endswith(subfield) or key.startswith(f"{category}_"):
            return True, False

    cand = None
    try:
        cand = extracted.get(category, {}).get(subfield)
    except Exception:
        cand = None

    if isinstance(cand, list) and cand:
        return (confidence >= conf_threshold, confidence < conf_threshold)

    if isinstance(cand, str) and cand.strip():
        lower = cand.lower()
        if subfield == "timeline":
            if re.search(r"\b(\d+\s*(years|year|months|month|weeks|week))\b", lower):
                if confidence >= (conf_threshold * 0.9):
                    return True, False
                return False, True
            return confidence >= conf_threshold, confidence < conf_threshold
        return confidence >= conf_threshold, confidence < conf_threshold

    norm_needed = subfield in {"hard_skills", "role_keywords", "soft_skills", "credentials"}
    if norm_needed and tokens:
        if subfield == "credentials":
            if any(
                re.match(r"^(b|m|ph)[a-z]{0,3}$", tok) or tok in HEURISTICS.get("credentials", [])
                for tok in tokens
            ):
                return True, False
        else:
            return True, False

    answers_text = extract_answer_text(answers)
    if subfield == "hard_skills":
        if "," in answers_text or len(extract_candidate_tokens(answers_text)) >= 3:
            return True, False
        if any(word in answers_text for word in ["framework", "library", "tool", "stack", "technologies"]):
            return True, False

    if subfield == "timeline":
        if re.search(r"\b(\d+\s*(years|year|months|month|weeks|week))\b", answers_text):
            return True, False
        if any(word in answers_text for word in ["available", "availability", "hours", "per week", "full-time", "part-time"]):
            return True, False

    if subfield == "credentials":
        if re.search(
            r"\b(ms|bs|bsc|ba|mba|phd|md|jd|certificate|certified|pmp|cfa)\b",
            answers_text,
        ):
            return True, False

    if subfield == "soft_skills":
        if any(word in answers_text for word in ["team", "lead", "manage", "communicate", "collaborate", "people"]):
            return True, False

    return False, False


def evaluate_missing_fields(
    session_state: Dict[str, Any],
    conf_threshold: float,
) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
    state = session_state.get("state", {})
    missing: Dict[str, List[str]] = {}
    confirmations: List[Tuple[str, str]] = []

    extracted = deepcopy(state.get("extracted", {}))

    for category, subfields in REQUIRED_SUBFIELDS.items():
        for subfield in subfields:
            satisfied, needs_confirmation = _satisfied_by_extractor(category, subfield, session_state, conf_threshold)
            if satisfied and not needs_confirmation:
                continue
            if needs_confirmation:
                value = None
                try:
                    value = extracted.get(category, {}).get(subfield)
                except Exception:
                    value = None
                summary = ""
                if isinstance(value, list):
                    summary = ", ".join(str(x) for x in value[:6])
                elif isinstance(value, str):
                    summary = value if len(value) < 200 else value[:200] + "..."
                confirmations.append((f"{category}_{subfield}", summary))
            else:
                missing.setdefault(category, []).append(subfield)

    return missing, confirmations


def generate_confirmation_questions(
    confirmations: Iterable[Tuple[str, str]],
    answers: Dict[str, str],
    max_questions: int,
) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []
    count = 0
    for cid, summary in confirmations:
        if count >= max_questions:
            break
        if "_" not in cid:
            continue
        category, subfield = cid.split("_", 1)
        if any(
            key == cid or key.endswith(subfield) or key.startswith(f"{category}_")
            for key in answers.keys()
        ):
            continue
        question_text = (
            f"I extracted: {summary} as your {subfield.replace('_', ' ')} ({category}). Is this correct? "
            "If not, please correct it briefly."
            if summary
            else f"Can you confirm your {subfield.replace('_', ' ')} ({category})? Provide a short answer."
        )
        questions.append({
            "id": cid,
            "question": question_text,
            "type": "text",
            "required": True,
        })
        count += 1
    return questions


def generate_deterministic_questions(
    missing: Dict[str, List[str]],
    max_questions: int,
) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []
    count = 0
    for category, subs in missing.items():
        for subfield in subs:
            if count >= max_questions:
                break
            readable = subfield.replace("_", " ")
            questions.append({
                "id": f"{category}_{subfield}",
                "question": f"Please provide your {readable} ({category}).",
                "type": "text",
                "required": True,
            })
            count += 1
        if count >= max_questions:
            break
    return questions


def filter_and_validate_questions(
    raw_questions: Iterable[Dict[str, Any]],
    missing: Dict[str, List[str]],
    previous_ids: Iterable[str],
    answers: Dict[str, str],
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    previous_set = set(previous_ids)
    missing_set = {f"{cat}_{sub}" for cat, subs in missing.items() for sub in subs}

    for question in raw_questions:
        qid = (question.get("id") or "").strip()
        text = (question.get("question") or "").strip()
        if not qid or not text:
            continue
        if "_" not in qid:
            continue
        if qid in seen_ids or qid in previous_set:
            continue
        category, subfield = qid.split("_", 1)
        if qid not in missing_set and not (category in missing and subfield in missing[category]):
            continue
        if any(
            key == qid or key.endswith(subfield) or key.startswith(f"{category}_")
            for key in answers.keys()
        ):
            continue
        if any(is_text_duplicate(existing["question"], text) for existing in filtered):
            continue
        seen_ids.add(qid)
        filtered.append({
            "id": qid,
            "question": text,
            "type": question.get("type", "text"),
            "required": bool(question.get("required", True)),
        })
    return filtered


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    parsed = None
    try:
        parsed = extract_json_from_text(text)
    except Exception:
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
    return parsed if isinstance(parsed, dict) else None


# ---------------------------------------------------------------------------
# Plan validation heuristics
# ---------------------------------------------------------------------------


def validate_plan_output(text: str) -> bool:
    """Heuristic validation for the generated plan text."""
    if not text or not isinstance(text, str):
        return False

    lower = text.lower()
    required_headings = [
        "honest assessment",
        "feasibility",
        "concrete",
        "role",
        "resume",
        "interview",
        "risks",
        "checkpoint",
        "one-line",
    ]
    found = sum(1 for heading in required_headings if heading in lower)
    if found >= 5:
        return True

    if re.search(r"##\s*1[\.\)]", text) or re.search(r"\b1\.\s*(honest|assessment)", lower):
        return True

    if re.search(r"##+\s*\d+\.\s*[A-Za-z ]{5,40}", text):
        return True

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
            keys = {k.lower() for k in obj.keys()}
            expected = {
                "honest_assessment",
                "feasibility",
                "roadmap",
                "role_recommendations",
                "resume",
                "interview",
                "risks",
            }
            if len(keys.intersection(expected)) >= 2:
                return True
        except Exception:
            pass

    if (
        re.search(r"\bones?[- ]line\b.*(next|step|do|start)", lower)
        or "one-line next step" in lower
    ) and found >= 3:
        return True

    return False
