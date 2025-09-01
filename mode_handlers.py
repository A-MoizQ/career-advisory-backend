# mode_handlers.py
import uuid
import os
import httpx
import logging
import re
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional, Tuple
import json
from copy import deepcopy

from response_parser import extract_json_from_text
from session_store import save_session, load_session, delete_session

logger = logging.getLogger("uvicorn.error")

# Models (env override)
EXTRACTOR_MODEL = os.environ.get("EXTRACTOR_MODEL", os.environ.get("CLARIFIER_MODEL", "gpt-4"))
CLARIFIER_MODEL = os.environ.get("CLARIFIER_MODEL", "gpt-4")
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "gpt-3.5-turbo")

# Tuning
CONFIDENCE_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.70"))
MAX_CLARIFY_ROUNDS = int(os.environ.get("MAX_CLARIFY_ROUNDS", "4"))
MAX_QUESTIONS_PER_ROUND = int(os.environ.get("MAX_QUESTIONS_PER_ROUND", "5"))

# Define the mandatory categories that MUST be covered before planning
REQUIRED_SUBFIELDS = {
    # 1. Foundational User Background
    "background": [
        "credentials",           # degrees, diplomas, certifications
        "hard_skills",           # technical skills, tools, frameworks
        "soft_skills",           # communication, leadership, adaptability
        "action_verbs",          # indicates past actions/projects
        "timeline"               # duration/time references (experience & availability)
    ],

    # 2. Career Vision & Ambitions
    "goal": [
        "role_keywords",         # specific role or domain aspirations
        "long_term_goals",       # long-term vision/future ambition
    ],

    # 3. Practical Constraints & Logistics
    "logistics": [
        "location",              # whereabouts or relocation readiness
        "availability",          # part-time/full-time, hours per week
        "relocation",            # moving or visa considerations
    ],

    # 4. Resources & Constraints
    "resources": [
        "budget",                # budget constraints or compensation notes
        "time_commitment",       # availability/training capacity
    ]
}

# Lightweight heuristics for fallback detection (kept small; extractor does the heavy lifting)
HEURISTICS = {
    "credentials": ["degree", "certification", "license", "credential", "diploma", "certified", "bsc", "bs", "ms", "mba", "phd", "md"],
    "hard_skills": ["python", "java", "sql", "cloud", "data", "analytics", "ml", "ai", "devops"],
    "soft_skills": ["communication", "leadership", "teamwork", "adaptability", "problem-solving", "critical thinking", "collaboration"],
    "action_verbs": ["developed", "led", "implemented", "optimized", "managed", "designed", "automated"],
    "role_keywords": ["engineer", "developer", "analyst", "scientist", "manager", "specialist"],
    "timeline": ["months", "years", "recent", "since", "duration", "available", "availability"],
    "location": ["remote", "onsite", "hybrid", "international", "local", "relocate"],
    "availability": ["hours", "per week", "full-time", "part-time", "contract", "freelance"],
    "budget": ["budget", "compensation", "paid", "salary", "stipend", "unpaid"],
    "long_term_goals": ["long term", "future", "eventually", "growth", "evolve", "aspire"],
}

# ------------------- Utility text/token helpers -------------------

def _extract_answer_text(answers: Dict[str, str]) -> str:
    return " \n".join(v.lower() for v in answers.values() if isinstance(v, str))

def _extract_candidate_tokens(text: str) -> List[str]:
    """
    Extract tokens from text preserving dots, hashes, pluses and dashes (e.g., node.js, c#, react-native).
    Returns lowercase tokens longer than 1 char.
    """
    if not text:
        return []
    tokens = re.findall(r"[A-Za-z0-9#\+\.\-/&]+", text.lower())
    tokens = [t for t in tokens if len(t) > 1]
    return tokens

def _normalize_skill_token(tok: str) -> str:
    """
    Simple normalization map for common variants; extendable.
    """
    t = tok.lower()
    t = t.replace(".js", "js").replace(".", "").replace(" ", "").replace("_", "")
    if t in ("reactjs", "reactnative"):
        return "react"
    if t.startswith("node"):
        return "nodejs"
    return t

def _is_text_duplicate(a: str, b: str, threshold: float = 0.78) -> bool:
    if not a or not b:
        return False
    a_norm = re.sub(r"\s+", " ", a.strip().lower())
    b_norm = re.sub(r"\s+", " ", b.strip().lower())
    if a_norm in b_norm or b_norm in a_norm:
        return True
    return SequenceMatcher(None, a_norm, b_norm).ratio() >= threshold

# ------------------- LLM extractor (step 1) -------------------

async def _call_extractor_llm(user_text: str, answers_text: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Call LLM to extract structured fields from raw user text + any previously saved answers.
    Returns parsed JSON or None on failure.
    Expected JSON schema:
    {
      "extracted": {
         "background": {"credentials": [...], "hard_skills": [...], "soft_skills": [...], "action_verbs": [...], "timeline": "2 years"},
         "goal": {"role_keywords": [...], "long_term_goals": "..."},
         "logistics": {"location": "...", "availability": "...", "relocation": "..."},
         "resources": {"budget": "...", "time_commitment": "..."}
      },
      "meta": {"confidence": 0.84, "notes": "short rationale"}
    }
    """
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
        "- Do NOT invent specifics that aren't present. If unsure, leave fields empty and set confidence lower.\n"
    )

    user_content = (
        "User raw message:\n" + (user_text or "<none>") +
        "\n\nPreviously collected answers (raw):\n" + (answers_text or "<none>") +
        "\n\nReturn the JSON now."
    )

    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": EXTRACTOR_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.0,
        "max_tokens": 600
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        parsed = None
        try:
            parsed = extract_json_from_text(raw)
        except Exception:
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = None
        if not parsed or not isinstance(parsed, dict):
            logger.warning("Extractor LLM returned invalid JSON.")
            return None
        return parsed
    except Exception as e:
        logger.warning(f"Extractor LLM call failed: {e}")
        return None

# ------------------- Merge & validation (step 2) -------------------

def _merge_extracted_into_session(session: Dict[str, Any], parsed: Dict[str, Any]):
    """
    Store the extractor's structured output into session state (canonicalized).
    """
    if not session:
        return
    state = session.setdefault("state", {})
    extracted = parsed.get("extracted", {}) if isinstance(parsed, dict) else {}
    meta = parsed.get("meta", {}) if isinstance(parsed, dict) else {}
    state["extracted"] = extracted
    state["extracted_meta"] = meta
    # Produce a flat set of tokens for easy checks
    combined = []
    for cat in ["background", "goal", "logistics", "resources"]:
        cdict = extracted.get(cat, {}) if isinstance(extracted.get(cat, {}), dict) else {}
        for sub, val in cdict.items():
            if isinstance(val, list):
                combined.extend([_normalize_skill_token(str(x)) for x in val if isinstance(x, str)])
            elif isinstance(val, str) and val.strip():
                # split into tokens
                toks = _extract_candidate_tokens(val)
                combined.extend([_normalize_skill_token(t) for t in toks])
    # also include answers raw tokens if present
    answers = state.get("answers", {})
    answers_text = _extract_answer_text(answers)
    combined.extend([_normalize_skill_token(t) for t in _extract_candidate_tokens(answers_text)])
    # de-dup while preserving insertion order
    seen = {}
    for t in combined:
        if t and t not in seen:
            seen[t] = True
    state["extracted_tokens"] = list(seen.keys())
    save_session(session)

def _validate_extracted_and_compute_missing(session: Dict[str, Any]) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
    """
    Using extracted data and deterministic rules, decide:
    - which required subfields remain missing (returns dict like REQUIRED_SUBFIELDS shape),
    - which fields are ambiguous/low-confidence and therefore need a confirmation question (returns list of (category_subfield, short_summary)).
    """
    state = session.get("state", {})
    answers = state.get("answers", {}) or {}
    answers_text = _extract_answer_text(answers)
    extracted = state.get("extracted", {}) or {}
    meta = state.get("extracted_meta", {}) or {}
    extracted_tokens = state.get("extracted_tokens", []) or []
    confidence = float(meta.get("confidence", 0.0) or 0.0)

    missing: Dict[str, List[str]] = {}
    confirmations: List[Tuple[str, str]] = []

    # heuristic function to decide if a subfield is satisfied
    def satisfied_by_extractor(cat: str, sub: str) -> bool:
        # 1) if user explicitly answered a key like background_hard_skills
        for k in answers.keys():
            if k == f"{cat}_{sub}" or k.endswith(sub) or k.startswith(cat + "_"):
                return True

        # 2) if extractor has a non-empty value and confidence is high enough
        try:
            val = extracted.get(cat, {}).get(sub)
        except Exception:
            val = None

        if isinstance(val, list) and len(val) > 0:
            if confidence >= CONFIDENCE_THRESHOLD:
                return True
            else:
                # Low confidence but non-empty -> mark for confirmation
                return False

        if isinstance(val, str) and val.strip():
            if sub == "timeline":
                # use regex detection
                if re.search(r"\b(\d+\s*(years|year|months|month|weeks|week))\b", val.lower()):
                    if confidence >= (CONFIDENCE_THRESHOLD * 0.9):
                        return True
                    else:
                        return False
                # short phrases like 'available' count only if confidence high
                return confidence >= CONFIDENCE_THRESHOLD
            else:
                if confidence >= CONFIDENCE_THRESHOLD:
                    return True
                else:
                    return False

        # 3) token-based fallback: e.g. tokens extracted from raw answers (react, nodejs, etc.)
        norm_needed = sub in ("hard_skills", "role_keywords", "soft_skills", "credentials")
        if norm_needed and any(t for t in extracted_tokens):
            # if at least 1 token present, consider it satisfied (conservative)
            # but for credentials require tokens like 'bs', 'ms', degree abbreviations
            if sub == "credentials":
                if any(re.match(r"^(b|m|ph)[a-z]{0,3}$", tok) or tok in HEURISTICS.get("credentials", []) for tok in extracted_tokens):
                    return True
            else:
                if len(extracted_tokens) >= 1:
                    return True

        # 4) content heuristics applied to raw answers_text
        if sub == "hard_skills":
            if "," in answers_text or len(_extract_candidate_tokens(answers_text)) >= 3:
                return True
            if any(w in answers_text for w in ["framework", "library", "tool", "stack", "technologies"]):
                return True

        if sub == "timeline":
            if re.search(r"\b(\d+\s*(years|year|months|month|weeks|week))\b", answers_text):
                return True
            if any(w in answers_text for w in ["available", "availability", "hours", "per week", "full-time", "part-time"]):
                return True

        if sub == "credentials":
            if re.search(r"\b(ms|bs|bsc|ba|mba|phd|md|jd|certificate|certified|pmp|cfa)\b", answers_text):
                return True

        if sub == "soft_skills":
            if any(w in answers_text for w in ["team", "lead", "manage", "communicate", "collaborate", "people"]):
                return True

        # fallback not satisfied
        return False

    # iterate required fields and fill missing/confirmations
    for cat, subs in REQUIRED_SUBFIELDS.items():
        for sub in subs:
            ok = satisfied_by_extractor(cat, sub)
            if ok:
                # satisfied by extractor or heuristics
                continue
            else:
                # check if extractor had some non-empty value but low confidence -> confirmation
                val = None
                try:
                    val = extracted.get(cat, {}).get(sub)
                except Exception:
                    val = None
                if val:
                    # ambiguous/low confidence -> confirmation
                    summary = ""
                    if isinstance(val, list):
                        summary = ", ".join(str(x) for x in val[:6])
                    elif isinstance(val, str):
                        summary = val if len(val) < 200 else val[:200] + "..."
                    confirmations.append((f"{cat}_{sub}", summary))
                else:
                    # genuinely missing
                    missing.setdefault(cat, []).append(sub)

    return missing, confirmations

# ------------------- Confirmation question generation (step 3) -------------------

def _generate_confirmation_questions(confirmations: List[Tuple[str, str]], answers: Dict[str, str], max_q: int = 3) -> List[Dict[str, Any]]:
    """
    Turn an extractor ambiguity into short confirm questions.
    confirmations: list of (category_subfield, short_summary)
    """
    out = []
    count = 0
    for cid, summary in confirmations:
        if count >= max_q:
            break
        if "_" not in cid:
            continue
        cat, sub = cid.split("_", 1)
        # If the user already answered explicitly for this id, skip
        if any(k == cid or k.endswith(sub) or k.startswith(cat + "_") for k in answers.keys()):
            continue
        if summary:
            q_text = f"I extracted: {summary} as your {sub.replace('_',' ')} ({cat}). Is this correct? If not, please correct it briefly."
        else:
            q_text = f"Can you confirm your {sub.replace('_',' ')} ({cat})? Provide a short answer."

        out.append({
            "id": cid,
            "question": q_text,
            "type": "text",
            "required": True
        })
        count += 1
    return out

# ------------------- Clarifier (step 4) - existing JSON clarifier with some adjustments -------------------

def _generate_deterministic_questions(missing: Dict[str, List[str]], answers: Dict[str, str], max_q: int = 3) -> List[Dict[str, Any]]:
    """Produce a conservative list of fallback clarifying questions (deterministic)."""
    out = []
    count = 0
    for cat, subs in missing.items():
        for sub in subs:
            if count >= max_q:
                break
            readable = sub.replace("_", " ")
            out.append({
                "id": f"{cat}_{sub}",
                "question": f"Please provide your {readable} ({cat}).",
                "type": "text",
                "required": True
            })
            count += 1
        if count >= max_q:
            break
    return out

def _filter_and_validate_questions(raw_questions: List[Dict[str, Any]], missing: Dict[str, List[str]],
                                   prev_q_ids: List[str], answers: Dict[str, str]) -> List[Dict[str, Any]]:
    """Validate that each question maps 1:1 to a missing subfield and is not a repeat."""
    filtered: List[Dict[str, Any]] = []
    seen_ids = set()

    # flatten missing into set of category_subfield strings for quick check
    missing_set = set(f"{cat}_{sub}" for cat, subs in missing.items() for sub in subs)

    for q in raw_questions:
        qid = (q.get("id") or "").strip()
        text = (q.get("question") or "").strip()
        if not qid or not text:
            continue
        # enforce format category_subfield
        if "_" not in qid:
            continue
        if qid in seen_ids:
            continue
        # do not ask for already answered subfields
        if any(k == qid or k.endswith(qid.split('_',1)[1]) or k.startswith(qid.split('_',1)[0]) for k in answers.keys()):
            continue
        # only allow questions for actually missing fields
        if qid not in missing_set:
            # Allow if the subfield portion is in missing values (some models emit cat and sub differently)
            cat, sub = qid.split("_", 1)
            if not (cat in missing and sub in missing[cat]):
                continue
        # avoid repeating previous questions
        if qid in prev_q_ids:
            continue
        # avoid near-duplicate text
        skip = False
        for existing in filtered:
            if _is_text_duplicate(existing["question"], text):
                skip = True
                break
        if skip:
            continue
        # passed basic checks
        seen_ids.add(qid)
        filtered.append({
            "id": qid,
            "question": text,
            "type": q.get("type", "text"),
            "required": bool(q.get("required", True))
        })
    return filtered

async def _call_clarifier_llm(missing: Dict[str, List[str]], condensed_block: str,
                              prev_q_ids: List[str], last_user_msg: Optional[str], api_key: str) -> Optional[Dict[str, Any]]:
    """
    Ask the clarifier LLM to output up to MAX_QUESTIONS_PER_ROUND JSON questions for missing subfields.
    Reuses earlier clarifier prompt but ensures strict JSON output.
    """
    missing_desc = "; ".join(f"{c}: {', '.join(s)}" for c, s in missing.items())
    system_prompt = f"""
You are a clarification generator for a career-advisory system. Your job: produce a compact, unambiguous JSON object listing exactly what is missing and up to {MAX_QUESTIONS_PER_ROUND} clarifying questions that map 1:1 to missing required subfields.

Context (do NOT include in your output except where specified):
- Missing required subfields: {missing_desc}
- Previously collected answers (truncated):\n{condensed_block}
- Previously asked question ids (do NOT repeat): {prev_q_ids}

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
"""
    user_example = {
        "role": "user",
        "content": (
            "User query/context: \n" + (last_user_msg or "<none>") + "\n\nWhen you output JSON, follow the schema exactly."
        )
    }

    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": CLARIFIER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            user_example
        ],
        "temperature": 0.0,
        "max_tokens": 600
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        parsed = None
        try:
            parsed = extract_json_from_text(raw)
        except Exception:
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = None
        return parsed
    except Exception as e:
        logger.warning(f"Clarifier LLM call failed: {e}")
        return None

# ------------------- Plan generation (final step) -------------------

def _validate_plan_output(text: str) -> bool:
    """
    Heuristic validation for the generated plan text.
    Returns True if the output appears to contain the required plan structure.
    Uses multiple checks:
      - presence of several required heading keywords (case-insensitive)
      - presence of '## 1.' or similar numbered headings
      - or a JSON object containing expected top-level keys (if planner returns JSON)
    This is intentionally conservative: it prefers false-negatives (ask again) over false-positives.
    """
    if not text or not isinstance(text, str):
        return False

    lower = text.lower()

    # 1) Required heading keywords (these are short checks; need several to pass)
    required_headings = [
        "honest assessment", "feasibility", "concrete", "role", "resume", "interview",
        "risks", "checkpoint", "one-line"
    ]
    found = 0
    for h in required_headings:
        if h in lower:
            found += 1
    if found >= 5:
        # found majority of headings -> accept
        return True

    # 2) Look for numbered section pattern "## 1." or "1. Honest assessment" etc.
    if re.search(r"##\s*1[\.\)]", text) or re.search(r"\b1\.\s*(honest|assessment)", lower):
        return True

    # 3) Look for explicit section headings written like "## 1. Honest assessment" (flexible)
    headings_pattern = re.compile(r"##+\s*\d+\.\s*[A-Za-z ]{5,40}")
    if headings_pattern.search(text):
        return True

    # 4) If planner returned JSON, check for plausible keys
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
            # Accept if it contains at least a couple of the expected keys anywhere
            keys = set(k.lower() for k in obj.keys())
            expected = {"honest_assessment", "feasibility", "roadmap", "role_recommendations", "resume", "interview", "risks"}
            if len(keys.intersection(expected)) >= 2:
                return True
        except Exception:
            pass

    # 5) Fallback: look for sentences that indicate an actionable next step (one-line next step)
    if re.search(r"\bones?[- ]line\b.*(next|step|do|start)", lower) or re.search(r"one[- ]line next step", lower):
        # if only this is present, still require at least 3 heading words earlier to accept
        if found >= 3:
            return True

    # Nothing convincing found — fail validation (so the system will re-prompt or try again)
    return False


async def _generate_plan_and_save(session: Dict[str, Any], api_key: str) -> Tuple[bool, Optional[str]]:
    """
    Use the collected state (validated answers & extracted tokens) to instruct the planner LLM to create the final plan.
    Stores plan text at session['state']['plan_text'] and sets plan_created flag if validated.
    Returns (success, plan_text_or_none)
    """
    if not session:
        return False, None
    state = session.setdefault("state", {})
    all_answers = state.get("answers", {})
    extracted = state.get("extracted", {})

    # Build planner prompt using the user's career_prompt template (we approximate here)
    planner_system = (
        "You are an expert, evidence-based, and brutally honest AI career advisor. "
        "You have been given the user's collected information below. "
        "Produce a prioritized, actionable plan following this structure exactly:\n"
        "1. Honest assessment\n2. Feasibility & fit\n3. Concrete, prioritized roadmap (30d,90d,6m,12m) with KPIs and resources\n4. Role & job-targeting recommendations (3-5 roles)\n5. Resume/LinkedIn guidance (5 bullets, 3 headline variants)\n6. Interview & hiring strategy\n7. Risks, trade-offs & fallback options\n8. Checkpoint schedule & metrics\n9. One-line next step\n\nCRITICAL: You are in the PLANNING PHASE: Do NOT ask any clarifying questions."
    )

    # Compose user content: include both extracted structured data and raw answers
    content_lines = ["=== USER EXTRACTED ==="]
    content_lines.append(json.dumps(extracted, ensure_ascii=False, indent=2))
    content_lines.append("\n=== USER REPORTED ANSWERS ===")
    for k, v in all_answers.items():
        content_lines.append(f"- {k}: {v}")
    content_lines.append("\n=== END ===")
    content = "\n".join(content_lines)

    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": PLANNER_MODEL,
        "messages": [
            {"role": "system", "content": planner_system},
            {"role": "user", "content": content}
        ],
        "temperature": 0.0,
        "max_tokens": 1600
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        plan_text = resp.json()["choices"][0]["message"]["content"]
        # Validate structure lightly
        valid = _validate_plan_output(plan_text)
        state["plan_text"] = plan_text
        if valid:
            state["plan_created"] = True
        else:
            # Keep plan_text but don't mark created
            state["plan_created"] = False
            logger.info("Plan generated but failed validation checks.")
        save_session(session)
        return valid, plan_text
    except Exception as e:
        logger.warning(f"Plan generation failed: {e}")
        return False, None

# ------------------- Controller: full pipeline in _get_next_action -------------------

async def _get_next_action(session: Dict[str, Any],
                           incoming_messages: List[Dict[str, Any]],
                           api_key: str) -> Dict[str, Any]:
    """
    Full pipeline coordinator:
      1) Extraction pass (LLM extractor)
      2) Deterministic validation & mapping (merge into session)
      3) Confirmation step (if extractor ambiguous)
      4) Clarifying questions (only for missing fields)
      5) If nothing missing -> CREATE_PLAN (and auto-generate plan and store)
    Returns dict with action and data:
      - {"action":"ASK_QUESTIONS", "questions":[...], "question_rounds":n}
      - {"action":"CREATE_PLAN", "question_rounds":n, "plan_valid":bool, "plan_text":...}
    """
    if not session:
        return {"action": "ERROR", "reason": "no session"}

    state = session.setdefault("state", {})
    answers = state.get("answers", {}) or {}
    rounds = state.get("question_rounds", 0)

    # If plan already exists in session, do nothing (allow follow-up)
    if state.get("plan_created"):
        return {"action": "PLAN_EXISTS", "plan_text": state.get("plan_text")}

    # Limit rounds
    if rounds >= MAX_CLARIFY_ROUNDS:
        # Force plan creation with assumptions mode
        # Try to generate plan now (best-effort) and return result
        ok, plan_text = await _generate_plan_and_save(session, api_key)
        return {"action": "CREATE_PLAN", "assumptions_mode": True, "plan_valid": ok, "plan_text": plan_text, "question_rounds": rounds}

    # find last user message content for context
    last_user_msg = None
    if incoming_messages:
        for m in reversed(incoming_messages):
            if m.get("role") == "user":
                last_user_msg = m.get("content")
                break

    # Step 1: Call extractor LLM (try/catch; fallback to no extractor)
    answers_text = _extract_answer_text(answers)
    extractor_output = await _call_extractor_llm(last_user_msg, answers_text, api_key)
    if extractor_output:
        try:
            _merge_extracted_into_session(session, extractor_output)
        except Exception as e:
            logger.warning(f"Failed to merge extractor output into session: {e}")

    # Step 2: deterministic validation & mapping
    missing, confirmations = _validate_extracted_and_compute_missing(session)

    # Step 3: If confirmations exist, prefer to ask those as single-shot confirmation questions
    if confirmations:
        # generate confirmation questions (max 3)
        confirmation_qs = _generate_confirmation_questions(confirmations, answers, max_q=3)
        # Filter against previously asked and session answers using _filter_and_validate_questions
        prev_qs = state.get("last_questions", [])
        prev_q_ids = [q.get("id") for q in prev_qs if isinstance(q, dict) and q.get("id")]
        filtered = _filter_and_validate_questions(confirmation_qs, missing, prev_q_ids, answers)
        if filtered:
            state["last_questions"] = filtered
            state["question_rounds"] = rounds + 1
            save_session(session)
            return {"action": "ASK_QUESTIONS", "questions": filtered, "question_rounds": rounds + 1}
        # else fall through to clarifier below

    # Step 4: If missing subfields remain, ask clarifier LLM for targeted JSON questions
    if missing:
        # Build condensed answers context
        condensed_answers = []
        for k, v in list(answers.items())[-10:]:
            condensed_answers.append(f"{k}: {v[:240]}")
        condensed_block = "\n".join(condensed_answers) or "<none>"

        prev_qs = state.get("last_questions", [])
        prev_q_ids = [q.get("id") for q in prev_qs if isinstance(q, dict) and q.get("id")]

        clarifier_parsed = await _call_clarifier_llm(missing, condensed_block, prev_q_ids, last_user_msg, api_key)
        # Validate clarifier output and filter
        raw_questions = []
        if clarifier_parsed and isinstance(clarifier_parsed, dict):
            raw_questions = clarifier_parsed.get("questions", []) if isinstance(clarifier_parsed.get("questions", []), list) else []
        filtered = _filter_and_validate_questions(raw_questions, missing, prev_q_ids, answers)
        if not filtered:
            # fallback deterministic
            fallback = _generate_deterministic_questions(missing, answers, max_q=3)
            state["last_questions"] = fallback
            state["question_rounds"] = rounds + 1
            save_session(session)
            return {"action": "ASK_QUESTIONS", "questions": fallback, "question_rounds": rounds + 1, "fallback": True}
        else:
            state["last_questions"] = filtered
            state["question_rounds"] = rounds + 1
            save_session(session)
            return {"action": "ASK_QUESTIONS", "questions": filtered, "question_rounds": rounds + 1}

    # Step 5: nothing missing, generate plan (best-effort)
    ok, plan_text = await _generate_plan_and_save(session, api_key)
    return {"action": "CREATE_PLAN", "plan_valid": ok, "plan_text": plan_text, "question_rounds": rounds}

# ------------------- CareerAdviceHandler wiring -------------------

class CareerAdviceHandler:
    def requires_session(self) -> bool:
        return True

    async def needs_clarification(self, session: Optional[Dict[str, Any]],
                                  incoming_messages: List[Dict[str, Any]],
                                  api_key: str) -> Optional[List[Dict[str, Any]]]:
        """
        Top-level entry. Orchestrates the full pipeline:
        - asks clarifying questions (if any)
        - if ready, creates the plan and stores it in session
        Returns:
          - list of question dicts if user needs to answer more,
          - None if plan created or follow-up mode.
        """
        if not session:
            return None
        state = session.setdefault("state", {})

        if state.get("plan_created"):
            # already have a plan; allow follow-up elsewhere
            return None

        decision = await _get_next_action(session, incoming_messages, api_key)
        action = decision.get("action")
        if action == "ASK_QUESTIONS":
            qs = decision.get("questions", [])
            # update session (already done inside _get_next_action) but ensure persisted
            state["last_questions"] = qs
            state["question_rounds"] = decision.get("question_rounds", state.get("question_rounds", 0) + 1)
            save_session(session)
            return qs

        if action in ("CREATE_PLAN", "PLAN_EXISTS"):
            # plan has been generated by _get_next_action (CREATE_PLAN) or already stored
            # The system that calls this handler should read session['state']['plan_text'] and send it back to user.
            # Here, return None to indicate no questions to ask.
            save_session(session)
            return None

        # any other actions -> default fallback question
        return [{"id": "fallback", "question": "Could you provide more details about your career goals and background?", "type": "text", "required": True}]

    def prepare_system_messages(self, session: Optional[Dict[str, Any]],
                                incoming_messages: List[Dict[str, Any]],
                                answers: Optional[Dict[str, str]]) -> List[str]:
        """
        Used by planner in case you need to call planner separately.
        """
        if not session:
            return []
        state = session.get("state", {})
        all_answers = state.get("answers", {})
        if not all_answers:
            return []
        assumptions_mode = state.get("assumptions_mode", False)
        missing = {}  # not used here
        lines = [
            "PLANNING PHASE: Produce the full structured plan NOW.",
            "Do NOT ask further questions. No preamble. Start directly with '## 1. Honest assessment'.",
        ]
        if assumptions_mode:
            lines.append("Some required details were missing; explicitly state assumptions where necessary.")
        lines.append("\n=== COLLECTED USER ANSWERS ===")
        for k, v in all_answers.items():
            lines.append(f"- {k}: {v}")
        lines.append("=== END ANSWERS ===")
        lines.append("If any critical piece is still unknown, state an assumption inline rather than asking.")
        return ["\n".join(lines)]

    def handle_llm_response(self, session: Optional[Dict[str, Any]],
                            structured: Optional[Dict[str, Any]],
                            raw_text: str) -> Dict[str, Any]:
        """
        Called after planner LLM returns text (if planner is called outside pipeline).
        We'll validate and persist plan state.
        """
        if not session:
            return {"persist": False}
        if _validate_plan_output(raw_text):
            session.setdefault("state", {})["plan_created"] = True
        else:
            session.setdefault("state", {})["plan_created"] = False
        session.setdefault("state", {})["plan_text"] = raw_text
        save_session(session)
        return {"persist": True, "session": session}

# ------------------- Minimal placeholders for other handlers (unchanged behavior) -------------------

class ResumeReviewHandler:
    def requires_session(self) -> bool:
        return False

    async def needs_clarification(self, session: Optional[Dict[str, Any]], incoming_messages: List[Dict[str, Any]], api_key: str) -> Optional[List[Dict[str, Any]]]:
        return None

    def prepare_system_messages(self, session: Optional[Dict[str, Any]], incoming_messages: List[Dict[str, Any]], answers: Optional[Dict[str, str]]) -> List[str]:
        return []

    def handle_llm_response(self, session: Optional[Dict[str, Any]], structured: Optional[Dict[str, Any]], raw_text: str) -> Dict[str, Any]:
        return {"persist": False, "session": None, "reply": None}

class JobHuntHandler:
    def requires_session(self) -> bool:
        return False

    async def needs_clarification(self, session: Optional[Dict[str, Any]], incoming_messages: List[Dict[str, Any]], api_key: str) -> Optional[List[Dict[str, Any]]]:
        return None

    def prepare_system_messages(self, session: Optional[Dict[str, Any]], incoming_messages: List[Dict[str, Any]], answers: Optional[Dict[str, str]]) -> List[str]:
        return []

    def handle_llm_response(self, session: Optional[Dict[str, Any]], structured: Optional[Dict[str, Any]], raw_text: str) -> Dict[str, Any]:
        return {"persist": False, "session": None, "reply": None}

class LearningRoadmapHandler:
    def requires_session(self) -> bool:
        return False

    async def needs_clarification(self, session: Optional[Dict[str, Any]], incoming_messages: List[Dict[str, Any]], api_key: str) -> Optional[List[Dict[str, Any]]]:
        return None

    def prepare_system_messages(self, session: Optional[Dict[str, Any]], incoming_messages: List[Dict[str, Any]], answers: Optional[Dict[str, str]]) -> List[str]:
        return []

    def handle_llm_response(self, session: Optional[Dict[str, Any]], structured: Optional[Dict[str, Any]], raw_text: str) -> Dict[str, Any]:
        return {"persist": False, "session": None, "reply": None}

class MockInterviewHandler:
    def requires_session(self) -> bool:
        return False

    async def needs_clarification(self, session: Optional[Dict[str, Any]], incoming_messages: List[Dict[str, Any]], api_key: str) -> Optional[List[Dict[str, Any]]]:
        return None

    def prepare_system_messages(self, session: Optional[Dict[str, Any]], incoming_messages: List[Dict[str, Any]], answers: Optional[Dict[str, str]]) -> List[str]:
        return []

    def handle_llm_response(self, session: Optional[Dict[str, Any]], structured: Optional[Dict[str, Any]], raw_text: str) -> Dict[str, Any]:
        return {"persist": False, "session": None, "reply": None}

MODE_HANDLERS = {
    "career_advice": CareerAdviceHandler(),
    "resume_review": ResumeReviewHandler(),
    "job_hunt": JobHuntHandler(),
    "learning_roadmap": LearningRoadmapHandler(),
    "mock_interview": MockInterviewHandler(),
}
