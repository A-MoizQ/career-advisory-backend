# mode_handlers.py
import uuid
import os
import httpx
import logging
from typing import Dict, Any, List, Optional

from response_parser import extract_json_from_text  # your existing parser

logger = logging.getLogger("uvicorn.error")

# Clarifier-generation settings
CLARIFIER_MODEL = os.environ.get("CLARIFIER_MODEL", "gpt-3.5-turbo")
CLARIFIER_TEMPERATURE = 0.0
CLARIFIER_MAX_ATTEMPTS = 3  # try several times if model misformats output


# Utility: enforce minimal shape for clarifiers returned by LLM
def _validate_clarifiers_obj(obj: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    Expected shape:
    {
      "clarifying_questions": [
        {"id":"q1", "question":"...", "type":"text", "required": true, "hint": "..."},
        ...
      ]
    }
    Returns list of {id, question, type, required, hint} or None
    """
    if not isinstance(obj, dict):
        return None
    if "clarifying_questions" not in obj:
        # allow some models to return "questions" or "items" as fallback
        if "questions" in obj:
            obj["clarifying_questions"] = obj["questions"]
        else:
            return None

    qs = obj.get("clarifying_questions")
    if not isinstance(qs, list) or len(qs) == 0:
        return None

    sanitized = []
    for i, q in enumerate(qs, start=1):
        if not isinstance(q, dict):
            # try to recover if element is just a string
            if isinstance(q, str):
                qtext = q.strip()
                if not qtext:
                    return None
                sanitized.append({"id": f"q{i}", "question": qtext, "type": "text", "required": True, "hint": ""})
                continue
            return None
        qid = q.get("id") or q.get("name") or f"q{i}"
        qtext = q.get("question") or q.get("prompt") or q.get("text")
        if not qtext or not isinstance(qtext, str):
            return None
        qtype = q.get("type") or "text"
        required = bool(q.get("required", True))
        hint = q.get("hint", "") or ""
        sanitized.append({"id": str(qid), "question": qtext.strip(), "type": qtype, "required": required, "hint": hint})
    # Limit number of clarifiers to reasonable count (1..7)
    if len(sanitized) > 7:
        sanitized = sanitized[:7]
    return sanitized


async def _validate_answers_with_llm(questions: List[Dict[str, Any]], answers: Dict[str, str], api_key: str) -> Dict[str, Any]:
    """
    Use LLM to validate that user answers are relevant and complete.
    Returns {"valid": bool, "issues": List[str], "follow_ups": List[Dict]}
    """
    # Build context for validation
    qa_pairs = []
    for q in questions:
        qid = q["id"]
        answer = answers.get(qid, "").strip()
        if answer:
            qa_pairs.append(f"Q: {q['question']}\nA: {answer}")
        else:
            qa_pairs.append(f"Q: {q['question']}\nA: [NO ANSWER]")
    
    qa_text = "\n\n".join(qa_pairs)
    
    system_prompt = (
        "You are an expert career advisor reviewing user responses to clarifying questions. "
        "Your job is to validate that answers are relevant, complete, and provide sufficient information for career planning.\n\n"
        
        "CRITICAL RULES:\n"
        "- Output ONLY a single JSON object with no surrounding text\n"
        "- Be strict but fair in validation\n"
        "- Flag vague, irrelevant, or insufficient answers\n\n"
        
        "Required JSON format:\n"
        '{"valid": true/false, "issues": ["reason1", "reason2"], "follow_ups": [{"id": "q_id", "question": "follow up question"}]}\n\n'
        
        "Validation criteria:\n"
        "- Background: Should mention education, experience, or skills\n"
        "- Goals: Should be specific (not just 'AI' but 'ML Engineer at tech company')\n"
        "- Timeline: Should include realistic timeframe\n"
        "- Commitment: Should specify hours/week available\n"
        "- Constraints: Should acknowledge any limitations\n\n"
        
        f"User Q&A to validate:\n{qa_text}\n\n"
        "Output validation JSON:"
    )
    
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": CLARIFIER_MODEL,
            "messages": [{"role": "system", "content": system_prompt}],
            "temperature": 0.0,
            "max_tokens": 500,
        }
        
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
        
        # Parse validation result
        parsed = extract_json_from_text(text)
        if parsed and "valid" in parsed:
            return {
                "valid": bool(parsed.get("valid", False)),
                "issues": parsed.get("issues", []),
                "follow_ups": parsed.get("follow_ups", [])
            }
    except Exception as e:
        logger.warning(f"Answer validation failed: {e}")
    
    # Fallback: basic validation
    issues = []
    for q in questions:
        answer = answers.get(q["id"], "").strip()
        if q["required"] and len(answer) < 10:
            issues.append(f"'{q['question']}' needs a more detailed answer")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "follow_ups": []
    }


class CareerAdviceHandler:
    def requires_session(self) -> bool:
        return True

    async def needs_clarification(self, session: Optional[Dict[str, Any]], incoming_messages: List[Dict[str, Any]], api_key: str) -> Optional[List[Dict[str, Any]]]:
        """
        CRITICAL: Always ask clarifying questions first unless we have sufficient answers.
        This enforces the workflow: clarify first, then plan.
        """
        # Count how many meaningful answers we have
        answered_count = 0
        if session:
            answers = session.get("state", {}).get("answers", {})
            answered_count = len([v for v in answers.values() if v and str(v).strip() and len(str(v).strip()) > 5])

        # ENFORCE: Must have at least 4-5 quality answers before proceeding to planning
        MIN_REQUIRED_ANSWERS = 4
        if answered_count < MIN_REQUIRED_ANSWERS:
            logger.info(f"Career advice: need clarification (only {answered_count}/{MIN_REQUIRED_ANSWERS} answers)")
            
            # Extract context from user messages
            context_text = ""
            if incoming_messages:
                last_user_msgs = [m for m in incoming_messages if m.get("role") == "user"]
                if last_user_msgs:
                    snippets = []
                    for m in last_user_msgs[-3:]:
                        content = m.get("content", "")
                        # Skip file upload messages and answer summaries
                        if not content.startswith("The user has uploaded") and not content.startswith("Answered clarifying"):
                            snippets.append(content[:800])
                    context_text = "\n\n".join(snippets)

            # If we have some answers, check if they need follow-up
            if session and session.get("state", {}).get("answers"):
                existing_answers = session["state"]["answers"]
                questions = session.get("state", {}).get("last_questions", [])
                
                if questions:
                    # Validate existing answers
                    validation = await _validate_answers_with_llm(questions, existing_answers, api_key)
                    
                    if not validation["valid"] and validation["follow_ups"]:
                        logger.info(f"Career advice: need follow-up questions due to validation issues")
                        return validation["follow_ups"]

            # Generate clarifying questions via LLM
            system_prompt = (
                "You are a strict clarifying-question generator for career advice. "
                "Your ONLY job is to generate 3-5 high-impact clarifying questions. "
                
                "CRITICAL RULES:\n"
                "- Output ONLY a single JSON object with no surrounding text\n"
                "- Do NOT provide any career advice, plans, or analysis\n"
                "- Do NOT include reasoning or explanations\n"
                "- Focus on questions that determine: career goals, background, constraints, timeline\n"
                
                "Required JSON format:\n"
                '{"clarifying_questions":[{"id":"bg","question":"What is your educational/work background?","type":"text","required":true,"hint":"Degree, experience, skills"}]}\n'
                
                "High-priority question topics:\n"
                "1. Educational/professional background\n"
                "2. Specific career goal/role target\n"
                "3. Timeline and time availability\n"
                "4. Current skills and experience level\n"
                "5. Geographic/financial constraints\n"
                
                f"User context: {context_text}\n\n"
                "Output the JSON object only:"
            )

            messages = [{"role": "system", "content": system_prompt}]
            if context_text:
                messages.append({"role": "user", "content": f"Generate clarifying questions for: {context_text}"})

            # Try to get valid clarifying questions from LLM
            for attempt in range(CLARIFIER_MAX_ATTEMPTS):
                try:
                    headers = {"Authorization": f"Bearer {api_key}"}
                    payload = {
                        "model": CLARIFIER_MODEL,
                        "messages": messages,
                        "temperature": CLARIFIER_TEMPERATURE,
                        "max_tokens": 400,
                    }

                    async with httpx.AsyncClient(timeout=20.0) as client:
                        resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                        resp.raise_for_status()
                        data = resp.json()
                        text = data["choices"][0]["message"]["content"]

                    # Parse the JSON response
                    parsed = extract_json_from_text(text)
                    if parsed:
                        validated = _validate_clarifiers_obj(parsed)
                        if validated:
                            logger.info(f"Generated {len(validated)} clarifying questions")
                            # Store questions in session for validation later
                            if session:
                                session.setdefault("state", {})["last_questions"] = validated
                            return validated

                except Exception as e:
                    logger.warning(f"Clarifier attempt {attempt + 1} failed: {e}")
                    continue

            # Fallback: Use deterministic questions if LLM fails
            logger.warning("Using fallback clarifying questions")
            fallback_questions = [
                {"id": "background", "question": "What is your current educational and professional background?", "type": "text", "required": True, "hint": "Degree, work experience, relevant skills"},
                {"id": "goal", "question": "What specific career role or field are you targeting?", "type": "text", "required": True, "hint": "e.g., AI Engineer, Data Scientist, specific company"},
                {"id": "timeline", "question": "What is your target timeline to transition into this career?", "type": "text", "required": True, "hint": "6 months, 1 year, 2 years"},
                {"id": "commitment", "question": "How many hours per week can you dedicate to learning and career development?", "type": "text", "required": True, "hint": "Be realistic about time availability"},
                {"id": "constraints", "question": "Do you have any constraints (financial, geographic, visa status, family)?", "type": "text", "required": True, "hint": "Anything that might affect your career transition"}
            ]
            if session:
                session.setdefault("state", {})["last_questions"] = fallback_questions
            return fallback_questions

        # If we have enough answers, validate they're sufficient for planning
        if session and session.get("state", {}).get("answers"):
            answers = session["state"]["answers"]
            questions = session.get("state", {}).get("last_questions", [])
            
            if questions:
                validation = await _validate_answers_with_llm(questions, answers, api_key)
                if not validation["valid"]:
                    logger.info(f"Career advice: answers insufficient, validation issues: {validation['issues']}")
                    if validation["follow_ups"]:
                        return validation["follow_ups"]
                    # If no specific follow-ups, ask original questions again
                    return questions

        # If we have enough answers, proceed to planning (no clarification needed)
        logger.info(f"Career advice: sufficient answers ({answered_count}), proceeding to planning")
        return None

    def prepare_system_messages(self, session: Optional[Dict[str, Any]], incoming_messages: List[Dict[str, Any]], answers: Optional[Dict[str, str]]) -> List[str]:
        """
        Prepare the context for final career planning with collected answers.
        """
        msgs: List[str] = []
        
        # Consolidate all answers
        all_answers = {}
        if session:
            all_answers.update(session.get("state", {}).get("answers", {}))
        if answers:
            all_answers.update(answers)

        if all_answers:
            # Provide the collected information to the LLM
            summary_lines = [
                "COLLECTED USER INFORMATION (use this to create the career plan):",
                ""
            ]
            for key, value in all_answers.items():
                summary_lines.append(f"• {key}: {value}")
            
            summary_lines.extend([
                "",
                "INSTRUCTIONS:",
                "- Use the above information to create a comprehensive career development plan",
                "- Follow the career advice prompt structure exactly", 
                "- Do NOT ask any more questions - provide the complete plan now",
                "- Be specific and actionable based on the user's background and goals",
                "- Address all aspects: skills gap, learning plan, job search strategy, timeline"
            ])
            
            msgs.append("\n".join(summary_lines))

        return msgs

    def handle_llm_response(self, session: Optional[Dict[str, Any]], structured: Optional[Dict[str, Any]], raw_text: str) -> Dict[str, Any]:
        """
        Handle the final response - clear session since planning is complete.
        """
        result: Dict[str, Any] = {}
        result["persist"] = False  # Clear session after providing final plan
        result["session"] = None
        result["reply"] = None  # Let the main handler process the response
        return result


# Simple handlers for other modes (no clarification needed)
class ResumeReviewHandler:
    def requires_session(self) -> bool:
        return False

    async def needs_clarification(self, session: Optional[Dict[str, Any]], incoming_messages: List[Dict[str, Any]], api_key: str) -> Optional[List[Dict[str, Any]]]:
        return None  # No clarification needed for resume review

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


# Map modes -> handler instances
MODE_HANDLERS = {
    "career_advice": CareerAdviceHandler(),
    "resume_review": ResumeReviewHandler(),
    "job_hunt": JobHuntHandler(), 
    "learning_roadmap": LearningRoadmapHandler(),
    "mock_interview": MockInterviewHandler(),
}