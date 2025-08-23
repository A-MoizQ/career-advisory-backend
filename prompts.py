import os
import base64
from typing import Dict

# Optional: load .env locally (dev only). Keep python-dotenv in dev requirements only.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Simple fallback markdown guide (shorter, less strict) ---
DEFAULT_MARKDOWN_GUIDE_SIMPLE = """
Format the assistant's reply as GitHub-Flavored Markdown (GFM).
- Use headings (## or ###) to separate sections.
- Use bullet lists or numbered lists for actionable items.
- Use pipe-style tables for tabular data; include a header row and a separator row (---).
- Keep table columns consistent; prefer short cell text.
- Do not output HTML or decorative box characters.
- If possible, return a single JSON block with keys `title`, `sections`, and `tables` (only if JSON fully represents the content).
""".strip()

def _load_text_from_env_or_file(env_name: str,
                                file_env_name: str = None,
                                b64_env_name: str = None,
                                default: str = "") -> str:
    # 1) Raw env var (works if you ensured exact newlines were preserved)
    v = os.environ.get(env_name)
    if v:
        return v

    # 2) Base64-encoded env var (recommended)
    if b64_env_name:
        b64 = os.environ.get(b64_env_name)
        if b64:
            try:
                return base64.b64decode(b64.encode()).decode('utf-8')
            except Exception:
                pass

    # 3) File path env var (recommended for Docker secrets / mounts)
    if file_env_name:
        path = os.environ.get(file_env_name)
        if path and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception:
                pass

    # fallback
    return default

# Load main markdown prompt from env, fallback to the simple default
# For production you will set MD_PROMPT as a secret in Render/Heroku
MD_PROMPT = _load_text_from_env_or_file(
    env_name="MD_PROMPT",
    file_env_name="MD_PROMPT_FILE",
    b64_env_name="MD_PROMPT_B64",
    default=DEFAULT_MARKDOWN_GUIDE_SIMPLE
)

# --- MODE prompts: prefer environment overrides per-mode ---
DEFAULT_MODE_PROMPTS: Dict[str, str] = {
    "career_advice": "You're a helpful AI career advisor.",
    "resume_review": "You are an expert resume reviewer. Analyze the provided resume text, identify its structure and provide actionable feedback.",
    "job_hunt": "You suggest job hunting strategies and tips.",
    "learning_roadmap": "You recommend learning paths based on goals.",
    "mock_interview": "You act as a mock interviewer and give feedback."
}


MODE_PROMPTS: Dict[str, str] = {
    "career_advice": _load_text_from_env_or_file(env_name="CAREER_PROMPT",file_env_name="CAREER_PROMPT_FILE",b64_env_name="CAREER_PROMPT_B64", default=DEFAULT_MODE_PROMPTS["career_advice"]),
    "resume_review": _load_text_from_env_or_file(env_name="RESUME_REVIEW_PROMPT",file_env_name="RESUME_REVIEW_PROMPT_FILE",b64_env_name="RESUME_REVIEW_PROMPT_B64", default=DEFAULT_MODE_PROMPTS["resume_review"]),
    "job_hunt": _load_text_from_env_or_file(env_name="JOB_HUNT_PROMPT",file_env_name="JOB_HUNT_PROMPT_FILE",b64_env_name="JOB_HUNT_PROMPT_B64", default=DEFAULT_MODE_PROMPTS["job_hunt"]),
    "learning_roadmap": _load_text_from_env_or_file(env_name="LEARNING_ROADMAP_PROMPT",file_env_name="LEARNING_ROADMAP_PROMPT_FILE",b64_env_name="LEARNING_ROADMAP_PROMPT_B64", default=DEFAULT_MODE_PROMPTS["learning_roadmap"]),
    "mock_interview": _load_text_from_env_or_file(env_name="MOCK_INTERVIEW_PROMPT",file_env_name="MOCK_INTERVIEW_PROMPT_FILE",b64_env_name="MOCK_INTERVIEW_PROMPT_B64", default=DEFAULT_MODE_PROMPTS["mock_interview"]),
}

# Export safe helper to build final system prompt
def build_system_prompt(mode: str) -> str:
    """
    Compose the system prompt for the model.
    Avoid printing/logging MD_PROMPT or MODE_PROMPTS (secrets).
    """
    mode_prompt = MODE_PROMPTS.get(mode, DEFAULT_MODE_PROMPTS["career_advice"])
    return mode_prompt + "\n\n" + MD_PROMPT
