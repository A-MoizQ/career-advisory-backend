# session_store.py
import os
import json
import time
import logging

logger = logging.getLogger("uvicorn.error")

REDIS_URL = os.environ.get("REDIS_URL")  # if provided, use Redis
_use_redis = False
_redis = None

if REDIS_URL:
    try:
        import redis
        _redis = redis.from_url(REDIS_URL, decode_responses=True)
        # quick ping
        _redis.ping()
        _use_redis = True
        logger.info("Using Redis session store")
    except Exception as e:
        logger.warning(f"Could not connect to Redis at {REDIS_URL}: {e}. Falling back to in-memory store.")

# in-memory store (dev only)
_sessions = {}

def save_session(session: dict):
    sid = session["session_id"]
    session["last_updated"] = int(time.time())
    if _use_redis:
        try:
            _redis.set(f"session:{sid}", json.dumps(session), ex=session.get("expires_at", None) and max(1, session["expires_at"] - int(time.time())))
        except Exception as e:
            logger.exception("Redis save_session failed; falling back to in-memory")
            _sessions[sid] = session
    else:
        _sessions[sid] = session

def load_session(session_id: str):
    if not session_id:
        return None
    if _use_redis:
        try:
            raw = _redis.get(f"session:{session_id}")
            if not raw:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.exception("Redis load_session failed; falling back to in-memory")
            return _sessions.get(session_id)
    else:
        return _sessions.get(session_id)

def delete_session(session_id: str):
    if not session_id:
        return
    if _use_redis:
        try:
            _redis.delete(f"session:{session_id}")
        except Exception:
            pass
    else:
        _sessions.pop(session_id, None)
