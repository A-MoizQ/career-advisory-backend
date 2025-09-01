# response_parser.py
import json
import re
import textwrap
from typing import Any, Dict, List, Optional

# keep your original box-drawing regex but used in sanitize
BOX_CHARS_RE = re.compile(r'[\u2500-\u257F\u2580-\u259F\u2500\u2502\u2510\u2514\u2518\u251C\u2524]+')
WRAP_WIDTH = 88  # wrap paragraphs for nicer display

# ---------------- Basic sanitation ----------------

def sanitize_raw_text(text: str) -> str:
    """
    Normalize line endings, remove box drawing chars and excessive blank lines,
    convert long dashes/equals separators to a simple '---' visual separator.
    """
    if not isinstance(text, str):
        return ""
    # Remove box-drawing characters
    text = BOX_CHARS_RE.sub("", text)
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace long runs of hyphens/equals with a single '---' line
    text = re.sub(r'\n[ \t]*[-=]{3,}[ \t]*\n', '\n\n---\n\n', text)
    # Collapse many blank lines to two
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip leading/trailing whitespace
    return text.strip() + "\n"

# ---------------- JSON extraction ----------------

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Try multiple robust strategies to extract a JSON object from model output:
    1) fenced ```json { ... } ```
    2) first balanced { ... } block that parses
    3) entire text if valid JSON
    Returns dict or None.
    """
    if not isinstance(text, str):
        return None
    # strategy 1: fenced json block (```json ... ```)
    m = re.search(r'```json\s*(\{.*?\})\s*```', text, re.S | re.I)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # strategy 1b: any ```...``` block that looks like json inside
    m2 = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text, re.S | re.I)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            pass

    # strategy 2: find first balanced {...} sequence that parses
    start_idx = text.find('{')
    if start_idx != -1:
        # attempt to find balanced braces parse windows (avoid O(n^2) huge loops)
        stack = 0
        for i in range(start_idx, len(text)):
            c = text[i]
            if c == '{':
                stack += 1
            elif c == '}':
                stack -= 1
                if stack == 0:
                    candidate = text[start_idx:i+1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        # continue scanning for next balanced block
                        pass

    # strategy 3: try parse entire trimmed text
    try:
        return json.loads(text.strip())
    except Exception:
        return None

# ---------------- Structured JSON -> Markdown ----------------

def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # remove control chars except newline and tab
    s = re.sub(r'[^\x09\x0A\x20-\x7E\u0080-\uFFFF]', '', s)
    return s.strip()

def _wrap_paragraph(text: str) -> List[str]:
    txt = _clean_text(text)
    if not txt:
        return []
    # wrap keeping paragraphs
    wrapped = textwrap.fill(txt, width=WRAP_WIDTH)
    return [wrapped]

def _format_bullets(bullets: List[Any]) -> List[str]:
    out = []
    for b in bullets:
        if b is None:
            continue
        line = _clean_text(str(b))
        if not line:
            continue
        wrapped = textwrap.fill(line, width=WRAP_WIDTH)
        wrapped = wrapped.replace("\n", "\n  ")
        out.append(f"- {wrapped}")
    return out

def _format_table(tbl: Dict[str, Any]) -> Optional[List[str]]:
    """
    tbl expected shape: {"headers": [...], "rows": [[...],[...]]}
    Returns GFM table lines or None if invalid.
    """
    headers = tbl.get("headers") or tbl.get("cols") or []
    rows = tbl.get("rows") or []
    if not isinstance(headers, list) or len(headers) == 0:
        return None
    # normalize headers to strings
    headers = [str(h).strip() for h in headers]
    good_rows = []
    for r in rows:
        if not isinstance(r, list):
            continue
        # skip rows with mismatched length
        if len(r) != len(headers):
            # try to coerce: flatten nested lists to semicolon lists; otherwise skip
            flat = []
            for c in r:
                if isinstance(c, list):
                    flat.append("; ".join(str(x) for x in c))
                else:
                    flat.append(str(c))
            if len(flat) == len(headers):
                good_rows.append([c.replace("\n"," ") for c in flat])
            else:
                # pad or truncate to fit
                if len(flat) < len(headers):
                    flat = flat + [''] * (len(headers) - len(flat))
                    good_rows.append([c.replace("\n"," ") for c in flat])
                else:
                    flat = flat[:len(headers)]
                    good_rows.append([c.replace("\n"," ") for c in flat])
        else:
            good_rows.append([str(c).replace("\n"," ") for c in r])
    # if no rows, it's okay; produce header only table
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [header_line, sep_line]
    for r in good_rows:
        cells = [c if c else " " for c in r]
        lines.append("| " + " | ".join(cells) + " |")
    return lines

def structured_json_to_markdown(obj: Dict[str, Any]) -> str:
    """
    Convert the model's structured JSON into clean, numbered GFM Markdown.
    This function expects the schema you described: title, sections (with heading/content/bullets), tables.
    It is forgiving about missing keys.
    """
    if not obj or not isinstance(obj, dict):
        return ""

    parts: List[str] = []
    title = obj.get("title") or obj.get("heading") or obj.get("name")
    if title:
        parts.append(f"# {_clean_text(title)}\n")

    sections = obj.get("sections") or obj.get("body") or []
    if isinstance(sections, dict):
        # sometimes sections provided as dict; convert to list
        sections = [sections]

    for idx, sec in enumerate(sections, start=1):
        if isinstance(sec, dict):
            heading = sec.get("heading") or sec.get("title") or sec.get("name") or f"Section {idx}"
            parts.append(f"## {idx}. {_clean_text(heading)}\n")
            content = sec.get("content")
            if isinstance(content, str) and content.strip():
                parts.extend(_wrap_paragraph(content))
                parts.append("")
            bullets = sec.get("bullets") or sec.get("items") or sec.get("list") or []
            if isinstance(bullets, list) and bullets:
                parts.extend(_format_bullets(bullets))
                parts.append("")
            # sometimes there are nested subsections
            subsecs = sec.get("sections") or sec.get("subsections") or []
            if isinstance(subsecs, list) and subsecs:
                for sidx, ssec in enumerate(subsecs, start=1):
                    if isinstance(ssec, dict):
                        sheading = ssec.get("heading") or ssec.get("title") or f"{idx}.{sidx}"
                        parts.append(f"### {idx}.{sidx} {_clean_text(sheading)}\n")
                        scontent = ssec.get("content") or ""
                        if scontent:
                            parts.extend(_wrap_paragraph(scontent))
                            parts.append("")
                        sbullets = ssec.get("bullets") or []
                        if sbullets:
                            parts.extend(_format_bullets(sbullets))
                            parts.append("")
        elif isinstance(sec, str):
            # raw markdown-like section
            if sec.strip():
                parts.extend(_wrap_paragraph(sec))
                parts.append("")

    # Tables handling
    tables = obj.get("tables") or []
    for t in tables:
        t_md = _format_table(t)
        if t_md:
            parts.append("\n".join(t_md))
            parts.append("")

    # If object carries a fallback 'markdown' key, append it (cleaned)
    fallback_md = obj.get("markdown")
    if isinstance(fallback_md, str) and fallback_md.strip():
        parts.append(_clean_text(fallback_md))

    # Join with consistent blank lines
    md = "\n".join(p.rstrip() for p in parts if p is not None).strip() + "\n"
    md = normalize_markdown(md)
    return md

# ---------------- Conservative pipe-block cleaning from earlier (kept) ----------------

def _clean_pipe_block(block_lines: List[str]) -> Optional[str]:
    """
    Try to salvage pipe-style table blocks; returns normalized GFM or None.
    """
    rows = []
    for r in block_lines:
        if re.fullmatch(r'\s*[-=]{2,}\s*', r):
            rows.append(['__SEP__'])
            continue
        parts = [c.strip().lstrip('- ').lstrip('• ').strip() for c in re.split(r'\|', r)]
        rows.append(parts)

    # remove separator-only rows
    rows = [r for r in rows if r != ['__SEP__']]
    if not rows or len(rows) < 1:
        return None

    max_cols = max(len(r) for r in rows)
    # pad rows
    for r in rows:
        if len(r) < max_cols:
            r += [''] * (max_cols - len(r))

    # compute columns that have any non-empty cell
    cols = list(zip(*rows))
    keep = [any(cell.strip() for cell in col) for col in cols]
    if sum(keep) < 2:
        return None

    new_rows = [[cell for cell, k in zip(row, keep) if k] for row in rows]
    header = new_rows[0]
    if not any(h.strip() for h in header):
        return None

    # Build normalized GFM table
    out = []
    out.append('| ' + ' | '.join([h if h else ' ' for h in header]) + ' |')
    out.append('| ' + ' | '.join(['---'] * len(header)) + ' |')
    for row in new_rows[1:]:
        out.append('| ' + ' | '.join([c if c else ' ' for c in row]) + ' |')
    return '\n'.join(out)

# ---------------- Normalize and repair markdown ----------------

def normalize_markdown(text: str) -> str:
    """
    Conservative normalization and repair for Markdown produced by LLMs:
    - sanitize raw text
    - remove stray fences
    - ensure blank line before headings
    - convert numeric item lines into '- ' bullet if needed
    - repair/insert table separators for pipe tables
    - wrap paragraphs
    """
    if not text:
        return ""

    text = sanitize_raw_text(text)

    # Remove leading/trailing code fences if present (we expect either clean JSON or markdown)
    text = re.sub(r'^\s*```(?:json)?\s*', '', text, flags=re.I)
    text = re.sub(r'\s*```\s*$', '', text)

    lines = text.splitlines()
    out: List[str] = []
    i = 0
    prev_blank = True  # force no extra blank at top
    while i < len(lines):
        ln = lines[i].rstrip()
        stripped = ln.strip()

        # Preserve explicit heading lines but ensure blank line before them
        if re.match(r'^\s*#{1,6}\s+', stripped) or re.match(r'^\s*\d+\.\s+\w', stripped):
            # convert "1. Title" -> "## 1. Title"
            if re.match(r'^\s*\d+\.\s+\w', stripped) and not stripped.startswith('#'):
                stripped = "## " + stripped
            if out and not prev_blank:
                out.append("")  # blank line before heading
            out.append(stripped)
            prev_blank = False
            i += 1
            continue

        # pipe table block handling
        if '|' in ln:
            j = i
            block = []
            while j < len(lines) and '|' in lines[j]:
                block.append(lines[j])
                j += 1
            cleaned = _clean_pipe_block(block)
            if cleaned:
                if out and not prev_blank:
                    out.append("")
                out.extend(cleaned.splitlines())
                out.append("")
                prev_blank = True
                i = j
                continue
            else:
                # keep block raw but ensure blank around
                if out and not prev_blank:
                    out.append("")
                out.extend(block)
                out.append("")
                prev_blank = True
                i = j
                continue

        # tab/space-separated table candidate handling (like 'ColA   ColB')
        if '\t' in ln or re.search(r'\S\s{2,}\S', ln):
            # collect block of such lines
            j = i
            block = []
            while j < len(lines) and lines[j].strip() and (('\t' in lines[j]) or re.search(r'\S\s{2,}\S', lines[j])):
                block.append(lines[j])
                j += 1
            if len(block) >= 2:
                # convert to pipe table if reasonable
                rows = []
                for r in block:
                    if '\t' in r:
                        parts = [c.strip() for c in re.split(r'\t+', r)]
                    else:
                        parts = [c.strip() for c in re.split(r'\s{2,}', r)]
                    rows.append(parts)
                max_cols = max(len(r) for r in rows)
                for r in rows:
                    if len(r) < max_cols:
                        r += [''] * (max_cols - len(r))
                col_nonempty = [any(row[c].strip() for row in rows) for c in range(max_cols)]
                if sum(col_nonempty) >= 2:
                    rows2 = [[cell for k,cell in enumerate(row) if col_nonempty[k]] for row in rows]
                    header = rows2[0]
                    table_lines = []
                    table_lines.append("| " + " | ".join([h if h else ' ' for h in header]) + " |")
                    table_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
                    for row in rows2[1:]:
                        table_lines.append("| " + " | ".join([c if c else ' ' for c in row]) + " |")
                    if out and not prev_blank:
                        out.append("")
                    out.extend(table_lines)
                    out.append("")
                    prev_blank = True
                    i = j
                    continue
            # fallback: copy lines
            out.extend(block)
            out.append("")
            prev_blank = True
            i = j
            continue

        # bullet/numbered lists preserve as is (normalize leading numbering)
        if re.match(r'^\s*[\-\*\+]\s+', stripped) or re.match(r'^\s*\d+\.\s+', stripped):
            # normalize numbers to '- ' bullets for consistent styling
            if re.match(r'^\s*\d+\.\s+', stripped):
                stripped = "- " + re.sub(r'^\s*\d+\.\s+', '', stripped)
            out.append(stripped)
            prev_blank = False
            i += 1
            continue

        # empty line
        if stripped == "":
            if not prev_blank:
                out.append("")
                prev_blank = True
            # else skip duplicate blank
            i += 1
            continue

        # default paragraph: wrap
        wrapped = textwrap.fill(stripped, width=WRAP_WIDTH)
        out.append(wrapped)
        prev_blank = False
        i += 1

    # final cleanup: collapse multiple blank lines
    final = "\n".join(out)
    final = re.sub(r'\n{3,}', '\n\n', final)
    # ensure heading is followed by blank line
    final = re.sub(r'(?m)^(#{1,6} .+)\n(?!\n)', r'\1\n\n', final)
    return final.strip() + "\n"

# ---------------- Top-level helper ----------------

def convert_structured_or_fix(raw_text: str, structured: Optional[Dict[str, Any]] = None) -> str:
    """
    If structured provided and valid, convert it; otherwise sanitize & normalize raw text.
    This is the function to call from main.py to ensure good GFM output.
    """
    if structured and isinstance(structured, dict):
        md = structured_json_to_markdown(structured)
        if md and len(md.strip()) > 5:
            return md
        # fall through if structured conversion failed
    # sanitize and normalize raw markdown returned by LLM
    sanitized = sanitize_raw_text(raw_text)
    normalized = normalize_markdown(sanitized)
    return normalized
