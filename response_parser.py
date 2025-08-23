import json
import re
from typing import Any, Dict, List, Optional

BOX_CHARS_RE = re.compile(r'[\u2500-\u257F\u2580-\u259F\u2500\u2502\u2510\u2514\u2518\u251C\u2524]+')


def sanitize_raw_text(text: str) -> str:
    # Remove box-drawing characters
    text = BOX_CHARS_RE.sub('', text)
    # Replace long runs of hyphens/equals with a single '---' line (table separators)
    text = re.sub(r'\n[ \t]*[-=]{3,}[ \t]*\n', '\n\n---\n\n', text)
    # Normalize Windows line endings and multiple blank lines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip() + '\n'

# Try to extract JSON object embedded in text (```json ... ``` or plain JSON)
def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    # 1) look for fenced json block
    m = re.search(r'```json\s*(\{.*?\})\s*```', text, re.S)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # 2) try to find first {...} that parses
    start = text.find('{')
    if start != -1:
        # try progressively larger slices - stop if too big
        for end in range(len(text), start, -1):
            if text[end-1] != '}':
                continue
            candidate = text[start:end]
            try:
                return json.loads(candidate)
            except Exception:
                continue
    # 3) try parsing entire trimmed text
    try:
        return json.loads(text.strip())
    except Exception:
        return None

# Convert structured JSON (from model) to clean markdown
def structured_json_to_markdown(obj: Dict[str, Any]) -> str:
    parts: List[str] = []
    title = obj.get('title') or obj.get('heading') or obj.get('name')
    if title:
        parts.append(f'# {title}\n')

    sections = obj.get('sections') or obj.get('body') or []
    for sec in sections:
        if isinstance(sec, dict):
            heading = sec.get('heading') or sec.get('title')
            if heading:
                parts.append(f'## {heading}\n')
            content = sec.get('content')
            bullets = sec.get('bullets') or sec.get('items') or []
            if isinstance(content, str) and content.strip():
                parts.append(content.strip() + '\n')
            if bullets:
                for b in bullets:
                    parts.append(f'- {b}')
                parts.append('')
        elif isinstance(sec, str):
            parts.append(sec)
            parts.append('')

    tables = obj.get('tables') or []
    for tbl in tables:
        headers = tbl.get('headers') or tbl.get('cols') or []
        rows = tbl.get('rows') or []
        if headers and isinstance(rows, list):
            parts.append('| ' + ' | '.join([h for h in headers]) + ' |')
            parts.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
            for r in rows:
                # ensure row length
                row_cells = [str(c) if c is not None else '' for c in r]
                if len(row_cells) < len(headers):
                    row_cells += [''] * (len(headers) - len(row_cells))
                parts.append('| ' + ' | '.join(row_cells) + ' |')
            parts.append('')

    # Fallback: some models return 'markdown' key
    md = obj.get('markdown')
    if md:
        parts.append(md)

    final = '\n'.join(parts)
    return final.strip() + '\n' if final.strip() else ''

# Conservative table cleaning / conversion of pipe blocks
def _clean_pipe_block(block_lines: List[str]) -> Optional[str]:
    # Split rows into cells, strip whitespace and remove accidental leading bullets
    rows = []
    for r in block_lines:
        # ignore lines that are just separators like '---' or '----'
        if re.fullmatch(r'\s*[-=]{2,}\s*', r):
            rows.append(['__SEP__'])
            continue
        parts = [c.strip().lstrip('- ').lstrip('• ').strip() for c in re.split(r'\|', r)]
        # remove leading/trailing empty cells created by pipes
        # but keep empties in-between for alignment
        rows.append(parts)

    # remove rows that are just separators
    rows = [r for r in rows if r != ['__SEP__']]

    if not rows or len(rows) < 2:
        return None

    max_cols = max(len(r) for r in rows)
    # pad rows
    for r in rows:
        if len(r) < max_cols:
            r += [''] * (max_cols - len(r))

    # compute columns to keep (at least one cell non-empty)
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

def normalize_markdown(text: str) -> str:
    """
    Conservative normalization:
    - sanitize artifacts first,
    - attempt to clean existing pipe tables,
    - convert ONLY consistent space/tab-separated blocks into tables,
    - avoid converting bullet lists.
    """
    text = sanitize_raw_text(text)
    lines = text.splitlines()
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # if bullet/list line, copy as-is
        if line.lstrip().startswith(('-', '*', '•', '+', '1.')):
            out.append(line)
            i += 1
            continue

        # PIPE blocks
        if '|' in line:
            j = i
            block = []
            while j < len(lines) and '|' in lines[j]:
                block.append(lines[j])
                j += 1
            cleaned = _clean_pipe_block(block)
            if cleaned:
                # ensure blank line above
                if out and out[-1].strip():
                    out.append('')
                out.extend(cleaned.splitlines())
                out.append('')
                i = j
                continue
            else:
                out.extend(block)
                i = j
                continue

        # SPACE/TAB table candidate
        if ('\t' in line) or re.search(r'\S\s{2,}\S', line):
            j = i
            block = []
            while j < len(lines) and lines[j].strip() and not lines[j].lstrip().startswith(('-', '*', '•', '+', '1.')) \
                  and (('\t' in lines[j]) or re.search(r'\S\s{2,}\S', lines[j])):
                block.append(lines[j])
                j += 1
            if len(block) >= 2:
                # split rows on tabs or 2+ spaces
                rows = []
                for r in block:
                    if '\t' in r:
                        parts = [c.strip().lstrip('- ').strip() for c in re.split(r'\t+', r)]
                    else:
                        parts = [c.strip().lstrip('- ').strip() for c in re.split(r'\s{2,}', r)]
                    rows.append(parts)
                max_cols = max(len(r) for r in rows)
                for r in rows:
                    if len(r) < max_cols:
                        r += [''] * (max_cols - len(r))
                # require at least 2 non-empty columns
                col_nonempty = [any(row[c].strip() for row in rows) for c in range(max_cols)]
                if sum(col_nonempty) >= 2:
                    # remove empty columns
                    rows2 = [[cell for k,cell in enumerate(row) if col_nonempty[k]] for row in rows]
                    # build table
                    header = rows2[0]
                    out_table = []
                    out_table.append('| ' + ' | '.join([h if h else ' ' for h in header]) + ' |')
                    out_table.append('| ' + ' | '.join(['---'] * len(header)) + ' |')
                    for row in rows2[1:]:
                        out_table.append('| ' + ' | '.join([c if c else ' ' for c in row]) + ' |')
                    if out and out[-1].strip():
                        out.append('')
                    out.extend(out_table)
                    out.append('')
                    i = j
                    continue
            # fallback copy
            out.append(line)
            i += 1
            continue

        # default copy
        out.append(line)
        i += 1

    normalized = '\n'.join(out)
    normalized = re.sub(r'\n{3,}', '\n\n', normalized)
    normalized = re.sub(r'(?m)([^\n])\n(#{1,6}\s)', r'\1\n\n\2', normalized)
    return normalized.strip() + '\n'
