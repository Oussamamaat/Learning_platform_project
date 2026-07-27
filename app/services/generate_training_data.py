"""
Generate Training Data for LoRA Fine-Tuning
────────────────────────────────────────────
Uses a local Ollama model as few-shot generator to produce
ChatML-formatted rows across 4 components.

Usage:
    python -m app.services.generate_training_data
    python -m app.services.generate_training_data --output-dir data/training --target-rows 7500
    python -m app.services.generate_training_data --target-rows 50  # smoke test
"""

import json
import logging
import os
import random
import re
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("data/training")
DEFAULT_MODEL = "hf.co/QuantFactory/Atlas-Chat-9B-GGUF:latest"
DEFAULT_TARGET_ROWS = 7500
DEFAULT_DEDUP_THRESHOLD = 0.95
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_BATCH_SIZE = 10
DEFAULT_EVAL_SPLIT = 0.1
DEDUP_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

COMPONENT_CONFIG = {
    "socratic": {"weight": 2500, "multi_turn_pct": 0.6},
    "code_switching": {"weight": 2500, "multi_turn_pct": 0.4},
    "grounded_refusal": {"weight": 2000, "multi_turn_pct": 0.3},
    "general_preservation": {"weight": 500, "multi_turn_pct": 0.2},
}

DOMAINS = ["industrial", "securite", "blockchain"]

DOMAIN_FOLDER_ALIASES = {"securite": "securite_physique"}

DOMAIN_STYLE_HINTS = {
    "industrial": (
        "Focus on workplace safety, PPE, LOTO, machine guarding, hazard communication, "
        "emergency procedures, and Moroccan labor law (Code du Travail)."
    ),
    "securite": (
        "Focus on physical security, surveillance, guarding, access control, "
        "incident reporting, and Law 27-06 on private security."
    ),
    "blockchain": (
        "Focus on blockchain compliance, AML/CFT, digital asset regulation, "
        "smart contracts, and Moroccan DPGF (Bill 42-25)."
    ),
}

# Production system prompt template — TRAINING DATA MUST MATCH THIS EXACTLY.
# Source: app/services/llm.py SYSTEM_PROMPT_TEMPLATE
PRODUCTION_SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert bilingual enterprise tutor specializing in {domain}.\n"
    "Guide the user in French and Moroccan Darija (Arabizi script) using a Socratic method.\n"
    "Use formal punctuation and capitalization.\n"
    "Ground all answers strictly in the provided context.\n"
    "If the context is insufficient, politely refuse and suggest what the user should study.\n"
    "Never invent facts. Only use information from the context below.\n\n"
    "CONTEXTE :\n"
    "{context}"
)

DOMAIN_LABELS = {
    "industrial": "industrial safety and workplace protocols",
    "securite": "physical security and surveillance procedures",
    "blockchain": "blockchain compliance and digital asset regulation",
}

# ---------------------------------------------------------------------------
# File Loaders
# ---------------------------------------------------------------------------


def load_file(path: Path) -> str:
    """Load a text file, return empty string if missing."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    logger.warning("File not found: %s", path)
    return ""


def load_raw_corpus(raw_dir: Path) -> list[dict]:
    """Load all .txt and .md files from raw/ corpus directory with domain tagging.

    Directory structure: raw/{scope}/{domain}/{media_type}/filename
    Example: raw/shared/blockchain/text/3.1_bill_42_25_draft.md
    """
    documents = []
    for ext in ["*.txt", "*.md"]:
        for file_path in raw_dir.rglob(ext):
            content = file_path.read_text(encoding="utf-8").strip()
            if not content:
                continue
            parts = file_path.relative_to(raw_dir).parts
            documents.append({
                "path": str(file_path),
                "content": content[:3000],
                "scope": parts[0] if len(parts) > 0 else "unknown",
                "domain": parts[1] if len(parts) > 1 else "unknown",
            })
    logger.info("Loaded %d raw corpus documents", len(documents))
    return documents


def docs_for_domain(raw_corpus: list[dict], domain: str) -> list[dict]:
    """Filter raw corpus to only documents matching the requested domain."""
    folder = DOMAIN_FOLDER_ALIASES.get(domain, domain)
    return [d for d in raw_corpus if d["domain"] == folder]


def load_manual_files(data_dir: Path) -> dict:
    """Load the 4 manual foundation files."""
    files = {
        "ortho_guide": load_file(data_dir / "ortho_guide.md"),
        "code_switching_rules": load_file(data_dir / "code_switching_rules.md"),
        "refusal_templates": load_file(data_dir / "refusal_templates.md"),
        "few_shot_examples": load_file(data_dir / "few_shot_examples.md"),
    }
    loaded = sum(1 for v in files.values() if v)
    logger.info("Loaded %d/4 manual files", loaded)
    return files


# ---------------------------------------------------------------------------
# Target Scaling
# ---------------------------------------------------------------------------


def scale_component_targets(target_rows: int) -> dict:
    """Scale component targets proportionally from total target_rows."""
    total_weight = sum(c["weight"] for c in COMPONENT_CONFIG.values())
    scaled = {}
    remaining = target_rows
    components = list(COMPONENT_CONFIG.keys())
    for i, (name, config) in enumerate(COMPONENT_CONFIG.items()):
        if i == len(components) - 1:
            target = remaining
        else:
            target = round(target_rows * config["weight"] / total_weight)
            remaining -= target
        scaled[name] = {"target": target, "multi_turn_pct": config["multi_turn_pct"]}
    return scaled


# ---------------------------------------------------------------------------
# Generation Prompts
# ---------------------------------------------------------------------------


def sample_one_few_shot(few_shot_str: str) -> str:
    """Extract a single random ChatML example from few_shot_examples.md."""
    try:
        match = re.search(r'```json\s*(.*?)\s*```', few_shot_str, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
            if isinstance(data, list) and len(data) > 0:
                selected = random.choice(data)
                return json.dumps(selected, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return few_shot_str[:1200]


def build_socratic_prompt(
    few_shot: str, ortho_guide: str, domain: str
) -> str:
    """Build compact Socratic prompt with single few-shot example."""
    style_hint = DOMAIN_STYLE_HINTS.get(domain, "")
    single_example = sample_one_few_shot(few_shot)

    return f"""Generate 1 Socratic enterprise tutor sample.
Domain: {domain} - {style_hint}
Language: 40% Darija Arabizi, 30% French, 30% Mixed. 3-5 turns.

Rules:
- Numerals: 2=ء, 3=ع, 5=خ, 7=ح, 8=غ, 9=ق.
- Technical terms in French (l-casque, LOTO, EPI, la vanne). Connectors in Darija.
- Use Arabizi Latin script ONLY. NEVER use Arabic script.
- EVERY assistant turn MUST first explain or state the relevant fact/rule/principle
  in 1-2 sentences, THEN end with ONE checking question. An assistant turn that is
  ONLY a question, with no explanation, is INVALID — this is teaching, not a quiz.

Example format:
{single_example}

Output JSON: an object with a "messages" array of alternating user/assistant turns."""


def build_code_switching_prompt(
    few_shot: str, code_switching_rules: str, domain: str
) -> str:
    """Build code-switching prompt with injected rules and domain hints."""
    style_hint = DOMAIN_STYLE_HINTS.get(domain, "")
    single_example = sample_one_few_shot(few_shot)
    return f"""Generate 1 code-switching enterprise tutor sample.
Domain: {domain} - {style_hint}

Rules:
- Numerals: 2=ء, 3=ع, 5=خ, 7=ح, 8=غ, 9=ق, 6=ط.
- Use Arabizi Latin script ONLY. NEVER Arabic script ("chno" not "شنو").
- Technical nouns in French with Darija article (l-casque, la vanne, les EPI).
- Connectors, verbs and questions in Darija. Switch at phrase boundaries only,
  never word-by-word.
- The assistant states the relevant fact, then asks one short follow-up.

Example format:
{single_example}

Output JSON: an object with a "messages" array of user/assistant turns."""


def build_grounded_refusal_prompt(
    refusal_templates: str, raw_context: str, domain: str
) -> str:
    """Build grounded-refusal prompt asking ONLY for user/assistant messages."""
    clean_context = raw_context[:1200].strip()

    return f"""Based ONLY on this CONTEXT:
---CONTEXT START---
{clean_context}
---CONTEXT END---

Refusal templates reference (use one if unanswerable):
{refusal_templates}

Generate 2 Q&A training samples as a JSON array:
Sample 1: A question answerable using the CONTEXT -> grounded answer citing the text.
Sample 2: A question NOT answerable using the CONTEXT -> polite refusal in Arabizi/French.

RULES:
- Use Latin Arabizi script ONLY (2=ء, 3=ع, 5=خ, 7=ح, 8=غ, 9=ق).
- Technical terms in French.
- DO NOT include system messages in your output. Only generate user and assistant roles.

Output JSON: an array of exactly 2 objects, each with a "messages" array
containing one user turn and one assistant turn."""


def build_general_preservation_prompt() -> str:
    return """Generate a single general-purpose Darija instruction-response pair.

This row preserves Darija fluency, cultural context, and general tutoring tone.
Topic: Moroccan workplace culture, general safety awareness, or everyday conversation.

CRITICAL LANGUAGE RULES:
- Darija MUST be written in Arabizi (Latin letters + numerals), NEVER in Arabic script.
- Use: 3=ع, 7=ح, 9=ق, 2=ء, 5=خ, 8=غ, 6=ط
- Example: "salam" not "سلام", "labas" not "لاباس", "chno" not "شنو"
- Primarily Darija Arabizi with some French.

Output JSON: an object with a "messages" array of user/assistant turns."""


# ---------------------------------------------------------------------------
# Ollama API Client
# ---------------------------------------------------------------------------


# Constrains generation to a valid row shape. Ollama enforces this during
# sampling, so malformed JSON becomes structurally impossible rather than
# something the repair layer has to clean up after the fact.
ROW_SCHEMA = {
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": ["user", "assistant"]},
                    "content": {"type": "string", "minLength": 15},
                },
                "required": ["role", "content"],
            },
        },
    },
    "required": ["messages"],
}

ROW_LIST_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "items": ROW_SCHEMA,
}


def call_ollama(
    prompt: str,
    model: str,
    ollama_url: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    retries: int = 3,
    num_ctx: int = 4096,
    timeout: int = 300,
    schema: Optional[dict] = None,
) -> Optional[str]:
    """Call Ollama API with retry logic and optional schema-constrained output."""
    url = f"{ollama_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # Holds the model resident between calls; without it Ollama can evict
        # and reload the 5.8GB Q4_K_M weights mid-run.
        "keep_alive": "30m",
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
        },
    }
    if schema is not None:
        payload["format"] = schema
    data = json.dumps(payload).encode("utf-8")

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                res_body = response.read().decode("utf-8")
                res_json = json.loads(res_body)
                return res_json.get("response")
        except urllib.error.URLError as e:
            logger.warning("Retry %d/%d Ollama error: %s", attempt + 1, retries, e)
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.warning("Retry %d/%d error: %s", attempt + 1, retries, e)
            time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Post-Processing
# ---------------------------------------------------------------------------


ROLE_CONTENT_RE = re.compile(
    r'"role"\s*:\s*"(system|user|assistant)"\s*,\s*"content"\s*:\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)
CONTENT_ROLE_RE = re.compile(
    r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"role"\s*:\s*"(system|user|assistant)"',
    re.DOTALL,
)


def _strip_fences(text: str) -> str:
    return re.sub(r'```(?:json)?', '', text).strip()


def _slice_to_start(text: str):
    """Drop any prose preamble before the first JSON container."""
    start_obj = text.find('{')
    start_arr = text.find('[')
    if start_obj == -1 and start_arr == -1:
        return None
    if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
        return text[start_obj:]
    return text[start_arr:]


def _fix_stray_after_string(text: str) -> str:
    """Drop junk between a closing quote and the next structural delimiter.

    The model routinely emits `"content": "text."` with a trailing period
    *outside* the quotes, which invalidates an otherwise perfect response.
    """
    return re.sub(r'"[ \t]*[^\s",:}\]\[{]{1,3}[ \t]*(?=[,}\]\n])', '"', text)


def _drop_trailing_commas(text: str) -> str:
    return re.sub(r',(\s*[}\]])', r'\1', text)


def _balance(text: str) -> str:
    """Close any containers/strings left open by a truncated response."""
    in_string = False
    escape = False
    stack = []
    for char in text:
        if escape:
            escape = False
            continue
        if char == '\\' and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char in '{[':
                stack.append('}' if char == '{' else ']')
            elif char in '}]':
                if stack and stack[-1] == char:
                    stack.pop()
    out = text
    if in_string:
        out += '"'
    out = re.sub(r',\s*$', '', out)
    return out + "".join(reversed(stack))


def repair_json(text: str):
    """Parse LLM JSON, applying progressively more aggressive repairs."""
    if not text or not text.strip():
        return None

    text = _slice_to_start(_strip_fences(text))
    if text is None:
        return None

    fixed = _drop_trailing_commas(_fix_stray_after_string(text))
    for candidate in (text, fixed, _balance(text), _balance(fixed)):
        try:
            return json.loads(candidate)
        except Exception:
            continue

    # Last resort: truncate to the final complete container and rebalance.
    for candidate in (fixed, text):
        last = max(candidate.rfind('}'), candidate.rfind(']'))
        if last == -1:
            continue
        sub = candidate[:last + 1]
        sub += '}' * max(0, sub.count('{') - sub.count('}'))
        sub += ']' * max(0, sub.count('[') - sub.count(']'))
        try:
            return json.loads(sub)
        except Exception:
            continue

    return None


def salvage_messages(text: str) -> list:
    """Recover conversations by regex when structural parsing fails entirely.

    Splits on `"messages"` keys so each block stays one conversation, which
    preserves multi-turn rows that a flat pair-scan would shred into
    single-turn fragments.
    """
    if not text:
        return []
    blocks = re.split(r'"messages"\s*:', _strip_fences(text))
    if len(blocks) > 1:
        blocks = blocks[1:]

    conversations = []
    for block in blocks:
        messages = [
            {"role": m.group(1), "content": m.group(2)}
            for m in ROLE_CONTENT_RE.finditer(block)
        ]
        if not messages:
            messages = [
                {"role": m.group(2), "content": m.group(1)}
                for m in CONTENT_ROLE_RE.finditer(block)
            ]
        if messages:
            conversations.append(_unescape_messages(messages))
    return conversations


def _unescape_messages(messages: list) -> list:
    """Resolve JSON escape sequences captured by the salvage regex."""
    out = []
    for msg in messages:
        raw = msg["content"]
        try:
            content = json.loads(f'"{raw}"')
        except Exception:
            content = raw
        out.append({"role": msg["role"], "content": content})
    return out


# Combining marks (harakat/shadda/sukun) and tatweel have no Arabizi
# equivalent -- they must be dropped, not transliterated.
ARABIC_DIACRITICS = re.compile('[ً-ْٰـ]')

# High-frequency Darija words get idiomatic spellings; the bare character map
# renders them unreadably ("chnw" rather than "chno"). Applied longest-first
# so a longer word is not clipped by a shorter one it contains.
DARIJA_WORD_MAP = {
    'كيفاش': 'kifash',
    'ماكاينش': 'makaynch',
    'هادشي': 'hadchi',
    'خاصك': 'khassek',
    'علاش': '3lash',
    'بغيتي': 'bghiti',
    'عندك': '3andek',
    'عندي': '3andi',
    'مزيان': 'mzyan',
    'واخا': 'wakha',
    'شكون': 'chkoun',
    'ديال': 'dyal',
    'غادي': 'ghadi',
    'ولكن': 'walakin',
    'شوية': 'chwiya',
    'كولشي': 'kolchi',
    'ماشي': 'machi',
    'دابا': 'daba',
    'بزاف': 'bzaf',
    'خاص': 'khass',
    'كاين': 'kayn',
    'بغيت': 'bghit',
    'صافي': 'safi',
    'عفاك': '3afak',
    'زعما': 'z3ma',
    'شنو': 'chno',
    'واش': 'wach',
    'حيت': '7it',
    'باش': 'bach',
    'هادي': 'hadi',
    'هاد': 'had',
    'فين': 'fin',
    'سمح': 'smeh',
    'راه': 'rah',
    'ملي': 'mli',
    'فاش': 'fach',
    'حنا': '7na',
    'معا': 'm3a',
    'على': '3la',
    'نيت': 'nit',
}

# Arabic punctuation has direct Latin equivalents. Leaving it in place was
# stamping Arabic question marks and commas into otherwise-Arabizi rows.
ARABIC_PUNCT_MAP = {
    '؟': '?', '،': ',', '؛': ';', '٪': '%',
    '٫': '.', '٬': ',', '۔': '.',
}

ARABIC_CHAR_MAP = {
    'ا': 'a', 'ب': 'b', 'ت': 't', 'ث': 'th', 'ج': 'j',
    'ح': '7', 'خ': 'kh', 'د': 'd', 'ذ': 'd', 'ر': 'r',
    'ز': 'z', 'س': 's', 'ش': 'ch', 'ص': 's', 'ض': 'd',
    'ط': 't', 'ظ': 'z', 'ع': '3', 'غ': 'gh', 'ف': 'f',
    'ق': '9', 'ك': 'k', 'ل': 'l', 'م': 'm', 'ن': 'n',
    'ه': 'h', 'و': 'w', 'ي': 'y', 'ة': 'a', 'ى': 'a',
    'أ': 'a', 'إ': 'i', 'ؤ': 'o', 'ئ': 'i', 'ء': '2',
    'آ': 'a', 'ٱ': 'a', 'گ': 'g', 'ڭ': 'g', 'ڤ': 'v',
    'پ': 'p', 'ژ': 'j', 'ﻻ': 'la',
    # Eastern Arabic numerals
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
}

_DARIJA_WORDS_SORTED = sorted(DARIJA_WORD_MAP.items(), key=lambda kv: -len(kv[0]))


def has_arabic_script(text: str) -> bool:
    """True if the text contains Arabic letters or numerals.

    Deliberately starts at U+0621 so the punctuation block (U+060C, U+061B,
    U+061F) does not by itself trigger transliteration.
    """
    return any('ء' <= c <= 'ۿ' for c in text)


def force_arabizi(text: str) -> str:
    """Convert Arabic-script Darija to Arabizi.

    Four passes: normalize punctuation, strip diacritics, replace known Darija
    words (longest first), then transliterate remaining characters. Punctuation
    is normalized even when no Arabic letters are present, because the model
    mixes Arabic punctuation into otherwise-Latin output.
    """
    if not text:
        return text

    for ar_p, lat_p in ARABIC_PUNCT_MAP.items():
        text = text.replace(ar_p, lat_p)

    if not has_arabic_script(text):
        return text

    text = ARABIC_DIACRITICS.sub('', text)

    for ar_word, lat_word in _DARIJA_WORDS_SORTED:
        text = text.replace(ar_word, lat_word)

    text = ''.join(ARABIC_CHAR_MAP.get(c, c) for c in text)
    return re.sub(r'[ \t]{2,}', ' ', text).strip()


def extract_question_heuristic(text: str) -> str:
    """Extract question text when LLM embeds the question inside system content."""
    if not text:
        return ""

    marker_match = re.search(
        r'(?:السؤال|سؤال|Question|So2al|Soal|Q)\s*[:\-]\s*(.*?)(?=(?:الأجوبة|الإجابة|Réponse|Answer|A\s*[:\-]|$))',
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if marker_match and marker_match.group(1).strip():
        return marker_match.group(1).strip()

    sentences = re.split(r'(?<=[.!\n])', text)
    q_sentences = [s.strip() for s in sentences if '\u061f' in s or '?' in s]
    if q_sentences:
        return " ".join(q_sentences)

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        return lines[-1]

    return text.strip()


def _trim_to_conversation(messages: list) -> list:
    """Trim to a well-formed turn sequence: starts at user, ends at assistant.

    The model often opens with an assistant turn or trails off on a user turn.
    Training on either teaches the wrong thing — a leading assistant turn
    teaches the model to speak unprompted, a trailing user turn has no target.
    Consecutive same-role turns are merged so alternation holds.
    """
    first_user = next(
        (i for i, m in enumerate(messages) if m["role"] == "user"), None
    )
    if first_user is None:
        return []
    last_assistant = next(
        (
            i
            for i in range(len(messages) - 1, first_user - 1, -1)
            if messages[i]["role"] == "assistant"
        ),
        None,
    )
    if last_assistant is None:
        return []

    window = messages[first_user:last_assistant + 1]

    merged = []
    for msg in window:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] = f"{merged[-1]['content']}\n{msg['content']}"
        else:
            merged.append(dict(msg))
    return merged


def normalize_row(
    row: dict,
    default_system_content: str,
    component: str,
    domain: str,
) -> dict:
    """Rebuild a row into the exact ChatML shape used at serving time.

    Recovers a missing user turn, transliterates to Arabizi, injects the
    production system prompt, and stamps component/domain from the generation
    context. Metadata is authored here rather than trusted from the model —
    a missing or misspelled `component` silently dropped the row at export.
    """
    messages = [m for m in row.get("messages", []) if isinstance(m, dict)]

    has_user = any(m.get("role") == "user" for m in messages)
    if not has_user and messages:
        raw_combined = " ".join(
            str(m.get("content", "")) for m in messages
        )
        extracted_q = extract_question_heuristic(raw_combined)
        if extracted_q:
            messages.insert(0, {"role": "user", "content": extracted_q})

    clean_messages = []
    transliterated = False
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            continue
        raw = msg.get("content", "")
        if not isinstance(raw, str):
            continue
        if has_arabic_script(raw):
            transliterated = True
        content = force_arabizi(raw)
        if content and content[0].islower():
            content = content[0].upper() + content[1:]
        clean_messages.append({"role": role, "content": content})

    clean_messages = _trim_to_conversation(clean_messages)

    clean_messages.insert(0, {"role": "system", "content": default_system_content})

    row["messages"] = clean_messages
    row["component"] = component
    row["domain"] = domain
    # Atlas-Chat emits Arabic-script Darija for most prompts, so this row's
    # Arabizi is machine transliteration rather than native model output.
    # Short vowels are absent from Arabic script and cannot be recovered, so
    # these rows read less naturally — flagged here so they can be filtered.
    row["transliterated"] = transliterated
    return row


MIN_ALNUM_CHARS = 4
_ALNUM_RE = re.compile(r'[0-9A-Za-z]')


def _has_substance(content: str) -> bool:
    """Reject placeholder turns such as the literal "..." from the prompt.

    The output-format block in every prompt uses "..." as a stand-in, and the
    model sometimes copies it verbatim. Such turns are non-empty so they pass a
    strip() check, but carry no content. Real turns hold well over 30
    alphanumeric characters, so this threshold has ample margin.
    """
    return len(_ALNUM_RE.findall(content)) >= MIN_ALNUM_CHARS


def validate_chatml(row: dict) -> bool:
    """Validate a normalized row: system + user + assistant, all non-empty.

    Run this *after* normalize_row — the model frequently omits the system
    message or the user turn, both of which normalization restores.
    """
    if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
        return False

    messages = row["messages"]
    if not all(isinstance(m, dict) for m in messages):
        return False

    roles = [m.get("role") for m in messages]
    if "assistant" not in roles or "user" not in roles:
        return False

    for m in messages:
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            return False
        if m["role"] != "system" and not _has_substance(content):
            return False

    return True


def deduplicate(rows: list, threshold: float = 0.95) -> list:
    """Remove near-duplicate rows using multilingual embedding similarity."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer(DEDUP_MODEL)
        # Compare only the conversation. The system message is byte-identical
        # across every row of a domain (and embeds the whole retrieved context
        # for grounded_refusal), so including it makes unrelated rows look
        # near-identical and collapses the dataset.
        texts = [
            " ".join(m["content"] for m in r["messages"] if m["role"] != "system")
            for r in rows
        ]
        embeddings = model.encode(texts, show_progress_bar=True)

        keep = []
        seen = []
        for i, emb in enumerate(embeddings):
            if not seen:
                keep.append(i)
                seen.append(emb)
                continue
            similarities = np.dot(seen, emb) / (
                np.linalg.norm(seen, axis=1) * np.linalg.norm(emb)
            )
            if float(similarities.max()) < threshold:
                keep.append(i)
                seen.append(emb)

        logger.info("Dedup: %d → %d rows (threshold=%.2f)", len(rows), len(keep), threshold)
        return [rows[i] for i in keep]
    except Exception as e:
        logger.warning("Dedup failed (keeping all rows): %s", e)
        return rows


def split_train_eval(rows: list, eval_split: float = 0.1) -> tuple:
    """Split rows into train and eval sets."""
    random.shuffle(rows)
    split_idx = int(len(rows) * (1 - eval_split))
    return rows[:split_idx], rows[split_idx:]


# ---------------------------------------------------------------------------
# Streaming Writer
# ---------------------------------------------------------------------------


class StreamingJsonlWriter:
    """Append-mode JSONL writer that flushes to disk incrementally."""

    def __init__(self, path: Path):
        self.path = path
        self.count = 0
        self._file = open(path, "a", encoding="utf-8")

    def write(self, row: dict):
        self._file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.count += 1
        if self.count % 50 == 0:
            self._file.flush()

    def close(self):
        self._file.flush()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Generation Core
# ---------------------------------------------------------------------------


def generate_component(
    component: str,
    config: dict,
    manual_files: dict,
    raw_corpus: list[dict],
    model: str,
    ollama_url: str,
    temperature: float,
    max_tokens: int,
    writer: StreamingJsonlWriter,
    max_attempt_factor: int = 6,
    arabizi_policy: str = "transliterate",
    concurrency: int = 1,
    resume_rows: Optional[list] = None,
) -> dict:
    """Generate all rows for one component, writing directly to disk."""
    target = config["target"]

    # Resume: rows already on disk count toward the target, and their text
    # keys seed the dedup set so a restart can't re-emit them.
    seen_texts = set()
    generated_count = 0
    if resume_rows:
        for row in resume_rows:
            seen_texts.add(" ".join(
                m["content"] for m in row.get("messages", [])
                if m.get("role") != "system"
            ))
        generated_count = len(resume_rows)
        logger.info(
            "Component: %s (target: %d, resuming from %d existing rows)",
            component, target, generated_count,
        )
    else:
        logger.info("Component: %s (target: %d rows)", component, target)

    attempts = 0
    max_attempts = target * max_attempt_factor
    parse_failures = 0
    salvaged = 0
    transliterated_count = 0
    rejected_arabic = 0

    def build_job() -> tuple:
        """Assemble one generation request: (prompt, system_content, domain)."""
        domain = random.choice(DOMAINS)
        domain_label = DOMAIN_LABELS.get(domain, domain)

        raw_doc = {"content": "No context available."}
        if component == "socratic":
            prompt = build_socratic_prompt(
                manual_files.get("few_shot_examples", ""),
                manual_files.get("ortho_guide", ""),
                domain,
            )
        elif component == "code_switching":
            prompt = build_code_switching_prompt(
                manual_files.get("few_shot_examples", ""),
                manual_files.get("code_switching_rules", ""),
                domain,
            )
        elif component == "grounded_refusal":
            domain_docs = docs_for_domain(raw_corpus, domain)
            raw_doc = random.choice(domain_docs) if domain_docs else raw_doc
            prompt = build_grounded_refusal_prompt(
                manual_files.get("refusal_templates", ""),
                raw_doc["content"],
                domain,
            )
        elif component == "general_preservation":
            prompt = build_general_preservation_prompt()
        else:
            raise ValueError(f"Unknown component: {component}")

        # Build system content in Python — model does NOT generate it
        if component == "grounded_refusal":
            clean_ctx = raw_doc["content"][:1200].strip()
            system_content = PRODUCTION_SYSTEM_PROMPT_TEMPLATE.format(
                domain=domain_label, context=clean_ctx,
            )
        else:
            system_content = PRODUCTION_SYSTEM_PROMPT_TEMPLATE.format(
                domain=domain_label, context="General enterprise knowledge.",
            )
        return prompt, system_content, domain

    # grounded_refusal asks for a pair (answerable + refusal); the rest
    # produce a single row.
    schema = ROW_LIST_SCHEMA if component == "grounded_refusal" else ROW_SCHEMA

    def run_job(job: tuple) -> tuple:
        """Execute one request. Runs on a worker thread — no shared state."""
        prompt, system_content, domain = job
        response = call_ollama(
            prompt, model, ollama_url, temperature, max_tokens, schema=schema
        )
        return response, system_content, domain

    executor = ThreadPoolExecutor(max_workers=concurrency) if concurrency > 1 else None

    while generated_count < target and attempts < max_attempts:
        # Size each wave to what's still missing, so we don't overshoot the
        # target by a full batch on the final iteration.
        remaining = target - generated_count
        wave = max(1, min(concurrency, remaining, max_attempts - attempts))
        jobs = [build_job() for _ in range(wave)]
        attempts += wave

        if executor is not None:
            results = list(executor.map(run_job, jobs))
        else:
            results = [run_job(j) for j in jobs]

        for response, system_content, domain in results:
            if generated_count >= target:
                break
            if not response:
                continue
            try:
                candidates = []
                parsed = repair_json(response)
                if isinstance(parsed, list):
                    candidates = [r for r in parsed if isinstance(r, dict)]
                elif isinstance(parsed, dict):
                    candidates = [parsed]

                # Normalize first, then validate: normalization restores the
                # system message and recovers a missing user turn, so validating
                # the raw model output discards rows that are recoverable.
                rows_to_write = []
                for row in candidates:
                    row = normalize_row(row, system_content, component, domain)
                    if validate_chatml(row):
                        rows_to_write.append(row)

                # Structural parsing lost the response (or produced nothing
                # usable): fall back to scraping message pairs from raw text.
                if not rows_to_write:
                    for messages in salvage_messages(response):
                        row = normalize_row(
                            {"messages": messages}, system_content,
                            component, domain,
                        )
                        if validate_chatml(row):
                            rows_to_write.append(row)
                    if rows_to_write:
                        salvaged += 1

                if not rows_to_write:
                    parse_failures += 1
                    continue

                for row in rows_to_write:
                    if row.get("transliterated") and arabizi_policy == "reject":
                        rejected_arabic += 1
                        continue
                    text_key = " ".join(
                        m["content"] for m in row["messages"]
                        if m["role"] != "system"
                    )
                    if text_key in seen_texts:
                        continue
                    seen_texts.add(text_key)
                    writer.write(row)
                    generated_count += 1
                    # Counted only for rows actually written, so the reported
                    # percentage stays relative to the emitted dataset.
                    if row.get("transliterated"):
                        transliterated_count += 1
                    if generated_count % 100 == 0:
                        logger.info(
                            "Progress: %d/%d (attempts: %d, parse_fails: %d, "
                            "salvaged: %d)",
                            generated_count, target, attempts,
                            parse_failures, salvaged,
                        )

            except Exception as e:
                logger.debug("Row processing failed: %s", e)
                parse_failures += 1
                continue

    if executor is not None:
        executor.shutdown(wait=True)

    if generated_count < target:
        logger.warning(
            "%s: only %d/%d rows after %d attempts — raise --max-attempt-factor "
            "or check model output quality",
            component, generated_count, target, attempts,
        )

    pct = 100.0 * transliterated_count / generated_count if generated_count else 0.0
    logger.info(
        "Generated %d rows (attempts: %d, parse_failures: %d, salvaged: %d, "
        "transliterated: %d = %.0f%%, rejected_arabic: %d)",
        generated_count, attempts, parse_failures, salvaged,
        transliterated_count, pct, rejected_arabic,
    )
    return {
        "generated": generated_count,
        "target": target,
        "attempts": attempts,
        "parse_failures": parse_failures,
        "salvaged": salvaged,
        "transliterated": transliterated_count,
        "transliterated_pct": round(pct, 1),
        "rejected_arabic": rejected_arabic,
    }


# ---------------------------------------------------------------------------
# Output Path Validation
# ---------------------------------------------------------------------------


def validate_output_path(output_dir: Path) -> Path:
    """Ensure output directory exists and warn if not on expected drive."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        resolved = output_dir.resolve()
        # Check if on a different drive than the project root
        project_root = Path.cwd().resolve()
        if resolved.drive and project_root.drive and resolved.drive != project_root.drive:
            logger.warning(
                "Output drive (%s) differs from project drive (%s). "
                "Ensure this is the intended fast SSD.",
                resolved.drive, project_root.drive,
            )
    except Exception:
        pass

    return output_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate training data for LoRA fine-tuning"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for training data",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Ollama model to use for generation",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API base URL",
    )
    parser.add_argument(
        "--target-rows",
        type=int,
        default=DEFAULT_TARGET_ROWS,
        help="Total target rows across all components (scales proportionally)",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=DEFAULT_DEDUP_THRESHOLD,
        help="Cosine similarity threshold for deduplication",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Generation temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Max tokens per generation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing manual foundation files",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("raw"),
        help="Directory containing raw corpus documents",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max-attempt-factor",
        type=int,
        default=6,
        help="Cap generation attempts per component at target * this factor",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Parallel in-flight requests. Only raise above 1 when the server "
            "has VRAM for that many KV-cache slots (set OLLAMA_NUM_PARALLEL to "
            "match); over-subscribing forces CPU offload and runs slower."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue a previous run: count existing *_raw.jsonl rows toward "
            "each component target instead of deleting them"
        ),
    )
    parser.add_argument(
        "--arabizi-policy",
        default="transliterate",
        choices=["transliterate", "reject"],
        help=(
            "How to handle Arabic-script model output: 'transliterate' converts "
            "it (higher yield, less natural Arabizi), 'reject' keeps only "
            "natively-Arabizi rows (much higher quality, far lower yield)"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level",
    )
    args = parser.parse_args()

    # Without this the module logger has no handler and every progress line
    # during a multi-hour run is silently discarded.
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    random.seed(args.seed)
    output_dir = validate_output_path(args.output_dir)

    logger.info("IBLOG Training Data Generator")
    logger.info(
        "Model: %s | Target: %d rows | Output: %s",
        args.model, args.target_rows, output_dir,
    )

    # Load manual files
    logger.info("Loading manual files...")
    manual_files = load_manual_files(args.data_dir)
    missing = [k for k, v in manual_files.items() if not v]
    if missing:
        logger.warning("Missing manual files: %s — prompts will use inline fallbacks", missing)

    # Load raw corpus for Component 3
    logger.info("Loading raw corpus...")
    raw_corpus = load_raw_corpus(args.raw_dir)
    for domain in DOMAINS:
        count = len(docs_for_domain(raw_corpus, domain))
        logger.info("  Corpus docs for %s: %d", domain, count)

    # Scale component targets from --target-rows
    component_config = scale_component_targets(args.target_rows)
    logger.info("Component targets: %s", {k: v["target"] for k, v in component_config.items()})

    # Remove stale output files before starting (append mode needs clean slate)
    for filename in ["train.jsonl", "eval.jsonl"]:
        stale = output_dir / filename
        if stale.exists():
            stale.unlink()
            logger.info("Removed stale file: %s", stale)

    if not args.resume:
        # Per-component raw files are appended to, so a fresh run must clear
        # them or a previous run's rows would be counted as this run's output.
        for component in component_config:
            stale = output_dir / f"{component}_raw.jsonl"
            if stale.exists():
                stale.unlink()
                logger.info("Removed stale file: %s", stale)

    # Generate per component with streaming writes
    component_stats = {}
    for component, config in component_config.items():
        raw_path = output_dir / f"{component}_raw.jsonl"

        resume_rows = None
        if args.resume and raw_path.exists():
            resume_rows = []
            with open(raw_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        resume_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        # A run killed mid-write can leave a truncated final
                        # line; drop it rather than aborting the resume.
                        logger.warning("Skipping malformed line in %s", raw_path)

        with StreamingJsonlWriter(raw_path) as writer:
            stats = generate_component(
                component,
                config,
                manual_files,
                raw_corpus,
                args.model,
                args.ollama_url,
                args.temperature,
                args.max_tokens,
                writer,
                args.max_attempt_factor,
                args.arabizi_policy,
                args.concurrency,
                resume_rows,
            )
        component_stats[component] = stats

    # Collect all raw rows for dedup
    all_rows = []
    for component in component_config:
        raw_path = output_dir / f"{component}_raw.jsonl"
        if raw_path.exists():
            with open(raw_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_rows.append(json.loads(line))
            raw_path.unlink()

    # Global deduplication
    logger.info("Total rows before dedup: %d", len(all_rows))
    all_rows = deduplicate(all_rows, args.dedup_threshold)

    # Split and export per component
    logger.info("Splitting and exporting...")
    train_all = []
    eval_all = []

    for component in component_config:
        comp_rows = [r for r in all_rows if r.get("component") == component]
        train, eval_set = split_train_eval(comp_rows, DEFAULT_EVAL_SPLIT)
        train_all.extend(train)
        eval_all.extend(eval_set)
        component_stats[component]["train"] = len(train)
        component_stats[component]["eval"] = len(eval_set)
        logger.info("%s: train=%d, eval=%d", component, len(train), len(eval_set))

    # Write final JSONL files
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    stats_path = output_dir / "component_stats.json"

    with open(train_path, "w", encoding="utf-8") as f:
        for row in train_all:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for row in eval_all:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(component_stats, f, indent=2, ensure_ascii=False)

    logger.info(
        "Done! Train: %s (%d rows), Eval: %s (%d rows), Stats: %s",
        train_path, len(train_all), eval_path, len(eval_all), stats_path,
    )


if __name__ == "__main__":
    main()
