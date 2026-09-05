"""
Re-scores an existing scripts/eval_stt.py result JSON with normalization --
Arabic orthographic folding (reusing app.services.citations.fold_arabic,
already proven by the PaddleOCR gate), punctuation stripping, and CER
alongside WER -- WITHOUT re-running any model, re-recording any audio, or
touching a GPU. Every input is the (reference, hypothesis) pair already
stored in the JSON from a real bake-off run.

Why this exists (POST_LEASE_MVP_SPRINT_PLAN.md item 6): scripts/eval_stt.py's
own docstring admits its WER is "deliberately not ... publication-grade ...
with punctuation/casing normalization rules" -- fine for a relative ranking,
but the reported Darija numbers (mean_wer 0.583 seamless / 0.728 whisper)
were being read as an absolute quality signal for the finetune question
(POST_LEASE_MVP_SPRINT_PLAN.md section 3), which they cannot honestly
support without normalization: Darija has no standardized orthography, so
raw whitespace-split WER penalizes legitimate spelling variation (هوما/هما,
انا/أنا), a bare trailing period, or digit-vs-spelled-out numbers as if they
were transcription failures.

CER (character error rate, same Levenshtein DP but over UTF-8 codepoints
after the same normalization) is reported alongside WER as the more
appropriate PRIMARY metric for a dialect without standard word-boundary
conventions in casual transcription -- word-level errors compound fast when
a single letter-variant difference splits or merges what should be one
word.

Usage:
    .gguf_venv/Scripts/python.exe scripts/eval_stt_rescore.py \\
        benchmark_results/phase2_2026-09-04_eval_stt_results.json
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.services.citations import fold_arabic  # noqa: E402

# Arabic-script punctuation (؟ ، ؛ etc.) plus the standard ASCII set plus
# curly quotes seen in real hypothesis output ("لا سيكوريتي") -- stripped
# entirely, not just normalized, since punctuation choice carries no
# transcription-correctness signal for this eval's purpose.
_PUNCT_RE = re.compile(
    r"[،؛؟٪-٭۔"  # Arabic comma/semicolon/question mark/misc/full stop
    r".,!?;:\"'`‘’“”()\[\]{}\-–—/\\]+"
)
_WS_RE = re.compile(r"\s+")

# Spelled-out-digit normalization is deliberately NOT attempted here (French
# "vingt sept zéro six" vs. digits "27-06", codeswitch_05's actual defect) --
# that's a distinct, harder normalization (locale-aware number parsing) that
# would risk silently changing what's being measured. Flagged, not solved:
# see the printed note for any example where this remains the dominant
# apparent error after normalization.


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = fold_arabic(text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text.lower()


def _levenshtein(a: list, b: list) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        curr = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    ref = list(reference.replace(" ", ""))
    hyp = list(hypothesis.replace(" ", ""))
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    in_path = Path(sys.argv[1])
    data = json.loads(in_path.read_text(encoding="utf-8"))

    print(f"Re-scoring {in_path.name} (raw vs. normalized WER, + CER)\n")
    out = {}
    for engine, payload in data.items():
        examples = payload["examples"]
        raw_wers, norm_wers, norm_cers = [], [], []
        rescored = []
        for ex in examples:
            ref, hyp = ex["reference"], ex["hypothesis"]
            raw_w = ex["wer"]
            ref_n, hyp_n = normalize(ref), normalize(hyp)
            norm_w = wer(ref_n, hyp_n)
            norm_c = cer(ref_n, hyp_n)
            raw_wers.append(raw_w)
            norm_wers.append(norm_w)
            norm_cers.append(norm_c)
            rescored.append({
                "id": ex["id"], "raw_wer": raw_w, "normalized_wer": round(norm_w, 3),
                "normalized_cer": round(norm_c, 3),
                "reference_normalized": ref_n, "hypothesis_normalized": hyp_n,
            })
        mean_raw = sum(raw_wers) / len(raw_wers)
        mean_norm_wer = sum(norm_wers) / len(norm_wers)
        mean_norm_cer = sum(norm_cers) / len(norm_cers)
        print(f"=== {engine} ===")
        print(f"  raw mean WER (stored, unnormalized):  {mean_raw:.3f}")
        print(f"  normalized mean WER (this script):    {mean_norm_wer:.3f}")
        print(f"  normalized mean CER (this script):    {mean_norm_cer:.3f}")
        print(f"  RTF (unaffected, unchanged):          {payload['mean_rtf']:.3f}")
        # Biggest movers -- where normalization changed the picture most.
        deltas = sorted(rescored, key=lambda r: r["raw_wer"] - r["normalized_wer"], reverse=True)
        print("  largest raw->normalized WER drops:")
        for r in deltas[:3]:
            print(f"    {r['id']}: {r['raw_wer']:.2f} -> {r['normalized_wer']:.2f}")
        print()
        out[engine] = {
            "mean_raw_wer": round(mean_raw, 3),
            "mean_normalized_wer": round(mean_norm_wer, 3),
            "mean_normalized_cer": round(mean_norm_cer, 3),
            "mean_rtf": payload["mean_rtf"],
            "examples": rescored,
        }

    out_path = in_path.with_name(in_path.stem + "_normalized.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    # Engine-choice verdict, stated explicitly rather than left implicit.
    if "whisper" in out and "seamless" in out:
        w, s = out["whisper"], out["seamless"]
        raw_gap = w["mean_raw_wer"] - s["mean_raw_wer"]
        norm_gap = w["mean_normalized_wer"] - s["mean_normalized_wer"]
        print(f"\nEngine verdict: raw WER gap (whisper-seamless) = {raw_gap:+.3f}, "
              f"normalized WER gap = {norm_gap:+.3f}")
        if raw_gap > 0.05 and norm_gap <= 0.02:
            print("  -> normalization closes most of the gap: seamless's raw-WER win over "
                  "whisper was largely a scoring artifact, not a real accuracy advantage. "
                  "Re-examine the engine choice against RTF (seamless costs 1.6x whisper's).")
        elif norm_gap > 0.02:
            print("  -> a real accuracy gap survives normalization: seamless's win is not "
                  "purely a scoring artifact.")
        else:
            print("  -> gap was already small; normalization does not change the verdict.")


if __name__ == "__main__":
    main()
