"""
Third pass: fix remaining Darija-specific naming in the export cells
(GGUF filename, generation_config comment, TRAINING_REPORT fields,
final summary print) and add the base-vs-adapter result into the report.
"""
import json
from pathlib import Path

nb = json.loads(Path("kaggle_finetune_fr_v1.ipynb").read_text(encoding="utf-8"))
cells = nb["cells"]


def set_source(idx, text):
    cells[idx]["source"] = text.splitlines(keepends=True)


def sub(idx, old, new):
    src = "".join(cells[idx]["source"])
    assert old in src, f"cell {idx}: pattern not found: {old[:60]!r}"
    set_source(idx, src.replace(old, new))


# Cell 37: generalize the Atlas-Chat-specific comment (mechanism is the same
# for stock Gemma-2-9B -- it ships the same <eos>-only generation_config).
sub(37,
    "# Atlas-Chat's generation_config lists only <eos> (1). Our rows terminate with\n"
    "# <end_of_turn> (107), so unpatched the served model never stops — it just\n"
    "# starts role-playing the user. Its repetition_penalty is also the STRING\n"
    '# "1.2" upstream, which some loaders reject.',
    "# Gemma-2-9B's generation_config lists only <eos> (1), same as Atlas-Chat\n"
    "# (which inherits it). Our rows terminate with <end_of_turn> (107), so\n"
    "# unpatched the served model never stops — it just starts role-playing\n"
    "# the user. repetition_penalty is also a STRING upstream on some configs,\n"
    "# which some loaders reject.")

# Cell 38: GGUF filename.
sub(38, 'GGUF_NAME = "atlas-darija-tutor-q4_k_m.gguf"',
        'GGUF_NAME = "iblog-tutor-fr-q4_k_m.gguf"')

# Cell 39: add language + base-vs-adapter fields to the training report.
sub(39,
    '''report = {
    "base_model": BASE_MODEL,''',
    '''report = {
    "language": "fr",
    "dataset": "fr_v3_merged",
    "base_model": BASE_MODEL,''')

sub(39,
    '''    "prompt_parity_checked": PARITY_CHECKED,
    "smoke_test_issues": issues,
}''',
    '''    "prompt_parity_checked": PARITY_CHECKED,
    "smoke_test_issues": issues,
    "base_vs_adapter": {
        "probes_run": len(results),
        "regressions": len(regressions),
        "regression_detail": [
            {"component": r["component"], "adapter_fabricated": r["adapter_fabricated"]}
            for r in regressions
        ],
        "passed": not BASE_VS_ADAPTER_REGRESSION,
    },
}''')

# Cell 42: final summary mentions base-vs-adapter status too.
sub(42,
    '''print(f"  smoke-test issues: {len(issues)}")
print()''',
    '''print(f"  smoke-test issues: {len(issues)}")
print(f"  base-vs-adapter: {report['base_vs_adapter']['regressions']}/"
      f"{report['base_vs_adapter']['probes_run']} regressions "
      f"({'PASS' if report['base_vs_adapter']['passed'] else 'FAIL'})")
print()''')

nb["cells"] = cells
Path("kaggle_finetune_fr_v1.ipynb").write_text(
    json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8"
)
print("patched cells 37, 38, 39, 42")
