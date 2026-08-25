"""
Real-Mermaid-parser CI gate for app/services/diagram_render.py.

Standing rule from the archived feasibility study
(_archive/docs_superseded/STRATEGY_V2.md:251): "Every mermaid block must
parse. Real parser, not a regex." Node 25.6.1 is available in this
environment (verified during development -- see
docs/architecture/diagram-generation.md), so every fixture here is piped
through the actual `mermaid.parse()` via frontend/scripts/verify_mermaid.mjs,
not a hand-rolled syntax approximation.

Deliberate divergence from the archived plan: it put this parser check at
*serve* time. Diagrams are now generated from a validated JSON spec and
rendered deterministically (app.services.diagram_render), so the shape of
the output is already fixed and covered here -- a serve-time re-check would
add a Node subprocess and latency to every diagram request for no
additional safety. The gate belongs in CI, not the hot path.

Skips (does not fail) when `node` is missing or frontend/node_modules/mermaid
hasn't been installed, so `pytest` still runs clean on a bare checkout that
hasn't run `npm install` in frontend/. Written alongside the diagram-
generation feature (2026-08-23).
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from app.services import diagram_render
from app.services.diagrams import (
    FlowchartSpec, FlowNode, FlowEdge,
    SequenceSpec, SeqMessage,
    MindmapSpec, MindBranch,
    PieSpec, PieSlice,
    XySpec,
    _heal_flowchart,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = REPO_ROOT / "frontend"
VERIFY_SCRIPT = FRONTEND_DIR / "scripts" / "verify_mermaid.mjs"

_NODE = shutil.which("node")
_MERMAID_INSTALLED = (FRONTEND_DIR / "node_modules" / "mermaid").exists()

pytestmark = pytest.mark.skipif(
    not _NODE or not _MERMAID_INSTALLED,
    reason="node or frontend/node_modules/mermaid not available -- run `npm install` in frontend/",
)


def _fixtures() -> list[dict]:
    fixtures = []

    # Escaping-heavy content: quotes, '#', ampersands, French accents --
    # exactly the characters that broke the raw fine-tune's Mermaid output
    # per _archive/docs_superseded/STRATEGY_V2.md:124-142.
    flow = FlowchartSpec(
        title="Consignation & sécurité",
        caption="Les étapes de consignation avant intervention.",
        direction="TD",
        nodes=[
            FlowNode(id="a", label='Couper l\'alimentation "générale" #1'),
            FlowNode(id="b", label="Vérifier l'absence de tension"),
            FlowNode(id="c", label="Verrouiller (cadenas & étiquette)"),
        ],
        edges=[
            FlowEdge(source="a", target="b", label="puis"),
            FlowEdge(source="b", target="c", label='validation "OK"'),
        ],
    )
    fixtures.append({"kind": "flowchart", "source": diagram_render.render("flowchart", flow)})

    # A HEALED fixture -- proves salvaged output (a dangling edge dropped,
    # not a phantom node invented) still parses.
    dirty = FlowchartSpec(
        title="T", caption="Légende suffisamment longue pour ce test.",
        nodes=[FlowNode(id="a", label="A"), FlowNode(id="b", label="B")],
        edges=[FlowEdge(source="a", target="b"), FlowEdge(source="a", target="ghost")],
    )
    healed, _ = _heal_flowchart(dirty, max_nodes=14)
    assert healed is not None
    fixtures.append({"kind": "flowchart-healed", "source": diagram_render.render("flowchart", healed)})

    seq = SequenceSpec(
        title="Échange",
        caption="Communication entre l'opérateur et le superviseur.",
        participants=["Opérateur", "Superviseur & Sécurité"],
        messages=[
            SeqMessage(**{"from": "Opérateur", "to": "Superviseur & Sécurité", "text": 'Demande d\'autorisation "urgente"'}),
            SeqMessage(**{"from": "Superviseur & Sécurité", "to": "Opérateur", "text": "Ratio: 50% valide"}),
        ],
    )
    fixtures.append({"kind": "sequence", "source": diagram_render.render("sequence", seq)})

    mind = MindmapSpec(
        title="Sécurité",
        caption="Carte mentale des équipements de protection.",
        root='Sécurité "générale" #1',
        branches=[
            MindBranch(label="EPI & protections", children=["Casque", 'Gants "anti-chimiques"']),
            MindBranch(label="Procédures", children=["Consignation"]),
        ],
    )
    fixtures.append({"kind": "mindmap", "source": diagram_render.render("mindmap", mind)})

    pie = PieSpec(
        title='Répartition "2026" & incidents',
        caption="Répartition des incidents par type.",
        slices=[PieSlice(label="Chutes & glissades", value=33.333), PieSlice(label='Machines "mobiles"', value=66.667)],
    )
    fixtures.append({"kind": "pie", "source": diagram_render.render("pie", pie)})

    xy = XySpec(
        title='Évolution "annuelle" & incidents',
        caption="Évolution du nombre d'incidents par mois.",
        x_labels=["Janvier & février", 'Mars "2026"'],
        y_axis_label="Nombre d'incidents",
        chart_type="bar",
        values=[12, 8],
    )
    fixtures.append({"kind": "xy", "source": diagram_render.render("xy", xy)})

    return fixtures


def test_every_rendered_fixture_parses_with_the_real_mermaid_parser(tmp_path):
    fixtures = _fixtures()
    fixtures_path = tmp_path / "fixtures.json"
    fixtures_path.write_text(json.dumps(fixtures, ensure_ascii=False), encoding="utf-8")

    result = subprocess.run(
        [_NODE, str(VERIFY_SCRIPT), str(fixtures_path)],
        cwd=str(FRONTEND_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        # Warm, this subprocess takes ~2s. Cold, it is dominated by Node
        # loading mermaid + jsdom off disk for the first time -- measured
        # >60s on Windows with an untouched node_modules (Defender scanning
        # thousands of small files), which silently turned this gate into a
        # TimeoutExpired on the first `pytest` of a session while every
        # fixture actually parsed fine. A CI gate that fails cold and passes
        # warm is worse than no gate, so this is sized for the cold case.
        timeout=300,
    )

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    results = [json.loads(line) for line in lines]
    failures = [r for r in results if not r.get("ok")]

    assert len(results) == len(fixtures), (
        f"expected {len(fixtures)} results, got {len(results)}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert not failures, "Mermaid parser rejected generated output:\n" + "\n".join(
        f"  [{f['kind']}] {f['error']}" for f in failures
    )
