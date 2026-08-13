"""
One-shot generator for tests/data/retrieval_eval.jsonl.

Not part of the runtime pipeline -- run once to produce the eval file, then
the file is the artifact that ships. Kept so the composition (30 FR / 12
Arabic-script / 8 Arabizi, 15 with prior_turn, 5 out-of-domain) is visible
and reproducible rather than hand-edited JSONL with no record of intent.
"""
import json
from pathlib import Path

# Each row: id, domain, language ("fr"|"ary"|"arz" for arabizi), query,
# gold_sources (basenames under raw/shared/<domain>/text/), gold_substring
# (must appear verbatim in the retrieved context for citation to work),
# prior_turn (natural-language previous user turn, or None).
ROWS = [
    # ---------------- industrial: French (10) ----------------
    dict(id="ind-fr-01", domain="industrial", language="fr",
         query="Quelle est l'obligation du travailleur en matière de sécurité au travail ?",
         gold_sources=["1.1_code_du_travail_health_safety.md"], gold_substring="Art. 283"),
    dict(id="ind-fr-02", domain="industrial", language="fr",
         query="Que fait le comité de sécurité et d'hygiène dans une entreprise ?",
         gold_sources=["1.1_code_du_travail_health_safety.md"], gold_substring="Art. 327-334"),
    dict(id="ind-fr-03", domain="industrial", language="fr",
         query="Quel est le rôle de la CNSS dans la prévention des accidents du travail ?",
         gold_sources=["1.2_cnss_prevention_guides.md"], gold_substring="CNSS"),
    dict(id="ind-fr-04", domain="industrial", language="fr",
         query="C'est quoi une norme NM chez IMANOR ?",
         gold_sources=["1.3_imanor_nm_standards_explainer.md"], gold_substring="NM"),
    dict(id="ind-fr-05", domain="industrial", language="fr",
         query="Qu'est-ce que le cycle PDCA dans la norme ISO 45001 ?",
         gold_sources=["1.4_iso_45001_explainer.md"], gold_substring="PDCA"),
    dict(id="ind-fr-06", domain="industrial", language="fr",
         query="Quelles sont les 6 étapes de la consignation LOTO ?",
         gold_sources=["1.5_lockout_tagout_procedures.md"], gold_substring="consignation",
         prior_turn="Explique-moi le principe du lockout/tagout."),
    dict(id="ind-fr-07", domain="industrial", language="fr",
         query="Quels équipements protègent la tête et les yeux sur un chantier ?",
         gold_sources=["1.6_ppe_requirements.md"], gold_substring="EPI"),
    dict(id="ind-fr-08", domain="industrial", language="fr",
         query="Quelle est la différence entre un protecteur fixe et un protecteur mobile ?",
         gold_sources=["1.7_machine_guarding_basics.md"], gold_substring="protecteurs",
         prior_turn="Pourquoi faut-il protéger les machines dangereuses ?"),
    dict(id="ind-fr-09", domain="industrial", language="fr",
         query="Comment doit-on stocker des matières dangereuses en toute sécurité ?",
         gold_sources=["1.8_hazardous_materials_handling.md"], gold_substring="ADR"),
    dict(id="ind-fr-10", domain="industrial", language="fr",
         query="Quels sont les éléments d'un plan d'évacuation d'urgence ?",
         gold_sources=["1.9_emergency_evacuation_procedures.md"], gold_substring="évacuation",
         prior_turn="Que doit-on faire en cas d'alarme incendie ?"),

    # ---------------- industrial: Arabic script (4) ----------------
    dict(id="ind-ar-01", domain="industrial", language="ary",
         query="شنو هي التزامات المشغل فيما يخص نظافة أماكن الشغل؟",
         gold_sources=["1.11_ar_code_travail_salama.md"], gold_substring="المادة 281"),
    dict(id="ind-ar-02", domain="industrial", language="ary",
         query="واش المشغل خاصو يوفر معدات الوقاية الشخصية للعمال؟",
         gold_sources=["1.11_ar_code_travail_salama.md"], gold_substring="المادة 283",
         prior_turn="شنو كايقول القانون على السلامة فالمصنع؟"),
    dict(id="ind-ar-03", domain="industrial", language="ary",
         query="شنو هي المراحل الخمس ديال فصل مصادر الطاقة؟",
         gold_sources=["1.12_ar_procedure_consignation.md"], gold_substring="المادة 3"),
    dict(id="ind-ar-04", domain="industrial", language="ary",
         query="واش يقدر شخص آخر يرفع القفل غير العون اللي حطو؟",
         gold_sources=["1.12_ar_procedure_consignation.md"], gold_substring="المادة 4",
         prior_turn="علاش كنديرو الإغلاق والوسم قبل الصيانة؟"),

    # ---------------- industrial: Arabizi (4) ----------------
    dict(id="ind-arz-01", domain="industrial", language="ary",
         query="chno howa dyal les EPI li khass el 3amel ylabss?",
         gold_sources=["1.6_ppe_requirements.md"], gold_substring="EPI"),
    dict(id="ind-arz-02", domain="industrial", language="ary",
         query="kifach kaydir l'consignation dyal la machine qbel s-siyana?",
         gold_sources=["1.5_lockout_tagout_procedures.md"], gold_substring="consignation"),
    dict(id="ind-arz-03", domain="industrial", language="ary",
         query="3lach khass ndiro evacuation fach kayn alarme?",
         gold_sources=["1.9_emergency_evacuation_procedures.md"], gold_substring="évacuation",
         prior_turn="chno kayn f plan dyal evacuation?"),
    dict(id="ind-arz-04", domain="industrial", language="ary",
         query="wach l'mou9awil khass ywafr chi protecteur 3la la machine?",
         gold_sources=["1.7_machine_guarding_basics.md"], gold_substring="protecteurs"),

    # ---------------- securite_physique: French (9) ----------------
    dict(id="sec-fr-01", domain="securite", language="fr",
         query="Quelles sont les sanctions prévues par la loi 27.06 sur le gardiennage ?",
         gold_sources=["2.1_loi_27_06_regulation.md"], gold_substring="Art. 26-35"),
    dict(id="sec-fr-02", domain="securite", language="fr",
         query="Quelles sont les conditions pour exercer comme agent de gardiennage ?",
         gold_sources=["2.1_loi_27_06_regulation.md"], gold_substring="Art. 4-12",
         prior_turn="Que dit la loi 27.06 sur les activités de gardiennage ?"),
    dict(id="sec-fr-03", domain="securite", language="fr",
         query="Quelles interdictions s'appliquent aux agents de sécurité ?",
         gold_sources=["2.1_loi_27_06_regulation.md"], gold_substring="Art. 21-25"),
    dict(id="sec-fr-04", domain="securite", language="fr",
         query="Qu'est-ce que la réforme de l'article 193 du Code du travail prévue par le projet de loi 032.26 ?",
         gold_sources=["2.2_loi_032_26_reform.md"], gold_substring="193"),
    dict(id="sec-fr-05", domain="securite", language="fr",
         query="Quels sont les différents niveaux de contrôle d'accès dans un site ?",
         gold_sources=["2.3_guarding_access_control_protocols.md"], gold_substring="contrôle d'accès"),
    dict(id="sec-fr-06", domain="securite", language="fr",
         query="Comment se déroule une ronde de surveillance ?",
         gold_sources=["2.3_guarding_access_control_protocols.md"], gold_substring="Rondes de surveillance",
         prior_turn="Quelles sont les tâches d'un agent en début de poste ?"),
    dict(id="sec-fr-07", domain="securite", language="fr",
         query="Comment classifie-t-on un incident de sécurité ?",
         gold_sources=["2.4_incident_reporting_procedures.md"], gold_substring="Classification des incidents"),
    dict(id="sec-fr-08", domain="securite", language="fr",
         query="Que doit contenir la rubrique 'actions entreprises' d'un rapport d'incident ?",
         gold_sources=["2.4_incident_reporting_procedures.md"], gold_substring="Actions entreprises",
         prior_turn="Comment est structuré un rapport d'incident de sécurité ?"),
    dict(id="sec-fr-09", domain="securite", language="fr",
         query="Comment gère-t-on l'accès des visiteurs sur un site sécurisé ?",
         gold_sources=["2.3_guarding_access_control_protocols.md"], gold_substring="visiteurs"),

    # ---------------- securite_physique: Arabic script (4) ----------------
    dict(id="sec-ar-01", domain="securite", language="ary",
         query="شنو كايقول الباب الثاني ديال القانون 27.06 على شروط مزاولة المهنة؟",
         gold_sources=["2.6_ar_loi_27_06.md"], gold_substring="الباب الثاني"),
    dict(id="sec-ar-02", domain="securite", language="ary",
         query="شنو هوما حدود الصلاحيات ديال عون الحراسة؟",
         gold_sources=["2.6_ar_loi_27_06.md"], gold_substring="الباب الثالث",
         prior_turn="واش عون الحراسة عندو صلاحيات محدودة؟"),
    dict(id="sec-ar-03", domain="securite", language="ary",
         query="كيفاش كيتصرح بالحوادث الأمنية فالمسطرة الداخلية؟",
         gold_sources=["2.7_ar_procedure_controle_acces.md"], gold_substring="المادة 4"),
    dict(id="sec-ar-04", domain="securite", language="ary",
         query="واش التسجيلات ديال الكاميرات كيبقاو سريين؟",
         gold_sources=["2.7_ar_procedure_controle_acces.md"], gold_substring="المادة 6",
         prior_turn="شكون عندو الحق يشوف تسجيلات المراقبة؟"),

    # ---------------- securite_physique: Arabizi (2) ----------------
    dict(id="sec-arz-01", domain="securite", language="ary",
         query="chno khass ndiro bach nakhdo carte d'acces jdida?",
         gold_sources=["2.7_ar_procedure_controle_acces.md"], gold_substring="المادة 2"),
    dict(id="sec-arz-02", domain="securite", language="ary",
         query="wach kayn 3oqoubat f qanoun 27.06 3la li ma khassrch chart dyal gardiennage?",
         gold_sources=["2.1_loi_27_06_regulation.md"], gold_substring="Art. 26-35"),

    # ---------------- blockchain: French (8) ----------------
    dict(id="bc-fr-01", domain="blockchain", language="fr",
         query="Comment le projet de loi 42.25 définit-il les actifs numériques ?",
         gold_sources=["3.1_bill_42_25_draft.md"], gold_substring="actifs numériques",
         prior_turn="Le Maroc a-t-il un cadre légal pour les actifs numériques ?"),
    dict(id="bc-fr-02", domain="blockchain", language="fr",
         query="Quelles sont les obligations d'un PSAV au titre de la lutte anti-blanchiment ?",
         gold_sources=["3.2_fatf_aml_cft_guidance.md"], gold_substring="PSAV",
         prior_turn="C'est quoi la Recommandation 15 du GAFI ?"),
    dict(id="bc-fr-03", domain="blockchain", language="fr",
         query="Qu'est-ce que la Règle du Voyage (Travel Rule) du GAFI ?",
         gold_sources=["3.2_fatf_aml_cft_guidance.md"], gold_substring="Travel Rule"),
    dict(id="bc-fr-04", domain="blockchain", language="fr",
         query="Quelle a été la mise en garde de Bank Al-Maghrib sur les cryptomonnaies en 2017 ?",
         gold_sources=["3.3_bam_ammc_statements.md"], gold_substring="2017"),
    dict(id="bc-fr-05", domain="blockchain", language="fr",
         query="Quelles sont les caractéristiques fondamentales d'une blockchain ?",
         gold_sources=["3.4_blockchain_smart_contract_fundamentals.md"], gold_substring="blockchain"),
    dict(id="bc-fr-06", domain="blockchain", language="fr",
         query="Quels sont les cas d'usage des smart contracts en entreprise ?",
         gold_sources=["3.4_blockchain_smart_contract_fundamentals.md"], gold_substring="Smart Contracts",
         prior_turn="C'est quoi un smart contract ?"),
    dict(id="bc-fr-07", domain="blockchain", language="fr",
         query="Quelle est la différence entre un utility token et un security token ?",
         gold_sources=["3.5_consensus_mechanisms_tokens.md"], gold_substring="Security Tokens"),
    dict(id="bc-fr-08", domain="blockchain", language="fr",
         query="Pourquoi la norme ISO/TC 307 est-elle utile pour la formation ?",
         gold_sources=["3.6_iso_tc_307_standards_explainer.md"], gold_substring="ISO/TC 307"),

    # ---------------- blockchain: Arabic script (3) ----------------
    dict(id="bc-ar-01", domain="blockchain", language="ary",
         query="شنو كايهضر عليه الباب الثاني ديال القانون 42-25 على الترخيص؟",
         gold_sources=["3.7_ar_projet_loi_42_25.md"], gold_substring="الباب الثاني"),
    dict(id="bc-ar-02", domain="blockchain", language="ary",
         query="شنو هوما العقوبات المنصوص عليها فمشروع القانون ديال الأصول المشفرة؟",
         gold_sources=["3.7_ar_projet_loi_42_25.md"], gold_substring="الباب الرابع",
         prior_turn="شنو هي التزامات اليقظة ديال المتعاملين فالأصول المشفرة؟"),
    dict(id="bc-ar-03", domain="blockchain", language="ary",
         query="كيفاش كيتصنف المخاطر فمسطرة اليقظة ومكافحة غسل الأموال؟",
         gold_sources=["3.8_ar_procedure_kyc_aml.md"], gold_substring="المادة 3"),

    # ---------------- blockchain: Arabizi (1) ----------------
    dict(id="bc-arz-01", domain="blockchain", language="ary",
         query="chno khass ndiro bach ntsna b l'identite dyal l'client (KYC)?",
         gold_sources=["3.8_ar_procedure_kyc_aml.md"], gold_substring="المادة 2"),

    # ---------------- out-of-domain: must retrieve nothing above threshold (5) ----------------
    dict(id="ood-01", domain="industrial", language="fr",
         query="Quelle est la recette traditionnelle du tajine aux pruneaux ?",
         gold_sources=[], gold_substring=None),
    dict(id="ood-02", domain="securite", language="fr",
         query="Quel est le délai légal de notification d'une fuite de données personnelles (RGPD) ?",
         gold_sources=[], gold_substring=None),
    dict(id="ood-03", domain="blockchain", language="fr",
         query="Quel a été le score du dernier match du Raja Casablanca ?",
         gold_sources=[], gold_substring=None),
    dict(id="ood-04", domain="industrial", language="ary",
         query="شنو هو أحسن رياضة باش نخسر الوزن؟",
         gold_sources=[], gold_substring=None),
    dict(id="ood-05", domain="securite", language="ary",
         query="chno howa charh dyal token dyal utility f l'blockchain?",
         gold_sources=[], gold_substring=None),
]


def main():
    fr = sum(1 for r in ROWS if r["language"] == "fr")
    ar = sum(1 for r in ROWS if r["language"] == "ary" and all(ord(c) < 0x0600 or ord(c) > 0x06FF for c in r["query"]))
    # crude arabizi/arabic split: arabic-script rows contain Arabic-block characters
    arabic_script = sum(1 for r in ROWS if r["language"] == "ary" and any(0x0600 <= ord(c) <= 0x06FF for c in r["query"]))
    arabizi = sum(1 for r in ROWS if r["language"] == "ary" and not any(0x0600 <= ord(c) <= 0x06FF for c in r["query"]))
    prior = sum(1 for r in ROWS if r.get("prior_turn"))
    ood = sum(1 for r in ROWS if not r["gold_sources"])

    print(f"total={len(ROWS)} fr={fr} arabic_script={arabic_script} arabizi={arabizi} "
          f"prior_turn={prior} out_of_domain={ood}")
    assert len(ROWS) == 50, f"expected 50 rows, got {len(ROWS)}"
    assert fr == 30, f"expected 30 french, got {fr}"
    assert arabic_script == 12, f"expected 12 arabic-script, got {arabic_script}"
    assert arabizi == 8, f"expected 8 arabizi, got {arabizi}"
    assert prior == 15, f"expected 15 prior_turn, got {prior}"
    assert ood == 5, f"expected 5 out-of-domain, got {ood}"

    out_path = Path(__file__).resolve().parents[1] / "tests" / "data" / "retrieval_eval.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in ROWS:
            row.setdefault("prior_turn", None)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
