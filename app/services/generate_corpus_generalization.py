"""
Domain-Generalization Corpus Generator
──────────────────────────────────────
Writes short reference documents for verticals the platform does not yet have
clients in — medical, legal, automotive, HR, logistics, hospitality.

Why these exist
───────────────
The fine-tune teaches a *behaviour*: Socratic scaffolding, French technical
vocabulary carried by Darija grammar, grounding, and refusal. The subject
matter is supposed to be irrelevant — it arrives through RAG at inference.

But a LoRA trained only on industrial safety, physical security, and
blockchain risks binding that behaviour to those topics, so a medical tenant
gets `la consignation LOTO` in a cardiology answer. Training against a handful
of unrelated verticals teaches the model that the pattern is domain
independent.

These are deliberately short: they are vehicles for behaviour, not knowledge
bases. Their content is generic professional reference material, and no client
is expected to use them directly.

Both scripts are separate: `generate_corpus.py` builds the real client corpus,
`generate_corpus_arabic.py` the Arabic-script sources.

Usage:
    python -m app.services.generate_corpus_generalization
"""

from pathlib import Path

RAW = Path("raw")
SCOPE = "generalization"

DOCS: list[tuple] = []


def add(domain: str, filename: str, content: str):
    DOCS.append((domain, filename, content))


# ── medical ────────────────────────────────────────────────────────────────

add(
    "medical",
    "med_1_hygiene_hospitaliere.md",
    """# Hygiène hospitalière — Précautions standard

**Référence interne :** Protocole HYG-01

## 1. Hygiène des mains

L'hygiène des mains constitue la mesure la plus efficace pour prévenir les
infections associées aux soins. La friction hydro-alcoolique est réalisée
avant et après chaque contact avec un patient.

## 2. Équipements de protection

Le port des gants est obligatoire lors de tout contact avec un liquide
biologique. Le masque chirurgical et la surblouse sont requis selon le niveau
de risque évalué.

## 3. Gestion des déchets

Les déchets d'activités de soins à risque infectieux sont éliminés dans des
conteneurs dédiés. Les objets piquants et tranchants sont déposés
immédiatement dans un collecteur rigide.

## 4. Traçabilité

Chaque acte de désinfection fait l'objet d'un enregistrement dans le registre
du service.
""",
)

add(
    "medical",
    "med_2_administration_medicaments.md",
    """# Administration des médicaments — Règles de sécurité

**Référence interne :** Protocole MED-04

## 1. La règle des cinq

Avant toute administration, l'infirmier vérifie : le bon patient, le bon
médicament, la bonne dose, la bonne voie d'administration, et le bon moment.

## 2. La prescription

Aucun médicament n'est administré sans une prescription médicale écrite,
datée et signée. La prescription orale est réservée aux situations d'urgence
et régularisée dans les meilleurs délais.

## 3. La posologie

La posologie est adaptée au poids et à la fonction rénale du patient. Tout
doute sur un calcul de dose impose une double vérification par un second
professionnel.

## 4. La traçabilité

L'administration est consignée immédiatement après l'acte. Toute erreur ou
tout événement indésirable fait l'objet d'une déclaration.
""",
)


# ── legal ──────────────────────────────────────────────────────────────────

add(
    "legal",
    "leg_1_contrat_travail.md",
    """# Le contrat de travail — Notions essentielles

**Référence :** Support de formation juridique interne

## 1. La formation du contrat

Le contrat de travail suppose trois éléments : une prestation de travail, une
rémunération, et un lien de subordination juridique.

## 2. Les types de contrat

Le contrat à durée indéterminée constitue la forme normale de la relation de
travail. Le contrat à durée déterminée est limité aux cas prévus par la loi et
comporte un terme précis.

## 3. La période d'essai

La période d'essai permet à chaque partie d'apprécier la relation. Sa durée
maximale varie selon la catégorie professionnelle du salarié.

## 4. La rupture

La rupture à l'initiative de l'employeur exige un motif valable et le respect
d'une procédure comprenant la convocation, l'entretien préalable, et la
notification écrite.
""",
)

add(
    "legal",
    "leg_2_protection_donnees.md",
    """# Protection des données personnelles — Obligations

**Référence :** Support de formation conformité

## 1. Le principe de finalité

Les données personnelles sont collectées pour une finalité déterminée,
explicite et légitime. Elles ne peuvent être traitées ultérieurement de
manière incompatible avec cette finalité.

## 2. La minimisation

Seules les données strictement nécessaires à la finalité poursuivie sont
collectées. La durée de conservation est limitée à ce qui est nécessaire.

## 3. Les droits des personnes

Toute personne dispose d'un droit d'accès, de rectification, d'opposition et
de suppression concernant les données la concernant.

## 4. La sécurité

Le responsable du traitement met en œuvre les mesures techniques et
organisationnelles appropriées. Toute violation de données fait l'objet d'une
notification à l'autorité compétente.
""",
)


# ── automotive ─────────────────────────────────────────────────────────────

add(
    "automotive",
    "auto_1_systeme_freinage.md",
    """# Le système de freinage — Diagnostic et entretien

**Référence interne :** Fiche technique MEC-07

## 1. Les composants

Le système de freinage comprend le maître-cylindre, les étriers, les
plaquettes, les disques, et le circuit hydraulique.

## 2. Le contrôle des plaquettes

L'épaisseur minimale des plaquettes est vérifiée à chaque révision. Une
épaisseur inférieure au seuil constructeur impose un remplacement immédiat.

## 3. Le liquide de frein

Le liquide de frein est hygroscopique et absorbe l'humidité. Il est remplacé
selon la périodicité prescrite par le constructeur, indépendamment du
kilométrage.

## 4. La purge du circuit

La présence d'air dans le circuit provoque une pédale spongieuse. La purge est
réalisée dans l'ordre prescrit, en commençant par l'étrier le plus éloigné du
maître-cylindre.
""",
)

add(
    "automotive",
    "auto_2_diagnostic_electronique.md",
    """# Diagnostic électronique — Méthode

**Référence interne :** Fiche technique MEC-12

## 1. La lecture des codes défaut

La valise de diagnostic est connectée à la prise OBD du véhicule. Les codes
défaut sont relevés avant toute intervention et consignés sur l'ordre de
réparation.

## 2. L'interprétation

Un code défaut désigne un symptôme, non une pièce défectueuse. Le diagnostic
impose de vérifier le faisceau, les connecteurs et la masse avant de conclure
au remplacement d'un capteur.

## 3. Les données en temps réel

L'analyse des paramètres en fonctionnement permet de distinguer un défaut
permanent d'un défaut intermittent.

## 4. La validation

Après réparation, les codes sont effacés et un essai routier est réalisé pour
confirmer la disparition du défaut.
""",
)


# ── human resources ────────────────────────────────────────────────────────

add(
    "rh",
    "rh_1_entretien_professionnel.md",
    """# L'entretien professionnel — Conduite

**Référence interne :** Guide RH-02

## 1. L'objet

L'entretien professionnel est consacré aux perspectives d'évolution du salarié
et à ses besoins en formation. Il se distingue de l'entretien d'évaluation de
la performance.

## 2. La préparation

Le responsable et le collaborateur préparent l'entretien séparément à partir
d'une trame commune communiquée à l'avance.

## 3. La conduite

L'entretien se déroule dans un lieu neutre, sans interruption. Le responsable
adopte une posture d'écoute et fonde ses observations sur des faits.

## 4. La formalisation

Un compte rendu est rédigé et signé par les deux parties. Les engagements pris
font l'objet d'un suivi lors de l'entretien suivant.
""",
)


# ── logistics ──────────────────────────────────────────────────────────────

add(
    "logistique",
    "log_1_gestion_stock.md",
    """# Gestion des stocks — Principes

**Référence interne :** Procédure LOG-03

## 1. La réception

Tout produit réceptionné fait l'objet d'un contrôle quantitatif et qualitatif
avant mise en stock. Les écarts sont consignés sur le bon de réception.

## 2. La rotation

La règle du premier entré, premier sorti s'applique aux produits soumis à une
date de péremption.

## 3. L'inventaire

L'inventaire tournant permet de vérifier la fiabilité du stock sans
interrompre l'activité. Les écarts font l'objet d'une analyse.

## 4. Le stockage

Les conditions de stockage respectent les prescriptions du fournisseur,
notamment la température et l'hygrométrie. Les produits incompatibles sont
séparés physiquement.
""",
)


# ── hospitality ────────────────────────────────────────────────────────────

add(
    "hotellerie",
    "hot_1_securite_alimentaire.md",
    """# Sécurité alimentaire — Bonnes pratiques

**Référence interne :** Procédure HAC-01

## 1. La chaîne du froid

La chaîne du froid est maintenue de la réception au service. Les températures
sont relevées deux fois par jour et consignées.

## 2. La marche en avant

Le circuit des denrées suit une progression du secteur sale vers le secteur
propre, sans croisement.

## 3. Les températures de cuisson

La cuisson à cœur atteint la température prescrite selon la nature du produit.
Le refroidissement rapide est réalisé dans le délai réglementaire.

## 4. La traçabilité

Les étiquettes des produits sont conservées. Un plat témoin est prélevé et
conservé pour chaque préparation servie.
""",
)


def main():
    from app.services.generate_corpus_index import write_corpus_doc

    written = 0
    for domain, filename, content in DOCS:
        target = RAW / SCOPE / domain / "text" / filename
        write_corpus_doc(target, content)
        print(f"  wrote {target}")
        written += 1
    domains = sorted({d for d, _, _ in DOCS})
    print(f"\n{written} documents across {len(domains)} generalization domains: "
          f"{', '.join(domains)}")


if __name__ == "__main__":
    main()
