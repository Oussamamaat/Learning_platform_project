"""
Synthetic Data Generator
───────────────────────
Generates raw text + Darija Q&A pairs about Moroccan road safety for:
  - RAG ingestion (as raw .txt files — ingestion.py handles chunking)
  - LoRA fine-tuning (as Darija JSONL training data)

Usage:
    python -m app.services.generate_data
"""

import json
from pathlib import Path

OUTPUT_DIR = Path("data/synthetic")


ROAD_SAFETY_TOPICS = [
    {
        "topic": "Signalisation Routière",
        "raw_text": (
            "La signalisation routière au Maroc comprend les panneaux de signalisation, "
            "les marquages au sol et les feux tricolores. Elle est régie par le Code de la "
            "Route marocain et vise à réguler la circulation, prévenir les accidents et guider "
            "les usagers de la route.\n\n"
            "Il existe trois catégories principales de panneaux de signalisation : les panneaux "
            "de danger (triangles rouges), les panneaux d'obligation (cercles bleus) et les "
            "panneaux d'interdiction (cercles rouges avec barre rouge). Chaque panneau a une "
            "signification précise que tout conducteur doit connaître.\n\n"
            "Les panneaux de danger sont des triangles à fond blanc avec une bordure rouge. "
            "Ils signalent un danger à proximité comme un virage, une intersection, un passage "
            "pour piétons ou un chantier. Le conducteur doit être vigilant et réduire sa vitesse.\n\n"
            "Les panneaux d'obligation sont bleus et imposent une action au conducteur. Par exemple, "
            "l'obligation de tourner à droite, l'obligation de céder le passage à un véhicule "
            "prioritaire, ou l'obligation de porter une ceinture de sécurité.\n\n"
            "Les panneaux d'interdiction sont ronds, blancs avec une bordure rouge et une barre "
            "rouge diagonale. Ils interdisent un comportement : interdiction de tourner à gauche, "
            "interdiction de dépasser, interdiction de s'arrêter, ou interdiction de circuler.\n\n"
            "Le marquage au sol comprend les lignes blanches continues (interdiction de dépasser), "
            "les lignes blanches en pointillés (dépassement autorisé), les zébras pour les passages "
            "piétons et les flèches de direction. Le conducteur doit respecter ces marquages.\n\n"
            "Les feux tricolores régissent les carrefours équipés. Le vert signifie le passage, "
            "l'orange (ou ambre) signifie ralentir et préparer l'arrêt, et le rouge signifie "
            "l'arrêt obligatoire. Au Maroc, la conduite à gauche est la norme."
        ),
    },
    {
        "topic": "Code de la Route",
        "raw_text": (
            "Le Code de la Route marocain est le texte législatif qui régit la circulation "
            "routière au Royaume du Maroc. Il définit les droits et obligations des usagers "
            "de la route, les règles de circulation et les sanctions applicables en cas "
            "d'infraction.\n\n"
            "La vitesse maximale autorisée au Maroc est de 120 km/h sur les autoroutes, "
            "100 km/h sur les routes nationales, 80 km/h sur les routes régionales et "
            "60 km/h en zone urbaine. Le non-respect de ces limites entraîne des amendes "
            "et des points sur le permis.\n\n"
            "L'âge minimum pour obtenir un permis de conduire au Maroc est de 18 ans. "
            "Le permis de conduire est délivré après une formation théorique et pratique, "
            "ainsi qu'un examen organisé par l'ONCFU ou une auto-école agréée.\n\n"
            "Tout conducteur doit avoir sur lui son permis de conduire, sa carte grise du "
            "véhicule, une attestation d'assurance et un contrôle technique en cours de "
            "validité. En cas de contrôle routier, ces documents doivent être présentés "
            "sur demande.\n\n"
            "L'alcool au volant est strictement interdit au Maroc. Le taux d'alcool autorisé "
            "est de 0,0 g/100ml de sang. Tout conducteur en état d'ébriété risque une amende, "
            "la suspension du permis et même une peine de prison en cas de récidive.\n\n"
            "L'usage du téléphone portable au volant est interdit sauf avec un kit mains "
            "libres. Le conducteur qui utilise son téléphone au volant risque une amende "
            "et des points de retrait sur son permis de conduire.\n\n"
            "La ceinture de sécurité est obligatoire pour le conducteur et tous les passagers "
            "avant du véhicule. Les enfants de moins de 10 ans doivent être installés sur "
            "des sièges spéciaux adaptés à leur taille et à leur poids."
        ),
    },
    {
        "topic": "Premiers Secours",
        "raw_text": (
            "Les premiers secours sont l'ensemble des gestes et actions qui visent à porter "
            "secours à une personne accidentée ou malade avant l'arrivée des secours "
            "professionnels. Au Maroc, le numéro d'urgence est le 150 pour la Police "
            "et les Pompiers.\n\n"
            "En cas d'accident de la route, la première chose à faire est de mettre en "
            "sécurité la zone : allumer les warnings, placer un triangle de signalisation "
            "à 50 mètres derrière le véhicule. Ne jamais déplacer une victime sauf en cas "
            "de danger immédiat.\n\n"
            "Alerter les secours en appelant le 150, mettre en sécurité la zone avec les "
            "warnings et le triangle. Ne jamais déplacer une victime sauf en cas de danger "
            "immédiat.\n\n"
            "Évaluer l'état de conscience de la victime. Si la victime est inconsciente mais "
            "respire, la placer en Position Latérale de Sécurité (PLS).\n\n"
            "La Position Latérale de Sécurité (PLS) consiste à placer une victime inconsciente "
            "qui respire sur le côté pour lui permettre de respirer librement et d'éviter "
            "qu'elle ne s'étouffe avec sa propre salive ou vomissements.\n\n"
            "L'arrêt cardiaque se manifeste par une perte de conscience, l'absence de "
            "respiration normale et l'absence de pouls. Il faut immédiatement appeler les "
            "secours (150) et pratiquer un massage cardiaque en attendant les secours.\n\n"
            "Les gestes de premiers secours incluent : alerter les secours (150), protéger "
            "la zone, évaluer l'état de la victime, pratiquer les gestes de survie (massage "
            "cardiaque, PLS) et surveiller la victime en attendant les secours."
        ),
    },
    {
        "topic": "Sécurité des Piétons",
        "raw_text": (
            "Les piétons ont le droit de priorité sur les passages piétons (zebra crossings). "
            "Tout conducteur doit ralentir et s'arrêter pour laisser traverser un piéton "
            "engagé sur un passage piéton. Le non-respect de cette règle est passible "
            "d'une amende.\n\n"
            "Les passages piétons sont signalés par des marquages zébra blancs au sol et "
            "souvent par un panneau de signalisation. Les piétons doivent utiliser ces "
            "passages pour traverser la route en toute sécurité.\n\n"
            "Les accidents impliquant des piétons sont parmi les plus graves au Maroc. "
            "En zone urbaine, la plupart des accidents de piétons se produisent aux "
            "carrefours et passages piétons. Il est essentiel pour les conducteurs de "
            "rester vigilants.\n\n"
            "Les piétons doivent être visibles, surtout la nuit. Ils doivent porter des "
            "vêtements clairs ou réfléchissants lorsqu'ils marchent sur la route. Les "
            "conducteurs doivent être particulièrement attentifs aux piétons dans les "
            "zones mal éclairées.\n\n"
            "Lorsqu'un piéton traverse la route en dehors d'un passage piéton, il doit "
            "s'assurer qu'aucun véhicule n'approche. Cependant, le conducteur reste "
            "responsable de la sécurité et doit être prêt à s'arrêter en cas de danger."
        ),
    },
    {
        "topic": "Sécurité des Enfants",
        "raw_text": (
            "Les enfants de moins de 10 ans ne peuvent pas voyager seuls dans un véhicule "
            "sans la supervision d'un adulte. Ils doivent être installés sur des sièges "
            "spéciaux adaptés à leur taille et à leur poids.\n\n"
            "Le siège auto est obligatoire pour les enfants de moins de 5 ans. Le siège "
            "doit être homologué et correctement installé sur le siège arrière du véhicule. "
            "L'utilisation du siège avant pour un enfant de moins de 12 ans est déconseillée.\n\n"
            "Les enfants doivent apprendre les règles de sécurité routière dès le plus jeune "
            "âge. Il est important de leur enseigner à traverser la route en regardant des "
            "deux côtés, à respecter les feux tricolores et à ne jamais courir sur la chaussée.\n\n"
            "Le transport d'enfants en van ou camion est interdit sauf si le véhicule est "
            "équipé de sièges spéciaux. Les enfants ne doivent jamais être laissés seuls "
            "dans un véhicule, surtout par temps chaud."
        ),
    },
    {
        "topic": "Conduite en Zone Urbaine",
        "raw_text": (
            "En zone urbaine, la vitesse maximale est de 60 km/h. Les conducteurs doivent "
            "être particulièrement attentifs aux piétons, aux vélos et aux deux-roues. "
            "Les passages piétons et les carrefours sont des zones à risque.\n\n"
            "Le stationnement est réglementé en zone urbaine. Il est interdit de stationner "
            "à moins de 5 mètres d'un carrefour, d'un passage piéton, d'un arrêt de bus "
            "ou devant une entrée de garage. Le stationnement gênant est passible d'une "
            "amende et d'une mise en fourrière.\n\n"
            "Les zones scolaires sont des zones où la vitesse est limitée à 20-30 km/h. "
            "Les conducteurs doivent être extrêmement prudents dans ces zones, surtout aux "
            "heures d'arrivée et de départ des écoles.\n\n"
            "L'utilisation des clignotants est obligatoire lors des virages, des changements "
            "de voie et des dépassements. Les conducteurs doivent signaler leurs intentions "
            "aux autres usagers de la route."
        ),
    },
    {
        "topic": "Conduite sur Route",
        "raw_text": (
            "Sur les routes nationales et régionales, la vitesse maximale est de 80 à "
            "100 km/h selon le type de route. Les dépassements ne sont autorisés que sur "
            "les portions de route où la ligne blanche est en pointillés.\n\n"
            "Lors des conditions météorologiques défavorables (pluie, brouillard, neige), "
            "les conducteurs doivent réduire leur vitesse et augmenter la distance de "
            "sécurité. Les feux de brouillard arrière doivent être utilisés en cas de "
            "visibilité réduite.\n\n"
            "Les véhicules de secours (ambulances, pompiers, police) ont la priorité sur "
            "tous les autres usagers. Lorsqu'un véhicule de secours approche avec ses "
            "sirènes allumées, les conducteurs doivent se dévier sur la droite et s'arrêter.\n\n"
            "Les camions et véhicules lourds doivent respecter des règles spécifiques : "
            "vitesse réduite, utilisation des bandes de circulation lentes sur les routes "
            "à forte déclivité, et arrêts réguliers pour éviter la surchauffe des freins."
        ),
    },
    {
        "topic": "Accidents de la Route",
        "raw_text": (
            "En cas d'accident de la route, il est obligatoire de s'arrêter et de porter "
            "secours aux victimes. Le conducteur responsable doit fournir ses coordonnées "
            "et celles de son assurance à l'autre conducteur. Un procès-verbal doit être "
            "établi par les forces de l'ordre.\n\n"
            "Le taux d'accidentalité au Maroc est parmi les plus élevés au monde. Les "
            "principales causes d'accidents sont : l'excès de vitesse, l'alcool au volant, "
            "le téléphone portable, le non-respect des feux et panneaux, et le dépassement "
            "imprudent.\n\n"
            "Les accidents de la route causent des milliers de décès et de blessés graves "
            "chaque année au Maroc. Les jeunes conducteurs de 18 à 30 ans sont les plus "
            "touchés par les accidents mortels.\n\n"
            "Après un accident, il faut : arrêter le véhicule, mettre les warnings, placer "
            "le triangle à 50 mètres, évaluer les victimes, appeler les secours (150) et "
            "ne pas déplacer les victimes gravement blessées."
        ),
    },
    {
        "topic": "Équipements de Sécurité",
        "raw_text": (
            "Le triangle de signalisation est obligatoire dans tout véhicule. Il doit être "
            "placé à 50 mètres derrière le véhicule en cas de panne ou d'accident sur route, "
            "et à 100 mètres sur autoroute.\n\n"
            "Le gilet jaune réfléchissant est obligatoire pour tout conducteur. En cas de "
            "panne ou d'accident, le conducteur doit enfiler son gilet jaune avant de sortir "
            "du véhicule pour assurer sa visibilité.\n\n"
            "Le kit de premiers secours est obligatoire dans tout véhicule au Maroc. Il doit "
            "contenir des pansements, des bandes, des ciseaux, des gants jetables et un "
            "guide de premiers secours.\n\n"
            "L'extincteur est obligatoire dans les véhicules de transport en commun et "
            "recommandé pour les véhicules particuliers. Il doit être vérifié régulièrement "
            "et changé selon la date de péremption.\n\n"
            "Les pneus doivent être en bon état et adaptés aux conditions de conduite. Les "
            "pneus usés augmentent considérablement les risques d'accidents, surtout par "
            "temps pluvieux."
        ),
    },
]


def generate_training_texts() -> list[dict]:
    """Generate raw text blocks for RAG ingestion (ingestion.py handles chunking)."""
    texts = []
    for topic_data in ROAD_SAFETY_TOPICS:
        texts.append({
            "content": topic_data["raw_text"],
            "source_name": f"road_safety_{topic_data['topic'].lower().replace(' ', '_')}",
            "topic": topic_data["topic"],
            "language": "fr",
        })
    return texts


def generate_qa_pairs() -> list[dict]:
    """Generate Darija instruction-response pairs for LoRA fine-tuning."""
    qa_pairs = []

    qa_pairs.append({
        "instruction": "شنو هي signalisation routière فالمغرب؟",
        "response": "ال signalisation routière فالمغرب كتشمل les panneaux de signalisation, les marquages au sol, et les feux tricolores. governed by le Code de la Route Moroccan. كتهدف pour régler la circulation, prévenir les accidents, et guider les usagers. There are 3 catégories principales : panneaux de danger (triangles rouges), panneaux d'obligation (cercles bleus), and panneaux d'interdiction (cercles rouges)."
    })
    qa_pairs.append({
        "instruction": "شحال هي la vitesse maximale f autoroute f l'Maroc?",
        "response": "La vitesse maximale f autoroute f l'Maroc hiya 120 km/h. F les routes nationales c'est 100 km/h, f les routes régionales 80 km/h, and f zone urbaine 60 km/h. L'infracción dial hadchi kaytsab amende w points f le permis."
    })
    qa_pairs.append({
        "instruction": "شنو هو zebra crossing?",
        "response": "Le zebra crossing, ou passage piéton, hiya la zone marquée de lignes blanches f la route li les piétons 3andhom le droit de traverser. Les conducteurs doivent s'arrêter pour laisser passer les piétons li msalqin f passage."
    })
    qa_pairs.append({
        "instruction": "شكون numero d'urgence f l'Maroc?",
        "response": "Le numéro d'urgence f l'Maroc hiya 150 pour la Police w les Pompiers. F cas d'accident, khassk t'appeler had l'nummero pour alerter les secours."
    })
    qa_pairs.append({
        "instruction": "كيفاش ndir la PLS?",
        "response": "La Position Latérale de Sécurité (PLS) : 1) Vérifier wach la victime respire, 2) Basculer la tête en arrière pour libérer les voies aériennes, 3) Mettre le bras dial la victime à 90 degrés, 4) Plier la jambe opposée, 5) Tourner la victime sur le côté, 6) Ouvrir la bouche pour drainage des liquides."
    })
    qa_pairs.append({
        "instruction": "Wach kayn el téléphoner au volant?",
        "response": "La, l'utilisation du téléphone portable au volant mhch el bahr au Maroc, sauf avec un kit mains libres. Le conducteur li kayelteb f telephone khassou ytkhallef amende w points retrait f le permis."
    })
    qa_pairs.append({
        "instruction": "Shhal men snina khass ncon f 3endek pour passer le permis?",
        "response": "L'âge minimum pour passer le permis au Maroc hiya 18 snina. Khassk t3ellem théorique w pratique f auto-école agréée,然后 teqdar tnedi l'examen."
    })
    qa_pairs.append({
        "instruction": "Shno les documents obligatoires f voiture?",
        "response": "Tout conducteur khassou 3ando : le permis de conduire, la carte grise du véhicule, l'attestation d'assurance valide, w le contrôle technique f validité. Les documents khass ybano f cas de contrôle routier."
    })
    qa_pairs.append({
        "instruction": "Shno ndir f cas d'accident?",
        "response": "F cas d'accident : 1) Arrêter le véhicule, 2) Allumer les warnings, 3) Placer le triangle à 50 mètres, 4) Évaluer les victimes, 5) Appeler les secours (150), 6) Ma tnemch les victimes gravement blessées, 7) Donner tes coordonnées w dial ton assurance."
    })
    qa_pairs.append({
        "instruction": "L'alcool au volant kayn el bahr?",
        "response": "Iyeh, l'alcool au volant strictement interdit au Maroc. Le taux autorisé hiya 0,0 g/100ml de sang. Tout conducteur f état d'ébriété ykoun的风险 amende, suspension du permis, w même prison f cas de récidive."
    })
    qa_pairs.append({
        "instruction": "Wach les enfants khasshom siège auto?",
        "response": "Iyeh, le siège auto obligatoire pour les enfants de moins de 5 ans. Khassou ykoun homologué w correctement installé sur le siège arrière. Le siège avant mhch el bahr pour un enfant de moins de 12 ans."
    })
    qa_pairs.append({
        "instruction": "Shno hiya el gilet jaune?",
        "response": "Le gilet jaune réfléchissant obligatoire pour tout conducteur au Maroc. F cas de panne ou d'accident, khass tlabes le gilet avant de sortir du véhicule pour assurer ta visibilité aux autres usagers."
    })
    qa_pairs.append({
        "instruction": "Shno huwa panneau de danger?",
        "response": "Les panneaux de danger hiya triangles à fond blanc avec bordure rouge. Kaychalw dangerous f la proximité : virage, intersection, passage piéton, chantier. Le conducteur khass ykoun vigilant w ynaqqes la vitesse."
    })
    qa_pairs.append({
        "instruction": "Shno huwa panneau d'interdiction?",
        "response": "Les panneaux d'interdiction hiya ronds, blancs avec bordure rouge w barre rouge diagonale. Kayhèmlo interdire un comportement : interdiction de tourner à gauche, dépasser, s'arrêter, ou circuler."
    })
    qa_pairs.append({
        "instruction": "Shno huwa panneau d'obligation?",
        "response": "Les panneaux d'obligation hiya bleus w kayfرضw une action : obligation de tourner à droite, céder le passage, porter la ceinture de sécurité, ou utiliser un casque pour les deux-roues."
    })
    qa_pairs.append({
        "instruction": "Shno hiya conduite défensive?",
        "response": "La conduite défensive c'est anticiper les dangers w adopter un comportement prudent f la route. Cela inclut : respecter les limitations de vitesse, maintenir une distance de sécurité, être vigilant aux autres usagers, w adapter sa conduite aux conditions météo."
    })
    qa_pairs.append({
        "instruction": "3lash la ceinture de sécurité mhina important?",
        "response": "La ceinture de sécurité obligatoire pour le conducteur w tous les passagers avant. Elle réduit de 50% le risque de décès f cas d'accident. Les enfants de moins de 10 ans khasshom ykounw f sièges spéciaux."
    })
    qa_pairs.append({
        "instruction": "Shno hiya les warnings?",
        "response": "Les warnings hiya feux de détresse du véhicule. Kayclignotent des deux côtés simultanément. Khass ytkounw activés f cas de panne, d'accident ou de danger pour signaler ta présence aux autres conducteurs."
    })
    qa_pairs.append({
        "instruction": "Shhal de distance de sécurité khass nkhalli?",
        "response": "La distance de sécurité khass tكون suffisante pour permettre l'arrêt f cas de danger. En règle générale, on recommande de garder au moins 2 secondes de distance par rapport au véhicule qui précède, w plus f cas de pluie ou de brouillard."
    })
    qa_pairs.append({
        "instruction": "Kifach ndir f pluie?",
        "response": "F cas de pluie : réduire la vitesse, augmenter la distance de sécurité, utiliser les feux de croisement, éviter les freinages brusques, w être particulièrement attentif aux aquaplaning (perte d'adhérence des pneus sur l'eau)."
    })

    return qa_pairs


def save_texts_as_files(texts: list[dict], output_dir: Path):
    """Save raw text as .txt files for RAG ingestion (ingestion.py handles chunking)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for text_data in texts:
        filename = text_data["topic"].lower().replace(" ", "_") + ".txt"
        filepath = output_dir / filename
        filepath.write_text(text_data["content"], encoding="utf-8")
        print(f"  Created: {filepath}")


def save_qa_as_jsonl(qa_pairs: list[dict], output_dir: Path):
    """Save Q&A pairs as JSONL for LoRA fine-tuning."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / "training_data.jsonl"

    with open(filepath, "w", encoding="utf-8") as f:
        for pair in qa_pairs:
            json.dump(pair, f, ensure_ascii=False)
            f.write("\n")

    print(f"  Created: {filepath} ({len(qa_pairs)} Q&A pairs)")


def main():
    print("=" * 60)
    print("Synthetic Data Generator — Moroccan Road Safety")
    print("=" * 60)

    # 1. Generate raw text files for RAG (ingestion.py handles chunking)
    print("\n[1/2] Generating raw text files for RAG ingestion...")
    texts = generate_training_texts()
    save_texts_as_files(texts, OUTPUT_DIR / "documents")
    print(f"  Total files: {len(texts)}")

    # 2. Generate Darija Q&A pairs for fine-tuning
    print("\n[2/2] Generating Darija Q&A pairs for fine-tuning...")
    qa_pairs = generate_qa_pairs()
    save_qa_as_jsonl(qa_pairs, OUTPUT_DIR / "training")
    print(f"  Total Q&A pairs: {len(qa_pairs)}")

    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Raw text files: data/synthetic/documents/ (for RAG ingestion)")
    print(f"  Training data: data/synthetic/training/training_data.jsonl (for LoRA)")
    print("=" * 60)


if __name__ == "__main__":
    main()
