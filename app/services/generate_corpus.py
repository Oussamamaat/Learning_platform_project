"""
Knowledge Base Corpus Generator
───────────────────────────────
Generates the complete raw content corpus for the MVP knowledge base:
  - Industrial domain (8+ files — Code du Travail, ISO 45001, LOTO, PPE, etc.)
  - Sécurité physique (4+ files — Law 27.06, 032.26, guarding, incident reporting)
  - Blockchain      (6 files — Bill 42.25, FATF, BAM, fundamentals, consensus, ISO/TC 307)

Usage:
    python -m app.services.generate_corpus
"""

from pathlib import Path

RAW = Path("raw")

# ─── File descriptors ───────────────────────────────────────────────────────
# Each entry: (relative_path, title, content, rights_status, domain, scope)

FILES: list[dict] = []

def add(path: str, title: str, content: str, rights: str, domain: str, scope: str = "shared"):
    FILES.append(dict(
        path=path, title=title, content=content,
        rights=rights, domain=domain, scope=scope,
    ))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — INDUSTRIAL
# ═══════════════════════════════════════════════════════════════════════════════

add(
    "shared/industrial/text/1.1_code_du_travail_health_safety.md",
    "Code du Travail — Dispositions relatives à la santé et sécurité au travail",
    """# Code du Travail — Santé et Sécurité au Travail (Maroc)

**Source:** Bulletin Officiel — Loi n° 65-99 relative au Code du Travail (promulguée par Dahir n° 1-03-194 du 11 septembre 2003)
**Statut:** Public, à vérifier sur http://www.sgg.gov.ma/

---

## Obligations générales de l'employeur (Livre III, Titre I)

L'employeur est tenu de prendre toutes les mesures nécessaires pour garantir la sécurité et protéger la santé des travailleurs. Ces mesures incluent :
- La prévention des risques professionnels
- L'information et la formation des travailleurs
- La mise en place d'une organisation et de moyens adaptés

## Obligations du travailleur (Art. 283)

Chaque travailleur doit prendre soin de sa sécurité et de sa santé ainsi que de celles des autres personnes concernées par ses actes ou omissions au travail, conformément à sa formation et aux instructions de son employeur.

## Services de médecine du travail (Art. 313-326)

Tout établissement employant des salariés doit organiser un service de médecine du travail. L'effectif minimum est d'un médecin du travail à temps plein pour 2 000 salariés dans les entreprises industrielles.

## Comité de sécurité et d'hygiène (Art. 327-334)

Les établissements occupant au moins 50 salariés doivent instituer un comité de sécurité et d'hygiène chargé d'analyser les risques professionnels et de proposer des mesures de prévention.

## Déclarations obligatoires

- Déclaration des accidents du travail dans les 48 heures
- Registre de sécurité tenu à jour
- Affichage des consignes de sécurité

> **Note d'implémentation:** Ce fichier résume les dispositions clés. Se référer au texte officiel du Bulletin Officiel pour la rédaction exacte des articles et pour les mises à jour éventuelles (notamment les amendements post-2020).
""",
    "Public (Bulletin Officiel — à vérifier)",
    "industrial",
)

add(
    "shared/industrial/text/1.2_cnss_prevention_guides.md",
    "CNSS — Guides de prévention des accidents du travail",
    """# CNSS — Guides de Prévention des Accidents du Travail

**Source:** CNSS (Caisse Nationale de Sécurité Sociale) — publications officielles
**Statut:** Public
**URL:** https://www.cnss.ma/

---

## Rôle de la CNSS en prévention

La CNSS participe à la prévention des accidents du travail et des maladies professionnelles à travers :
- Des campagnes d'information et de sensibilisation
- Des guides thématiques par secteur d'activité
- Des statistiques et analyses des accidents déclarés
- Des actions de formation auprès des entreprises cotisantes

## Thèmes couverts par les guides CNSS

1. **Prévention des chutes de hauteur** — Travaux en hauteur, échafaudages, échelles
2. **Manutention manuelle et mécanique** — Gestes et postures, chariots élévateurs
3. **Risques électriques** — Intervention sur installations électriques, habilitation
4. **Incendie et explosion** — Stockage de produits dangereux, moyens d'extinction
5. **Risques chimiques** — Fiches de données de sécurité (FDS), étiquetage
6. **Bruit et vibrations** — Protection auditive, limitation de l'exposition
7. **Travail sur écran** — Ergonomie du poste informatique, pauses

## Taux de cotisation AT/MP

Le taux de cotisation accidents du travail / maladies professionnelles est variable selon le secteur d'activité et l'effectif de l'entreprise. Il est calculé sur la base de la masse salariale déclarée.

> **Note d'implémentation:** Résumé à partir des publications CNSS accessibles au public. Les documents complets sont disponibles sur le site officiel de la CNSS.
""",
    "Public (CNSS)",
    "industrial",
)

add(
    "shared/industrial/text/1.3_imanor_nm_standards_explainer.md",
    "IMANOR (NM) — Normes marocaines — Guide explicatif",
    """# Normes Marocaines (NM) — IMANOR — Guide Explicatif

**Source:** IMANOR (Institut Marocain de Normalisation)
**Statut:** Normes payantes — contenu original explicatif (ne reproduit pas le texte normatif)

---

## Qu'est-ce qu'une norme NM ?

Les normes marocaines (NM) sont des documents de référence établis par IMANOR qui définissent des spécifications techniques, des méthodes d'essai, des terminologies ou des exigences de qualité applicables aux produits, services et processus au Maroc.

## Normes NM pertinentes pour la sécurité industrielle

| Référence | Titre | Domaine |
|-----------|-------|---------|
| NM 03.1.001 | Systèmes de management de la santé et de la sécurité au travail — Exigences | SST |
| NM EN 166 | Protection individuelle de l'œil — Spécifications | EPI |
| NM EN 388 | Gants de protection contre les risques mécaniques | EPI |
| NM EN 397 | Casques de protection | EPI |
| NM EN 20345 | Chaussures de sécurité | EPI |

## Comment utiliser ces normes dans la formation

Les normes sont citées à titre de référence pour indiquer aux apprenants :
1. Les spécifications techniques que les équipements doivent respecter
2. Les critères de conformité pour les inspections
3. Les bases des procédures de certification

> **Important:** Les textes intégraux des normes NM sont protégés par le droit d'auteur et ne peuvent être reproduits. Ce document fournit un aperçu pédagogique. Pour l'application opérationnelle, se procurer les normes auprès d'IMANOR (www.imanor.gov.ma).
""",
    "Paid (IMANOR) — explication originale uniquement",
    "industrial",
)

add(
    "shared/industrial/text/1.4_iso_45001_explainer.md",
    "ISO 45001 — Structure et exigences — Contenu pédagogique original",
    """# ISO 45001:2018 — Systèmes de Management de la Santé et de la Sécurité au Travail

**Source:** Rédaction originale in-house
**Statut:** Contenu original — explication pédagogique de la norme (aucun texte normatif reproduit)

---

## Qu'est-ce que l'ISO 45001 ?

L'ISO 45001 est la norme internationale pour les systèmes de management de la santé et de la sécurité au travail (SST). Elle fournit un cadre permettant aux organisations de gérer les risques SST et d'améliorer leurs performances en matière de sécurité.

## Structure HLS (High-Level Structure)

La norme suit la structure commune à toutes les normes de systèmes de management ISO :

1. **Domaine d'application** — Périmètre du système de management
2. **Références normatives** — Documents de référence
3. **Termes et définitions** — Vocabulaire spécialisé
4. **Contexte de l'organisme** — Enjeux internes et externes
5. **Leadership et participation des travailleurs** — Engagement de la direction
6. **Planification** — Identification des risques et opportunités
7. **Support** — Ressources, compétences, communication
8. **Réalisation des activités opérationnelles** — Maîtrise des processus
9. **Évaluation des performances** — Surveillance, audit, revue de direction
10. **Amélioration** — Actions correctives, amélioration continue

## Cycle PDCA (Plan-Do-Check-Act)

| Phase | Activités clés |
|-------|----------------|
| PLAN | Identifier les dangers, évaluer les risques, établir des objectifs SST |
| DO | Mettre en œuvre les contrôles opérationnels, former le personnel |
| CHECK | Surveiller, mesurer, auditer, analyser les résultats |
| ACT | Prendre des actions correctives, améliorer en continu |

## Bénéfices attendus

- Réduction des accidents et incidents
- Conformité légale et réglementaire
- Amélioration de la culture sécurité
- Engagement des travailleurs
- Avantage compétitif (appels d'offres, certification)
""",
    "Original (in-house)",
    "industrial",
)

add(
    "shared/industrial/text/1.5_lockout_tagout_procedures.md",
    "Lockout/Tagout (LOTO) — Procédures de consignation — Contenu formation",
    """# Lockout/Tagout (LOTO) — Consignation des Énergies

**Source:** Rédaction originale in-house
**Statut:** Contenu original

---

## Qu'est-ce que la consignation ?

La consignation (Lockout/Tagout) est une procédure de sécurité qui consiste à verrouiller et étiqueter les sources d'énergie d'une machine ou d'un équipement avant d'effectuer des travaux de maintenance, de nettoyage ou de réparation.

## Les 6 étapes de la consignation

### Étape 1 — Préparation
- Identifier le personnel concerné
- Connaître le type d'énergie à isoler (électrique, mécanique, hydraulique, pneumatique, thermique, chimique)
- Disposer du kit de consignation adapté

### Étape 2 — Arrêt de la machine
- Mettre la machine à l'arrêt selon la procédure normale
- S'assurer que tous les mouvements sont arrêtés
- Attendre l'arrêt complet des éléments rotatifs

### Étape 3 — Isolation des sources d'énergie
- Ouvrir le sectionneur électrique et poser le cadenas
- Fermer les vannes et poser les cadenas
- Purger les circuits (pneumatiques, hydrauliques)
- Déposer les fusibles si nécessaire

### Étape 4 — Pose du cadenas et de l'étiquette
- Chaque intervenant pose son cadenas personnel (un seul cadenas par personne)
- L'étiquette mentionne : nom, prénom, date, heure, raison de la consignation
- L'étiquette est remplie lisiblement et attachée solidement

### Étape 5 — Vérification de l'absence d'énergie
- Tenter de redémarrer la machine (elle ne doit pas démarrer)
- Contrôler l'absence de tension avec un VAT (vérificateur d'absence de tension)
- Purger les énergies résiduelles
- Vérifier que tous les mouvements sont impossibles

### Étape 6 — Intervention et déconsignation
- Réaliser l'intervention en sécurité
- Retirer les outils et déchets
- Vérifier que personne n'est exposé
- Retirer les cadenas (chacun retire son propre cadenas)
- Réarmer la machine et tester

## Règles d'or

1. **Jamais** retirer le cadenas d'une autre personne
2. **Jamais** contourner un dispositif de consignation
3. Chaque intervenant pose **son propre** cadenas
4. La déconsignation n'est faite que par le poseur du cadenas
5. En cas d'équipe : un cadenas d'équipe en plus des cadenas individuels

## Situations d'exception

- Perte de clé : procédure spécifique avec le responsable sécurité
- Intervention urgente : dérogation écrite signée par le responsable
- Sous-traitants : coordination obligatoire avec l'équipe de maintenance interne
""",
    "Original (in-house)",
    "industrial",
)

add(
    "shared/industrial/text/1.6_ppe_requirements.md",
    "Équipements de Protection Individuelle (EPI) — Exigences et bonnes pratiques",
    """# Équipements de Protection Individuelle (EPI)

**Source:** Rédaction originale in-house
**Statut:** Contenu original

---

## Principes généraux

Les EPI sont le dernier niveau de protection dans la hiérarchie des mesures de prévention. Ils n'éliminent pas le danger mais protègent le travailleur contre les risques résiduels.

## Classification des EPI

### Protection de la tête
- **Casque de sécurité (NM EN 397)** — Chantiers, zones de manutention, hauteur
- Vérifier la date de péremption (généralement 5 ans)
- Remplacer après un choc violent même sans dommage visible

### Protection des yeux et du visage
- **Lunettes de sécurité** — Projections, poussières, UV
- **Écran facial** — Projections chimiques, meulage, plasma
- **Masque de soudage** — Arc électrique, rayonnement UV/IR

### Protection auditive
- **Bouchons d'oreilles** — Jusqu'à 30 dB d'atténuation
- **Casque antibruit** — Jusqu'à 35 dB d'atténuation
- Obligatoire au-delà de 85 dB(A) sur 8h

### Protection respiratoire
- **Demi-masque FFP1/FFP2/FFP3** — Poussières, aérosols
- **Masque complet avec cartouche** — Gaz, vapeurs chimiques
- Nécessite un test d'ajustement et une formation

### Protection des mains
- **NM EN 388** — Risques mécaniques (abrasion, coupure, déchirure, perforation)
- **NM EN 374** — Risques chimiques
- **NM EN 407** — Risques thermiques
- **NM EN 60903** — Risques électriques (gants isolants)

### Protection des pieds
- **NM EN ISO 20345** — Chaussures de sécurité (embout 200J)
- **NM EN ISO 20346** — Chaussures de protection (embout 100J)
- **NM EN ISO 20347** — Chaussures de travail (sans embout)

### Protection du corps
- Vêtements haute visibilité (NM EN ISO 20471)
- Vêtements de protection chimique (Type 3/4/5/6)
- Vêtements ignifugés (NM EN ISO 11612)

## Responsabilités

### Employeur
- Fournir les EPI gratuitement
- Assurer la formation à leur utilisation
- Vérifier le port effectif
- Tenir à jour les fiches de suivi

### Travailleur
- Utiliser les EPI conformément à la formation
- Signaler toute détérioration
- Ranger et entretenir les EPI
- Ne pas modifier les EPI

## Entretien et durée de vie

| EPI | Durée de vie indicative | Points de contrôle |
|-----|------------------------|-------------------|
| Casque | 5 ans | Fissures, déformation, UV |
| Lunettes | 1-2 ans | Rayures, fissures |
| Harnais | 5 ans / après choc | Coutures, connecteurs |
| Gants | Variable selon usage | Trous, déchirures |
""",
    "Original (in-house)",
    "industrial",
)

add(
    "shared/industrial/text/1.7_machine_guarding_basics.md",
    "Protection des machines — Principes fondamentaux",
    """# Protection des Machines — Principes Fondamentaux

**Source:** Rédaction originale in-house
**Statut:** Contenu original

---

## Pourquoi protéger les machines ?

Les machines représentent une source majeure d'accidents du travail :
- Zones de pincement, de cisaillement, de coupe
- Projections de matière
- Énergies dangereuses
- Mouvements inattendus

## Principes de conception des protecteurs

1. **Ne pas créer de nouveaux dangers** — Arêtes vives, points de pincement
2. **Ne pas être contourné facilement** — Fixation solide, outillage requis
3. **Permettre la maintenance** — Accessible sans démontage complet
4. **Résister aux conditions d'utilisation** — Robustesse, durabilité

## Types de protecteurs

### Protecteurs fixes
- Montés de façon permanente (soudure, boulons)
- Ne peuvent être retirés qu'avec un outil
- Avantage : protection permanente
- Inconvénient : maintenance moins accessible

### Protecteurs mobiles
- Articulés ou coulissants
- Associés à un dispositif de verrouillage
- La machine s'arrête si le protecteur est ouvert
- Exemples : portes de four, capots de broyeur

### Protecteurs réglables
- S'adaptent à différentes pièces ou opérateurs
- Réglage manuel sans outil
- Exemples : guides de sciage, butées

## Dispositifs de protection

### Barrières immatérielles (cellules photoélectriques)
- Détection de présence dans la zone dangereuse
- Arrêt immédiat de la machine
- Utilisation : presses, robots, lignes d'assemblage

### Tapis de sécurité
- Détection de présence par pression
- Arrêt de la machine si une personne est dans la zone
- Utilisation : robots, zones d'accès

### Commandes bimanuelles
- Nécessitent l'utilisation des deux mains
- Maintien des mains hors de la zone dangereuse
- Utilisation : presses, cisailles

### Dispositifs à validation
- Permettent un mouvement contrôlé en mode maintenance
- L'opérateur doit maintenir une pression
- Utilisation : réglage, apprentissage robot

## Marquage CE et conformité

- Directive Machines 2006/42/CE (reprise dans la réglementation marocaine)
- Marquage CE : conformité aux exigences essentielles de sécurité
- Notice d'instructions en français obligatoire
- Déclaration CE de conformité

## Vérifications périodiques

- Contrôle visuel quotidien par l'opérateur
- Vérification semestrielle par le service maintenance
- Inspection annuelle par un organisme agréé
- Tenue d'un registre de maintenance
""",
    "Original (in-house)",
    "industrial",
)

add(
    "shared/industrial/text/1.8_hazardous_materials_handling.md",
    "Manutention des matières dangereuses — Procédures et consignes",
    """# Manutention des Matières Dangereuses

**Source:** Rédaction originale in-house
**Statut:** Contenu original

---

## Classification des matières dangereuses (ONU)

| Classe | Type | Exemples |
|--------|------|----------|
| 1 | Matières explosibles | Dynamite, munitions |
| 2 | Gaz | Propane, oxygène, acétylène |
| 3 | Liquides inflammables | Essence, solvants, alcools |
| 4 | Solides inflammables | Soufre, phosphore |
| 5 | Matières comburantes | Peroxydes, nitrates |
| 6 | Matières toxiques | Pesticides, cyanures |
| 7 | Matières radioactives | Sources scellées |
| 8 | Matières corrosives | Acides, bases fortes |
| 9 | Matières dangereuses diverses | Amiante, piles au lithium |

## Étagement et signalisation ADR

- Pictogrammes de danger (NF EN ISO 7010)
- Plaque de danger orange sur les véhicules
- Fiche de sécurité (FDS) obligatoire pour chaque produit
- Document de transport ADR

## Stockage sécurisé

### Règles de compatibilité
- Ne pas stocker ensemble des matières incompatibles
- Exemple : acides ≠ bases, oxydants ≠ inflammables
- Distance de sécurité entre produits incompatibles : minimum 2 mètres

### Rétention
- Bac de rétention pour tout liquide dangereux
- Capacité minimale : 100% du plus grand contenant ou 25% du total
- Vidange et contrôle réguliers

### Ventilation
- Local ventilé (naturelle ou mécanique)
- Détection de gaz pour les produits dégageant des vapeurs toxiques
- Extraction basse pour les gaz lourds, haute pour les gaz légers

## Procédures d'urgence

1. **Fuite ou déversement** — Confiner la zone, utiliser le kit anti-déversement
2. **Incendie** — Utiliser l'extincteur adapté (poudre, CO₂, mousse)
3. **Contact cutané** — Rincer à l'eau 15 minutes, retirer les vêtements souillés
4. **Inhalation** — Évacuer à l'air libre, appeler les secours (150)
5. **Ingestion** — Ne pas faire vomir, appeler le centre antipoison

## Équipements obligatoires
- Kit anti-déversement à proximité immédiate
- Douche de sécurité et lave-œil opérationnels (vérification hebdomadaire)
- Extincteur adapté aux produits stockés
- Ventilation forcée dans les locaux confinés

## Formation obligatoire
- Formation à la lecture des FDS
- Formation à l'utilisation des EPI adaptés
- Formation aux gestes d'urgence
- Recyclage annuel obligatoire
""",
    "Original (in-house)",
    "industrial",
)

add(
    "shared/industrial/text/1.9_emergency_evacuation_procedures.md",
    "Procédures d'évacuation d'urgence — Plan et consignes",
    """# Procédures d'Évacuation d'Urgence

**Source:** Rédaction originale in-house
**Statut:** Contenu original

---

## Principes généraux

Tout établissement doit disposer d'un plan d'évacuation adapté à ses activités, à ses effectifs et à sa structure. Le plan doit être testé par des exercices réguliers.

## Éléments du plan d'évacuation

### 1. Détection et alarme
- Déclencheur manuel (bris de glace) à chaque issue
- Détecteurs automatiques (fumée, chaleur, gaz)
- Alarme sonore spécifique (signal différent de la sonnerie de travail)
- Alarme visuelle pour les malentendants

### 2. Consignes d'évacuation
**À faire :**
- Rester calme et ne pas courir
- Couper les machines et les sources d'énergie
- Suivre les itinéraires balisés
- Se rendre au point de rassemblement
- Compter les présences
- Attendre les consignes des secours

**À ne pas faire :**
- Utiliser les ascenseurs
- Revenir en arrière pour récupérer des objets
- S'arrêter aux vestiaires
- Rester aux abords des issues

### 3. Itinéraires et issues
- Minimum 2 issues distinctes par niveau
- Largeur minimale : 0,90 m (1,20 m pour les ERP)
- Portes s'ouvrant dans le sens de l'évacuation
- Blocs autonomes d'éclairage de sécurité (BAES)
- Signalisation conforme (NF EN ISO 7010)

### 4. Point de rassemblement
- À l'extérieur du bâtiment, distance minimale = hauteur du bâtiment + 5 m
- Accessible aux véhicules de secours
- Signalé par un panneau
- Abrité des intempéries si possible

## Exercices d'évacuation

| Type | Fréquence | Objectif |
|------|-----------|----------|
| Exercice partiel | Trimestriel | Tester un secteur, une équipe |
| Exercice général | Annuel | Évacuation complète, chronométrage |
| Exercice surprise | Variable | Tester la réactivité réelle |

## Rôles et responsabilités

### Guide-évacuation (serre-file)
- Vérifie que tous les occupants ont quitté la zone
- Ferme les portes sans les verrouiller
- Rassemble les personnes au point de rendez-vous

### Chargé d'évacuation
- Coordonne l'évacuation générale
- Vérifie les communications avec les secours
- Maintient la discipline au point de rassemblement

### Chef d'établissement
- Décide du déclenchement de l'alarme
- Organise les exercices
- Met à jour le plan d'évacuation

## Registre et documentation
- Consignes d'évacuation affichées dans chaque zone
- Plan d'évacuation à chaque niveau
- Registre des exercices (date, durée, observations)
- Fiche d'intervention des secours à l'entrée du bâtiment
""",
    "Original (in-house)",
    "industrial",
)

add(
    "tenant_placeholder/industrial/text/1.10_client_sop_synthetic.md",
    "[Synthétique] Procédure opératoire normalisée — Client industriel fictif",
    """# [DOCUMENT SYNTHÉTIQUE — NE PAS UTILISER AVEC DES DONNÉES RÉELLES]

## Procédure Opératoire Normalisée (PON) — Industrie Chimique « ChemCo Maroc »

**Client fictif :** ChemCo Maroc — Usine de production de détergents
**Document :** PON-SEC-042 — Intervention sur cuve de mélange

---

## 1. Objet
Cette procédure décrit les opérations de nettoyage et de maintenance de la cuve de mélange CM-204 située dans l'atelier de production.

## 2. Personnel concerné
- Opérateur de production (habilitation minimale : OP-2)
- Technicien de maintenance (habilitation minimale : MT-3)
- Responsable sécurité

## 3. Équipements de protection requis
- Casque de sécurité
- Lunettes de protection chimique
- Gants nitrile (résistance chimique classe 3)
- Combinaison anti-statique
- Chaussures de sécurité (S3)

## 4. Phases de l'intervention

### 4.1 Préparation
- Obtenir le permis de travail auprès du responsable sécurité
- Vérifier la disponibilité des EPI
- S'assurer que la cuve est vide et purgée

### 4.2 Consignation (LOTO)
- Arrêter l'agitateur
- Fermer les vannes d'alimentation (V-204A, V-204B)
- Poser les cadenas individuels
- Purger le circuit de chauffage
- Vérifier l'absence de pression

### 4.3 Nettoyage
- Ouvrir le trou d'homme
- Ventiler la cuve pendant 30 minutes minimum
- Contrôler l'atmosphère (explosimètre, détecteur H2S)
- Procéder au nettoyage par rinçage

### 4.4 Maintenance
- Inspection des soudures internes
- Contrôle de l'épaisseur de la cuve
- Remplacement du joint du trou d'homme si nécessaire

### 4.5 Remise en service
- Retirer les cadenas
- Refermer le trou d'homme
- Tester l'étanchéité
- Remplir le registre de maintenance

## 5. Anomalies à signaler immédiatement
- Corrosion anormale détectée
- Fissure ou déformation de la cuve
- Dysfonctionnement de la vanne de sécurité
- Odeur de produit chimique persistante après nettoyage

## 6. Documents associés
- PON-SEC-001 : Permis de travail
- PON-SEC-015 : Procédure LOTO générale
- FDS-204 : Fiche de données de sécurité du produit nettoyant
""",
    "Synthétique (placeholder — ne pas utiliser en production)",
    "industrial",
    "tenant_placeholder",
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SÉCURITÉ PHYSIQUE
# ═══════════════════════════════════════════════════════════════════════════════

add(
    "shared/securite_physique/text/2.1_loi_27_06_regulation.md",
    "Loi n° 27.06 — Réglementation des activités de gardiennage et de transport de fonds",
    """# Loi n° 27.06 — Activités de Gardiennage et de Transport de Fonds

**Source:** Bulletin Officiel
**Statut:** Public — à vérifier sur http://www.sgg.gov.ma/
**Note:** Cette loi encadre strictement les activités de sécurité privée au Maroc.

---

## Champ d'application (Art. 1-3)

La loi n° 27.06 régit les activités suivantes :
- La surveillance et le gardiennage de biens meubles et immeubles
- Le transport et la surveillance de fonds, bijoux et valeurs
- La protection des personnes
- L'installation et la maintenance de systèmes de sécurité

## Conditions d'exercice (Art. 4-12)

Toute entreprise exerçant ces activités doit :
- Obtenir un agrément délivré par l'autorité gouvernementale chargée de la sécurité
- Justifier d'une capacité technique et financière suffisante
- Employer des agents de sécurité agréés individuellement
- Souscrire une assurance responsabilité civile professionnelle

## Obligations des agents de sécurité (Art. 13-20)

- Être de nationalité marocaine
- Être âgé d'au moins 21 ans
- N'avoir aucune condamnation pénale incompatible avec la profession
- Suivre une formation professionnelle agréée
- Porter un uniforme réglementaire (pas de tenue similaire aux forces publiques)
- Être titulaire d'une carte professionnelle

## Interdictions (Art. 21-25)

- Port d'arme non autorisé
- Exercice d'activités de police judiciaire
- Détention de fichiers de données personnelles non autorisés
- Sous-traitance non déclarée au ministère de tutelle

## Sanctions (Art. 26-35)

- Amende de 10 000 à 100 000 MAD pour exercice illégal
- Suspension ou retrait de l'agrément en cas de manquement grave
- Peines d'emprisonnement en cas de violation des interdictions

> **Note d'implémentation:** Résumé des dispositions principales. Le texte officiel intégral est disponible au Bulletin Officiel.
""",
    "Public (Bulletin Officiel)",
    "securite_physique",
)

add(
    "shared/securite_physique/text/2.2_loi_032_26_reform.md",
    "Projet de loi n° 032.26 — Réforme du Code du Travail (sécurité privée)",
    """# Projet de Loi n° 032.26 — Réforme de l'Article 193 du Code du Travail

**Source:** Parlement / Bulletin Officiel
**Statut:** Projet de loi en cours d'examen (2026) — Ne pas présenter comme définitif

---

## Contexte

Le projet de loi n° 032.26 vise à amender l'article 193 du Code du Travail concernant l'aménagement du temps de travail des agents de sécurité privée. Il s'inscrit dans la réforme plus large du secteur de la sécurité privée au Maroc.

## Modifications proposées (à confirmer avec le texte final)

Les principales évolutions envisagées portent sur :
1. **Temps de travail** — Adaptation des règles de durée légale pour tenir compte de la nature spécifique du travail de gardiennage (postes de longue durée, travail de nuit)
2. **Travail de nuit** — Majoration spécifique pour les agents de sécurité travaillant la nuit (actuellement non encadrée de façon spécifique)
3. **Repos compensateur** — Nouveau dispositif de repos compensateur pour les heures supplémentaires
4. **Périodes d'astreinte** — Clarification du statut des périodes d'astreinte par rapport au temps de travail effectif

## Calendrier législatif

- Dépôt : Premier trimestre 2026
- Examen en commission : Deuxième trimestre 2026
- Vote prévu : Fin 2026 (sujet à modification)

> **Important:** Ce texte est un PROJET DE LOI en cours d'examen. Il n'a pas encore force de loi. Les informations ci-dessus sont basées sur les annonces publiques et les comptes rendus parlementaires. Mettre à jour dès la publication au Bulletin Officiel.
""",
    "Public (Projet de loi — statut à vérifier)",
    "securite_physique",
)

add(
    "shared/securite_physique/text/2.3_guarding_access_control_protocols.md",
    "Protocoles de gardiennage et de contrôle d'accès — Contenu formation",
    """# Protocoles de Gardiennage et Contrôle d'Accès

**Source:** Rédaction originale in-house
**Statut:** Contenu original

---

## Principes fondamentaux du contrôle d'accès

### Objectifs
- Vérifier l'identité des personnes entrant sur le site
- Contrôler les véhicules et marchandises
- Prévenir les intrusions et les vols
- Tenir un registre des entrées et sorties

### Niveaux de contrôle
1. **Niveau 1 — Entrée piétonne** — Vérification visuelle du badge, pointage
2. **Niveau 2 — Entrée véhicule** — Contrôle du véhicule, inspection du coffre
3. **Niveau 3 — Zone sensible** — Contrôle biométrique + fouille
4. **Niveau 4 — Accès restreint** — Escorte obligatoire, autorisation nominative

## Procédures de gardiennage

### Prise de poste
- Arriver 15 minutes avant le début du poste
- Vérifier l'état du matériel (talkie-walkie, badge, registre)
- Lire le cahier de consignes
- Signer la fiche de prise de poste

### Rondes de surveillance
- Fréquence définie par le plan de sécurité
- Parcours variable (ne pas être prévisible)
- Points de contrôle à poinçonner ou scanner
- En cas d'anomalie : stopper la ronde et alerter le poste central

### Gestion des visiteurs
- Enregistrement obligatoire (nom, société, heure d'entrée, heure de sortie)
- Remise d'un badge visiteur distinct (identifiable)
- Escorte obligatoire dans les zones sensibles
- Signature du registre à la sortie

### Gestion des colis et marchandises
- Vérification du bon de livraison
- Inspection visuelle (portail détecteur, miroir d'inspection)
- Enregistrement dans le registre des entrées marchandises
- Orientation vers le quai de déchargement approprié

## Communication et reporting

### Canaux de communication
- Talkie-walkie (canal dédié par zone)
- Téléphone d'urgence (ligne directe avec le PC sécurité)
- Application mobile de gardiennage (si déployée)

### Rapports
- Rapport quotidien (incidents, anomalies, statistiques)
- Rapport d'incident immédiat (dans l'heure suivant l'événement)
- Main-courante électronique ou papier

## Situations d'urgence

### Intrusion
1. Alerter le PC sécurité
2. Verrouiller les accès concernés
3. Observer sans intervenir (ne pas confronter)
4. Guider les forces de l'ordre

### Incendie
1. Déclencher l'alarme
2. Évacuer les visiteurs
3. Faciliter l'accès des pompiers
4. Pointer les présences au point de rassemblement

### Agression
1. Alerter les secours (police : 19, ambulance : 150)
2. Mettre en sécurité le personnel et le public
3. Préserver la scène (ne rien déplacer)
4. Rédiger un rapport circonstancié
""",
    "Original (in-house)",
    "securite_physique",
)

add(
    "shared/securite_physique/text/2.4_incident_reporting_procedures.md",
    "Procédures de signalement et de rapport d'incidents",
    """# Procédures de Signalement et Rapport d'Incidents

**Source:** Rédaction originale in-house
**Statut:** Contenu original

---

## Définition d'un incident de sécurité

Un incident de sécurité est tout événement qui :
- Menace ou compromet la sécurité des personnes, des biens ou des informations
- Constitue une violation des procédures établies
- Perturbe le fonctionnement normal des opérations

## Classification des incidents

| Niveau | Type | Délai de rapport | Destinataires |
|--------|------|------------------|---------------|
| **Niveau 1** | Anomalie mineure (porte non verrouillée, badge perdu) | Fin de poste | Supérieur hiérarchique |
| **Niveau 2** | Incident modéré (tentative d'intrusion, vol mineur) | 1 heure | Responsable sécurité |
| **Niveau 3** | Incident grave (agression, incendie, vol important) | Immédiat | Direction + forces de l'ordre |
| **Niveau 4** | Incident critique (prise d'otage, explosion, décès) | Immédiat | Direction générale + autorités |

## Structure du rapport d'incident

### En-tête
- Date et heure de l'incident
- Lieu précis (bâtiment, zone, étage)
- Agent ayant constaté l'incident
- Numéro de référence unique

### Description des faits
- Chronologie détaillée (quoi, quand, où, qui)
- Faits objectifs uniquement (pas d'interprétation)
- Témoins éventuels (nom, fonction, coordonnées)

### Actions entreprises
- Mesures immédiates prises
- Personnes alertées (nom, heure)
- Matériel ou équipement mobilisé

### Suivi
- Mesures correctives proposées
- Préconisations pour éviter la récidive
- Date de clôture prévue

## Circuits de signalement

### Voie hiérarchique normale
Agent → Chef de poste → Responsable sécurité → Direction

### Voie d'urgence (incidents graves)
Agent → Responsable sécurité (direct) → Direction générale

### Signalement externe
- Police : 19
- Pompiers : 15
- SAMU : 150
- DGSSI (cybersécurité) : selon procédure interne

## Confidentialité

- Les rapports d'incident sont confidentiels
- Diffusion limitée aux personnes ayant besoin d'en connaître
- Conservation minimale : 5 ans
- Destruction conforme à la réglementation (RGPD, Loi 09-08)

## Registre des incidents

Tout site doit tenir un registre chronologique des incidents comprenant :
- Numéro d'ordre
- Date et heure
- Nature de l'incident
- Suite donnée
- Date de clôture
""",
    "Original (in-house)",
    "securite_physique",
)

add(
    "tenant_placeholder/securite_physique/text/2.5_client_security_sop.md",
    "[Synthétique] Procédure de sécurité — Site client fictif",
    """# [DOCUMENT SYNTHÉTIQUE — NE PAS UTILISER AVEC DES DONNÉES RÉELLES]

## Plan de Sécurité du Site — Société « SecureLog Maroc »

**Client fictif :** SecureLog Maroc — Plateforme logistique
**Document :** PDS-SEC-001 — Version 2.3

---

## 1. Périmètre du site
- Entrepôt de stockage : 12 000 m²
- Bureaux administratifs : 800 m²
- Parking poids lourds : 60 places
- Accès unique principal + 1 accès pompiers

## 2. Effectif sécurité
- 1 chef de poste (8h-18h, jours ouvrés)
- 2 agents d'accueil (24h/24, 7j/7)
- 2 agents en ronde (ronde horaire, 24h/24)
- 1 opérateur vidéo (8h-20h)

## 3. Équipements
- 24 caméras IP (enregistrement 30 jours)
- Contrôle d'accès badge (portail + porte piétonne)
- Portique détecteur de métaux (entrée personnel)
- Talkies-walkies (canal 3)
- Système anti-intrusion périmétrique

## 4. Consignes particulières
- Interdiction stricte de quitter le poste sans être relevé
- Ronde extérieure toutes les 2 heures (pointage au lecteur RFID)
- Visiteurs : badge distinctif jaune, escorte obligatoire en zone de stockage
- Livraisons : contrôle du bon de livraison + inspection visuelle

## 5. Contacts d'urgence
| Service | Numéro |
|---------|--------|
| Police secteur | 05 22 12 34 56 |
| Pompiers | 15 |
| Responsable sécurité (24h) | 06 12 34 56 78 |
| Direction site | 06 98 76 54 32 |
""",
    "Synthétique (placeholder — ne pas utiliser en production)",
    "securite_physique",
    "tenant_placeholder",
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BLOCKCHAIN
# ═══════════════════════════════════════════════════════════════════════════════

add(
    "shared/blockchain/text/3.1_bill_42_25_draft.md",
    "Projet de loi n° 42.25 — Actifs numériques et blockchain (Maroc)",
    """# Projet de Loi n° 42.25 — Actifs Numériques et Technologie Blockchain

**Source:** Ministère de l'Économie et des Finances / Presse
**Statut:** Projet de loi en cours d'examen — Ne pas présenter comme définitif
**Date de la source:** 2025 (statut à revoir mensuellement)

---

## Contexte

Le Maroc s'est doté d'une feuille de route pour encadrer les actifs numériques et la technologie blockchain. Le projet de loi n° 42.25 constitue le premier cadre législatif dédié à ces technologies.

## Principales dispositions envisagées

### Définition des actifs numériques
- Distinction entre crypto-actifs, stablecoins et tokens utilitaires
- Définition juridique de la blockchain comme registre distribué
- Qualification des smart contracts

### Régime des prestataires
- Agrément obligatoire pour les plateformes d'échange
- Exigences de fonds propres minimaux
- Obligation de séparation des fonds clients
- Assurance responsabilité civile professionnelle

### Lutte contre le blanchiment (AML/CFT)
- Application des recommandations du GAFI (FATF)
- KYC renforcé pour les transactions > seuil défini
- Déclaration des transactions suspectes à l'UTRF
- Tenue de registres pendant 5 ans minimum

### Fiscalité
- Régime fiscal applicable aux plus-values sur cession d'actifs numériques
- TVA sur les prestations de services liées aux actifs numériques
- Déclaration obligatoire des avoirs détenus à l'étranger

## Calendrier prévisionnel
- Examen en Conseil de gouvernement : 2025-2026
- Adoption par le Parlement : À confirmer
- Entrée en vigueur : Après publication au Bulletin Officiel

> **Important:** Ce texte est un PROJET DE LOI dont le statut évolue rapidement. Les informations ci-dessus sont basées sur les annonces publiques et les communiqués de presse. À vérifier et mettre à jour régulièrement.
""",
    "Public (projet de loi — statut à vérifier mensuellement)",
    "blockchain",
)

add(
    "shared/blockchain/text/3.2_fatf_aml_cft_guidance.md",
    "GAFI (FATF) — Recommandations AML/CFT pour les actifs virtuels",
    """# GAFI/FATF — Recommandations AML/CFT Applicables aux Actifs Virtuels

**Source:** FATF (Financial Action Task Force / GAFI) — publications officielles
**Statut:** Public
**URL:** https://www.fatf-gafi.org/

---

## Recommandation 15 — Nouvelles technologies

La Recommandation 15 du GAFI stipule que les pays doivent évaluer les risques de blanchiment de capitaux et de financement du terrorisme associés aux activités ou opérations impliquant des actifs virtuels, et appliquer les mesures de lutte contre le blanchiment et le financement du terrorisme.

## Définitions clés (Glossaire GAFI)

- **Actif virtuel** : Représentation numérique de valeur qui peut être échangée ou transférée numériquement et qui peut être utilisée à des fins de paiement ou d'investissement
- **Prestataire de services sur actifs virtuels (PSAV)** : Toute personne physique ou morale qui exerce une ou plusieurs activités définies par le GAFI
- **VASP** : Virtual Asset Service Provider

## Obligations des PSAV/VASP

1. **Enregistrement ou agrément** — Obtention d'une licence dans le pays d'établissement
2. **KYC / Due Diligence** — Identification et vérification des clients
3. **Tenue de registres** — Conservation des données pendant 5 ans minimum
4. **Déclaration de transactions suspectes** — Obligation de déclaration à la cellule de renseignement financier
5. **Voyage Rule (Règle du voyage)** — Transmission des informations sur le donneur d'ordre et le bénéficiaire pour les transferts d'actifs virtuels

## La Règle du Voyage (Travel Rule)

Depuis juin 2019, le GAFI exige que les PSAV obtiennent, détiennent et transmettent les informations suivantes pour toute transaction d'actifs virtuels :
- Nom et adresse du donneur d'ordre
- Numéro de compte (ou identifiant de portefeuille)
- Nom et adresse du bénéficiaire
- Numéro de compte (ou identifiant de portefeuille) du bénéficiaire

## Mise en œuvre au Maroc

L'UTRF (Unité de Traitement du Renseignement Financier) est l'autorité compétente pour recevoir les déclarations de transactions suspectes. Le cadre législatif en cours d'élaboration (projet de loi 42.25) devrait transposer ces recommandations en droit marocain.

> **Note:** Résumé basé sur les publications GAFI accessibles au public. Les documents officiels sont disponibles sur www.fatf-gafi.org.
""",
    "Public (FATF)",
    "blockchain",
)

add(
    "shared/blockchain/text/3.3_bam_ammc_statements.md",
    "Bank Al-Maghrib / AMMC — Communications officielles sur les crypto-actifs",
    """# Bank Al-Maghrib et AMMC — Communications sur les Actifs Numériques

**Source:** Bank Al-Maghrib (www.bkam.ma) et AMMC (www.ammc.ma)
**Statut:** Public

---

## Bank Al-Maghrib — Position sur les crypto-actifs

### Mise en garde de 2017
Bank Al-Maghrib a émis plusieurs communiqués mettant en garde le public contre les risques associés aux crypto-actifs :
- Volatilité extrême
- Absence de garantie
- Risques de fraude et d'escroquerie
- Utilisation potentielle pour des activités illicites

### Communiqué de 2021
Réaffirmation de la mise en garde et rappel que les crypto-actifs n'ont pas cours légal au Maroc et ne bénéficient d'aucune garantie de l'État.

### Projets exploratoires (2022-2025)
BAM explore :
- Un éventuel CBDC (Central Bank Digital Currency / Monnaie numérique de banque centrale)
- Les applications de la blockchain pour les paiements et les transferts de fonds
- Les partenariats avec des fintechs marocaines

## AMMC — Position sur les actifs numériques

L'Autorité Marocaine du Marché des Capitaux a inclus dans sa stratégie 2023-2025 une réflexion sur l'encadrement des actifs numériques et des ICO (Initial Coin Offerings).

### Points clés
- Nécessité d'un cadre réglementaire adapté pour protéger les investisseurs
- Distinction entre tokens de paiement, tokens utilitaires et security tokens
- Appel à une coordination internationale

### Orientations actuelles
- Attente du cadre législatif (projet de loi 42.25)
- Coopération avec Bank Al-Maghrib et le Ministère des Finances
- Veille réglementaire internationale (ESMA, IOSCO)

## Conclusion pour la formation

Le cadre réglementaire marocain des actifs numériques est en construction. Les apprenants doivent comprendre que :
1. Les crypto-actifs ne sont pas réglementés à ce jour au Maroc
2. Les autorités mettent en garde contre les risques
3. Un cadre légal est en préparation (projet 42.25)
4. Les banques et institutions financières s'intéressent aux technologies blockchain sous-jacentes

> **Note:** Synthèse des communications publiques disponibles. Se référer aux sites officiels pour les textes intégraux et les mises à jour.
""",
    "Public (BAM / AMMC)",
    "blockchain",
)

add(
    "shared/blockchain/text/3.4_blockchain_smart_contract_fundamentals.md",
    "Blockchain et Smart Contracts — Notions fondamentales (contenu pédagogique original)",
    """# Blockchain et Smart Contracts — Notions Fondamentales

**Source:** Rédaction originale in-house
**Statut:** Contenu original

---

## Qu'est-ce qu'une blockchain ?

Une blockchain est un registre distribué, décentralisé et immuable qui enregistre des transactions de manière chronologique et sécurisée.

### Caractéristiques fondamentales
1. **Distribué** — Le registre est répliqué sur plusieurs nœuds (ordinateurs)
2. **Décentralisé** — Pas d'autorité centrale de contrôle
3. **Immuable** — Une fois écrites, les données ne peuvent être modifiées
4. **Transparent** — Les transactions sont visibles par tous les participants
5. **Sécurisé** — Cryptographie asymétrique et mécanismes de consensus

## Types de blockchain

| Type | Accès | Validation | Exemple |
|------|-------|------------|---------|
| **Publique** | Ouvert à tous | Tout nœud peut valider | Bitcoin, Ethereum |
| **Privée** | Restreint à une organisation | Nœuds autorisés | Hyperledger Fabric |
| **Consortium** | Géré par plusieurs organisations | Nœuds autorisés de plusieurs entités | R3 Corda |
| **Hybride** | Combinaison public/privé | Selon configuration | Dragonchain |

## Mécanismes de consensus

Le consensus est le processus par lequel les nœuds du réseau s'accordent sur l'état du registre.

- **Proof of Work (PoW)** : Les mineurs résolvent un problème mathématique (Bitcoin)
- **Proof of Stake (PoS)** : Les validateurs sont choisis selon leur mise (Ethereum 2.0)
- **Delegated Proof of Stake (DPoS)** : Les validateurs sont élus par les détenteurs de tokens
- **Practical Byzantine Fault Tolerance (PBFT)** : Tolérance aux pannes dans les blockchains privées
- **Raft** : Consensus simple pour blockchains d'entreprise

## Smart Contracts

Un smart contract est un programme autonome qui s'exécute automatiquement sur une blockchain lorsque des conditions prédéfinies sont remplies.

### Caractéristiques
- Auto-exécution : pas d'intervention humaine
- Transparence : le code est visible sur la blockchain
- Immuabilité : le code ne peut être modifié après déploiement
- Décentralisation : exécution par l'ensemble des nœuds

### Cas d'usage en entreprise
- **Gestion de la chaîne d'approvisionnement** : Traçabilité des produits
- **Finance décentralisée (DeFi)** : Prêts, échanges, assurances
- **Vote électronique** : Votes sécurisés et vérifiables
- **Propriété intellectuelle** : Gestion des droits d'auteur
- **Assurances** : Déclenchement automatique des indemnités

### Limites et risques
- Bugs dans le code (exploit DAO, 2016)
- Oracles (données externes potentiellement manipulables)
- Coûts de déploiement et d'exécution (gas fees)
- Problème de mise à jour et d'upgrade
- Irréversibilité des transactions
""",
    "Original (in-house)",
    "blockchain",
)

add(
    "shared/blockchain/text/3.5_consensus_mechanisms_tokens.md",
    "Mécanismes de consensus et tokens — Vue d'ensemble (contenu pédagogique original)",
    """# Tokens et Mécanismes de Consensus — Vue d'Ensemble

**Source:** Rédaction originale in-house
**Statut:** Contenu original

---

## Tokens : Définition et typologie

Un token est une unité de valeur émise sur une blockchain existante. Il représente un actif ou un droit.

### Catégories de tokens

#### 1. Cryptomonnaies (Coins)
- Monnaies numériques natives de leur blockchain
- Utilisées comme moyen d'échange et réserve de valeur
- Exemples : Bitcoin (BTC), Ether (ETH), XRP

#### 2. Tokens utilitaires (Utility Tokens)
- Donnent accès à un service ou un produit
- Utilisés dans les écosystèmes décentralisés
- Exemples : UNI (Uniswap), FIL (Filecoin)

#### 3. Security Tokens
- Représentent une participation dans un actif sous-jacent
- Soumis à la réglementation financière
- Exemples : Tokenisation immobilière, actions tokenisées

#### 4. Stablecoins
- Indexés sur une valeur stable (USD, EUR, or)
- Mécanismes : collateralisation, algorithmique
- Exemples : USDT (Tether), USDC (Circle), DAI (MakerDAO)

#### 5. NFTs (Non-Fungible Tokens)
- Tokens uniques et non interchangeables
- Utilisés pour l'art numérique, la propriété intellectuelle
- Standards : ERC-721, ERC-1155

## Mécanismes de consensus détaillés

### Proof of Work (PoW)
- **Principe** : Les mineurs résolvent un problème cryptographique
- **Avantages** : Sécurité éprouvée, décentralisation
- **Inconvénients** : Consommation énergétique élevée, scalabilité limitée
- **Utilisé par** : Bitcoin, Litecoin, Dogecoin (historiquement)

### Proof of Stake (PoS)
- **Principe** : Les validateurs mettent en jeu leurs tokens
- **Avantages** : Efficacité énergétique, scalabilité
- **Inconvénients** : Risque de centralisation (whales)
- **Utilisé par** : Ethereum 2.0, Cardano, Solana, Polkadot

### Delegated Proof of Stake (DPoS)
- **Principe** : Les détenteurs de tokens élisent des délégués
- **Avantages** : Rapidité, scalabilité
- **Inconvénients** : Centralisation (peu de délégués)
- **Utilisé par** : EOS, TRON, BitShares

### Practical Byzantine Fault Tolerance (PBFT)
- **Principe** : Tolérance aux fautes byzantines dans un réseau à nœuds connus
- **Avantages** : Finalité rapide, faible latence
- **Inconvénients** : Communication intensive, adapté aux petits réseaux
- **Utilisé par** : Hyperledger Fabric, R3 Corda, Zilliqa

## Comparaison des performances

| Critère | PoW | PoS | DPoS | PBFT |
|---------|-----|-----|------|------|
| TPS | 3-7 (BTC) / 15-30 (ETH) | 30-1000+ | 1000-5000+ | 1000-10000+ |
| Finalité | ~60 min | ~15 min | ~1 min | ~5 sec |
| Énergie | Très élevée | Faible | Faible | Très faible |
| Décentralisation | Élevée | Moyenne | Faible | Faible |
| Cas d'usage | Cryptomonnaie | Généraliste | DApps haute vitesse | Enterprise |

## Standards de tokens

- **ERC-20** — Standard de token fongible (Ethereum)
- **ERC-721** — Standard de token non fongible (NFT)
- **ERC-1155** — Standard multi-token (fongible + non fongible)
- **BEP-20** — Standard équivalent sur Binance Smart Chain
- **SPL** — Standard Solana Program Library (Solana)
""",
    "Original (in-house)",
    "blockchain",
)

add(
    "shared/blockchain/text/3.6_iso_tc_307_standards_explainer.md",
    "ISO/TC 307 — Normes blockchain et registres distribués — Guide explicatif",
    """# ISO/TC 307 — Normes Blockchain et Registres Distribués

**Source:** ISO (International Organization for Standardization) — Comité technique TC 307
**Statut:** Normes payantes — contenu original explicatif (ne reproduit pas le texte normatif)

---

## Qu'est-ce que l'ISO/TC 307 ?

Le comité technique ISO/TC 307 « Blockchain and distributed ledger technologies » est responsable de l'élaboration de normes internationales pour les technologies blockchain et les registres distribués (DLT).

## Normes publiées (sélection)

| Référence | Titre | Statut |
|-----------|-------|--------|
| ISO 22739:2024 | Blockchain and distributed ledger technologies — Vocabulary | Publiée |
| ISO/TR 23244:2020 | Privacy and personally identifiable information protection considerations | Publiée |
| ISO/TR 23455:2019 | Smart contracts — Overview of existing smart contract platforms | Publiée |
| ISO 23257:2022 | Reference architecture | Publiée |
| ISO/TS 23258:2021 | Taxonomy and ontology | Publiée |
| ISO/TS 23635:2022 | Guidance on governance | Publiée |

## Normes en développement

| Référence | Titre | État d'avancement |
|-----------|-------|-------------------|
| ISO/DIS 22747 | Interoperability | Enquête |
| ISO/AWI 22748 | Security management | En préparation |
| ISO/NP 22749 | Identity management | Proposition |
| ISO/NP 22750 | Smart contract — Legally binding | Proposition |

## Pourquoi ces normes sont importantes pour la formation

1. **Vocabulaire commun** (ISO 22739) — Définit les termes techniques que les apprenants doivent maîtriser
2. **Architecture de référence** (ISO 23257) — Structure standardisée pour comprendre comment les systèmes blockchain sont organisés
3. **Gouvernance** (ISO/TS 23635) — Cadre pour la gestion des systèmes blockchain en entreprise
4. **Smart contracts** (ISO/TR 23455) — Aperçu des plateformes et bonnes pratiques

## Comment utiliser cette norme en formation

Les normes ISO/TC 307 sont utilisées comme références pour :
1. Enseigner la terminologie standardisée (vocabulaire commun)
2. Présenter les architectures de référence
3. Discuter des considérations de gouvernance et de sécurité
4. Préparer les apprenants aux standards qui feront foi dans l'industrie

> **Important:** Les textes intégraux des normes ISO sont protégés par le droit d'auteur et ne peuvent être reproduits sans licence. Ce document fournit un aperçu pédagogique des normes disponibles sous l'ISO/TC 307.
""",
    "Paid (ISO) — explication originale uniquement",
    "blockchain",
)


# ─── Write everything ────────────────────────────────────────────────────────
def main():
    from app.services.generate_corpus_index import append_to_index, init_index, write_corpus_doc

    init_index()
    count = 0
    for f in FILES:
        path = RAW / f["path"]
        write_corpus_doc(path, f["content"])
        append_to_index(path, f["title"], f["rights"], f["domain"], f["scope"])
        count += 1
        print(f"  OK {path}")

    print(f"\nTotal files created: {count}")


if __name__ == "__main__":
    main()