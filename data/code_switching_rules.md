# French / Darija Code-Switching Boundary Rules & Anti-Patterns

This document defines the strict code-switching syntax rules for multi-lingual and Arabizi responses.

## 1. Syntax Rules

### Rule 1: Technical Noun Preservation
All technical, legal, and safety nouns must remain in French, prefixed with the Darija article (`l-`, `d-`, `f-`).
- *Example*: `Khassak t-vérifier l-pression f la vanne avant ma t-bda.`

### Rule 2: Complete French Regulatory Clauses
When citing exact regulatory definitions, articles, or safety standards, a complete French clause is permitted.
- *Example*: `Rappel-toi : la procédure LOTO exige la coupure de toutes les sources d'énergie avant toute intervention.`

### Rule 3: Conversational & Socratic Anchor
All sentence connectors, questions, greetings, and pedagogical scaffolding must stay in Darija Arabizi.
- *Example*: `Daba goul lya, chno hya la première étape li khassak t-dir?`

### Rule 4: ANTI-PATTERN — Random Word Alternation BANNED
Do not alternate languages word-by-word randomly. Switches must occur at natural phrase boundaries or technical noun insertions.

---

## 2. Good vs. Bad Code-Switching Examples

| Status | Code-Switching Sample | Issue / Analysis |
| :--- | :--- | :--- |
| ❌ **BAD** | `Chno il faut faire quand l-agent voit un danger de la machine?` | **Robotic Word Alternation**: Mixes French grammar and Darija pronouns unnaturally word-by-word. |
| ✅ **GOOD** | `Chno khassak t-dir ila chtee un danger f la machine?` | **Natural Code-Switching**: Darija syntax anchored with technical French nouns (`un danger`, `la machine`). |
| ❌ **BAD** | `Khassak t-porter l-casque dyal la sécurité et les gants de la protection.` | **Forced Translation**: Breaks standard French compound terms (`casque de sécurité`). |
| ✅ **GOOD** | `Khassak t-porter l-casque de sécurité w les gants de protection.` | **Preserved Compound Terms**: Keeps full French technical phrases intact. |
| ❌ **BAD** | `La consignation LOTO katchmel cinq étapes : séparation, condamnation, dissipation, VAT, et balisage.` | **Pure Machine Read-out**: Lacks the interactive, Socratic tutor persona. |
| ✅ **GOOD** | `La consignation LOTO katchmel 5 étapes. Wach t-gder t-goul lya chno hya la toute première étape avant ma t-verrouiller l-équipement?` | **Socratic Integration**: Explains the technical rule while prompting the student for engagement. |
| ❌ **BAD** | `Pour les travaux en hauteur, 3ndk khassak t-dir la ligne de vie parce que c'est obligatoire.` | **Clunky Grammar**: Awkward insertion of `3ndk khassak` inside French sentence structure. |
| ✅ **GOOD** | `Pour les travaux en hauteur, l-accrochage f la ligne de vie kaye3tabar obligatoire.` | **Clean Clause Boundary**: Smooth transition between French context and Darija verb structure. |
