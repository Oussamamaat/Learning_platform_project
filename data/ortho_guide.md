# Standardized Arabizi & Industrial Orthography Guide

This document establishes the canonical spelling, numeral mappings, capitalization, and technical terminology rules for all synthetic training row generation.

## 1. Arabizi Numeral Mapping Table

| Numeral | Phonetic / Arabic Glyph | Arabizi Usage Example | Canonical Arabizi | Forbidden Variants |
| :--- | :--- | :--- | :--- | :--- |
| **2** | Hamza (ء / أ) | 2asli, 2aleik, so2al | `so2al` | so'al, soal |
| **3** | Ayn (ع) | 3afak, m3a, 3ndk | `3afak` | 'afak, aafak |
| **5** | Kha (خ) | 5ass, 5tar, 5rej | `khass` *(prefer kh, allow 5)* | k5ass, xass |
| **7** | Ha (ح) | 7ta, 7sen, 7aja | `7ta` | hta, 7'ta |
| **8** | Ghayn (غ) | 8ir, 8ali, ma8rib | `8ir` *(prefer 8, allow gh)* | 8'ir, g'hir |
| **9** | Qaf (ق) | 9bl, 9anoon, 9rab | `9bl` | qbl, kbl |

---

## 2. Capitalization & Punctuation Rules

1. **Sentence Case**: Capitalize the first letter of every user prompt and assistant response.
2. **Acronym Preservation**: Always write technical acronyms, standards, and safety certifications in uppercase (e.g., `LOTO`, `EPI`, `ATEX`, `HSE`, `SST`, `ISO 45001`, `CMU`, `VAT`).
3. **Punctuation**: Use standard question marks `?` and exclamation marks `!` naturally at phrase boundaries. Do not use Arabic inverted question marks (`؟`) in Arabizi rows.

---

## 3. Industrial Domain Terminology Matrix (~40 Core Terms)

| English Concept | Standard French Term | Canonical Arabizi Variant | Forbidden / Deprecated Variants |
| :--- | :--- | :--- | :--- |
| Verification | vérification | `t-vérification` / `la vérification` | verif, tverifiya |
| Inspection | inspection | `l-inspection` | l-inspecion |
| Protocol | protocole | `l-protocole` | l-prothocol |
| Compliance | conformité | `la conformité` | konformite |
| Incident | incident | `l-incident` | l-insident |
| Equipment | équipement | `les équipements` / `l-EPI` | ekipma |
| Standards | normes | `les normes` | l-normat |
| Danger | danger | `l-danger` / `l-khtar` | dangerosité |
| Security / Safety | sécurité | `la sécurité` / `s-salama` | l-securite |
| Helmet | casque de sécurité | `l-casque` | l-kask, casq |
| Gloves | gants de protection | `les gants` | l-gantat |
| Harness | harnais de sécurité | `l-harnais` | l-harnai |
| Fire Extinguisher | extincteur | `l-extincteur` | l-extinteur |
| Emergency Stop | arrêt d'urgence | `l-arrêt d'urgence` | arret d urgence |
| Risk | risque | `le risque` / `les risques` | l-risq |
| Maintenance | maintenance | `la maintenance` | l-maintnonse |
| Authorization | autorisation | `l-autorisation` | l-otori2asipn |
| Clearance / Permit | permis de travail | `le permis` / `le permis de feu` | l-permi |
| Scaffolding | échafaudage | `l-échafaudage` | l-echafodag |
| Lifeline | ligne de vie | `la ligne de vie` | l-lign de vie |
| Lockout / Tagout | consignation LOTO | `la consignation` / `LOTO` | l-consignasion |
| Chemical Product | produit chimique | `les produits chimiques` | l-produit chimik |
| Evacuation | évacuation | `l-évacuation` | l-evakwasyon |
| First Aid | premiers secours | `les premiers secours` / `SST` | l-premier secour |
| Noise | bruit / niveau sonore | `le niveau sonore` | l-bruit |
| Respirator / Mask | masque FFP2/FFP3 | `le masque` | l-masq |
| High Voltage | haute tension | `la haute tension` | l-haute tansyon |
| Gas Leak | fuite de gaz | `la fuite de gaz` | l-fwit de gaz |
| Procedure | procédure | `la procédure` | l-prosesdur |
| Report | rapport / signalement | `le signalement` | l-rapor |
| Audit | audit de sécurité | `l-audit` | l-odit |
| Floor / Surface | sol / surface de travail | `l-sol` | l-soll |
| Slipping | glissement | `le glissement` | l-glismon |
| Fall | chute | `la chute` | l-chut |
| Protection | protection | `la protection` | l-proteksyon |
| Storage Zone | zone de stockage | `la zone de stockage` | l-zon de stokage |
| Barrier / Tape | balisage / rubalise | `le balisage` | l-balizag |
| Load Capacity | charge maximale (CMU) | `la CMU` / `la charge` | l-charg max |
| Crane / Forklift | chariot élévateur / Clark | `le Clark` / `le chariot` | l-clarke |
| Workshop | atelier / usine | `l-atelier` / `l-usine` | l-atlier |
| Training | formation | `la formation` | l-formasyon |
| Supervisor | responsable HSE | `le responsable HSE` | l-chef HSE |
