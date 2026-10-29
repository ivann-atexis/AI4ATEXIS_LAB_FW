# AI4ATEXIS_LAB_FW : consignes pour les agents

Lire le [README](../README.md) pour l'architecture et les commandes actuelles.
Ce depot est un banc de compatibilite et de demonstration, pas un serveur
d'inference de production.

## Choisir le workflow

Charger le skill correspondant avant de realiser la tache :

| Demande | Skill |
|---|---|
| Ajouter, integrer ou adapter un modele, implementer chargement/inference | [ai4atexis-lab-fw-add-model](skills/ai4atexis-lab-fw-add-model/SKILL.md) |
| Dependances, conflits de versions, Docker, CUDA, runtime specialise | [ai4atexis-lab-fw-runtime](skills/ai4atexis-lab-fw-runtime/SKILL.md) |
| Creer ou modifier une demo CLI, Gradio ou FastAPI | [ai4atexis-lab-fw-apps](skills/ai4atexis-lab-fw-apps/SKILL.md) |
| Tester, valider, smoke test, benchmark ou comparaison de resultats | [ai4atexis-lab-fw-test-benchmark](skills/ai4atexis-lab-fw-test-benchmark/SKILL.md) |

Une integration peut utiliser plusieurs skills, seulement pour les parties
necessaires. Consulter les fichiers references, sans recopier tout le depot.

## Invariants du projet

- Un adaptateur par modele ; une image par ensemble de dependances compatibles,
  pas une image obligatoire par modele.
- Respecter [les types](../common/types.py) et
  [ModelSession](../common/runtime.py). Le coeur ne doit pas imposer PyTorch,
  Transformers, une modalite ou une generation autoregressive.
- Reutiliser les helpers communs et le template. Ne pas ajouter de registre
  central, de branche conditionnelle par nom de modele ou de nouvelle couche
  d'abstraction pour une simple integration.
- Par defaut, travailler **sans poids** : pas de telechargement de snapshot,
  de chargement reel, de smoke test modele, de serveur modele ou de benchmark
  GPU sans demande explicite. `--offline` utilise encore des poids et peut
  lancer le GPU ; ce n'est pas un test sans modele.
- Verifier identifiant, API et revision du modele dans des sources officielles.
  Ne pas inventer une API, des versions compatibles ou un SHA.
- Ne pas modifier le Python/CUDA de l'hote. Conserver les dependances dans le
  runtime choisi ; ne pas commettre de tokens, de poids ou de donnees sensibles.
- Distinguer tests simules, image construite et inference reelle dans le bilan.
  Une etape non executee ou bloquee doit etre annoncee comme telle.
