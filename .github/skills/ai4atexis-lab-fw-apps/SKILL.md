---
name: ai4atexis-lab-fw-apps
description: 'Creer, adapter ou corriger une mini-app de modele AI4ATEXIS_LAB_FW : demo CLI, chat Gradio, formulaire multimodal, decisions JSON, API FastAPI, OpenAI-compatible, OpenWebUI, streaming SSE ou specification OpenAPI/Swagger. Use when building model demos, connecting OpenWebUI, changing app inputs/outputs, chat history or API contracts while reusing the already loaded ModelSession.'
argument-hint: 'Modele, app cli/gradio/api, entrees et sortie attendue'
---

# Creer des mini-apps AI4ATEXIS_LAB_FW

## Contrat

Chaque app expose `run(session: ModelSession, options: AppOptions) -> None`
dans `models/<nom>/apps/<cli|gradio|api>/app.py`.
La session est deja chargee : appeler `session.infer(input_data)` et exploiter
`InferenceResult`, sans appeler directement le backend ni recharger le modele.
Pour un flux, utiliser `session.stream(input_data, cancel)` ; ne pas contourner
le verrou ni lancer un second chargement.

Lire les [types](../../../common/types.py), la
[session](../../../common/runtime.py), le
[lanceur](../../../scripts/run.py) et les
[apps communes](../../../common/apps). Confirmer dans l'adaptateur le vrai
schema d'entree/sortie avant de concevoir l'interface.

## Procedure

1. Identifier la demande : reutilisation d'une app commune, demo specialisee
   d'un modele ou changement commun a tous les modeles. Preferer un simple
   point d'entree vers une implementation existante :

   ```python
   from common.apps.cli import run

   __all__ = ["run"]
   ```

   Pour Gradio, choisir `run_chat`, `run_decisions` ou `run_generic` selon
   le contrat, pas selon le nom ou la taille du modele. Le chat exige une
   sortie texte ; le formulaire generique convient a une sortie JSON.
2. Si une nouvelle interface est necessaire, la placer dans l'adaptateur.
   N'extraire un helper commun que s'il est reellement partage. Separer la
   construction de l'interface de son lancement pour la tester sans serveur.
   Utiliser `options.host`, `options.port` et `options.input_data` ; ne pas
   inventer des options que le lanceur ne fournit pas.
3. Respecter les particularites :
   - **CLI** : conserver une sortie JSON exploitable et les logs sur stderr.
     Le lanceur fournit deja texte, fichier JSON et image locale via
     `--prompt`, `--input`, `--image`. Ces options ne concernent pas les apps
     web. Ne pas forcer une boucle interactive pour une inference simple.
   - **Chat** : transmettre de vrais messages avec roles ; reutiliser le
     traitement de l'historique fourni par Gradio. Ne pas concatener
     arbitrairement l'historique ni appliquer le chat template dans l'app :
     c'est la responsabilite de l'adaptateur.
   - **Decisions / multimodal** : representer les champs reels du modele
     et afficher ses reponses structurees. d1 utilise un etat, des questions
     et une image facultative, pas des parametres de generation.
   - **API** : conserver `GET /health`, `POST /infer`, le corps
     `input_data` et la reponse `result.to_dict()`. Reutiliser
     [l'app FastAPI](../../../common/apps/api.py) si possible.
     Pour une extension, valider les donnees au bord et documenter toute
     modification de contrat plutot que casser les clients silencieusement.
4. Garder les controles communs : `ModelSession` serialise les inferences.
   Ne pas contourner ce verrou, activer un partage Gradio public ou publier
   le port sur toutes les interfaces de l'hote sans demande explicite.
   Garder les analytics Gradio desactives.
5. Pour une image, reutiliser les
   [helpers d'entree](../../../common/utils.py) : image PIL deja chargee en
   CLI/Gradio, base64 brut dans `image_base64` pour l'API commune.
   Ne pas ouvrir un chemin serveur ou une URL arbitraire transmis dans une
   requete. Conserver les limites d'entree ; pas de token dans le navigateur.
6. Montrer les erreurs de saisie explicitement. Dans l'API commune,
   `ValueError` donne une reponse 422 et les pannes du runtime restent des
   erreurs serveur. Ne pas capturer toutes les exceptions pour retourner
   une reponse 200 vide. Ne pas journaliser tokens, images ou documents prives.
7. Ajouter les tests avant tout essai reel : partir de
   [tests des apps](../../../tests/test_apps.py) et de
   [fake_session](../../../tests/helpers.py).
   Verifier la transmission exacte des entrees a `session.infer`, la forme
   JSON de sortie, les erreurs, le parcours image et l'historique si utiles.
   Construire Gradio sans `.launch()` et utiliser `TestClient` pour FastAPI.
   Les callbacks ne doivent pas executer d'exemples au chargement de la demo.

## OpenAI, OpenWebUI et OpenAPI

- Distinguer **OpenAPI** (specification JSON `/openapi.json`, Swagger `/docs`)
  du protocole **OpenAI-compatible** (`/v1/models`, `/v1/chat/completions`).
  Pour choisir un LLM dans OpenWebUI, ajouter une connexion OpenAI avec une
  URL finissant par `/v1`, pas importer le schema comme un outil.
- Reutiliser [les routes communes](../../../common/apps/openai.py), deja
  incluses par `common.apps.api`. Ne pas recopier une implementation par
  modele ni casser `/health` et `/infer`.
- Activer `api.chat: true` uniquement si l'adaptateur accepte des messages
  texte et des surcharges `generation` par requete. d1 et les autres
  modeles non-chat restent accessibles via `/infer`, sans annonce de chat.
- Respecter le sous-ensemble supporte : roles system/user/assistant,
  temperature, top_p, max_tokens ou max_completion_tokens, stop, n=1.
  Refuser les fonctionnalites non implementees, notamment tools, images
  dans le chat et mode JSON contraint. Ne pas ignorer silencieusement les
  parametres ni muter la configuration partagee.
- Le resultat chat exige un texte et un `finish_reason` reel. Les compteurs
  viennent du backend, pas d'un calcul approximatif sur le texte.
- Un hook `stream` optionnel produit des `StreamChunk(text=...)` puis un
  seul `StreamChunk(result=...)`. Respecter l'evenement d'annulation et
  fermer le generateur et son worker avant de liberer le verrou.
  Reutiliser `stream_causal` si adapte. Pas de faux streaming obtenu en
  decoupant une reponse deja complete.
- Preserver les deltas SSE, l'evenement final, l'usage facultatif et `[DONE]`.
  Apres une erreur de flux, emettre une erreur, pas une fin de succes.
  Conserver la fermeture explicite de la reponse, y compris si un `send`
  ASGI 2.4 echoue lors d'une deconnexion.
- Pour lancer OpenWebUI avec le framework, utiliser
  [compose.openwebui.yaml](../../../compose.openwebui.yaml) : mode API force,
  readiness du modele, image locale sans pull, donnees persistantes et
  connexion automatique vers `http://lab:7860/v1`.
  Pour raccorder une UI existante, utiliser plutot
  [compose.openwebui.external.yaml](../../../compose.openwebui.external.yaml)
  et son reseau externe, comme explique dans le README.
  `localhost` dans un conteneur ne designe pas l'hote. Ne pas exposer cette
  API sans authentification sur le LAN pour contourner un probleme reseau.

## Validation et livraison

```bash
HF_HUB_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=False \
  python -m unittest tests.test_apps -v
```

Ajouter les nouveaux tests specialises a cette selection. Une session simulee
est suffisante pour tester l'app sans poids. Si une validation visuelle est
necessaire, utiliser aussi une session simulee, un port local et nettoyer le
serveur temporaire ensuite.

Pour le protocole et le cycle de vie du flux :

```bash
HF_HUB_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=False \
  python -m unittest tests.test_openai tests.test_streaming -v
```

Verifier aussi la specification OpenAPI generee ; elle doit annoncer les
reponses JSON et SSE et leurs erreurs, pas seulement retourner un HTTP 200.

Un lancement reel reste une etape distincte, uniquement si demande :

```bash
MODEL=mon-modele APP=gradio docker compose up lab
# ou APP=api ; choisir le RUNTIME adapte si necessaire.
```

Documenter le schema d'entree, les commandes et les limites dans le README.
Rappeler que ces apps sont des demos locales sans authentification de
production. Indiquer clairement si seule l'interface simulee a ete validee.
