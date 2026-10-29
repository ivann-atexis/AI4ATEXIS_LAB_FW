# AI4ATEXIS_LAB_FW

Banc de developpement et de compatibilite **day-0**, en Transformers direct
pour les premiers adaptateurs. Ce n'est pas un serveur vLLM/SGLang, ni une
abstraction qui impose la generation de texte a tous les modeles.

**Le modele et son environnement sont deux choses distinctes :**

- un **adaptateur** definit comment charger et executer un modele ;
- un **runtime Docker** fournit ses bibliotheques et ses versions ;
- plusieurs adaptateurs compatibles partagent une image ;
- un modele avec des dependances incompatibles utilise une autre image,
  sans modifier les commandes, les autres adaptateurs ou le coeur du framework.

Le nom du projet est **AI4ATEXIS_LAB_FW**. Les identifiants Docker utilisent
`ai4atexis_lab_fw`, car Docker exige des minuscules. Les noms de skills
utilisent le prefixe `ai4atexis-lab-fw-`, au format minuscules et tirets requis.

## Ce qui est inclus

| Adaptateur | Chargement / inference | Mini-app Gradio |
|---|---|---|
| `lfm3` | `LiquidAI/LFM3-1.7B`, `AutoModelForCausalLM`, chat template puis `generate()` | Chat texte avec historique |
| `d1-2b` | `LiquidAI/d1-2B`, `AutoModel`, `system_one()` en une passe | Etat texte, image facultative, questions JSON, decisions |
| `template` | Point de depart a copier ; pas un modele pret a charger | Formulaire texte / objet JSON |

Chaque adaptateur contient aussi une app CLI, une app FastAPI et un smoke test.
Les apps sont de petits points d'entree vers les implementations communes :
on peut remplacer une app d'un modele sans dupliquer les autres.
L'app API propose aussi un chat compatible OpenAI / OpenWebUI pour les
adaptateurs qui declarent `api.chat: true` : LFM3 et le template texte fourni.
Le streaming est reel, via le backend, pas une reponse complete decoupee apres
la generation. LFM3 expose aussi les appels d'outils natifs via cette API ;
OpenWebUI execute les outils et renvoie leurs resultats. d1 reste une API
de decisions via `/infer`.

**Aucun poids n'est livre ou telecharge par la creation du projet, la construction
de l'image ou les tests automatises.** En revanche, lancer une vraie inference,
un smoke test de modele ou un benchmark telecharge le snapshot configure s'il
n'est pas deja en cache. `--offline` interdit ce telechargement ; le chargement
reel echoue explicitement si les fichiers necessaires manquent dans le cache.

## Organisation

```text
.
|-- compose.yaml
|-- compose.openwebui.yaml    # optionnel : API + OpenWebUI avec l'image locale
|-- compose.openwebui.external.yaml  # optionnel : raccorder un OpenWebUI existant
|-- .env.example
|-- docker/
|   |-- Dockerfile
|   |-- requirements.in       # versions principales choisies
|   `-- requirements.txt      # verrou transitif avec empreintes SHA256
|-- common/
|   |-- apps/                # CLI, Gradio et FastAPI reutilisables
|   |-- benchmark.py
|   |-- config.py
|   |-- device.py
|   |-- hub.py
|   |-- logging.py
|   |-- runtime.py
|   |-- transformers.py      # helpers optionnels, pas une obligation du coeur
|   |-- types.py
|   `-- utils.py
|-- models/
|   |-- lfm3/
|   |-- d1-2b/
|   `-- template/
|       |-- config.yaml
|       |-- load.py
|       |-- inference.py
|       |-- test.py
|       |-- requirements.in  # optionnel : runtime specialise
|       `-- apps/
|           |-- cli/app.py
|           |-- gradio/app.py
|           `-- api/app.py
|-- scripts/
|   |-- run.py
|   |-- download.py
|   `-- benchmark.py
|-- tests/                   # tests hors ligne avec doubles de modeles
`-- results/
```

## 1. Preparer l'environnement, sans poids

Prerequis : Docker Engine, Docker Compose >= 2.30, NVIDIA Container Toolkit,
un GPU NVIDIA et un pilote compatible CUDA 13.0 pour l'image fournie.
L'image NGC choisie est multiarchitecture ARM64 / AMD64 et prend en charge
Blackwell. Son digest est fixe dans le [Dockerfile](docker/Dockerfile).
Les versions PyTorch / torchvision viennent de cette image : on ne remplace
pas ses binaires CUDA par des roues PyPI potentiellement incompatibles.

```bash
cp .env.example .env
docker compose config --quiet
docker compose build lab

# Teste contrats, adaptateurs simules, API et construction des interfaces.
# Pas de poids, pas d'inference reelle.
docker compose run --rm -e HF_HUB_OFFLINE=1 lab \
  python -m unittest discover -s tests -v

docker compose run --rm lab python scripts/run.py --list-models
```

La construction telecharge une image de base volumineuse et les dependances
Python, **pas les modeles**. Elle ne modifie pas Python/CUDA sur la machine hote.

Les sources sont copiees dans l'image, pas montees en direct : apres une
modification de code ou de configuration, reconstruire avec
`docker compose build lab`. Le cache HF et les resultats sont conserves.

Si Docker indique `permission denied` sur son socket, faire configurer l'acces
au daemon par l'administrateur avant de construire ou lancer des conteneurs.
`docker compose config` fonctionne sans acces au daemon.

## 2. Lancer les modeles, quand les poids seront souhaites

Les commandes de cette section peuvent telecharger plusieurs Go au premier
lancement. Mettre `HF_TOKEN` dans `.env` uniquement si necessaire. Ce fichier
est ignore par Git et exclu du contexte Docker ; ne pas mettre de token dans
les YAML des modeles.

### CLI

```bash
docker compose run --rm lab python scripts/run.py lfm3
docker compose run --rm lab python scripts/run.py lfm3 \
  --prompt "Explique simplement ce qu'est un banc de test."

docker compose run --rm lab python scripts/run.py d1-2b \
  --prompt "The delivery is late and two replacement parts are missing."
```

Pour d1, le texte devient l'etat et les questions proviennent de son
[config.yaml](models/d1-2b/config.yaml). Une entree plus riche peut etre
fournie dans un fichier JSON :

```json
{
  "state": "The sensor failed during the acceptance test.",
  "questions": {
    "maintenance": {
      "type": "noul",
      "instructions": "Does this report require maintenance?"
    }
  }
}
```

Monter explicitement les entrees locales dans le conteneur :

```bash
docker compose run --rm -v "$PWD/inputs:/inputs:ro" lab \
  python scripts/run.py d1-2b --input /inputs/request.json \
  --image /inputs/photo.png
```

Le dossier `inputs` est un exemple a creer si necessaire, ignore par Git et
exclu de l'image. `--prompt`, `--input`
et `--image` concernent les inferences CLI, pas les apps web.

### Gradio

```bash
MODEL=lfm3 APP=gradio docker compose up lab
# ou, apres avoir arrete le premier :
MODEL=d1-2b APP=gradio docker compose up lab
```

Ouvrir <http://127.0.0.1:7860>. d1 permet de choisir le theme, la question et
les categories sans modifier de JSON :

1. L'app s'ouvre directement sur **Question personnalisee**, pre-remplie avec
   le theme et la question chat / chien / aucun / les deux, sans le texte
   d'exemple de maintenance.
2. Modifier si besoin le **Theme / contexte** facultatif, la **Question**, et les categories
   avec leurs descriptions. Le theme est ajoute aux instructions du modele,
   pas utilise comme une description de ce que contient l'image.
3. Fournir un texte dans **Etat / document**, une image, ou les deux, puis
   cliquer sur **Analyser**.

Le bouton **Exemple image : chat / chien / aucun / les deux** remplit le
formulaire et selectionne le mode personnalise. Il efface le texte d'exemple
de maintenance sans supprimer l'image deja chargee ni lancer d'inference.
Le bouton de remise a zero vide le theme, la question, les categories
(une ligne vide non cochee), le texte, l'image, le resultat et son temps de
reponse, et laisse
le mode personnalise actif. Il annule aussi la reponse d'une analyse en cours,
sans recharger le modele. Le bouton d'exemple permet ensuite de retrouver
le formulaire pre-rempli.
Le champ **Temps de reponse (s)** affiche la duree de la derniere inference
reussie, arrondie a la milliseconde. La mesure entoure `session.infer()` cote
serveur, avec synchronisation GPU si applicable. Elle inclut le traitement
par l'adaptateur et une eventuelle attente du verrou de session, mais pas le
chargement du modele, le transfert de l'image, l'attente dans la file Gradio
ou l'affichage dans le navigateur. C'est un temps de reponse, pas un debit de
generation de tokens : d1 prend des decisions sans generer de texte.
Charger une image puis cliquer sur **Analyser** : les categories proposees
sont `chat` (sans chien), `chien` (sans chat), `aucun` (ni chat ni chien)
et `les_deux` (chat et chien presents ensemble). Tout reste editable.
Il s'agit d'une classification globale de l'image, pas d'une detection avec
localisation ou comptage des animaux. La qualite des predictions doit etre
verifiee avec le vrai modele sur vos images ; les tests de l'app utilisent
une session simulee, sans poids.

Le mode personnalise envoie uniquement une question `classification` de type
`choice` : il ne conserve pas les questions maintenance/intervention/urgence.
Une seule categorie est choisie par analyse. Le mode **Questions configurees**
reste disponible et conserve les questions initiales et leurs tableaux de
categories et de scores. Dans les deux formulaires :

- Cocher/decocher **Utiliser** dans les tableaux des categories et des scores.
- Double-cliquer sur une cellule pour modifier un nom ou une description.
  Ajouter une ligne avec **+ Ajouter un critere**, la remplir puis cocher
  **Utiliser** pour activer un nouveau critere.
- Pour les scores, garder les niveaux ordonnes du plus faible au plus eleve.
  Chaque question doit conserver au moins un critere actif ; les noms actifs
  doivent etre uniques et les categories doivent avoir une description.
- Cliquer sur **Analyser** pour appliquer cette selection a la requete,
  sans recharger le modele ni modifier le fichier de configuration.

Le mode **JSON avance** permet toujours de personnaliser toutes les questions.
Seul le mode selectionne est utilise ; les trois modes sont independants et
ne se synchronisent pas. Aucun de ces reglages ne modifie la configuration
sauvegardee. Le choix du mode remplace l'ancienne case
**Utiliser le JSON a la place des tableaux**.

LFM3 utilise une vraie conversation, pas la
concatenation artisanale d'un historique. Le partage public et les analytics
Gradio sont desactives.

### FastAPI

```bash
MODEL=d1-2b APP=api docker compose up lab

curl http://127.0.0.1:7860/health
curl http://127.0.0.1:7860/infer \
  -H 'Content-Type: application/json' \
  -d '{"input_data":{"state":"The test equipment has stopped working."}}'
```

Documentation interactive **OpenAPI / Swagger** :
<http://127.0.0.1:7860/docs>. Specification JSON :
<http://127.0.0.1:7860/openapi.json>.
LFM3 accepte `{"input_data":"Bonjour"}` ou
`{"input_data":{"messages":[{"role":"user","content":"Bonjour"}]}}`.

Une image API se transmet dans un champ `image_base64` contenant du base64
brut, a cote de `input_data`. Les chemins et URL d'images provenant d'une
requete ne sont pas ouverts par l'adaptateur d1. La CLI et Gradio lui
transmettent directement une image PIL deja chargee.

Ce sont des **demos de laboratoire**, sans authentification ni infrastructure
de production. Le port publie par Compose est limite a `127.0.0.1`. Une seule
inference a la fois est executee par session pour eviter les acces concurrents
au meme modele. Pour exposer ces apps, ajouter une couche adaptee
d'authentification, limites et supervision.

### OpenWebUI et API compatible OpenAI

**OpenAPI** decrit les routes (Swagger, `/openapi.json`).
**OpenAI-compatible** designe le protocole de chat attendu par OpenWebUI.
Les deux sont disponibles dans l'app `api`, sans service d'inference
supplementaire. Les appels reutilisent le modele deja charge dans la session.

| Route | Usage |
|---|---|
| `GET /health` | Etat du modele charge |
| `POST /infer` | Contrat generique, texte ou decisions, inchange |
| `GET /v1/models` | Uniquement le modele charge, s'il est compatible chat |
| `POST /v1/chat/completions` | Conversation texte, reponse JSON ou streaming SSE |
| `GET /openapi.json`, `GET /docs` | Specification OpenAPI et Swagger |

#### Lancer LFM3 et OpenWebUI ensemble (recommande)

La variante [compose.openwebui.yaml](compose.openwebui.yaml) ajoute
directement OpenWebUI a la stack. Aucun reseau externe ni conteneur
OpenWebUI preexistant n'est necessaire.

```bash
MODEL=lfm3 HF_HUB_OFFLINE=1 \
  docker compose -f compose.yaml -f compose.openwebui.yaml up -d --build
```

Si Docker exige `sudo`, utiliser :

```bash
sudo env MODEL=lfm3 HF_HUB_OFFLINE=1 \
  docker compose -f compose.yaml -f compose.openwebui.yaml up -d --build
```

Cette commande :

- reconstruit l'image du framework avec les sources actuelles ;
- lance `lab` en mode **API**, meme si `.env` contient `APP=gradio` ;
- attend que `/health` confirme que le modele est pret ;
- cree un conteneur OpenWebUI depuis l'image locale exacte
  `ghcr.io/open-webui/open-webui:main`, avec **`pull_policy: never`** ;
- configure la connexion OpenAI vers `http://lab:7860/v1` sur le reseau
  Docker interne commun, sans charger une seconde instance de LFM3.

L'image OpenWebUI n'est ni reconstruite ni telechargee. Si ce tag n'est pas
present dans le meme daemon Docker, le lancement echoue explicitement au
lieu de le telecharger. Le tag `main` n'est pas mis a jour automatiquement :
c'est bien la version deja disponible localement qui est utilisee.
La reconstruction de `lab` peut consulter les registres et index de paquets ;
elle ne telecharge pas de poids. `HF_HUB_OFFLINE=1` exige que le snapshot
LFM3 soit deja dans le cache du framework.

Ouvrir **<http://127.0.0.1:3000>**, creer le premier compte administrateur et
selectionner `LiquidAI/LFM3-1.7B`. Si le port est occupe, definir
`OPENWEBUI_PORT` dans `.env`. La connexion est preconfiguree pour une nouvelle
base ; les reglages modifies ensuite dans l'administration OpenWebUI restent
persistants et peuvent prendre le dessus sur les variables d'environnement.

Les comptes, conversations et fichiers OpenWebUI sont conserves dans le
volume Docker **`openwebui-data`**, prefixe par le nom du projet Compose.
Il est distinct du cache HF du modele et n'importe pas les donnees d'une
autre installation OpenWebUI. Le login reste active ; aucune cle ou aucun
mot de passe administrateur n'est impose dans les fichiers du projet.

OpenWebUI est configure en mode HF hors ligne, avec Ollama desactive.
Le chat utilise uniquement notre API locale. Les fonctions RAG, embeddings
ou transcription ne sont pas garanties : elles exigent leurs propres
modeles deja presents et/ou un backend adapte, qui n'est pas fourni par
notre API de chat. `OFFLINE_MODE` et `HF_HUB_OFFLINE` evitent les
telechargements HF automatiques ; ils ne sont pas un pare-feu reseau.

Pour suivre ou arreter la stack, garder les memes fichiers Compose :

```bash
docker compose -f compose.yaml -f compose.openwebui.yaml ps
docker compose -f compose.yaml -f compose.openwebui.yaml logs -f lab openwebui
docker compose -f compose.yaml -f compose.openwebui.yaml down
```

Les volumes sont conserves par cet arret. **Ne pas ajouter `-v`** pour un
simple arret : cela supprimerait aussi les donnees persistantes.

Pour raccourcir les commandes suivantes, on peut ajouter dans `.env` :

```dotenv
COMPOSE_FILE=compose.yaml:compose.openwebui.yaml
```

Puis utiliser simplement `docker compose up -d --build`. Tant que ce choix
est actif, le mode API est force. Pour revenir au fonctionnement CLI/Gradio
du framework seul, arreter la stack puis retirer ce choix ou utiliser
explicitement `docker compose -f compose.yaml ...`.

#### Lancer uniquement l'API, sans OpenWebUI

La nouvelle API necessite une reconstruction : les sources ne sont pas
montees en direct. La construction ne telecharge pas de poids.

```bash
docker compose build lab
MODEL=lfm3 APP=api HF_HUB_OFFLINE=1 docker compose up -d lab
```

Cette derniere commande **remplace l'app Gradio du service `lab` et recharge
le modele une fois**. Elle n'est pas executee automatiquement lors d'une
modification de code. Le mode HF hors ligne exige le snapshot deja en cache.
Pour conserver le choix aux prochains lancements, mettre `APP=api` dans
votre `.env`, au lieu de changer de mode seulement pour une commande.

```bash
curl http://127.0.0.1:7860/v1/models

curl http://127.0.0.1:7860/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"LiquidAI/LFM3-1.7B","messages":[{"role":"user","content":"Bonjour"}],"max_tokens":64,"temperature":0}'

curl -N http://127.0.0.1:7860/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"LiquidAI/LFM3-1.7B","messages":[{"role":"user","content":"Bonjour"}],"stream":true,"stream_options":{"include_usage":true},"max_tokens":64}'
```

Le nom court `lfm3` est accepte comme alias ; la decouverte et les reponses
utilisent l'identifiant canonique. Il n'y a pas de chargement automatique
d'un autre modele quand le champ `model` change.

#### Connecter un OpenWebUI Docker existant

Une image dans `docker images` ne signifie pas qu'un conteneur tourne.
Identifier le conteneur et son reseau ; remplacer `open-webui` ci-dessous par
son vrai nom :

```bash
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}'
docker inspect --format '{{range $name, $_ := .NetworkSettings.Networks}}{{println $name}}{{end}}' open-webui
```

Choisir un **reseau Docker personnalise deja partageable**, pas les reseaux
integres `bridge`, `host` ou `none`. Le fichier
[compose.openwebui.external.yaml](compose.openwebui.external.yaml) y rattache seulement `lab` :
il ne cree ni ne remplace ton OpenWebUI.

```bash
OPENWEBUI_NETWORK=mon-reseau MODEL=lfm3 APP=api HF_HUB_OFFLINE=1 \
  docker compose -f compose.yaml -f compose.openwebui.external.yaml up -d lab
```

Conserver aussi `OPENWEBUI_NETWORK=mon-reseau` dans `.env` si ce raccordement
doit etre reutilise. Si aucun reseau adapte n'existe, en creer un et y
connecter le conteneur OpenWebUI avant de lancer cette variante :

```bash
docker network create ai4atexis-openwebui
docker network connect ai4atexis-openwebui open-webui
```

Dans ce cas, utiliser `OPENWEBUI_NETWORK=ai4atexis-openwebui`. Pour que
l'association survive a une recreation du conteneur OpenWebUI, declarer
egalement ce reseau externe dans son propre Compose. Ne pas recreer
OpenWebUI a partir de son image sans conserver son volume de donnees.

Si Docker exige `sudo`, placer les variables dans `.env` ou utiliser
`sudo env MODEL=lfm3 APP=api ... docker compose ...` : un simple
`APP=api sudo docker ...` peut perdre la variable.

Dans OpenWebUI : **Administration / Settings > Connections > OpenAI**,
ajouter une connexion :

- **URL de base** : `http://ai4atexis-lab-fw:7860/v1` sur le reseau partage ;
- **cle API** : vide, ou une valeur factice comme `local` si le formulaire
  l'exige ; cette API de laboratoire ne verifie pas de cle ;
- **modele** : `LiquidAI/LFM3-1.7B`, normalement decouvert par `/v1/models`.

Utiliser une connexion **OpenAI**, pas une connexion Ollama ni l'import
d'un outil OpenAPI, pour selectionner LFM3 comme modele de conversation.

Si OpenWebUI fonctionne directement sur l'hote ou en reseau `host`, utiliser
`http://127.0.0.1:7860/v1`, sans la variante de reseau. Dans un conteneur
bridge, `localhost` designe ce conteneur. `host.docker.internal` ne suffit
pas non plus si le service de l'hote est publie uniquement sur sa boucle
locale. Depuis une autre machine, prevoir un acces protege (proxy
authentifie, TLS ou tunnel adapte), sans ouvrir cette demo sur le LAN.

#### Perimetre du protocole

- Messages `system`, `user`, `assistant`, avec texte brut ou parties
  `{"type":"text","text":"..."}`. L'historique est transmis au chat template.
- Parametres pris en charge : `temperature` (0 a 2), `top_p` (> 0 a 1),
  `max_tokens` **ou** `max_completion_tokens`, `stop` (au plus quatre textes)
  et `n=1`. `temperature=0` choisit une generation gloutonne.
- Ces parametres remplacent les valeurs de generation **pour la requete
  seulement**, sans modifier la configuration partagee.
- `stream=true` emet des deltas SSE, un `finish_reason` (`stop`, `length`
  ou `tool_calls` pour un appel natif)
  et `[DONE]`. Avec `stream_options.include_usage=true`, un dernier evenement
  a `choices: []` contient les vrais compteurs, ou `usage: null` s'ils sont
  inconnus. Le champ facultatif `user` est accepte comme metadonnee client,
  sans effet sur la generation et sans journalisation de sa valeur.
- Une deconnexion demande l'annulation, attend l'arret du worker et libere
  le verrou de session. L'arret est cooperatif entre etapes de generation :
  il ne peut pas interrompre un kernel GPU deja lance.
- Les erreurs de requete ont une enveloppe `error` compatible OpenAI.
  Une panne apres ouverture du flux emet une erreur SSE, sans faux
  `finish_reason` de succes ni `[DONE]`.
- Les outils sont pris en charge seulement par les adaptateurs qui declarent
  `api.tool_calling: true` et implementent le protocole natif, comme LFM3.
  `tools: null`, `tools: []` et `tool_choice: "none"` restent compatibles
  avec un chat sans outils. Un adaptateur sans cette capacite refuse les
  demandes actives avec une erreur explicite `tools_not_supported`.
- Pas de mode JSON contraint pour les reponses texte, d'images/audio dans
  ce chat, d'embeddings ou de Responses API. Les options non implementees
  sont refusees explicitement, pas ignorees. Les images d1 restent disponibles
  via `/infer`. Desactiver les fonctions OpenWebUI qui exigent ces capacites.

#### Outils natifs avec LFM3 et OpenWebUI

LFM3 utilise son chat template officiel pour recevoir les schemas d'outils
et l'historique, puis produit des appels au format XML propre au modele.
L'adaptateur [LFM3](models/lfm3/tools.py) les convertit en `tool_calls`
OpenAI : identifiant unique, nom de fonction et `function.arguments` sous
forme de chaine JSON. La reponse annonce `finish_reason: "tool_calls"`.
Le framework **n'execute aucune fonction, commande ou requete externe**
pour satisfaire un appel : c'est OpenWebUI ou le client API qui l'execute,
avec ses propres permissions et confirmations.

Dans OpenWebUI, les outils integres peuvent etre reactives pour LFM3 apres
reconstruction de l'image du framework. Sur OpenWebUI 0.12.4, ils peuvent
etre proposes automatiquement, meme pour un chat ordinaire. Une ancienne
image du framework, anterieure au support natif, les refusait avec
`Extra inputs are not permitted` sur le champ `tools`.

Pour appliquer les modifications, sans effacer comptes ni conversations :

```bash
sudo env MODEL=lfm3 HF_HUB_OFFLINE=1 \
  docker compose -f compose.yaml -f compose.openwebui.yaml up -d --build
```

Le cycle API est le suivant :

1. Envoyer `messages` et `tools` (schemas OpenAI de type `function`).
2. Recuperer `choices[0].message.tool_calls`, puis executer les fonctions
   dans le client, jamais dans le serveur du framework.
3. Ajouter le message assistant recu, avec ses appels, dans l'historique.
4. Ajouter un message `role: "tool"` par resultat, avec le meme
   `tool_call_id`. Garder les schemas `tools` dans la requete suivante.
5. Envoyer cet historique pour que LFM3 utilise les resultats et reponde
   ou demande d'autres outils.

Les chaines JSON d'arguments de l'historique sont decodees en dictionnaires
avant le chat template, comme l'exige LFM3. Le contenu et
`reasoning_content` du tour assistant sont preserves. Les references
`tool_reference` et le champ `defer_loading` sont transmis au template
pour les clients utilisant une recherche d'outils ; le client implemente
lui-meme cette recherche.

`tool_choice` accepte `auto` (defaut avec outils), `none`, `required` ou
`{"type":"function","function":{"name":"nom_outil"}}`.
Les demandes obligatoires ajoutent une instruction au modele et sont
controlees sur le resultat ; ce n'est pas un decodage contraint des logits.
`parallel_tool_calls=false` refuse une sortie comportant plusieurs appels.
Sinon, plusieurs appels sont permis, chacun avec son propre identifiant ;
leur execution parallele ou sequentielle reste la responsabilite du client.

Les arguments sont convertis selon le JSON Schema fourni (chaines, nombres,
booleens, tableaux, objets, null), puis valides avant toute reponse d'appel.
Les schemas peuvent utiliser des references locales ; les references
externes sont refusees pour eviter tout chargement reseau implicite.
Un outil inconnu, des arguments invalides ou un bloc XML tronque provoquent
une erreur, pas un appel incomplet a executer. Si la generation est trop
courte, augmenter `max_tokens`.

En streaming, le texte ordinaire est emis progressivement. Les blocs
d'appels sont retenus jusqu'a validation du resultat, puis emis comme
deltas SSE indexes `delta.tool_calls`, avant la fin `tool_calls` et
`[DONE]`. Les compteurs restent ceux des tokens reellement generes par
LFM3, balises comprises. Un exemple d'appel cite dans un bloc de code ou
dans une section de raisonnement n'est pas traite comme une action.

Le template reste volontairement sur `api.tool_calling: false` : pour un
autre modele, il faut son vrai formateur et son vrai parseur d'appels,
pas seulement activer un booleen. Dans OpenWebUI, desactiver les outils
integres pour ces modeles tant que leur adaptateur ne les implemente pas.

Reference :
[connexion OpenAI-compatible d'OpenWebUI](https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-openai-compatible/).

## 3. Telechargement, smoke tests et benchmarks reels

```bash
# Telecharge uniquement ; n'importe pas et n'execute pas le code du modele.
docker compose run --rm lab python scripts/download.py lfm3
docker compose run --rm lab python scripts/download.py d1-2b

# Verifie chargement + inference et les assertions propres au modele.
docker compose run --rm lab python scripts/run.py lfm3 --test
docker compose run --rm lab python scripts/run.py d1-2b --test

# Une seule session chargee, warmup exclu des mesures.
docker compose run --rm lab python scripts/benchmark.py lfm3 --runs 5 --warmup 1
docker compose run --rm lab python scripts/benchmark.py d1-2b --runs 5 --warmup 1

# Une fois le snapshot complet en cache :
docker compose run --rm lab python scripts/run.py lfm3 --offline
```

Options communes : `--config fichier.yaml` (configuration complete, pas un
patch), `--device`, `--dtype`, `--seed`, `--revision`, `--offline`, `--debug`.
Le chargeur interprete device/dtype : le coeur ne les impose pas aux backends.
Pour les chargeurs PyTorch fournis, `auto` choisit CUDA, puis MPS, puis CPU ;
un `cuda` explicite indisponible provoque une erreur, pas un repli silencieux.
`dtype: auto` choisit bf16 si possible sur CUDA, sinon fp16, et fp32 sur CPU.

Les rapports JSON sont ecrits atomiquement dans [results/](results/), avec :

- configuration, revision HF immuable, seed et empreinte du code ;
- versions effectives des paquets, Python, architecture, device et dtype ;
- temps de chargement separe, nombre de warmups et mesures individuelles ;
- latences de bout en bout, moyenne, mediane, p95 et requetes/seconde ;
- tokens generes/seconde lorsque le modele en produit reellement ;
- pics de memoire de l'allocateur PyTorch CUDA et etats `nvidia-smi`
  avant/apres la mesure.

Ces etats GPU sont des instantanes de la machine, pas un echantillonnage
continu ni une mesure isolee des autres processus. Sur GB10, la memoire
unifiee peut apparaitre `N/A` dans `nvidia-smi` : cette valeur est preservee.
Les pics PyTorch ne couvrent pas les allocations d'un backend non PyTorch.
Les metriques indisponibles sont indiquees comme telles.

Les temps comprennent preparation, inference et decodage. CUDA/MPS est
synchronise pour les modeles PyTorch. Ce n'est pas un benchmark de service
concurrent, ni une mesure TTFT/streaming. Pour d1, les sorties sont des
decisions, pas des tokens generes : le debit de tokens est `null`.

## 4. Ajouter un modele : aucun registre central

```bash
cp -r models/template models/mon-modele
```

Modifier en general seulement :

1. `config.yaml` : identifiant, revision et parametres propres au modele ;
2. `load.py` : `load_model(config) -> (model, processor)` ;
3. `inference.py` :
   `infer(model, processor, input_data, config) -> InferenceResult`.

Le nom du dossier est decouvert automatiquement, y compris les noms avec
tirets. Les apps du template sont deja presentes. Adapter aussi `test.py`
et/ou les apps si des assertions ou une interface specialisees sont utiles.

Le contrat commun se trouve dans [common/types.py](common/types.py) :

```python
from common.types import InferenceResult

def load_model(config):
    # Ici : votre bibliotheque, votre modele, et eventuellement un processor.
    return model, processor

def infer(model, processor, input_data, config):
    # Texte, decisions, embeddings ou autre resultat JSON.
    return InferenceResult(
        output={"decisions": ["inspect"]},
        input_tokens=None,
        output_tokens=None,
    )
```

`input_data` est une chaine ou un dictionnaire. Le dictionnaire peut contenir
les donnees JSON propres au modele ; une app specialisee peut y ajouter une
image PIL chargee. `output` doit etre serialisable en JSON, sans NaN/Inf.
Ne pas inventer de compteurs : utiliser `None` quand ils ne s'appliquent pas.

Les apps recoivent une `ModelSession` deja chargee et appellent simplement :

```python
result = session.infer(input_data)
```

Le template texte inclut `api.chat: true` et un hook optionnel de streaming
dans [inference.py](models/template/inference.py). L'app API herite donc
automatiquement des routes OpenAI, sans recopier une implementation HTTP.
Un adaptateur chat doit accepter `{"messages": [...], "generation": {...}}`
avec les surcharges limitees du protocole et retourner une sortie texte avec
`finish_reason="stop"`, `"length"` ou `"tool_calls"`. Ce champ supplementaire du resultat
interne ne change pas la reponse JSON historique de `/infer`.
Lorsqu'un appel natif est present, un champ `tool_calls` est ajoute a
la reponse generique sans modifier `output` et `usage`.

Pour le streaming, le hook est un generateur :
`stream(model, processor, input_data, config, cancel)`.
Il produit des `StreamChunk(text=...)`, puis exactement un
`StreamChunk(result=InferenceResult(...))` final. Il doit respecter
`cancel` (un `threading.Event`) et nettoyer ses ressources dans `finally`.
La session maintient le meme verrou pendant tout le flux. Le helper
`stream_causal` couvre ce contrat pour les modeles causaux Transformers.
Un adaptateur chat sans ce hook reste utilisable sans streaming ; une
requete `stream=true` est alors refusee.

Pour un modele non-chat, **desactiver `api.chat` et retirer ou remplacer
le hook de streaming herite du template**. d1 n'annonce ainsi aucun modele
dans `/v1/models` et conserve son contrat de decisions via `/infer`.

Le coeur n'appelle jamais `generate()`, `system_one()` ou une API de backend.
Les helpers Transformers sont optionnels. Un adaptateur ONNX, diffusion,
vision, audio ou embeddings peut donc remplacer completement le chargement
et l'inference. Son chargeur gere ses seeds et son accelerateur. Les entrees
et sorties differentes passent par le meme contrat ; la demo JSON du template
reste disponible, ou peut etre remplacee par une demo adaptee.

Pour les depots Hugging Face, `download_snapshot(config)` exige un SHA complet
et met en cache **le snapshot entier**, pas seulement des fichiers supposes
etre des poids Transformers. Le chargement se fait ensuite sur ce chemin
local, ce qui epingle aussi tokenizer, processor et code distant. Un chargeur
utilisant une autre source n'est pas oblige d'utiliser ce helper.

## 5. Ajouter des dependances ou un runtime different

### Meme environnement : aucune autre image

Un modele compatible avec les bibliotheques communes utilise directement
`ai4atexis_lab_fw:transformers`. Changer `MODEL` ne reconstruit pas l'image et ne
duplique pas les dependances.

### Dependances supplementaires : un verrou et un tag distincts

Le [requirements.in du template](models/template/requirements.in) inclut les
versions principales communes. Ajouter des versions exactes propres au modele,
puis generer **son verrou complet**, plutot que modifier une image en cours
d'execution :

```bash
# uv 0.13.6 est un outil de maintenance, pas une dependance des apps.
uv pip compile --universal --python-version 3.12 --generate-hashes \
  --no-header --no-emit-index-url \
  models/mon-modele/requirements.in \
  -o models/mon-modele/requirements.txt

RUNTIME=mon-runtime \
RUNTIME_REQUIREMENTS=models/mon-modele/requirements.txt \
docker compose build lab

RUNTIME=mon-runtime MODEL=mon-modele APP=gradio docker compose up lab
```

On peut aussi conserver `RUNTIME`, `RUNTIME_REQUIREMENTS` et `MODEL` dans `.env`.
Le verrou de ce runtime **remplace** le verrou par defaut lors de la
construction : il n'est pas installe apres coup en cassant silencieusement les
versions existantes. L'image commune garde son propre tag.

Pour une version de Transformers incompatible avec la version commune,
retirer l'include commun du `requirements.in` specialise et definir son
ensemble coherent de dependances. Conserver au minimum PyYAML pour le coeur,
et les dependances des apps communes que le modele souhaite reutiliser.
Le resolveur doit reussir avant de construire l'image.

### Autre base, backend ou bibliotheques systeme : Dockerfile dedie

Definir un Dockerfile propre au modele ou a une famille de modeles, puis :

```bash
RUNTIME=autre-backend DOCKERFILE=models/mon-modele/Dockerfile \
docker compose build lab

RUNTIME=autre-backend MODEL=mon-modele APP=cli docker compose up lab
```

Ce Dockerfile doit copier le projet dans `/workspace`, installer le coeur et
les apps souhaites, et fournir un executable `python`. Il peut remplacer
completement PyTorch. Conserver `/cache/huggingface` pour le cache HF et
`/workspace/results` pour les rapports si ces fonctionnalites sont utilisees.
La reservation GPU du Compose fourni convient a cette machine NVIDIA ; un
runtime CPU pur sur une machine sans GPU necessitera une variante Compose
sans `gpus: all`.

**Le nombre d'images suit les incompatibilites de dependances, pas le nombre
de modeles.** Les poids restent dans le volume `hf-cache`, hors des images.

## Utiliser le framework avec une IA

Des skills de projet sont disponibles dans [.github/skills/](.github/skills/).
Les [instructions Copilot](.github/copilot-instructions.md) orientent l'agent
vers le workflow adapte, sans charger tous les details en permanence.

| Skill | Usage |
|---|---|
| [ai4atexis-lab-fw-add-model](.github/skills/ai4atexis-lab-fw-add-model/SKILL.md) | Integrer un modele avec le bon contrat, ses apps et ses tests |
| [ai4atexis-lab-fw-runtime](.github/skills/ai4atexis-lab-fw-runtime/SKILL.md) | Reutiliser une image ou isoler des dependances incompatibles |
| [ai4atexis-lab-fw-apps](.github/skills/ai4atexis-lab-fw-apps/SKILL.md) | Creer une demo CLI, Gradio ou FastAPI adaptee au modele |
| [ai4atexis-lab-fw-test-benchmark](.github/skills/ai4atexis-lab-fw-test-benchmark/SKILL.md) | Valider hors ligne ou effectuer une mesure reelle autorisee |

Exemples dans un client prenant en charge les Agent Skills, comme Copilot :

```text
/ai4atexis-lab-fw-add-model Ajoute le modele officiel <organisation/modele>,
avec ses apps et tests, sans telecharger les poids.

/ai4atexis-lab-fw-runtime Ce modele demande une version de Transformers
incompatible avec le runtime commun. Prepare un environnement isole.

/ai4atexis-lab-fw-apps Cree une demo Gradio adaptee aux entrees de ce modele.

/ai4atexis-lab-fw-test-benchmark Valide les adaptateurs et les apps sans poids.
```

On peut aussi formuler la demande en langage naturel ; les descriptions
permettent a l'agent de selectionner le skill pertinent. Leur disponibilite
depend de la prise en charge et de la decouverte des skills par le client.
Si les nouveaux skills n'apparaissent pas, recharger la fenetre VS Code et
ouvrir une nouvelle conversation dans ce workspace.

Les workflows distinguent toujours tests simules, construction Docker et
inference reelle. Le perimetre par defaut reste **sans poids** ; un smoke
test ou benchmark reel ne se lance pas simplement pour verifier des fichiers.

## Reproductibilite et limites

- Image de base epinglee par digest multiarchitecture.
- Dependances Python transitives verrouillees avec hashes.
- Snapshots LFM3/d1 epingles sur des commits verifies au 29 octobre 2026.
- Seed configuree dans chaque adaptateur PyTorch fourni.
- Environnement effectif et source identifies dans chaque benchmark.

Une seed ne garantit pas des sorties identiques bit a bit entre GPU,
bibliotheques, kernels ou ordres de requetes. Les apps consomment une suite
aleatoire par session ; ne pas confondre cela avec une reseed par requete.
LFM3 peut aussi emettre des traces de raisonnement malgre
`enable_thinking: false`, selon le comportement documente du modele.

Les adaptateurs ont `trust_remote_code: true` car les checkpoints officiels
utilisent du code personnalise. Ce code s'executera dans le conteneur lors
du chargement : verifier son origine et la licence Liquid avant utilisation.
Un conteneur n'est pas une garantie d'innocuite pour du code non fiable.

Le fine-tuning, la quantification, les entrees video/audio et les runtimes
optimises ne sont pas implementes dans cette premiere version. Ils peuvent
etre ajoutes sous forme d'adaptateurs ou de workflows distincts.

References officielles :

- [LFM3-1.7B, revision utilisee](https://huggingface.co/LiquidAI/LFM3-1.7B/tree/ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7)
- [d1-2B, revision utilisee](https://huggingface.co/LiquidAI/d1-2B/tree/cfe5760ff5d5998f91d9ac56bd4b2dbd96996d41)
- [Image NVIDIA PyTorch 25.09](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-09.html)
