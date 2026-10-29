# ModelForge - AI4ATEXIS_LAB_FW

Banc de developpement et de compatibilite **day-0**, en Transformers direct
pour les premiers adaptateurs. Ce n'est pas un serveur vLLM/SGLang, ni une
abstraction qui impose la generation de texte a tous les modeles.

**Le modele et son environnement sont deux choses distinctes :**

- un **adaptateur** definit comment charger et executer un modele ;
- un **runtime Docker** fournit ses bibliotheques et ses versions ;
- plusieurs adaptateurs compatibles partagent une image ;
- un modele avec des dependances incompatibles utilise une autre image,
  sans modifier les commandes, les autres adaptateurs ou le coeur du framework.

## Ce qui est inclus

| Adaptateur | Chargement / inference | Mini-app Gradio |
|---|---|---|
| `lfm3` | `LiquidAI/LFM3-1.7B`, `AutoModelForCausalLM`, chat template puis `generate()` | Chat texte avec historique |
| `d1-2b` | `LiquidAI/d1-2B`, `AutoModel`, `system_one()` en une passe | Etat texte, image facultative, questions JSON, decisions |
| `template` | Point de depart a copier ; pas un modele pret a charger | Formulaire texte / objet JSON |

Chaque adaptateur contient aussi une app CLI, une app FastAPI et un smoke test.
Les apps sont de petits points d'entree vers les implementations communes :
on peut remplacer une app d'un modele sans dupliquer les autres.

**Aucun poids n'est livre ou telecharge par la creation du projet, la construction
de l'image ou les tests automatises.** En revanche, lancer une vraie inference,
un smoke test de modele ou un benchmark telecharge le snapshot configure s'il
n'est pas deja en cache. `--offline` interdit ce telechargement ; le chargement
reel echoue explicitement si les fichiers necessaires manquent dans le cache.

## Organisation

```text
.
|-- compose.yaml
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

Ouvrir <http://127.0.0.1:7860>. d1 permet de charger une image et de modifier
les questions dans l'interface. LFM3 utilise une vraie conversation, pas la
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

Documentation interactive : <http://127.0.0.1:7860/docs>.
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
`modelforge:transformers`. Changer `MODEL` ne reconstruit pas l'image et ne
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
