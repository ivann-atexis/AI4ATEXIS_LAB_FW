---
name: ai4atexis-lab-fw-runtime
description: 'Configurer ou corriger un environnement AI4ATEXIS_LAB_FW : Docker Compose, image dediee, requirements, conflits Transformers/PyTorch/CUDA, version de backend, ARM64/AMD64 ou ajout de dependances. Use when selecting, extending, pinning or isolating a model runtime without changing other models or the host Python environment.'
argument-hint: 'Modele ou famille, dependances requises, runtime existant ou nouveau'
---

# Gerer les runtimes AI4ATEXIS_LAB_FW

## Principe

Un runtime est un ensemble coherent de dependances, pas un synonyme de modele.
Reutiliser une image si possible ; isoler une incompatibilite reelle.
Ne pas mettre a niveau l'hote, modifier une image en cours d'execution ou
remplacer les binaires CUDA pour contourner un conflit.

## Procedure

1. Lire le [README](../../../README.md), le
   [Compose](../../../compose.yaml), le
   [Dockerfile](../../../docker/Dockerfile) et les
   [dependances principales](../../../docker/requirements.in).
   Confirmer les besoins dans les sources officielles du modele. Identifier
   Python, backend, architecture CPU, GPU/pilote et bibliotheques systeme.
   Ne pas supposer que le daemon Docker ou le GPU sont accessibles.
2. Choisir la solution la plus simple :

   | Situation | Solution |
   |---|---|
   | Les dependances actuelles suffisent | Reutiliser `ai4atexis_lab_fw:transformers` |
   | Paquets supplementaires compatibles | Verrou complet specialise avec l'include commun |
   | Versions de bibliotheques incompatibles avec le runtime commun | Verrou independant et tag distinct |
   | Autre version de Python, base, backend ou bibliotheques systeme | Dockerfile dedie et tag distinct |

   Ne pas creer une nouvelle image uniquement parce qu'un nouveau modele
   arrive. Ne pas toucher aux versions communes pour satisfaire un seul
   adaptateur.
3. Pour un verrou specialise, partir de
   [requirements.in du template](../../../models/template/requirements.in).
   L'include `../../docker/requirements.in` conserve les versions communes
   seulement si elles sont compatibles. Sinon le retirer et declarer un
   ensemble independant coherent, avec PyYAML pour le coeur et les paquets
   des apps communes effectivement reutilisees.
4. Epingler les versions directes, puis generer le verrou transitif avec
   hashes. Ne pas editer manuellement les hashes ni installer un second jeu
   de dependances apres le verrou. Exemple, depuis la racine :

   ```bash
   uv pip compile --universal --python-version 3.12 --generate-hashes \
     --no-header --no-emit-index-url \
     models/mon-modele/requirements.in \
     -o models/mon-modele/requirements.txt
   ```

   Adapter la version de Python a la base choisie. Utiliser la version de
   maintenance de `uv` documentee dans le README. N'installer un outil
   manquant que dans un environnement isole, apres un besoin constate.
   Examiner les conflits du resolveur ; pas de `--no-deps` pour les masquer.
5. Verifier que la resolution ne remplace pas involontairement les PyTorch
   et torchvision CUDA fournis par la base NVIDIA. Un nouveau besoin
   incompatible avec ces binaires necessite une base coherente, pas une
   roue generique ajoutee par-dessus. Verifier aussi les roues ARM64/AMD64
   requises ; la resolution universelle seule ne prouve pas leur existence
   ni le fonctionnement CUDA.
6. Donner au runtime un tag distinct et un verrou selectionne explicitement :

   ```bash
   RUNTIME=mon-runtime \
   RUNTIME_REQUIREMENTS=models/mon-modele/requirements.txt \
   docker compose config --quiet
   ```

   `RUNTIME` choisit seulement le tag d'image et son identification.
   `RUNTIME_REQUIREMENTS` remplace le verrou utilise a la construction.
   `MODEL` choisit l'adaptateur ; `APP` choisit sa mini-app.
   Ces variables sont independantes et ne sont pas un registre automatique.
7. Si la base doit changer, creer un Dockerfile dedie, epingle par digest.
   Il doit fournir `python`, copier le projet dans `/workspace`, installer
   le coeur et les apps requises, et conserver les chemins de cache/resultats
   utiles. Aucun telechargement de poids ni secret pendant le build.
   Selectionner ce fichier avec `DOCKERFILE=models/mon-modele/Dockerfile`.
   Le contexte de construction reste la racine du depot.
8. Documenter les variables et commandes correspondant a ce runtime.
   Ne pas ecraser un `.env` utilisateur existant. Les tokens n'ont leur
   place ni dans les requirements, ni dans le Dockerfile, ni dans Git.
   Ne pas afficher un Compose resolu contenant un token dans des logs.

## Construction et execution, si demandees

La generation d'un verrou consulte les index de paquets ; un build telecharge
une image et des dependances, mais pas de poids. Ne pas lancer un build
volumineux si la demande porte seulement sur sa preparation.

```bash
RUNTIME=mon-runtime \
RUNTIME_REQUIREMENTS=models/mon-modele/requirements.txt \
docker compose build lab

# Charge vraiment le modele et peut telecharger des poids : accord requis.
RUNTIME=mon-runtime \
RUNTIME_REQUIREMENTS=models/mon-modele/requirements.txt \
MODEL=mon-modele APP=gradio docker compose up lab
```

Pour un Dockerfile dedie, garder egalement `DOCKERFILE` coherent entre
configuration, construction et execution. Les sources sont copiees dans
l'image : une modification exige une reconstruction avant les tests Docker.
Le Compose fourni reserve un GPU NVIDIA ; un runtime CPU sans GPU exige
une variante sans cette reservation, pas un faux succes CUDA.

En cas d'acces Docker refuse, signaler le blocage sans modifier les permissions
du socket, les groupes ou le pilote de la machine. Valider ce qui reste
possible, notamment `docker compose config --quiet`.

## Critere de fin

Fournir le choix de runtime motive, le verrou reproductible ou le Dockerfile
necessaire, les commandes coherentes et l'etat reel des validations.
Un resolveur satisfait ou un Compose valide ne prouve pas qu'une image se
construit, ni qu'un modele fonctionne sur GPU.
