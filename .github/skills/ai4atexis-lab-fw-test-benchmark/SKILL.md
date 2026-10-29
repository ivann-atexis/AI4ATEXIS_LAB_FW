---
name: ai4atexis-lab-fw-test-benchmark
description: 'Tester ou valider AI4ATEXIS_LAB_FW, un adaptateur, une app ou les contrats OpenAI/OpenWebUI, outils natifs, OpenAPI et streaming SSE ; distinguer tests hors ligne, smoke tests reels et benchmarks GPU. Use when running tests, verifying tool-call round trips, diagnosing a failed inference, checking streaming cancellation, measuring latency/memory/throughput or comparing model results with reproducible evidence.'
argument-hint: 'Modele ou surface, validation sans poids ou reelle, metriques souhaitees'
---

# Tester et benchmarker AI4ATEXIS_LAB_FW

## Choisir le bon niveau

| Niveau | Poids / inference reelle | Usage |
|---|---|---|
| Configuration, aide CLI, decouverte | Non | Verifier le cablage |
| Tests unitaires avec mocks | Non | Niveau par defaut |
| Construction Docker | Non, mais image et paquets telecharges | Verifier le runtime |
| Smoke test `scripts/run.py <modele> --test` | Oui | Compatibilite chargement + inference |
| App ou benchmark reel | Oui | Comportement et performances observes |

Une demande vague de validation reste sans poids. Demander l'accord avant
de changer de niveau si le perimetre n'est pas explicite. `HF_HUB_OFFLINE=1`
et `--offline` interdisent les telechargements HF, mais n'empechent ni le
chargement des poids en cache, ni l'utilisation du GPU. Ils ne constituent
pas une sandbox reseau pour un autre backend.

## Procedure hors ligne

1. Lire les tests touches et les commandes du [README](../../../README.md).
   Distinguer [tests/](../../../tests), executes par `unittest`, des fichiers
   `models/<nom>/test.py`, appeles par `--test` avec un modele reel.
   Verifier l'environnement disponible sans le modifier.
2. Choisir le plus petit ensemble utile, en une seule execution du runner.
   Exemples a adapter a la surface modifiee :

   ```bash
   HF_HUB_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=False \
     python -m unittest tests.test_adapters -v

   HF_HUB_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=False \
     python -m unittest tests.test_framework tests.test_apps -v
   ```

   Ajouter les tests du nouvel adaptateur s'il y en a. Pour une modification
   transversale, utiliser une seule decouverte complete :

   ```bash
   HF_HUB_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=False \
     python -m unittest discover -s tests -v
   ```

   Pour les routes OpenAI et le cycle de vie du streaming :

   ```bash
   HF_HUB_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=False \
     python -m unittest tests.test_openai tests.test_streaming tests.test_tools -v
   ```

   Verifier la decouverte du seul modele charge et compatible chat, le
   schema JSON, les erreurs, les parametres par requete sans mutation,
   les deltas avant fin de generation, les raisons d'arret, l'usage reel
   et la fermeture du worker/verrou sur deconnexion ASGI 2.3 et 2.4.
   Un test qui lit une reponse complete ne prouve pas un streaming reel.
   Une erreur de flux ne doit produire ni fin de succes ni `[DONE]`.
   d1 ne doit pas etre annonce comme chat.
   Pour un modele `api.tool_calling: true`, verifier aussi le cycle
   appel/resultat sans executer de vrai outil, les types JSON des arguments,
   les ids de l'historique, choix/parallelisme et deltas SSE indexes.
   Des appels cites dans du code/raisonnement ou mal formes ne doivent pas
   devenir des actions. Tester les references de schema locales et le refus
   des references externes, sans acces reseau implicite.

3. Simuler les chargeurs, les appels HF et le backend. Reutiliser
   [OfflineTestCase](../../../tests/helpers.py). Verifier formes exactes,
   erreurs et valeurs mesurables, pas seulement l'absence d'exception.
   Les sorties de benchmark simulees prouvent les calculs, pas la vitesse
   du modele. Ne pas ajouter d'acces reseau ou de poids aux tests unitaires.
4. Si les dependances locales manquent, utiliser le runtime isole approprie.
   Ne pas installer les paquets dans le Python/CUDA de l'hote pour faire
   passer les tests. Une image existante peut etre testee sans poids :

   ```bash
   docker compose run --rm -e HF_HUB_OFFLINE=1 lab \
     python -m unittest discover -s tests -v
   ```

   Choisir le bon `RUNTIME` si necessaire et ne pas tester une image contenant
   des sources perimees. Une construction requise se traite via
   [ai4atexis-lab-fw-runtime](../ai4atexis-lab-fw-runtime/SKILL.md).
5. Valider le cablage pertinent sans charger de modele :

   ```bash
   docker compose config --quiet
   python scripts/run.py --list-models
   python scripts/run.py --help
   python scripts/benchmark.py --help
   ```

   Ces verifications ne remplacent pas les tests du comportement.
   Pour OpenAPI, inspecter `/openapi.json` avec `TestClient` et verifier les
   routes et schemas, dont JSON/SSE pour Chat Completions. Pour le reseau
   OpenWebUI integre, valider le Compose combine : image locale avec
   `pull_policy: never`, volume persistant, ports locaux, mode API force
   et dependance a l'etat healthy du modele. Le reseau partage ne doit
   pas exiger `OPENWEBUI_NETWORK` dans cette variante.
   Pour une UI externe, valider `compose.openwebui.external.yaml` avec
   `OPENWEBUI_NETWORK` defini ; cela ne prouve pas que le reseau existe.
   En cas de blocage Docker ou d'import manquant, signaler l'etape bloquee
   au lieu de declarer une validation reussie.

## Smoke test et benchmark, seulement si autorises

1. Confirmer le modele, la revision immuable, le runtime, l'entree, les
   ressources disponibles et l'autorisation ou non de telecharger des poids.
   Consulter les rapports deja disponibles avant de relancer une mesure.
2. Charger une seule session pour un benchmark. Reutiliser
   [le script](../../../scripts/benchmark.py) et
   [les mesures communes](../../../common/benchmark.py), pas une boucle
   separee par app. Par defaut, interdire les nouveaux telechargements :

   ```bash
   docker compose run --rm lab \
     python scripts/run.py mon-modele --test --offline

   docker compose run --rm lab \
     python scripts/benchmark.py mon-modele \
       --runs 5 --warmup 1 --offline
   ```

   Choisir le `RUNTIME` approprie. Si le cache manque, annoncer l'echec ;
   ne pas retirer automatiquement `--offline`. Pour une execution avec
   telechargement explicitement autorise, verifier aussi `HF_HUB_OFFLINE`
   et `local_files_only` dans la configuration.
3. Verifier le rapport persiste dans `results/`, et pas seulement stdout :
   modele, revision, configuration, input, seed, empreinte du code,
   environnement, warmups, mesures et forme des sorties. Une sortie propre
   ne prouve pas l'atteinte d'un seuil de qualite : tester le critere demande.
4. Interpreter les metriques correctement :
   - latence de bout en bout, chargement separe et warmup exclu ;
   - synchronisation CUDA/MPS pour les modeles PyTorch ;
   - requetes/seconde, pas nombre de decisions/seconde ;
   - tokens/seconde fondes sur les vrais compteurs ; `null` si non applicable,
     notamment pour d1 sans generation ;
   - pics allocateur PyTorch distincts de la memoire totale de la machine ;
   - `nvidia-smi` avant/apres : instantanes, pas monitoring continu ni
     mesure isolee des autres processus ; conserver `N/A` sur GB10.
5. Pour une comparaison, aligner tache, entrees, sortie attendue, precisions,
   generation, warmups, nombre de runs et ressources, ou documenter les
   differences. Ne pas comparer des tokens/seconde de LFM3 a des decisions
   d1 comme une meme grandeur. La seed ne garantit pas une identite bit a bit.

## Bilan attendu

Rapporter la commande exacte sans secret, les tests passes/echoues, le rapport
cree s'il y en a et les limitations. Distinguer explicitement code valide,
inference reelle validee et performance mesuree. Pas de resultat invente,
de modele declare compatible via mocks seulement, ni de fallback silencieux
apres une erreur GPU.
