---
name: ai4atexis-lab-fw-add-model
description: 'Ajouter ou integrer un nouveau modele dans AI4ATEXIS_LAB_FW. Use when adding a model, creating an adapter, day-0 support, implementing load.py or inference.py, or adapting text, vision, decisions, embeddings, audio, diffusion or another backend. Preserve the shared contract and work without model weights by default.'
argument-hint: 'Nom du modele, source officielle, modalites et perimetre de validation'
---

# Ajouter un modele AI4ATEXIS_LAB_FW

## Objectif et limites

Livrer un adaptateur executable par les commandes existantes, ses trois
mini-apps et ses tests, sans imposer son backend aux autres modeles.
Par defaut : code, configuration, documentation et tests simules uniquement.
Ne pas telecharger de poids, appeler un chargeur reel ou lancer le GPU.

## Procedure

1. **Identifier le vrai modele.** Confirmer le nom du dossier, l'identifiant
   officiel, les modalites, le format d'entree/sortie et la licence. Lire la
   model card et l'exemple d'inference officiels. Pour HF, consulter les
   metadonnees pour obtenir le SHA complet et les versions minimales.
   Ces consultations ne justifient pas un telechargement de snapshot ni
   l'execution du code distant. Si la reference est ambigue, demander la
   precision plutot qu'integrer un autre modele.
2. **Lire les contrats actuels :**
   [types](../../../common/types.py),
   [session](../../../common/runtime.py),
   [configuration](../../../common/config.py) et
   [template](../../../models/template).
   Choisir le parcours de reference approprie :
   [LFM3](../../../models/lfm3) pour un chat causal,
   [d1](../../../models/d1-2b) pour des decisions non generatives.
   Ce sont des exemples, pas des architectures a imposer.
3. **Creer le dossier a partir du template**, sans ecraser un adaptateur
   existant. Exemple depuis la racine :

   ```bash
   cp -r models/template models/mon-modele
   ```

   Choisir un nom qui commence par un caractere alphanumerique et contient
   seulement des caracteres alphanumeriques, `_` ou `-`. La decouverte est
   automatique. Les noms avec tirets passent par `importlib` ; ne pas creer
   d'import Python statique invalide ni de registre central.
4. **Renseigner la configuration.** Retirer les placeholders. Fournir
   `model_id`, un `example_input` realiste et les options propres au modele.
   Pour le helper HF : `revision` est un SHA immuable de 40 caracteres, pas
   `main`. Mettre une seed valide ; laisser le chargeur interpreter device
   et dtype. Ne pas ajouter des parametres `generation` a un modele qui ne
   genere pas. `trust_remote_code` reste faux sauf necessite officielle
   verifiee et documentee. Le template texte active `api.chat: true` pour
   OpenAI/OpenWebUI : le desactiver pour les modeles non-chat.
5. **Implementer uniquement le chargement dans `load.py`.**

   ```python
   def load_model(config: Config) -> tuple[ModelType, ProcessorType | None]:
       ...
   ```

   `ModelType` et `ProcessorType` representent les vrais types du backend,
   pas de nouvelles classes a ajouter pour respecter cet exemple. Utiliser
   les types disponibles ; ne pas ajouter de casts pour masquer des erreurs.
   Importer les bibliotheques lourdes au chargement, sans effet de bord a
   l'import du module. Reutiliser les helpers
   [Transformers](../../../common/transformers.py) s'ils conviennent ; sinon
   implementer le backend dans l'adaptateur. Pour HF, utiliser
   [download_snapshot](../../../common/hub.py), puis charger modele et
   tokenizer/processor depuis ce chemin local avec `local_files_only=True`.
   Respecter le mode hors ligne, initialiser les seeds et le mode evaluation.
   Aucun `.to("cuda")` ou dtype impose en ignorant la configuration.
6. **Implementer uniquement l'inference dans `inference.py`.**

   ```python
   def infer(
       model: ModelType,
       processor: ProcessorType | None,
       input_data: InputData,
       config: Config,
   ) -> InferenceResult:
       ...
   ```

   Valider les entrees ; signaler les erreurs plutot que retourner un faux
   resultat. `input_data` est une chaine ou un dictionnaire. Retourner une
   sortie JSON, sans tenseur brut, NaN ou Inf. Convertir les valeurs du
   backend a la frontiere de l'adaptateur. Ne pas forcer `generate()` :
   utiliser la vraie API du modele. Pour un causal LM, respecter son chat
   template et ne decoder que les tokens nouveaux. Compter les tokens reels ;
   utiliser `None` si inconnus, et `0` seulement si le compteur est vraiment
   nul. Par exemple, d1 annonce zero token de sortie genere.
   Pour un adaptateur chat, accepter aussi les surcharges `generation`
   de la requete sans modifier `config`, et renseigner `finish_reason`
   (`stop` ou `length`) sur les resultats texte.
7. **Conserver les trois apps du template**, puis adapter seulement ce qui
   est necessaire via [ai4atexis-lab-fw-apps](../ai4atexis-lab-fw-apps/SKILL.md).
   Les apps utilisent `session.infer(input_data)`, sans charger leur propre
   modele. Conserver le formulaire generique si une demo specialisee n'est
   pas utile ; ne pas transformer automatiquement tous les modeles en chat.
   L'app API commune inclut deja `/infer`, les routes OpenAI conditionnelles
   et la documentation OpenAPI ; ne pas les recopier dans le modele.
8. **Choisir le runtime**, sans modifier les versions des modeles existants.
   Reutiliser l'image commune si compatible. Sinon suivre
   [ai4atexis-lab-fw-runtime](../ai4atexis-lab-fw-runtime/SKILL.md).
   Le choix du runtime est explicite dans Compose ou `.env`, pas deduit
   automatiquement d'un champ YAML ajoute au modele.
9. **Ajouter deux niveaux de tests.** Adapter `models/mon-modele/test.py` :
   `test(session) -> InferenceResult`, avec assertions propres a la sortie
   reelle. Ajouter separement des tests hors ligne dans
   `tests/test_mon_modele.py`, avec chargeur, snapshot et backend simules.
   Reutiliser [OfflineTestCase](../../../tests/helpers.py). Couvrir les
   entrees invalides, la forme JSON, les compteurs et la transmission de la
   configuration. Un test de sortie constante ne remplace pas ces assertions.
10. **Documenter** l'adaptateur, les sources officielles, son runtime, les
    limites et les commandes dans le [README](../../../README.md).

## Streaming optionnel pour un modele chat

Le template fournit deja un generateur
`stream(model, processor, input_data, config, cancel)` dans `inference.py`.
Il delegue au helper causal et retourne des `StreamChunk(text=...)`, puis
exactement un `StreamChunk(result=InferenceResult(...))` final.
La session partage son verrou entre inference et flux.

Pour un autre backend chat, adapter ce hook a sa vraie API de streaming,
respecter `cancel` (`threading.Event`) et fermer les ressources dans `finally`.
Le resultat final doit conserver les vrais compteurs et la raison d'arret.
Ne pas simuler un streaming en decoupant un resultat complet. Sans hook,
le chat non-streaming reste possible et `stream=true` est refuse explicitement.

Pour des decisions, embeddings, diffusion ou un autre modele non-chat,
desactiver `api.chat` et **retirer ou remplacer le hook herite du template**,
sinon il resterait branche sur le backend causal. Les apps generiques et
`/infer` continuent de fonctionner sans compatibilite chat.

## Validation sans poids

Adapter le nom du module de test a celui qui vient d'etre cree. Utiliser le
Python de l'environnement approprie, pas installer les dependances sur l'hote.

```bash
python scripts/run.py --list-models
HF_HUB_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=False \
  python -m unittest tests.test_mon_modele -v
```

Ajouter les tests communs concernes si un helper commun a change.
Pour un adaptateur chat, couvrir aussi decouverte `/v1/models`, contrat
Chat Completions, surcharges par requete et eventuel streaming. Prendre
[les tests OpenAI](../../../tests/test_openai.py) et
[les tests de streaming](../../../tests/test_streaming.py) comme references,
toujours avec des doubles, sans charger de poids.
Ne pas executer `python scripts/run.py mon-modele --test` pour une validation
sans poids : cette commande charge vraiment le modele.

## Bilan attendu

Indiquer le contrat d'entree/sortie, le runtime choisi, les fichiers ajoutes,
les sources verifiees et les tests executes. Lister distinctement ce qui
reste non verifie : construction Docker, chargement, inference GPU ou demo
reelle. Ne pas declarer le modele compatible sur la seule base de mocks.
