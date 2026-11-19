# Cine Benchmark

Classement des films et séries partagés dans un groupe d'avis (type « On a regardé pour vous »).
Pour chaque publication et ses commentaires, l'appli :

1. **anonymise** le texte : noms des auteurs, mentions `@…`, liens et boutons Facebook
   (« J'aime », « Répondre », horodatages…) sont supprimés avant tout stockage ;
2. **identifie l'œuvre** (titre, type, année) ;
3. **note chaque avis de 0 à 10** à partir de son texte ;
4. **classe les œuvres** avec une moyenne bayésienne : une œuvre doit accumuler plusieurs avis
   pour dépasser la moyenne générale (un seul 10/10 ne bat pas trente 8,5/10).

## Source des données

L'appli ne va pas chercher les données sur Facebook (connexion obligatoire, collecte automatisée
interdite par les conditions de Meta, plus d'API pour les groupes depuis 2024). On importe par
**copier-coller** : sur Facebook, déplier les commentaires d'une publication, tout sélectionner,
copier, puis coller dans la page « Importer ». L'appli sépare la publication des commentaires
toute seule.

## Démarrage

```bash
cd cine-benchmark
pip install -e ".[dev]"            # ajouter ,claude pour l'analyse par Claude
uvicorn cine_benchmark.app:app --reload
# puis ouvrir http://127.0.0.1:8000
```

La base SQLite est créée dans `data/cine.db` (non versionnée) ; autre emplacement : `CINE_DB=/chemin/cine.db`.

### Import en lot (ligne de commande)

```bash
cine-import exemples/collage_facebook.txt     # collages bruts séparés par une ligne =====
cine-import avis.json --brut                  # [{"publication": "...", "commentaires": ["..."], "titre": "facultatif"}]
```

Les doublons (même publication et mêmes commentaires) sont ignorés.

## Deux moteurs de notation

| Moteur | Quand | Qualité |
|---|---|---|
| **Lexique** (par défaut) | toujours, hors ligne, gratuit | bon sur les avis courts ; gère négations (« pas génial »), intensifs (« vraiment nul »), « mais » (la fin de la phrase compte double), émojis et notes explicites (« 8/10 », « ★★★★ ») ; ne comprend pas l'ironie |
| **Claude** | si `ANTHROPIC_API_KEY` est défini | comprend l'ironie et les nuances, reconnaît le titre officiel même mal écrit, ignore les commentaires hors sujet |

Avec Claude, chaque publication fait l'objet d'un appel (modèle `claude-opus-5-5`, modifiable via
`CINE_MODELE`). Le repli automatique côté serveur (`fallbacks`) est activé : si le modèle décline
une requête, l'API la rejoue sur un modèle de repli. Si l'appel échoue (réseau, quota), l'import
se fait quand même avec le lexique. Seul le texte déjà anonymisé est envoyé.

Un avis sans opinion (« @Paul regarde ça », une question) n'est pas noté et ne compte pas dans la moyenne.

## Corriger les données

Sur la fiche d'une œuvre : corriger le titre, le type ou l'année. Si le nouveau titre existe déjà,
les deux fiches sont **fusionnées** (utile quand deux personnes écrivent « The Bear » et « the bear »).
Une publication importée par erreur peut être supprimée.

## API JSON

- `GET /api/classement?type=série&q=dark&tri=score&min_avis=3`
- `GET /api/oeuvre/{id}`

## Tests

```bash
pytest
```
