# Halal Trading — système interne de prévision et de trading automatisé

Application interne qui prévoit, pour chaque actif autorisé, la probabilité
d'atteindre un objectif de gain, achète les meilleures opportunités et revend à
l'objectif, au stop ou à l'échéance. Actions et crypto.

**Deux niveaux de conformité halal**, imposés par le code (module `compliance`),
pas par un prompt ni par le modèle :

1. **Nos opérations — tolérance zéro sur l'intérêt** : compte « cash », pas de
   marge, pas de levier, pas de vente à découvert, pas de solde négatif, aucun
   intérêt reçu ou payé, pas de staking.
2. **Les sociétés — norme AAOIFI** : dette à intérêt ≤ 30 % de la capitalisation,
   placements à intérêt ≤ 30 %, revenus non conformes (intérêts compris) ≤ 5 % du
   CA, avec purification des dividendes. Exclusions d'activité totales : banque,
   assurance, crédit, alcool, porc, jeux d'argent, tabac, adulte, armement, drogues.
   Un profil `zero_tolerance` reste disponible dans `config/compliance.yaml`.

## Architecture

```
Données ─▶ Features ─▶ Modèles ML ─▶ Décision ─▶ CONFORMITÉ ─▶ Exécution broker
(OHLCV,    (techniques,  (P(objectif    (espérance   (bloque tout    (IBKR actions,
 fondamentaux) volatilité)  atteint)      de gain)     ordre non       Coinbase crypto)
                                                       conforme)            │
         Surveillance du compte (flux d'intérêt) ◀─────────────────────────┘
```

### Modèles : un modèle par classe d'actifs, pas un par actif

- **Un modèle « actions »** entraîné sur l'ensemble des actions halal, et
  **un modèle « crypto »**, avec un pipeline et une interface communs.
- Pourquoi séparer : les deux marchés n'ont ni les mêmes horaires (bourse vs
  24h/24), ni la même volatilité, ni les mêmes moteurs (résultats et
  fondamentaux pour les actions ; tendance du BTC et liquidité pour la crypto),
  ni les mêmes frais. Les objectifs de gain et les stops diffèrent aussi.
- Pourquoi ne pas faire un modèle par actif : trop peu de données par actif,
  forte tendance au sur-apprentissage. Un modèle commun par classe apprend des
  régularités sur des centaines d'actifs, et reçoit l'identité de l'actif sous
  forme de caractéristiques (secteur, volatilité, taille...).

### Cible de prévision

Pour chaque actif : probabilité d'atteindre **+X % avant −Y % dans N jours**
(« triple barrier »), avec X et Y adaptés à la volatilité de l'actif.
Décision : `espérance = P(gain) × objectif − P(perte) × stop − frais`.

## Conformité (`src/halal_trading/compliance`)

| Composant | Rôle |
|---|---|
| `policy.py` + `config/compliance.yaml` | Politique tolérance zéro. Refuse tout seuil > 0, tout instrument interdit, tout stablecoin. |
| `screening.py` | Filtre des sociétés (activité + finances) et des crypto. Donnée manquante = refus. |
| `universe.py` | Univers validé, re-filtrage obligatoire tous les 100 jours. |
| `order_guard.py` | Contrôle de chaque ordre : actif validé, pas de marge, pas de levier, pas de vente à découvert, solde jamais négatif (frais compris). |
| `account_guard.py` | Au démarrage : compte « cash », prêt de titres, rémunération du cash, staking et récompenses désactivés. Sinon le bot ne démarre pas. |
| `ledger_monitor.py` | Analyse l'historique du compte : tout intérêt reçu ou payé arrête le bot. |
| `purification.py` | Registre des montants à donner (intérêt reçu par erreur, part non conforme d'un dividende). |
| `audit.py` | Journal JSONL de toutes les décisions, pour revue par un savant. |

### Filtre des sociétés

Exclusion si l'une de ces conditions est vraie :
- activité interdite, détectée dans le secteur, l'industrie **et** la
  description d'activité (ex. LVMH : « Luxury Goods » mais vins et spiritueux) ;
- un ratio financier dépasse le seuil du profil actif ;
- donnée nécessaire inconnue (une estimation prudente — borne haute — est
  utilisée quand une ligne détaillée manque, et signalée dans le rapport) ;
- comptes de plus de 400 jours ;
- instrument autre qu'une action ordinaire (actions de préférence, ETF...).

Le rapport donne pour chaque société conforme son **taux de purification** :
la part de chaque dividende à donner.

Les locations simples (loyers) ne sont pas comptées comme dette par défaut :
point à faire valider (`count_operating_leases_as_debt`).

### Crypto

Liste blanche explicite, **vide par défaut** : chaque crypto doit être ajoutée
dans `config/compliance.yaml` avec le nom de la personne qui l'a validée.
Stablecoins, prêt, staking, rendement, dérivés et levier sont toujours interdits.
Cotation en monnaie fiat uniquement (EUR, USD).

## Collecte des données (`src/halal_trading/data`)

Option A (gratuite, prototype) : Coinbase pour la crypto, Yahoo Finance (yfinance)
pour les actions. Configuration dans `config/data_sources.yaml`.

```bash
halal-data crypto     # prix crypto journaliers depuis 2017 (Coinbase, sans clé)
halal-data stocks     # prix actions ajustés, dividendes, divisions (Yahoo)
halal-data screen     # filtre halal tolérance zéro de l'univers de départ
halal-data quality    # contrôle qualité des données stockées
```

- Stockage local Parquet dans `data/` (non versionné), mise à jour incrémentale.
- Chaque filtrage archive les fondamentaux bruts datés : on construit ainsi
  notre propre historique du statut halal, trimestre après trimestre.
- **Limite de Yahoo** : il ne fournit pas les « autres revenus non conformes »,
  ni la distinction crédit-bail / location simple. Sans vérification manuelle
  (`config/manual_reviews.yaml`, avec auteur et source), la société est exclue.
  Une vérification ne complète que les champs manquants, jamais une valeur
  fournie par la source.

## Modèle de prévision (`src/halal_trading/model`)

```bash
halal-model backtest stocks   # validation walk-forward + simulation, frais déduits
halal-model predict stocks    # espérance de gain par action au dernier cours
```

- **Cible** : rendement réel du trade avec sortie à l'objectif (3σ), au stop
  (1,5σ) ou à l'échéance (20 jours de cotation). Achat à l'ouverture du
  lendemain de la décision, stop prioritaire si les deux sont touchés le même jour.
- **Modèles** : référence naïve, linéaire (Ridge), gradient boosting. Le
  linéaire est retenu (plus stable).
- **Validation** : entraînement sur le passé, test sur l'année suivante, purge
  des étiquettes qui chevauchent la période de test.

### Résultats (actions halal, 2012 → 2026)

| | Rendement annuel | Sharpe |
|---|---|---|
| Sorties serrées (10 j, 1σ) | 7 à 16 % | 0,5 à 1,0 |
| Réglage retenu, choisi sur 2012-2019, testé sur 2020-2026 | +15 % à +30 % selon le mode | 0,8 à 1,4 |
| Acheter et garder les mêmes actions, 2020-2026 | +17 % | 0,83 |

Limites : univers de 23 actions choisies parmi les gagnantes d'aujourd'hui
(biais du survivant : tous les chiffres sont gonflés), pouvoir de sélection
faible (IC ≈ 0,02), perte maximale de 30 à 40 %. Crypto : avec seulement BTC et
ETH, le modèle n'apporte rien.

## Feuille de route

1. ~~Socle et module de conformité~~
2. ~~Module de collecte (prix, fondamentaux, filtrage, qualité)~~
3. ~~Features, labels « triple barrière », backtest walk-forward avec frais~~
4. ~~Modèles (référence, linéaire, gradient boosting)~~ — élargir l'univers pour réduire le biais du survivant
5. Moteur de décision et de risque, paper trading (compte démo IBKR)
6. Connecteurs réels, petit budget, dashboard interne et alertes

## Développement

```bash
pip install -e ".[dev]"
pytest
```
