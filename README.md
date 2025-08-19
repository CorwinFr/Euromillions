# 🎯 Projet de Prédiction EuroMillions

Ce projet implémente **deux approches complémentaires** pour prédire les tirages d'EuroMillions, combinant statistiques avancées, machine learning et intelligence artificielle.

## 🏆 Les Deux Approches

### 1. 🧮 Le Classement Malin (Stats + ML)

**Philosophie** : Traiter chaque numéro comme un joueur de football avec sa propre "forme" et ses caractéristiques.

#### Comment ça marche ?

Chaque numéro (1-50) et étoile (1-12) est analysé selon plusieurs critères :

- **📈 Forme récente** : Fréquence d'apparition sur différentes fenêtres temporelles (10, 25, 50, 100, 200 tirages)
- **⏰ Temps depuis la dernière apparition** : Combien de tirages se sont écoulés depuis la dernière sortie
- **🤝 Co-occurrences** : Avec quels autres numéros ce numéro sort-il habituellement ?
- **📊 Tendances exponentielles** : Pondération décroissante des tirages anciens (demi-vie configurable)

#### Architecture Technique

1. **🎯 Ranker (LGBMRanker/LambdaMART)** : Classe les numéros du plus au moins prometteur
2. **🔍 Classifieur (LGBMClassifier)** : Donne une probabilité de sortie pour chaque numéro
3. **⚖️ Calibration OOF** : Utilise la régression isotonique pour calibrer les probabilités
4. **🔄 Ensembles multi-fenêtres** : Combine plusieurs modèles entraînés sur différentes périodes
5. **🎲 Génération de portefeuille** : Crée 20 grilles diversifiées via échantillonnage Gumbel-Top-k

#### Fonctionnalités Avancées

- **GPU LightGBM** : Accélération automatique sur GPU avec fallback CPU
- **Backtest rolling-origin** : Validation sur 20 périodes avec métriques NDCG@K et Recall@K
- **Visualisations** : Graphiques PNG des performances historiques
- **Mapping automatique** : Support des CSV français avec colonnes `date_de_tirage`, `boule_1`, etc.

### 2. 🦜 Le Perroquet Séquentiel (IA Transformer)

**Philosophie** : Un modèle Transformer qui "lit" l'historique des tirages et "écrit" des combinaisons plausibles.

#### Comment ça marche ?

1. **📚 Apprentissage séquentiel** : Le modèle lit une fenêtre de 60 tirages passés
2. **🧠 Architecture Transformer** : Encodeur-décodeur avec attention multi-têtes
3. **🎯 Génération contrainte** : 
   - Numéros uniques pour les boules (1-50)
   - Étoiles indépendantes (1-12)
   - Top-K sampling pour la diversité
4. **🔀 Aléa contrôlé** : Balance entre cohérence historique et exploration

#### Architecture Technique

- **Encodeur** : Traite la séquence d'historique (W=60 tirages × 7 numéros + séparateurs)
- **Décodeur** : Génère séquentiellement les 5 boules puis 2 étoiles
- **Positional Encoding** : Comprend l'ordre temporel des tirages
- **Masquage causal** : Empêche le modèle de "tricher" en regardant le futur

#### Optimisations GPU

- **🚀 CUDA forcé** : Utilisation obligatoire du GPU si disponible
- **⚡ Précision mixte (AMP)** : Économie de mémoire GPU
- **📊 Monitoring temps réel** : Suivi de l'utilisation mémoire GPU
- **🔧 Optimisations avancées** : cuDNN benchmark, TF32 sur Ampere

## 🗂️ Structure du Projet

```
Poc_Euromillions/
├── 📊 CSV/                              # Données historiques
├── 🧮 euromillions_pro_pipeline.py     # Approche 1: Stats + ML
├── 🦜 Poc_Euromillions.py              # Approche 2: Transformer
├── 🔮 predict_euromillions.py          # Script de prédiction seule
├── 🏃 run_euromillions.py              # Orchestrateur principal
├── 📋 requirements.txt                 # Dépendances
├── 📈 out_euro_final/                  # Résultats approche 1
└── 🎯 euromillions.csv                 # Dataset principal
```

## 🚀 Installation et Utilisation

### Prérequis

```bash
pip install -r requirements.txt
```

### Approche 1 : Le Classement Malin

```bash
# Exécution complète avec backtest et génération
python euromillions_pro_pipeline.py

# Avec GPU forcé
python euromillions_pro_pipeline.py --gpu

# CSV personnalisé
python euromillions_pro_pipeline.py --csv mon_fichier.csv --out mes_resultats/
```

**Sorties** :
- `backtest_results.csv` : Métriques de performance historique
- `pred_numbers_next.csv` / `pred_stars_next.csv` : Scores détaillés
- `portfolio_20_tickets.csv` : 20 grilles diversifiées
- Graphiques PNG : NDCG, Recall, hits cumulés

### Approche 2 : Le Perroquet Séquentiel

```bash
# Entraînement + génération
python Poc_Euromillions.py

# Prédiction seule (modèle pré-entraîné)
python predict_euromillions.py
```

**Sorties** :
- `euromillions_model.pth` : Modèle sauvegardé
- `predictions_euromillions.csv` : 10 prédictions générées
- Logs détaillés d'entraînement avec métriques GPU

## ⚙️ Configuration

### Approche 1 (Stats + ML)

```python
@dataclass
class Config:
    pool_numbers: int = 50              # Numéros 1-50
    pool_stars: int = 12                # Étoiles 1-12
    windows_numbers: Tuple = (180, 360, 720)  # Fenêtres temporelles
    half_life_numbers: int = 260        # Demi-vie pondération
    n_tickets: int = 20                 # Grilles dans le portefeuille
    gpu_try: bool = True                # Tentative GPU automatique
```

### Approche 2 (Transformer)

```python
class Config:
    W = 60                              # Fenêtre d'historique
    D_MODEL = 256                       # Dimension du modèle
    NUM_EPOCHS = 50                     # Époques d'entraînement
    BATCH_SIZE = 128                    # Taille de batch (GPU)
    FORCE_GPU = True                    # GPU obligatoire
    USE_MIXED_PRECISION = True          # AMP pour économie mémoire
```

## 📊 Métriques et Validation

### Approche 1 : Backtest Rolling-Origin

- **NDCG@5** (numéros) et **NDCG@2** (étoiles) : Mesure la qualité du classement
- **Recall@5/2** : Proportion de vrais positifs dans le top-K
- **Hits cumulés** : Évolution du nombre de prédictions correctes
- **20 splits temporels** : Validation robuste sur données historiques

### Approche 2 : Génération Diversifiée

- **Top-K Sampling** : Balance cohérence/exploration (K=5)
- **Contraintes de validité** : Respect des règles EuroMillions
- **Monitoring GPU** : Utilisation mémoire temps réel
- **Reproductibilité** : Seed fixe pour résultats déterministes

## 🎯 Stratégies de Prédiction

### Fusion Intelligente (Approche 1)

```python
# Combinaison pondérée ranker + classifieur calibré
score_final = 0.65 × score_ranker + 0.35 × proba_calibrée
```

### Génération Séquentielle (Approche 2)

```python
# Génération pas-à-pas avec contraintes
for position in [boule1, boule2, ..., etoile2]:
    proba = softmax(transformer_output)
    proba[numéros_déjà_tirés] = 0  # Contrainte unicité
    numéro = top_k_sample(proba, k=5)
```

## 🔧 Optimisations Techniques

### GPU et Performance

- **Détection automatique CUDA** avec fallback CPU gracieux
- **Batch processing** optimisé pour GPU (128 échantillons)
- **Pin memory** et workers multiples pour transferts CPU↔GPU
- **Gradient clipping** et scheduling adaptatif du learning rate

### Robustesse et Monitoring

- **Gestion d'erreurs GPU** : Retry automatique en mode CPU
- **Monitoring mémoire** : Affichage temps réel de l'utilisation
- **Sauvegarde/reprise** : Checkpoints complets avec métadonnées
- **Validation croisée temporelle** : Respect de la chronologie

## 🎲 Génération de Portefeuilles

### Diversification Intelligente

L'approche 1 génère 20 grilles diversifiées en :

1. **Mélangeant** probabilités calibrées (80%) + scores de rang (20%)
2. **Échantillonnant** via Gumbel-Top-k pour éviter les doublons
3. **Garantissant** la diversité avec maximum 1000 tentatives

### Exemple de Sortie

```
Ticket 01: N [7, 12, 23, 34, 45] | S [3, 8]
Ticket 02: N [2, 15, 28, 39, 47] | S [5, 11]
...
```

## 📈 Résultats et Interprétation

### Métriques Typiques

- **NDCG@5 numéros** : ~0.15-0.25 (aléatoire = 0.10)
- **Recall@5 numéros** : ~0.60-0.80 (5 prédictions sur 5 attendus)
- **Hits cumulés** : Tendance croissante sur le backtest

### Interprétation

- **Scores élevés** = Numéros/étoiles favoris selon l'historique
- **Portefeuille diversifié** = Couverture large des possibilités
- **Backtest positif** = Stratégie historiquement performante

---

## 🤖 Philosophie du Projet

Ce projet illustre deux paradigmes complémentaires :

1. **🧮 L'approche analytique** : Décompose le problème, extrait des features, optimise des métriques
2. **🦜 L'approche générative** : Apprend les patterns implicites, génère de nouvelles séquences

Ensemble, elles offrent une vision complète du défi de prédiction loterie : entre analyse rationnelle et intuition artificielle.

---

*"Dans l'incertitude absolue, seule la méthode peut nous guider."* 🎯
https://www.linkedin.com/in/guillaume-clement-erp-cloud/
