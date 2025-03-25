[Guillaume CLEMENT le 25/03/2025](https://www.linkedin.com/feed/update/urn:li:activity:7310182405049335809/)


# Euromillions Transformer Predictor (Exemple Éducatif)

Ce projet illustre l’entraînement d’un modèle **Transformer** pour prédire (de manière purement fictive) les prochains tirages Euromillions. L’objectif est de démontrer les principes du **Deep Learning** et l’utilisation d’un Transformer en PyTorch, sans prétendre réellement prévoir un phénomène aléatoire.

## Table des Matières
- [Contexte](#contexte)
- [Avertissements Importants](#avertissements-importants)
- [Caractéristiques Techniques](#caractéristiques-techniques)
- [Logique d’Apprentissage](#logique-dapprentissage)
- [Choix Techniques](#choix-techniques)
- [Installation et Exécution](#installation-et-exécution)
- [Usage et Résultats](#usage-et-résultats)
- [Conclusion](#conclusion)

## Contexte

**Objectif principal** : Utiliser un Transformer – technologie similaire à celle utilisée par ChatGPT ou Mistral pour prédire des mots – dans un contexte ludique : anticiper (sans validité réelle) des tirages Euromillions.

**Exemple d’apprentissage** :
- Lecture des anciens tirages (boules + étoiles).
- Conversion en séquences numériques.
- Entraînement d’un modèle pour « deviner » le tirage suivant.

Ce projet est strictement pédagogique : il ne prétend pas fournir de véritables prédictions gagnantes.

## Avertissements Importants
- **Projet Purement Fictif** : Aucune garantie réelle de prédiction.
- **Aucun Conseil Financier** : Ne pariez jamais votre argent avec ce modèle.
- **Hasard vs. Modèle** : La loterie reste purement aléatoire, même avec des modèles complexes.
- **Démonstration Pédagogique** : Objectif pédagogique uniquement.

## Caractéristiques Techniques

- **Langage** : Python (3.7+ recommandé)
- **Bibliothèques clés** :
  - PyTorch (Deep Learning)
  - Pandas (manipulation des données)
  - NumPy (calcul numérique)
- **Modèle** : Transformer Seq2Seq avec `batch_first=True`

## Logique d’Apprentissage

### Chargement et Préparation des Données
- Lecture CSV des tirages passés (5 boules + 2 étoiles).
- Tri chronologique et conversion en séquences numériques.

### Fenêtrage
- Définir une fenêtre historique `W` (par ex. `W=100`).
- Séquences d’entrée : `[i, i+1, ..., i+W-1]`
- Cible à prédire : tirage `i+W`

### Séquence d’Entrée (src)
- Conversion des nombres en tokens, séparés par un token spécial (`0`).

### Séquence de Sortie (tgt)
- Introduction d’un token de début (`0`) avec décalage (shift).

### Entraînement
- Utilisation de `Dataset` et `DataLoader`.
- Masque triangulaire pour éviter la fuite d’information future.
- Calcul de la perte (CrossEntropy).

### Validation
- Division des données en Train / Validation (80/20 par défaut).
- Validation à chaque époque pour contrôler le surapprentissage.

### Génération
- Méthode d’échantillonnage : `top-k` / `top-p (nucleus)`.
- Application de règles de filtrage (ex. éviter les doublons).

## Choix Techniques
- **`batch_first=True`** : Simplifie la gestion des dimensions `[batch_size, seq_len, ...]`.
- **Fenêtre Glissante (W)** : Plus grande fenêtre augmente la richesse contextuelle mais diminue le nombre d’échantillons et augmente les besoins en ressources.
- **Dimension du Modèle (D_MODEL)** : Plus grand D_MODEL augmente la capacité du modèle mais aussi sa consommation en mémoire et son temps d'entraînement.
- **Masque Triangulaire** : Simule l'autoregression.
- **Top-K / Top-P** : Introduit de la diversité dans les générations, évitant un argmax strict.

## Installation et Exécution

### Cloner le dépôt :
```bash
git clone https://github.com/votre-utilisateur/euromillions-transformer.git
cd euromillions-transformer
```

### Installer les dépendances (dans un venv ou conda recommandé) :
```bash
pip install -r requirements.txt
```

- Assurez-vous que le fichier `requirements.txt` contient `torch`, `pandas`, `numpy`, etc.

### Fichier CSV :
- Placer `euromillions.csv` dans le répertoire du projet (ou adapter le chemin).

### Exécution du script :
```bash
python main_euromillions.py
```

## Usage et Résultats

Le script réalise :
- Lecture et préparation des données.
- Entraînement du modèle (nombre d’époques configurable).
- Évaluation des performances en validation (perte moyenne).
- Génération de tirages fictifs via `top-k` / `top-p` sampling.

Exemple de sortie :
```text
Prédiction n°1 : [12, 45, 3, 17, 36, 2, 9]
Prédiction n°2 : [10, 49, 6, 22, 28, 1, 3]
...
```
> Ces résultats restent essentiellement aléatoires dans ce contexte.

## Conclusion

Ce projet est strictement pédagogique, démontrant les capacités techniques d’un Transformer tout en rappelant l’imprévisibilité d’un phénomène réellement aléatoire.

**Points Clés :**
- DataLoader pour un apprentissage structuré.
- Division claire entre entraînement et validation.
- Masque triangulaire pour l'autoregression.
- Utilisation de top-k/top-p pour génération diversifiée.

Amusez-vous à explorer différents paramètres (`W`, `D_MODEL`, `TOP_K`, etc.) et appliquez ces techniques à des cas pratiques : prévisions temporelles, complétion de séquences à dépendances longues, etc.

