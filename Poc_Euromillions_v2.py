###########################################################
# SCRIPT ÉDUCATIF D'ILLUSTRATION
# --------------------------------------
# Exemple d'utilisation d'un Transformer
# pour prédire (fictivement) le prochain tirage Euromillions.
#
# NOTE : ceci est un exemple purement pédagogique.
#        Il n'y a aucune garantie de prévision correcte
#        des résultats Euromillions.
#
# Auteur : [Votre Nom]
# Date   : [Date du jour]
###########################################################

import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

###########################################################
# 1) PARAMÈTRES GÉNÉRAUX
###########################################################

CSV_PATH            = "euromillions.csv"      # Chemin vers le fichier CSV
SEPARATOR           = ";"                     # Séparateur CSV
DATE_COLUMN         = "date_de_tirage"        # Nom de la colonne date

# Hyperparamètres
W                  = 100    # Fenêtre d'historique (réduite pour multiplier les échantillons)
NUM_EPOCHS         = 50    # Nombre d'époques d'entraînement (un entraînement plus long)
BATCH_SIZE         = 64    # Taille de batch (pour une meilleure stabilité des gradients)
MAX_PREDICTIONS    = 10    # Nombre de prédictions (tirages) à générer
TOP_K              = 10    # Pour le top-k sampling (élargit le choix des tokens)
TOP_P              = 0.9   # Pour le top-p (nucleus) sampling, valeur classique
USE_TOP_P          = True  # Activation du top-p

VOCAB_SIZE         = 51   # On autorise 0..50 (0 = token spécial)
# On pourrait autoriser moins/plus pour expérimenter,
# ex. si on limite vraiment 1..50 pour boules et 1..12 pour étoiles.

# Taille du Transformer (paramètres « plus sérieux »)
# TD_MODEL    = 512   # Dimension d'embedding plus élevée pour capter des interactions complexes
# TNHEAD      = 8     # Nombre de têtes de self-attention (doit diviser D_MODEL, ici 512/8 = 64 par tête)
# TNUM_LAYERS = 6     # Nombre de couches, ce qui permet d'approfondir le réseau

# On augmente la taille du Transformer :
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2

# Chemin éventuel pour sauvegarder le "meilleur" modèle
BEST_MODEL_PATH = "models/best_lottery_transformer.pth"

###########################################################
# 2) LECTURE DU CSV + PRÉPARATION DES DONNÉES
###########################################################

df = pd.read_csv(CSV_PATH, sep=SEPARATOR, engine="python")

# Convertir la colonne date et trier
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce', dayfirst=True)
df = df.dropna(subset=[DATE_COLUMN])   # retire lignes sans date
df = df.sort_values(by=DATE_COLUMN)    # tri chronologique

# Colonnes boules/étoiles (à adapter si nécessaire)
cols_numbers = ['boule_1','boule_2','boule_3','boule_4','boule_5','etoile_1','etoile_2']
data = df[cols_numbers].dropna().astype(int).copy()

# On en fait une liste de listes (chaque tirage)
draws = data.values.tolist()

print(f"Nombre de tirages valides trouvés : {len(draws)}")
if len(draws) > 0:
    print("Exemple de tirage :", draws[0])

if len(draws) <= W:
    raise ValueError(f"Pas assez de tirages ({len(draws)}) pour une fenêtre W={W}.")

###########################################################
# 3) CONSTRUCTION DES SEQUENCES (entrées / sorties)
#    + SPLIT TRAIN/VALID
###########################################################

# Séparation train / validation
# Exemple : 80% pour train, 20% pour validation
train_ratio = 0.8
N = len(draws)
split_index = int(N * train_ratio)

draws_train = draws[:split_index]
draws_valid = draws[split_index:]

def build_sequences(draws_list, window_size):
    """
    Construit les échantillons (input, output).
    - draws_list : liste de tirages
    - window_size : taille de la fenêtre W
    Retourne (X, Y, Y_in)
       * X  : séquence concaténée de W tirages (avec séparateur 0)
       * Y  : tirage suivant (7 nombres)
       * Y_in : [0] + 6 nombres (pour la partie décodeur)
    """
    sequences_in = []
    sequences_out = []
    sequences_out_in = []

    separateur = 0  # token spécial
    token_debut = 0

    length = len(draws_list)
    for i in range(length - window_size):
        past_draws = draws_list[i : i + window_size]
        next_draw  = draws_list[i + window_size]

        seq_in = []
        for d in past_draws:
            seq_in.extend(d) 
            seq_in.append(separateur)
        # Optionnel : on peut enlever le dernier séparateur
        # seq_in = seq_in[:-1]

        seq_out = next_draw
        # Pour le décodeur, on prépare la séquence d'entrée (shift right)
        seq_out_in = [token_debut] + seq_out[:-1]

        sequences_in.append(seq_in)
        sequences_out.append(seq_out)
        sequences_out_in.append(seq_out_in)

    X = np.array(sequences_in, dtype=int)
    Y = np.array(sequences_out, dtype=int)
    Y_in = np.array(sequences_out_in, dtype=int)

    return X, Y, Y_in

X_train, Y_train, Y_in_train = build_sequences(draws_train, W)
X_valid, Y_valid, Y_in_valid = build_sequences(draws_valid, W)

print(f"Train set: {len(X_train)} échantillons")
print(f"Valid set: {len(X_valid)} échantillons")

# Dimensions attendues
seq_in_len = W * 8   # car chaque tirage=7 nombres + 1 séparateur => 8
seq_out_len = 7      # 5 boules + 2 étoiles

###########################################################
# 4) DATASET + DATALOADER
###########################################################
class EuroMillionsDataset(Dataset):
    """Dataset PyTorch pour nos séquences Euromillions."""
    def __init__(self, X, Y, Y_in):
        super().__init__()
        self.X = X
        self.Y = Y
        self.Y_in = Y_in

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_item = self.X[idx]
        y_item = self.Y[idx]
        y_in_item = self.Y_in[idx]
        return torch.tensor(x_item, dtype=torch.long), \
               torch.tensor(y_item, dtype=torch.long), \
               torch.tensor(y_in_item, dtype=torch.long)

train_dataset = EuroMillionsDataset(X_train, Y_train, Y_in_train)
valid_dataset = EuroMillionsDataset(X_valid, Y_valid, Y_in_valid)

train_loader = DataLoader(train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

valid_loader = DataLoader(valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False)

###########################################################
# 5) MODÈLE TRANSFORMER
###########################################################
class PositionalEncoding(nn.Module):
    """Encodage positionnel sinusoïdal."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x : [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class LotteryTransformer(nn.Module):
    """Transformer encodeur/décodeur pour la prédiction de séquences."""
    def __init__(self, d_model=256, nhead=8, num_layers=4, vocab_size=51):
        super().__init__()
        self.d_model = d_model
        # Embeddings
        self.embedding_src = nn.Embedding(vocab_size, d_model)
        self.embedding_tgt = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead,
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers,
            dim_feedforward=4*d_model,
            dropout=0.1,
            batch_first=True  # On active batch_first
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, tgt_mask=None):
        """
        src: [batch_size, seq_len_src]
        tgt: [batch_size, seq_len_tgt]
        """
        # Embedding
        src_emb = self.embedding_src(src) * math.sqrt(self.d_model)
        tgt_emb = self.embedding_tgt(tgt) * math.sqrt(self.d_model)

        # + position
        src_emb = self.pos_encoding(src_emb)
        tgt_emb = self.pos_encoding(tgt_emb)

        # Passage dans le Transformer
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        # [batch_size, seq_len_tgt, d_model]
        out = self.fc_out(out)  # [batch_size, seq_len_tgt, vocab_size]
        return out

###########################################################
# 6) INITIALISATION DU MODÈLE, CRITÈRE, OPTIMISEUR
###########################################################
model = LotteryTransformer(d_model=D_MODEL, nhead=NHEAD, 
                           num_layers=NUM_LAYERS, vocab_size=VOCAB_SIZE)

# Sélection du device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Device utilisé :", device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

###########################################################
# 7) FONCTIONS UTILITAIRES
###########################################################

def generate_subsequent_mask(size: int, device: torch.device):
    """
    Crée un masque triangulaire pour le décodeur.
    PyTorch attend un masque [T, T] même si batch_first=True.
    """
    return nn.Transformer.generate_square_subsequent_mask(size).to(device)

def evaluate_model(model, data_loader, criterion):
    """
    Évalue la perte sur un data_loader donné.
    Retourne la perte moyenne.
    """
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for src_batch, tgt_out_batch, tgt_in_batch in data_loader:
            src_batch = src_batch.to(device)
            tgt_in_batch = tgt_in_batch.to(device)
            tgt_out_batch = tgt_out_batch.to(device)

            # Masque triangulaire (seq_out_len = 7)
            seq_len = tgt_in_batch.size(1)
            tgt_mask = generate_subsequent_mask(seq_len, device)

            output = model(src_batch, tgt_in_batch, tgt_mask=tgt_mask)
            # output => [batch_size, seq_out_len, vocab_size]

            out_flat = output.reshape(-1, VOCAB_SIZE)
            tgt_out_flat = tgt_out_batch.reshape(-1)
            loss = criterion(out_flat, tgt_out_flat)

            batch_size = src_batch.size(0)
            total_loss += loss.item() * batch_size
            count += batch_size

    return total_loss / count if count > 0 else float('inf')

###########################################################
# 8) BOUCLE D'ENTRAÎNEMENT + VALIDATION
###########################################################
best_val_loss = float('inf')
folder = os.path.dirname(BEST_MODEL_PATH)
if folder:
    os.makedirs(folder, exist_ok=True)

print("\n=== ENTRAÎNEMENT DU MODÈLE ===")
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    nb_samples = 0

    for src_batch, tgt_out_batch, tgt_in_batch in train_loader:
        src_batch = src_batch.to(device)
        tgt_in_batch = tgt_in_batch.to(device)
        tgt_out_batch = tgt_out_batch.to(device)

        batch_current_size = src_batch.size(0)

        # Masque triangulaire (seq_out_len = 7)
        seq_len = tgt_in_batch.size(1)
        tgt_mask = generate_subsequent_mask(seq_len, device)

        optimizer.zero_grad()
        output = model(src_batch, tgt_in_batch, tgt_mask=tgt_mask)

        out_flat = output.reshape(-1, VOCAB_SIZE)
        tgt_out_flat = tgt_out_batch.reshape(-1)
        loss = criterion(out_flat, tgt_out_flat)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_current_size
        nb_samples += batch_current_size

    epoch_loss /= nb_samples if nb_samples > 0 else 1.0

    # Validation
    val_loss = evaluate_model(model, valid_loader, criterion)

    # Enregistrement si meilleure validation
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)

    print(f"Epoch {epoch}/{NUM_EPOCHS} "
          f"- Perte train : {epoch_loss:.4f}, "
          f"- Perte valid : {val_loss:.4f}, "
          f"best_val_loss={best_val_loss:.4f}")

print("=== ENTRAÎNEMENT TERMINÉ ===\n")
print(f"Meilleure perte en validation : {best_val_loss:.4f}")
print(f"Modèle sauvegardé dans : {BEST_MODEL_PATH}")

###########################################################
# 9) FONCTIONS DE GÉNÉRATION (CONTRAINTE + TOP-K / TOP-P)
###########################################################

def top_k_top_p_filtering(logits, top_k=5, top_p=0.9):
    """
    Filtre les logits pour faire du top-k + top-p (nucleus) sampling.
    - logits : 1D tensor [vocab_size]
    - top_k  : on ne garde que les k meilleurs
    - top_p  : on ne garde que les tokens dont la somme cumulative >= p
    Retourne : un tensor de probabilités filtrées (shape [vocab_size]).
    """
    # Convertit en probas
    probs = torch.softmax(logits, dim=-1).clone()

    # Tri décroissant
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # On garde le top-k
    sorted_probs[(top_k):] = 0.0

    # On applique le top-p si on veut
    if USE_TOP_P:
        cutoff = (cumulative_probs > top_p).nonzero(as_tuple=True)[0]
        if len(cutoff) > 0:
            first_idx = cutoff[0].item()
            # On met à 0 tout ce qui suit
            sorted_probs[first_idx+1:] = 0.0

    # On remet tout en place
    new_probs = torch.zeros_like(probs)
    new_probs[sorted_indices] = sorted_probs

    # Normalisation
    new_probs /= new_probs.sum() + 1e-9
    return new_probs

def generate_valid_draw(model, src_tensor, seq_out_len=7, 
                        top_k=5, top_p=0.9):
    """
    Génère un tirage valide (5 boules + 2 étoiles) via sampling contraint.
    - src_tensor : [1, seq_in_len]
    - seq_out_len : 7 (5 boules + 2 étoiles)
    - top_k / top_p : pour filtrer les logits.
    """
    model.eval()
    token_debut = 0
    separateur = 0

    predicted_draw = []
    tgt_input = torch.tensor([[token_debut]], dtype=torch.long, device=device)

    for step in range(seq_out_len):
        seq_len_tgt = tgt_input.size(1)
        tgt_mask = generate_subsequent_mask(seq_len_tgt, device)
        with torch.no_grad():
            output = model(src_tensor, tgt_input, tgt_mask=tgt_mask)
            logits = output[0, -1, :]  # dernière position => [vocab_size]

        # On interdit le token 0 (séparateur)
        logits[0] = -9999.0

        # Contrôle distinct boules/étoiles
        if step < 5:
            # Boules => 1..50 distincts
            for used in predicted_draw:
                logits[used] = -9999.0
            # On met hors de 1..50 à -inf
            logits[51:] = -9999.0
        else:
            # Étoiles => 1..12 distinctes
            for used in predicted_draw:
                logits[used] = -9999.0
            logits[13:] = -9999.0

        # top-k / top-p filtering
        filtered_probs = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        next_token = int(torch.multinomial(filtered_probs, 1))

        predicted_draw.append(next_token)
        next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        tgt_input = torch.cat([tgt_input, next_token_tensor], dim=1)

    return predicted_draw

###########################################################
# 10) CHARGEMENT DU MEILLEUR MODÈLE + GÉNÉRATION
###########################################################
# On recharge le "meilleur" modèle sauvegardé
if os.path.exists(BEST_MODEL_PATH):
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()
    print(f"Meilleur modèle chargé depuis {BEST_MODEL_PATH}")
else:
    print("Aucun modèle sauvegardé trouvé, on utilise le modèle actuel.")

# Préparation du dernier contexte (W derniers tirages complets)
last_draws = draws[-W:]
src_seq = []
for d in last_draws:
    src_seq.extend(d)
    src_seq.append(0)
src_seq = src_seq[:W*8]  # On s'assure d'avoir la taille fixe

src_tensor = torch.tensor(src_seq, dtype=torch.long, device=device).unsqueeze(0) 
# shape => [1, W*8]

# Exemples de prédictions
print(f"\n=== GÉNÉRATION DE {MAX_PREDICTIONS} TIRAGES ===")
predictions_all = []
for i in range(MAX_PREDICTIONS):
    draw_pred = generate_valid_draw(model, src_tensor, seq_out_len=7,
                                    top_k=TOP_K, top_p=TOP_P)
    predictions_all.append(draw_pred)
    print(f"Prédiction n°{i+1} : {draw_pred}")

###########################################################
# 11) ANALYSE RAPIDE DES DISTRIBUTIONS (optionnel)
###########################################################
# On compare la fréquence historique de chaque nombre (1..50)
# et celle observée dans nos prédictions récentes.

# A) Fréquence historique (sur l'ensemble)
freq_historic = {n: 0 for n in range(1, 51)}
freq_historic_star = {n: 0 for n in range(1, 13)}
for draw_ in draws:
    for b in draw_[:5]:
        freq_historic[b] += 1
    for e in draw_[5:]:
        freq_historic_star[e] += 1

# B) Fréquence dans nos prédictions
freq_pred = {n: 0 for n in range(1, 51)}
freq_pred_star = {n: 0 for n in range(1, 13)}
for draw_ in predictions_all:
    for b in draw_[:5]:
        if 1 <= b <= 50:
            freq_pred[b] += 1
    for e in draw_[5:]:
        if 1 <= e <= 12:
            freq_pred_star[e] += 1

print("\n=== COMPARAISON DE DISTRIBUTION (boules 1..50) ===")
for n in range(1, 51):
    if freq_pred[n] > 0:
        print(f"Numéro {n} => historique={freq_historic[n]}, prédiction={freq_pred[n]}")

print("\n=== COMPARAISON DE DISTRIBUTION (étoiles 1..12) ===")
for n in range(1, 13):
    if freq_pred_star[n] > 0:
        print(f"Étoile {n} => historique={freq_historic_star[n]}, prédiction={freq_pred_star[n]}")

###########################################################
# COMMENTAIRES FINALS
# -------------------
#  - Le script reste un exemple pédagogique. Nous manipulons un Transformer
#    pour "prédire" des tirages, ce qui n'a pas de validité réelle.
#  - Nous avons ajouté un split train/valid, un DataLoader, un système
#    de sauvegarde du meilleur modèle, et des fonctions de génération
#    (top-k et éventuellement top-p).
#  - Si vous souhaitez entraîner plus longtemps, augmentez NUM_EPOCHS,
#    ou explorez différentes valeurs de W, D_MODEL, etc.
#  - Vous pouvez aussi évaluer la perplexité sur le set de validation
#    ou implémenter un mécanisme d'early-stopping plus sophistiqué.
###########################################################
