################################################################################
# SCRIPT DE PRÉDICTION EUROMILLIONS - VERSION PROFESSIONNELLE
#
# Améliorations :
# - Structure modulaire (fonctions, classes, config)
# - Utilisation de torch.utils.data.Dataset & DataLoader
# - Séparation Entraînement / Validation pour le suivi du surapprentissage
# - Boucle d'entraînement améliorée : tqdm, gradient clipping, scheduler
# - Reproductibilité (seed) et flexibilité (device auto)
# - Forçage de CUDA si disponible
#
# Auteur : Gemini (basé sur le code initial)
# Date   : 18/08/2025
################################################################################

import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import List, Dict

# NOUVEAU: Une classe ou un dictionnaire pour centraliser tous les paramètres.
# C'est beaucoup plus propre et facile à modifier.
class Config:
    CSV_PATH = "euromillions.csv"
    SEPARATOR = ","
    DATE_COLUMN = "date_de_tirage"
    COLS_NUMBERS = ['boule_1','boule_2','boule_3','boule_4','boule_5','etoile_1','etoile_2']
    
    # Paramètres GPU
    FORCE_GPU = True                   # Forcer l'utilisation du GPU
    CUDA_DEVICE = 0                    # Index du GPU à utiliser (0 par défaut)
    
    # Paramètres de séquençage
    W = 60                             # Fenêtre d'historique (nb de tirages passés)
    VOCAB_SIZE = 51                    # 0 (token spécial) + 1-50 (nombres)
    
    # Tokens spéciaux
    separator_token = 0                # Token de séparation entre tirages
    start_token = 0                    # Token de début pour le décodeur
    
    # Hyperparamètres du modèle
    D_MODEL = 256                      # Dimension principale du Transformer
    NHEAD = 8                          # Nombre de têtes d'attention
    NUM_ENCODER_LAYERS = 4             # Nb de couches dans l'encodeur
    NUM_DECODER_LAYERS = 4             # Nb de couches dans le décodeur
    DIM_FEEDFORWARD = 4 * D_MODEL      # Dimension de la couche feed-forward
    DROPOUT = 0.1
    
    # Paramètres d'entraînement (optimisés pour GPU)
    NUM_EPOCHS = 50
    BATCH_SIZE = 128                   # Augmenté pour une meilleure utilisation GPU
    LEARNING_RATE = 1e-4               # Taux d'apprentissage plus faible, souvent meilleur pour les Transformers
    VALIDATION_SPLIT = 0.15            # 15% des données pour la validation
    CLIP_GRAD_NORM = 1.0               # Valeur pour le gradient clipping
    SEED = 42                          # Pour la reproductibilité
    
    # Optimisations GPU
    USE_MIXED_PRECISION = True         # Utiliser la précision mixte (AMP) pour économiser la mémoire
    PIN_MEMORY = True                  # Optimisation pour le transfert CPU->GPU
    NUM_WORKERS = 4                    # Nombre de workers pour le DataLoader
    
    # Paramètres de génération
    MAX_PREDICTIONS = 10
    TOP_K = 5
    
    # Paramètres de sauvegarde
    MODEL_SAVE_PATH = "euromillions_model.pth"  # Chemin de sauvegarde du modèle
    SAVE_MODEL = True                           # Sauvegarder le modèle après entraînement
    LOAD_MODEL = False                          # Charger un modèle pré-entraîné
    GENERATION_ONLY = False                     # Mode génération seule (sans entraînement)

# --- 1. FONCTIONS UTILITAIRES ET PRÉPARATION DES DONNÉES ---

def setup_gpu_device(config: Config):
    """Configure et force l'utilisation du GPU si demandé."""
    print("=== 🔧 CONFIGURATION GPU 🔧 ===")
    
    # Vérification de la disponibilité CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA disponible : {'✅ Oui' if cuda_available else '❌ Non'}")
    
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"Nombre de GPUs détectés : {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i} : {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Configuration du device
    if config.FORCE_GPU and cuda_available:
        if config.CUDA_DEVICE < torch.cuda.device_count():
            device = torch.device(f"cuda:{config.CUDA_DEVICE}")
            torch.cuda.set_device(config.CUDA_DEVICE)
            print(f"🚀 Utilisation forcée du GPU {config.CUDA_DEVICE}")
            
            # Optimisations GPU
            torch.backends.cudnn.benchmark = True  # Optimise les convolutions
            torch.backends.cuda.matmul.allow_tf32 = True  # Permet TF32 sur Ampere
            
        else:
            print(f"⚠️  GPU {config.CUDA_DEVICE} non disponible, utilisation du GPU 0")
            device = torch.device("cuda:0")
            torch.cuda.set_device(0)
    elif config.FORCE_GPU and not cuda_available:
        print("❌ ERREUR : GPU forcé mais CUDA non disponible !")
        print("   Veuillez installer PyTorch avec support CUDA ou désactiver FORCE_GPU")
        raise RuntimeError("GPU forcé mais CUDA non disponible")
    else:
        device = torch.device("cpu")
        print("🖥️  Utilisation du CPU")
    
    # Affichage des informations du device sélectionné
    if device.type == "cuda":
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
        memory_allocated = torch.cuda.memory_allocated(current_device) / (1024**3)
        print(f"Device actuel : {gpu_name} (GPU {current_device})")
        print(f"Mémoire GPU : {memory_allocated:.2f} GB / {gpu_memory:.1f} GB utilisée")
    
    print("=" * 35)
    return device

def set_seed(seed: int):
    """Fixe la graine aléatoire pour la reproductibilité."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # Note: cudnn.benchmark est géré dans setup_gpu_device()

def load_and_prepare_data(config: Config) -> List[List[int]]:
    """Charge les données depuis le CSV, les nettoie et les retourne."""
    print("1. Chargement et préparation des données...")
    df = pd.read_csv(config.CSV_PATH, sep=config.SEPARATOR, engine="python")
    df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN], errors='coerce', dayfirst=True)
    df = df.dropna(subset=[config.DATE_COLUMN] + config.COLS_NUMBERS)
    df = df.sort_values(by=config.DATE_COLUMN)
    
    data = df[config.COLS_NUMBERS].astype(int)
    draws = data.values.tolist()
    
    print(f"   -> Nombre de tirages valides trouvés : {len(draws)}")
    if len(draws) <= config.W:
        raise ValueError(f"Pas assez de tirages ({len(draws)}) pour une fenêtre W={config.W}.")
    
    return draws

def save_model(model, optimizer, scheduler, epoch, config, device):
    """Sauvegarde le modèle, l'optimiseur et les métadonnées."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'config': {
            'VOCAB_SIZE': config.VOCAB_SIZE,
            'D_MODEL': config.D_MODEL,
            'NHEAD': config.NHEAD,
            'NUM_ENCODER_LAYERS': config.NUM_ENCODER_LAYERS,
            'NUM_DECODER_LAYERS': config.NUM_DECODER_LAYERS,
            'DIM_FEEDFORWARD': config.DIM_FEEDFORWARD,
            'DROPOUT': config.DROPOUT,
            'W': config.W,
            'separator_token': config.separator_token,
            'start_token': config.start_token
        },
        'device': str(device)
    }
    
    torch.save(checkpoint, config.MODEL_SAVE_PATH)
    print(f"✅ Modèle sauvegardé dans {config.MODEL_SAVE_PATH}")

def load_model(config, device):
    """Charge un modèle pré-entraîné."""
    try:
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device)
        
        # Vérification de compatibilité
        saved_config = checkpoint['config']
        if (saved_config['VOCAB_SIZE'] != config.VOCAB_SIZE or 
            saved_config['D_MODEL'] != config.D_MODEL):
            raise ValueError("Configuration du modèle sauvegardé incompatible")
        
        # Création du modèle
        model = LotteryTransformer(
            vocab_size=saved_config['VOCAB_SIZE'],
            d_model=saved_config['D_MODEL'],
            nhead=saved_config['NHEAD'],
            num_encoder_layers=saved_config['NUM_ENCODER_LAYERS'],
            num_decoder_layers=saved_config['NUM_DECODER_LAYERS'],
            dim_feedforward=saved_config['DIM_FEEDFORWARD'],
            dropout=saved_config['DROPOUT']
        ).to(device)
        
        # Chargement des poids
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Création de l'optimiseur et du scheduler
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        # Chargement des états
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        
        print(f"✅ Modèle chargé depuis {config.MODEL_SAVE_PATH} (epoch {epoch})")
        return model, optimizer, scheduler, epoch
        
    except FileNotFoundError:
        print(f"❌ Fichier modèle {config.MODEL_SAVE_PATH} non trouvé")
        return None, None, None, 0
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        return None, None, None, 0

# NOUVEAU: La manière professionnelle de gérer les données avec PyTorch.
class LotteryDataset(Dataset):
    """
    Dataset PyTorch pour les tirages de loterie.
    Transforme la liste de tirages en séquences (input, target_in, target_out)
    utilisables par le DataLoader.
    """
    def __init__(self, draws: List[List[int]], window_size: int):
        self.draws = draws
        self.W = window_size
        self.separator_token = 0
        self.start_token = 0
        
        if len(draws) <= self.W:
            raise ValueError("La liste de tirages est plus courte que la fenêtre (W).")

    def __len__(self) -> int:
        return len(self.draws) - self.W

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        past_draws = self.draws[idx : idx + self.W]
        next_draw = self.draws[idx + self.W]

        # Séquence d'entrée (encoder)
        seq_in = []
        for d in past_draws:
            seq_in.extend(d)
            seq_in.append(self.separator_token)
        # seq_in = seq_in[:-1] # Optionnel: retirer le dernier séparateur
        
        # Séquences de sortie (decoder)
        seq_out = next_draw
        seq_out_in = [self.start_token] + seq_out[:-1] # Décalé à droite pour l'entrée du décodeur

        return {
            "src": torch.tensor(seq_in, dtype=torch.long),
            "tgt_in": torch.tensor(seq_out_in, dtype=torch.long),
            "tgt_out": torch.tensor(seq_out, dtype=torch.long)
        }


# --- 2. DÉFINITION DU MODÈLE TRANSFORMER ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: [batch_size, seq_len, d_model] (batch_first=True) """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class LotteryTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_encoder_layers: int, 
                 num_decoder_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # NOTE: batch_first=True est maintenant recommandé pour de meilleures performances d'inférence
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True 
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        # src: [batch_size, src_len], tgt: [batch_size, tgt_len] (batch_first=True)
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        # output: [batch_size, tgt_len, d_model]
        return self.fc_out(output)


# --- 3. BOUCLE D'ENTRAÎNEMENT ET D'ÉVALUATION ---

def print_gpu_memory_usage(device):
    """Affiche l'utilisation de la mémoire GPU."""
    if device.type == "cuda":
        memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)
        memory_cached = torch.cuda.memory_reserved(device) / (1024**3)
        memory_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"   💾 Mémoire GPU : {memory_allocated:.2f}GB allouée, {memory_cached:.2f}GB réservée / {memory_total:.1f}GB total")

def train_one_epoch(model, dataloader, optimizer, criterion, device, clip_value, scaler=None, use_amp=False):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Entraînement", leave=False)

    for batch in progress_bar:
        src = batch['src'].to(device, non_blocking=True)       # B, S
        tgt_in = batch['tgt_in'].to(device, non_blocking=True) # B, T
        tgt_out = batch['tgt_out'].to(device, non_blocking=True)# B, T

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1)).to(device)

        optimizer.zero_grad()
        
        # Utilisation de la précision mixte si activée
        if use_amp and scaler is not None:
            with autocast('cuda'):
                output = model(src, tgt_in, tgt_mask) # B, T, V
                loss = criterion(output.view(-1, output.size(-1)), tgt_out.reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(src, tgt_in, tgt_mask) # B, T, V
            loss = criterion(output.view(-1, output.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
        
        total_loss += loss.item()
        
        # Mise à jour de la barre de progression avec infos GPU
        postfix = {"loss": loss.item()}
        if device.type == "cuda":
            memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)
            postfix["GPU_mem"] = f"{memory_allocated:.1f}GB"
        progress_bar.set_postfix(postfix)

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device, non_blocking=True)
            tgt_in = batch['tgt_in'].to(device, non_blocking=True)
            tgt_out = batch['tgt_out'].to(device, non_blocking=True)
            
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1)).to(device)
            output = model(src, tgt_in, tgt_mask)
            
            loss = criterion(output.view(-1, output.size(-1)), tgt_out.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


# --- 4. FONCTION DE GÉNÉRATION ---

def generate_valid_draw_topk(model, src_tensor, device, config, seq_out_len=7):
    model.eval()
    predicted_draw = []
    
    # B, S (où B=1) - batch_first=True
    src_tensor = src_tensor.unsqueeze(0).to(device)
    
    # L'entrée du décodeur commence avec le token de début
    tgt_input = torch.tensor([[config.start_token]], dtype=torch.long, device=device) # B=1, T=1

    for step in range(seq_out_len):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        
        with torch.no_grad():
            output = model(src_tensor, tgt_input, tgt_mask) # B, T, V
            # On prend les logits du dernier token prédit
            logits = output[0, -1, :] # V

        probs = torch.softmax(logits, dim=-1)
        probs[config.separator_token] = 0.0 # On interdit le token 0

        # Application des contraintes (numéros uniques, plages valides)
        if step < 5: # Génération des 5 boules (1-50)
            for num in predicted_draw: probs[num] = 0.0
            probs[51:] = 0.0 # Interdit les numéros > 50
        else: # Génération des 2 étoiles (1-12)
            # Les étoiles peuvent avoir les mêmes numéros que les boules
            for num in predicted_draw[5:]: probs[num] = 0.0
            probs[13:] = 0.0 # Interdit les étoiles > 12

        # Top-K Sampling
        if probs.sum() < 1e-9: # Fallback si toutes les probas sont nulles
             next_token = torch.tensor(1, device=device)
        else:
            topk_probs, topk_indices = torch.topk(probs, config.TOP_K)
            next_token = topk_indices[torch.multinomial(topk_probs, 1).item()]
        
        predicted_draw.append(next_token.item())
        # Ajout du token prédit à l'entrée du décodeur pour le prochain pas
        tgt_input = torch.cat([tgt_input, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

    return predicted_draw


# --- 5. ORCHESTRATEUR PRINCIPAL ---

def main():
    """Fonction principale qui orchestre le chargement, l'entraînement et la génération."""
    config = Config()
    set_seed(config.SEED)

    # Configuration et forçage du GPU
    device = setup_gpu_device(config)
    
    # 1. Chargement et préparation des données
    all_draws = load_and_prepare_data(config)
    
    # Tentative de chargement d'un modèle pré-entraîné
    model, optimizer, scheduler, start_epoch = None, None, None, 0
    if config.LOAD_MODEL or config.GENERATION_ONLY:
        model, optimizer, scheduler, start_epoch = load_model(config, device)
    
    # Si pas de modèle chargé ou mode entraînement, on initialise tout
    if model is None and not config.GENERATION_ONLY:
        print("🔄 Initialisation d'un nouveau modèle...")
        
        # Préparation des datasets seulement si on va s'entraîner
        full_dataset = LotteryDataset(all_draws, config.W)
        indices = list(range(len(full_dataset)))
        train_indices, val_indices = train_test_split(indices, test_size=config.VALIDATION_SPLIT, random_state=config.SEED)
        
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        
        # Création des DataLoaders avec optimisations GPU
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True,
            pin_memory=config.PIN_MEMORY and device.type == "cuda",
            num_workers=config.NUM_WORKERS if device.type == "cuda" else 0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.BATCH_SIZE,
            pin_memory=config.PIN_MEMORY and device.type == "cuda",
            num_workers=config.NUM_WORKERS if device.type == "cuda" else 0
        )
        print(f"   -> {len(train_dataset)} échantillons d'entraînement, {len(val_dataset)} de validation.")
        print(f"   -> DataLoaders configurés avec pin_memory={config.PIN_MEMORY and device.type == 'cuda'}, num_workers={config.NUM_WORKERS if device.type == 'cuda' else 0}")
        
        # 3. Initialisation du modèle, de l'optimiseur et de la fonction de perte
        model = LotteryTransformer(
            vocab_size=config.VOCAB_SIZE,
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            num_encoder_layers=config.NUM_ENCODER_LAYERS,
            num_decoder_layers=config.NUM_DECODER_LAYERS,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
    elif model is None and config.GENERATION_ONLY:
        print("❌ Mode génération seule activé mais aucun modèle trouvé !")
        print(f"   Veuillez d'abord entraîner un modèle ou vérifier le chemin : {config.MODEL_SAVE_PATH}")
        return
    
    criterion = nn.CrossEntropyLoss()
    
    # Initialisation du scaler pour la précision mixte
    scaler = None
    use_amp = config.USE_MIXED_PRECISION and device.type == "cuda"
    if use_amp:
        scaler = GradScaler('cuda')
        print("   -> Précision mixte (AMP) activée pour économiser la mémoire GPU")

    # 4. Boucle d'entraînement (seulement si pas en mode génération seule)
    if not config.GENERATION_ONLY:
        print("\n=== 🚀 DÉBUT DE L'ENTRAÎNEMENT 🚀 ===")
        if start_epoch > 0:
            print(f"   Reprise de l'entraînement depuis l'epoch {start_epoch}")
        print_gpu_memory_usage(device)
        
        for epoch in range(start_epoch + 1, config.NUM_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, config.CLIP_GRAD_NORM, scaler, use_amp)
            val_loss = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            
            print(f"Epoch {epoch:02d}/{config.NUM_EPOCHS} | "
                  f"Perte Entraînement: {train_loss:.4f} | "
                  f"Perte Validation: {val_loss:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Affichage mémoire GPU tous les 10 epochs
            if epoch % 10 == 0 or epoch == 1:
                print_gpu_memory_usage(device)
                
        print("=== ✅ ENTRAÎNEMENT TERMINÉ ✅ ===\n")
        print_gpu_memory_usage(device)
        
        # Sauvegarde du modèle
        if config.SAVE_MODEL:
            save_model(model, optimizer, scheduler, config.NUM_EPOCHS, config, device)
    else:
        print("⚡ Mode génération seule activé - pas d'entraînement")
    
    # 5. Génération des prédictions
    print(f"=== 🔮 GÉNÉRATION DE {config.MAX_PREDICTIONS} TIRAGES 🔮 ===")
    last_w_draws = all_draws[-config.W:]
    src_seq_list = []
    for d in last_w_draws:
        src_seq_list.extend(d)
        src_seq_list.append(config.separator_token)

    src_tensor = torch.tensor(src_seq_list, dtype=torch.long)
    
    for i in range(config.MAX_PREDICTIONS):
        draw_pred = generate_valid_draw_topk(model, src_tensor, device, config)
        boules = sorted(draw_pred[:5])
        etoiles = sorted(draw_pred[5:])
        print(f"Prédiction n°{i+1:02d} : Boules {boules} - Étoiles {etoiles}")
    
    # Nettoyage final de la mémoire GPU
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print("\n🧹 Cache GPU nettoyé")
        print_gpu_memory_usage(device)


if __name__ == "__main__":
    main()