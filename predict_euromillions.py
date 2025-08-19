#!/usr/bin/env python3
"""
Script de prédiction Euromillions - Génération seule
Charge un modèle pré-entraîné et génère des prédictions
"""

import torch
import pandas as pd
from Poc_Euromillions import (
    Config, setup_gpu_device, load_and_prepare_data, 
    load_model, generate_valid_draw_topk, set_seed
)

def predict_only():
    """Fonction de prédiction sans entraînement."""
    # Configuration
    config = Config()
    config.GENERATION_ONLY = True
    config.LOAD_MODEL = True
    set_seed(config.SEED)
    
    print("🔮 === PRÉDICTIONS EUROMILLIONS === 🔮")
    print(f"Modèle : {config.MODEL_SAVE_PATH}")
    print(f"Prédictions : {config.MAX_PREDICTIONS}")
    
    # Configuration GPU
    device = setup_gpu_device(config)
    
    # Chargement des données
    all_draws = load_and_prepare_data(config)
    
    # Chargement du modèle
    model, _, _, epoch = load_model(config, device)
    
    if model is None:
        print("\n❌ ERREUR : Aucun modèle trouvé !")
        print("Vous devez d'abord entraîner un modèle avec :")
        print("  python Poc_Euromillions.py")
        return
    
    print(f"✅ Modèle chargé (entraîné sur {epoch} epochs)")
    
    # Préparation de la séquence source
    print(f"\n=== 🔮 GÉNÉRATION DE {config.MAX_PREDICTIONS} TIRAGES 🔮 ===")
    last_w_draws = all_draws[-config.W:]
    src_seq_list = []
    for d in last_w_draws:
        src_seq_list.extend(d)
        src_seq_list.append(config.separator_token)
    
    src_tensor = torch.tensor(src_seq_list, dtype=torch.long)
    
    # Génération des prédictions
    predictions = []
    for i in range(config.MAX_PREDICTIONS):
        draw_pred = generate_valid_draw_topk(model, src_tensor, device, config)
        boules = sorted(draw_pred[:5])
        etoiles = sorted(draw_pred[5:])
        prediction = {
            'numero': i + 1,
            'boules': boules,
            'etoiles': etoiles
        }
        predictions.append(prediction)
        print(f"Prédiction n°{i+1:02d} : Boules {boules} - Étoiles {etoiles}")
    
    # Sauvegarde des prédictions
    save_predictions(predictions)
    
    # Nettoyage GPU
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print("\n🧹 Cache GPU nettoyé")

def save_predictions(predictions):
    """Sauvegarde les prédictions dans un fichier CSV."""
    filename = "predictions_euromillions.csv"
    
    # Création du DataFrame
    data = []
    for pred in predictions:
        row = {
            'prediction_numero': pred['numero'],
            'boule_1': pred['boules'][0],
            'boule_2': pred['boules'][1],
            'boule_3': pred['boules'][2],
            'boule_4': pred['boules'][3],
            'boule_5': pred['boules'][4],
            'etoile_1': pred['etoiles'][0],
            'etoile_2': pred['etoiles'][1]
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"\n💾 Prédictions sauvegardées dans {filename}")

if __name__ == "__main__":
    predict_only()
