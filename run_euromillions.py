#!/usr/bin/env python3
"""
Script d'exécution simple pour l'analyse EuroMillions avec GPU
"""

import sys
from pathlib import Path
from euromillions_pro_pipeline import main, Config

def run_analysis():
    """Lance l'analyse EuroMillions avec les paramètres optimisés"""
    print("🎰 Lancement de l'analyse EuroMillions Pro avec GPU 🎰")
    print("=" * 60)
    
    # Configuration avec GPU activé
    config = Config(gpu_try=True)
    
    # Paramètres simulant les arguments CLI
    class Args:
        csv = "euromillions.csv"
        out = "resultats_euromillions"
        gpu = True
        demo = False
    
    # Sauvegarde des arguments sys.argv originaux
    original_argv = sys.argv.copy()
    
    try:
        # Simulation des arguments CLI pour le script principal
        sys.argv = [
            "euromillions_pro_pipeline.py",
            "--csv", "euromillions.csv",
            "--out", "resultats_euromillions",
            "--gpu"
        ]
        
        # Lancement du script principal
        main()
        
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution: {e}")
        return False
    finally:
        # Restauration des arguments originaux
        sys.argv = original_argv
    
    print("\n✅ Analyse terminée avec succès!")
    print(f"📊 Les résultats sont disponibles dans le dossier: resultats_euromillions/")
    return True

if __name__ == "__main__":
    success = run_analysis()
    sys.exit(0 if success else 1)

