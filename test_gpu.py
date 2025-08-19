#!/usr/bin/env python3
"""
Script de test pour vérifier la configuration GPU du script Euromillions
"""

import torch
import sys

def test_gpu_setup():
    """Test de la configuration GPU"""
    print("=== 🔧 TEST DE CONFIGURATION GPU 🔧 ===")
    
    # Vérification de PyTorch
    print(f"Version PyTorch : {torch.__version__}")
    
    # Vérification CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA disponible : {'✅ Oui' if cuda_available else '❌ Non'}")
    
    if cuda_available:
        print(f"Version CUDA : {torch.version.cuda}")
        print(f"Version cuDNN : {torch.backends.cudnn.version()}")
        
        gpu_count = torch.cuda.device_count()
        print(f"Nombre de GPUs : {gpu_count}")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"  GPU {i} : {props.name}")
            print(f"    Mémoire : {memory_gb:.1f} GB")
            print(f"    Compute Capability : {props.major}.{props.minor}")
        
        # Test d'allocation mémoire
        print("\n🧪 Test d'allocation mémoire GPU...")
        try:
            device = torch.device("cuda:0")
            test_tensor = torch.randn(1000, 1000, device=device)
            print("✅ Allocation GPU réussie")
            
            # Test de calcul
            result = torch.mm(test_tensor, test_tensor)
            print("✅ Calcul GPU réussi")
            
            # Nettoyage
            del test_tensor, result
            torch.cuda.empty_cache()
            print("✅ Nettoyage GPU réussi")
            
        except Exception as e:
            print(f"❌ Erreur GPU : {e}")
    else:
        print("\n⚠️  CUDA non disponible. Le script utilisera le CPU.")
        print("Pour utiliser le GPU :")
        print("1. Vérifiez que vous avez une GPU NVIDIA compatible")
        print("2. Installez les drivers NVIDIA récents")
        print("3. Installez PyTorch avec support CUDA :")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("=" * 50)
    
    return cuda_available

if __name__ == "__main__":
    gpu_available = test_gpu_setup()
    
    if gpu_available:
        print("🚀 Votre système est prêt pour l'entraînement GPU !")
        sys.exit(0)
    else:
        print("🖥️  Votre système utilisera le CPU pour l'entraînement.")
        sys.exit(1)
