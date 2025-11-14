"""
Script de teste para verificar se o ambiente está configurado corretamente
"""
import sys
from pathlib import Path

def test_imports():
    """Testa se todas as bibliotecas essenciais estão instaladas"""
    print("=" * 60)
    print("TESTE DE CONFIGURAÇÃO DO AMBIENTE")
    print("=" * 60)

    errors = []

    # Testar TensorFlow
    print("\n[1/6] Testando TensorFlow...")
    try:
        import tensorflow as tf
        print(f"  ✅ TensorFlow {tf.__version__} instalado")

        # Verificar GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  ✅ {len(gpus)} GPU(s) detectada(s): {[gpu.name for gpu in gpus]}")
        else:
            print("  ⚠️  Nenhuma GPU detectada. TensorFlow usará CPU.")
    except ImportError as e:
        errors.append(f"TensorFlow: {e}")
        print("  ❌ TensorFlow não instalado")

    # Testar Keras
    print("\n[2/6] Testando Keras...")
    try:
        from keras import __version__
        print(f"  ✅ Keras {__version__} instalado")
    except ImportError as e:
        errors.append(f"Keras: {e}")
        print("  ❌ Keras não instalado")

    # Testar OpenCV
    print("\n[3/6] Testando OpenCV...")
    try:
        import cv2
        print(f"  ✅ OpenCV {cv2.__version__} instalado")
    except ImportError as e:
        errors.append(f"OpenCV: {e}")
        print("  ❌ OpenCV não instalado")

    # Testar NumPy
    print("\n[4/6] Testando NumPy...")
    try:
        import numpy as np
        print(f"  ✅ NumPy {np.__version__} instalado")
    except ImportError as e:
        errors.append(f"NumPy: {e}")
        print("  ❌ NumPy não instalado")

    # Testar Pandas
    print("\n[5/6] Testando Pandas...")
    try:
        import pandas as pd
        print(f"  ✅ Pandas {pd.__version__} instalado")
    except ImportError as e:
        errors.append(f"Pandas: {e}")
        print("  ❌ Pandas não instalado")

    # Testar Matplotlib
    print("\n[6/6] Testando Matplotlib...")
    try:
        import matplotlib
        print(f"  ✅ Matplotlib {matplotlib.__version__} instalado")
    except ImportError as e:
        errors.append(f"Matplotlib: {e}")
        print("  ❌ Matplotlib não instalado")

    return errors


def test_project_structure():
    """Verifica se a estrutura do projeto está correta"""
    print("\n" + "=" * 60)
    print("VERIFICAÇÃO DA ESTRUTURA DO PROJETO")
    print("=" * 60)

    project_root = Path(__file__).parent

    required_dirs = [
        'config',
        'data',
        'models',
        'notebooks',
        'src',
        'src/emotion_recognition',
        'src/utils',
        'tests'
    ]

    required_files = [
        'config/config.yaml',
        'requirements.txt',
        'README.md',
        'src/__init__.py',
        'src/emotion_recognition/model.py',
        'src/emotion_recognition/data_loader.py',
        'src/emotion_recognition/train.py',
        'src/utils/config_loader.py'
    ]

    missing = []

    print("\nVerificando diretórios...")
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ (ausente)")
            missing.append(dir_path)

    print("\nVerificando arquivos...")
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (ausente)")
            missing.append(file_path)

    return missing


def test_dataset():
    """Verifica se o dataset FER-2013 está disponível"""
    print("\n" + "=" * 60)
    print("VERIFICAÇÃO DO DATASET")
    print("=" * 60)

    project_root = Path(__file__).parent
    fer2013_path = project_root / 'data' / 'fer2013.csv'

    if fer2013_path.exists():
        print(f"\n  ✅ Dataset encontrado: {fer2013_path}")

        # Verificar tamanho
        size_mb = fer2013_path.stat().st_size / (1024 * 1024)
        print(f"  📊 Tamanho: {size_mb:.2f} MB")

        return True
    else:
        print(f"\n  ❌ Dataset não encontrado: {fer2013_path}")
        print("\n  📥 Como obter o dataset:")
        print("     1. Acesse: https://www.kaggle.com/datasets/msambare/fer2013")
        print("     2. Baixe o arquivo fer2013.csv")
        print(f"     3. Coloque em: {fer2013_path}")

        return False


def test_config():
    """Testa se as configurações podem ser carregadas"""
    print("\n" + "=" * 60)
    print("TESTE DE CONFIGURAÇÃO")
    print("=" * 60)

    try:
        from src.utils.config_loader import config

        print("\n  ✅ Configuração carregada com sucesso")
        print(f"\n  📋 Classes de emoções: {config['dataset']['classes']}")
        print(f"  🎯 Número de classes: {config['dataset']['num_classes']}")
        print(f"  📐 Tamanho da imagem: {config['dataset']['img_size']}")
        print(f"  🏋️  Batch size: {config['training']['batch_size']}")
        print(f"  🔄 Epochs: {config['training']['epochs']}")

        return True
    except Exception as e:
        print(f"\n  ❌ Erro ao carregar configuração: {e}")
        return False


def main():
    """Função principal"""
    print("\n")

    # Teste 1: Imports
    import_errors = test_imports()

    # Teste 2: Estrutura
    missing_items = test_project_structure()

    # Teste 3: Dataset
    dataset_ok = test_dataset()

    # Teste 4: Config
    config_ok = test_config()

    # Resumo final
    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)

    if not import_errors and not missing_items and config_ok:
        print("\n✅ Ambiente configurado corretamente!")

        if dataset_ok:
            print("✅ Dataset disponível. Você pode começar o treinamento!")
            print("\n   Execute: python src/emotion_recognition/train.py")
        else:
            print("⚠️  Dataset não encontrado. Baixe-o antes de treinar.")
            print("\n   Veja instruções em: SETUP.md")
    else:
        print("\n❌ Há problemas na configuração:")

        if import_errors:
            print(f"\n  Bibliotecas faltando: {len(import_errors)}")
            print("  Execute: pip install -r requirements.txt")

        if missing_items:
            print(f"\n  Arquivos/diretórios faltando: {len(missing_items)}")
            for item in missing_items:
                print(f"    - {item}")

        if not config_ok:
            print("\n  Erro na configuração. Verifique config/config.yaml")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
