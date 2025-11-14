"""
Script de treinamento do modelo de reconhecimento de emoções
"""
import sys
from pathlib import Path

# Adicionar o diretório raiz ao path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from datetime import datetime

from src.emotion_recognition.model import create_emotion_model
from src.emotion_recognition.data_loader import EmotionDataLoader, download_fer2013_instructions
from src.utils.config_loader import config


def train_emotion_model():
    """Função principal de treinamento"""

    print("=" * 50)
    print("TREINAMENTO DO MODELO DE RECONHECIMENTO DE EMOÇÕES")
    print("=" * 50)

    # Carregar configurações
    paths = config.get_paths()
    data_dir = paths['data_dir']
    models_dir = paths['models_dir']
    logs_dir = paths['logs_dir']

    # Criar diretórios se não existirem
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Configurações do dataset
    dataset_config = config['dataset']
    img_size = tuple(dataset_config['img_size'])
    num_classes = dataset_config['num_classes']
    color_mode = dataset_config['color_mode']

    # Configurações de treinamento
    training_config = config['training']
    batch_size = training_config['batch_size']
    epochs = training_config['epochs']
    learning_rate = training_config['learning_rate']

    # Configurações do modelo
    model_config = config['model']
    input_shape = tuple(model_config['input_shape'])
    dropout_rate = model_config['dropout_rate']

    # Detectar formato do dataset (CSV vs Pastas)
    fer2013_csv = data_dir / 'fer2013.csv'
    train_folder = data_dir / 'train'
    test_folder = data_dir / 'test'

    use_folders = train_folder.exists() and test_folder.exists()
    use_csv = fer2013_csv.exists()

    if not use_folders and not use_csv:
        print("\n❌ Dataset FER-2013 não encontrado!")
        print("\nProcurando por:")
        print(f"  - CSV: {fer2013_csv}")
        print(f"  - Pastas: {train_folder} e {test_folder}")
        print(download_fer2013_instructions())
        return

    # Inicializar data loader
    data_loader = EmotionDataLoader(
        data_path=data_dir,
        img_size=img_size,
        num_classes=num_classes,
        color_mode=color_mode
    )

    augmentation_enabled = config['augmentation']['enabled']

    # Carregar dados de acordo com o formato disponível
    if use_folders:
        print("\n[1/5] Carregando dados de PASTAS...")
        print(f"  Train: {train_folder}")
        print(f"  Test: {test_folder}")

        validation_split = dataset_config.get('val_split', 0.2)
        train_gen, val_gen, test_gen = data_loader.create_data_generators_from_folders(
            data_dir=data_dir,
            batch_size=batch_size,
            validation_split=validation_split,
            augmentation=augmentation_enabled
        )

        # Para compatibilidade com código de avaliação
        X_test = None
        y_test = None

    else:  # use_csv
        print("\n[1/5] Carregando dados de CSV...")
        print(f"  Path: {fer2013_csv}")

        X_train, y_train, X_val, y_val, X_test, y_test = data_loader.load_fer2013_csv(fer2013_csv)

        # Criar data generators
        print("\n[2/5] Criando geradores de dados com augmentation...")
        train_gen, val_gen = data_loader.create_data_generators(
            X_train, y_train, X_val, y_val,
            batch_size=batch_size,
            augmentation=augmentation_enabled
        )
        test_gen = None

    # Criar modelo
    print("\n[3/5] Construindo modelo CNN...")
    model = create_emotion_model(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

    print("\nResumo do modelo:")
    model.summary()

    # Callbacks
    print("\n[4/5] Configurando callbacks...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"emotion_cnn_{timestamp}"

    callbacks = [
        # Salvar o melhor modelo
        ModelCheckpoint(
            filepath=str(models_dir / f"{model_name}_best.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),

        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=training_config['early_stopping']['patience'],
            restore_best_weights=True,
            verbose=1
        ),

        # Reduzir learning rate quando estagnado
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),

        # TensorBoard
        TensorBoard(
            log_dir=str(logs_dir / f"tensorboard_{timestamp}"),
            histogram_freq=1
        )
    ]

    # Treinar
    print("\n[5/5] Iniciando treinamento...")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Augmentation: {augmentation_enabled}")
    print()

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # Avaliar no conjunto de teste
    print("\n" + "=" * 50)
    print("AVALIAÇÃO NO CONJUNTO DE TESTE")
    print("=" * 50)

    if use_folders and test_gen is not None:
        # Avaliar usando generator de pastas
        test_loss, test_accuracy = model.evaluate(
            test_gen,
            steps=test_gen.samples // batch_size,
            verbose=1
        )
    else:
        # Avaliar usando arrays do CSV
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # Salvar modelo final
    final_model_path = models_dir / f"{model_name}_final.h5"
    model.save(final_model_path)
    print(f"\n✅ Modelo salvo em: {final_model_path}")

    # Salvar histórico
    history_path = models_dir / f"{model_name}_history.npz"
    np.savez(
        history_path,
        train_loss=history.history['loss'],
        train_accuracy=history.history['accuracy'],
        val_loss=history.history['val_loss'],
        val_accuracy=history.history['val_accuracy']
    )
    print(f"✅ Histórico salvo em: {history_path}")

    print("\n" + "=" * 50)
    print("TREINAMENTO CONCLUÍDO!")
    print("=" * 50)


if __name__ == "__main__":
    train_emotion_model()
