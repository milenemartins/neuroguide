"""
Carregamento e pré-processamento de dados para reconhecimento de emoções
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import cv2


class EmotionDataLoader:
    """
    Classe para carregar e preparar dados de emoções faciais

    Suporta:
    - FER-2013 dataset
    - Data augmentation
    - Normalização
    - Split train/val/test
    """

    def __init__(
        self,
        data_path: str,
        img_size: Tuple[int, int] = (48, 48),
        num_classes: int = 7,
        color_mode: str = 'grayscale'
    ):
        """
        Inicializa o data loader

        Args:
            data_path: Caminho para o arquivo CSV ou diretório com imagens
            img_size: Tamanho das imagens (altura, largura)
            num_classes: Número de classes de emoções
            color_mode: 'grayscale' ou 'rgb'
        """
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.num_classes = num_classes
        self.color_mode = color_mode
        self.channels = 1 if color_mode == 'grayscale' else 3

    def load_fer2013_csv(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Carrega o dataset FER-2013 de um arquivo CSV

        O FER-2013 CSV tem o formato:
        emotion, pixels, Usage
        0, "0 1 2 3...", Training
        1, "0 1 2 3...", PublicTest
        ...

        Args:
            csv_path: Caminho para o arquivo fer2013.csv

        Returns:
            X_train, y_train, X_test, y_test
        """
        print(f"Carregando dataset de {csv_path}...")
        df = pd.read_csv(csv_path)

        # Separar por conjunto (Training, PublicTest, PrivateTest)
        train_data = df[df['Usage'] == 'Training']
        val_data = df[df['Usage'] == 'PublicTest']
        test_data = df[df['Usage'] == 'PrivateTest']

        # Função para converter pixels string para array
        def preprocess_data(data):
            pixels = data['pixels'].tolist()
            images = []

            for pixel_sequence in pixels:
                # Converter string de pixels para array
                face = [int(pixel) for pixel in pixel_sequence.split()]
                face = np.array(face).reshape(self.img_size[0], self.img_size[1])

                if self.channels == 1:
                    face = np.expand_dims(face, axis=-1)

                images.append(face)

            images = np.array(images, dtype='float32')
            # Normalizar pixels para [0, 1]
            images /= 255.0

            # One-hot encode labels
            labels = to_categorical(data['emotion'].values, num_classes=self.num_classes)

            return images, labels

        X_train, y_train = preprocess_data(train_data)
        X_val, y_val = preprocess_data(val_data)
        X_test, y_test = preprocess_data(test_data)

        print(f"Dados carregados:")
        print(f"  Training: {X_train.shape}")
        print(f"  Validation: {X_val.shape}")
        print(f"  Test: {X_test.shape}")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def create_data_generators_from_folders(
        self,
        data_dir: str,
        batch_size: int = 64,
        validation_split: float = 0.2,
        augmentation: bool = True
    ):
        """
        Cria geradores de dados lendo diretamente de pastas com estrutura:
        data_dir/
            train/
                emotion1/
                emotion2/
                ...
            test/
                emotion1/
                emotion2/
                ...

        Args:
            data_dir: Diretório raiz contendo 'train' e 'test'
            batch_size: Tamanho do batch
            validation_split: Proporção de validação do conjunto de treino
            augmentation: Se True, aplica data augmentation no treino

        Returns:
            train_generator, val_generator, test_generator
        """
        train_path = Path(data_dir) / 'train'
        test_path = Path(data_dir) / 'test'

        if not train_path.exists() or not test_path.exists():
            raise ValueError(f"Pastas 'train' ou 'test' não encontradas em {data_dir}")

        # Configurar data augmentation para treino
        if augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                shear_range=0.1,
                fill_mode='nearest',
                validation_split=validation_split
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=validation_split
            )

        # Apenas rescaling para teste
        test_datagen = ImageDataGenerator(rescale=1./255)

        # Training generator
        train_generator = train_datagen.flow_from_directory(
            str(train_path),
            target_size=self.img_size,
            batch_size=batch_size,
            color_mode=self.color_mode,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        # Validation generator
        val_generator = train_datagen.flow_from_directory(
            str(train_path),
            target_size=self.img_size,
            batch_size=batch_size,
            color_mode=self.color_mode,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

        # Test generator
        test_generator = test_datagen.flow_from_directory(
            str(test_path),
            target_size=self.img_size,
            batch_size=batch_size,
            color_mode=self.color_mode,
            class_mode='categorical',
            shuffle=False
        )

        # Exibir informações
        print(f"\n✓ Dados carregados de pastas:")
        print(f"  Training samples: {train_generator.samples}")
        print(f"  Validation samples: {val_generator.samples}")
        print(f"  Test samples: {test_generator.samples}")
        print(f"  Classes detectadas: {list(train_generator.class_indices.keys())}")

        return train_generator, val_generator, test_generator

    def create_data_generators(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 64,
        augmentation: bool = True
    ):
        """
        Cria geradores de dados com augmentation para treinamento

        Args:
            X_train: Imagens de treino
            y_train: Labels de treino
            X_val: Imagens de validação
            y_val: Labels de validação
            batch_size: Tamanho do batch
            augmentation: Se True, aplica data augmentation no treino

        Returns:
            train_generator, val_generator
        """
        if augmentation:
            # Data augmentation para treino
            train_datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator()

        # Apenas normalização para validação (já foi feita, mas mantemos consistência)
        val_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=batch_size,
            shuffle=True
        )

        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=batch_size,
            shuffle=False
        )

        return train_generator, val_generator

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Pré-processa uma única imagem para inferência

        Args:
            image: Imagem em formato numpy array

        Returns:
            Imagem pré-processada
        """
        # Redimensionar
        image = cv2.resize(image, self.img_size)

        # Converter para grayscale se necessário
        if self.color_mode == 'grayscale' and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Normalizar
        image = image.astype('float32') / 255.0

        # Adicionar dimensão de canal se grayscale
        if self.channels == 1 and len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        # Adicionar dimensão de batch
        image = np.expand_dims(image, axis=0)

        return image


def download_fer2013_instructions():
    """
    Instruções para baixar o dataset FER-2013
    """
    instructions = """
    ========================================
    Como obter o dataset FER-2013
    ========================================

    1. Acesse o Kaggle: https://www.kaggle.com/datasets/msambare/fer2013

    2. Faça login ou crie uma conta no Kaggle

    3. Baixe o arquivo 'fer2013.csv'

    4. Coloque o arquivo na pasta: data/fer2013.csv

    Alternativamente, use a API do Kaggle:

    ```bash
    pip install kaggle
    kaggle datasets download -d msambare/fer2013
    unzip fer2013.zip -d data/
    ```

    ========================================
    """
    return instructions
