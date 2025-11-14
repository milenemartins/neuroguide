"""
Arquitetura do modelo CNN para reconhecimento de emoções
"""
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from typing import Tuple, List


class EmotionCNN:
    """
    Rede Neural Convolucional para classificação de emoções faciais

    Arquitetura:
    - 4 blocos convolucionais com MaxPooling e BatchNormalization
    - 2 camadas densas com Dropout
    - Camada de saída com Softmax para 7 emoções
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (48, 48, 1),
        num_classes: int = 7,
        dropout_rate: float = 0.5
    ):
        """
        Inicializa o modelo

        Args:
            input_shape: Formato da imagem de entrada (altura, largura, canais)
            num_classes: Número de classes de emoções
            dropout_rate: Taxa de dropout para regularização
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model = None

    def build_model(self) -> keras.Model:
        """
        Constrói a arquitetura da CNN

        Returns:
            Modelo Keras compilado
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),

            # Bloco 1: Conv + Conv + MaxPool + BatchNorm
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),

            # Bloco 2: Conv + Conv + MaxPool + BatchNorm
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),

            # Bloco 3: Conv + Conv + MaxPool + BatchNorm
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),

            # Bloco 4: Conv + Conv + MaxPool + BatchNorm
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),

            # Flatten e camadas densas
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),

            # Camada de saída
            layers.Dense(self.num_classes, activation='softmax')
        ])

        self.model = model
        return model

    def compile_model(
        self,
        learning_rate: float = 0.001,
        optimizer: str = 'adam',
        loss: str = 'categorical_crossentropy',
        metrics: List[str] = None
    ):
        """
        Compila o modelo com otimizador e função de perda

        Args:
            learning_rate: Taxa de aprendizado
            optimizer: Nome do otimizador
            loss: Função de perda
            metrics: Lista de métricas para avaliar
        """
        if self.model is None:
            raise ValueError("Modelo não foi construído. Execute build_model() primeiro.")

        if metrics is None:
            metrics = ['accuracy']

        # Configurar otimizador
        if optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer.lower() == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer

        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )

    def summary(self):
        """Exibe o resumo da arquitetura do modelo"""
        if self.model is None:
            raise ValueError("Modelo não foi construído.")
        return self.model.summary()

    def get_model(self) -> keras.Model:
        """Retorna o modelo Keras"""
        return self.model


def create_emotion_model(
    input_shape: Tuple[int, int, int] = (48, 48, 1),
    num_classes: int = 7,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.001
) -> keras.Model:
    """
    Função auxiliar para criar e compilar o modelo de uma vez

    Args:
        input_shape: Formato da imagem de entrada
        num_classes: Número de classes de emoções
        dropout_rate: Taxa de dropout
        learning_rate: Taxa de aprendizado

    Returns:
        Modelo Keras compilado
    """
    emotion_cnn = EmotionCNN(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )

    emotion_cnn.build_model()
    emotion_cnn.compile_model(learning_rate=learning_rate)

    return emotion_cnn.get_model()
