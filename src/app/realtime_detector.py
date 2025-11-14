"""
Detector de emoções em tempo real usando webcam
"""
import sys
from pathlib import Path

# Adicionar o diretório raiz ao path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from keras.models import load_model
from typing import Tuple, Optional

from src.utils.config_loader import config


class RealtimeEmotionDetector:
    """
    Classe para detecção de emoções em tempo real via webcam

    Funcionalidades:
    - Captura de vídeo da webcam
    - Detecção de rostos usando Haar Cascade
    - Classificação de emoções usando modelo treinado
    - Exibição dos resultados em tempo real
    """

    def __init__(
        self,
        model_path: str,
        camera_id: int = 0,
        confidence_threshold: float = 0.5
    ):
        """
        Inicializa o detector em tempo real

        Args:
            model_path: Caminho para o modelo treinado (.h5)
            camera_id: ID da câmera (0 para webcam padrão)
            confidence_threshold: Threshold mínimo de confiança para exibir emoção
        """
        self.model_path = Path(model_path)
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold

        # Carregar configurações
        self.emotion_classes = config['dataset']['classes']
        self.img_size = tuple(config['dataset']['img_size'])

        # Carregar modelo
        print(f"Carregando modelo de {self.model_path}...")
        self.model = load_model(self.model_path)
        print("✅ Modelo carregado com sucesso!")

        # Carregar detector de rostos (Haar Cascade)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise ValueError("Erro ao carregar Haar Cascade para detecção de rostos")

        # Cores para cada emoção (BGR)
        self.emotion_colors = {
            'Raiva': (0, 0, 255),      # Vermelho
            'Nojo': (0, 128, 0),       # Verde escuro
            'Medo': (128, 0, 128),     # Roxo
            'Feliz': (0, 255, 255),    # Amarelo
            'Triste': (255, 0, 0),     # Azul
            'Surpreso': (255, 165, 0), # Laranja
            'Neutro': (128, 128, 128)  # Cinza
        }

    def preprocess_face(self, face: np.ndarray) -> np.ndarray:
        """
        Pré-processa um rosto detectado para o modelo

        Args:
            face: Imagem do rosto (região detectada)

        Returns:
            Rosto pré-processado para o modelo
        """
        # Converter para grayscale se necessário
        if len(face.shape) == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Redimensionar para o tamanho esperado pelo modelo
        face = cv2.resize(face, self.img_size)

        # Normalizar
        face = face.astype('float32') / 255.0

        # Adicionar dimensões (batch, height, width, channels)
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        return face

    def predict_emotion(self, face: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Prediz a emoção de um rosto

        Args:
            face: Imagem do rosto detectado

        Returns:
            (emoção, confiança, probabilidades de todas as classes)
        """
        # Pré-processar
        processed_face = self.preprocess_face(face)

        # Predição
        predictions = self.model.predict(processed_face, verbose=0)[0]

        # Emoção com maior probabilidade
        emotion_idx = np.argmax(predictions)
        emotion = self.emotion_classes[emotion_idx]
        confidence = predictions[emotion_idx]

        return emotion, confidence, predictions

    def draw_results(
        self,
        frame: np.ndarray,
        x: int, y: int, w: int, h: int,
        emotion: str,
        confidence: float,
        predictions: np.ndarray
    ) -> np.ndarray:
        """
        Desenha os resultados no frame

        Args:
            frame: Frame de vídeo
            x, y, w, h: Coordenadas do rosto detectado
            emotion: Emoção detectada
            confidence: Confiança da predição
            predictions: Array com todas as probabilidades

        Returns:
            Frame com anotações
        """
        # Cor baseada na emoção
        color = self.emotion_colors.get(emotion, (255, 255, 255))

        # Desenhar retângulo ao redor do rosto
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Texto com emoção e confiança
        text = f"{emotion}: {confidence * 100:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        # Calcular tamanho do texto para desenhar background
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Desenhar background do texto
        cv2.rectangle(
            frame,
            (x, y - text_h - 10),
            (x + text_w, y),
            color,
            -1
        )

        # Desenhar texto
        cv2.putText(
            frame,
            text,
            (x, y - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )

        # Desenhar barra de probabilidades (pequena)
        bar_height = 15
        bar_width = w
        bar_y = y + h + 10

        for i, (emotion_name, prob) in enumerate(zip(self.emotion_classes, predictions)):
            bar_x = x
            bar_fill = int(bar_width * prob)

            # Background da barra
            cv2.rectangle(
                frame,
                (bar_x, bar_y + i * (bar_height + 2)),
                (bar_x + bar_width, bar_y + i * (bar_height + 2) + bar_height),
                (50, 50, 50),
                -1
            )

            # Preenchimento da barra
            emotion_color = self.emotion_colors.get(emotion_name, (255, 255, 255))
            cv2.rectangle(
                frame,
                (bar_x, bar_y + i * (bar_height + 2)),
                (bar_x + bar_fill, bar_y + i * (bar_height + 2) + bar_height),
                emotion_color,
                -1
            )

            # Nome da emoção
            cv2.putText(
                frame,
                f"{emotion_name[:3]} {prob * 100:.0f}%",
                (bar_x + 5, bar_y + i * (bar_height + 2) + bar_height - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1
            )

        return frame

    def run(self):
        """
        Inicia a captura e detecção em tempo real

        Pressione 'q' para sair
        Pressione 's' para salvar um screenshot
        """
        # Iniciar captura de vídeo
        print(f"\nIniciando captura da câmera {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir a câmera {self.camera_id}")

        print("✅ Câmera iniciada!")
        print("\nControles:")
        print("  'q' - Sair")
        print("  's' - Salvar screenshot")
        print("\nAguardando detecção de rostos...\n")

        screenshot_count = 0

        try:
            while True:
                # Capturar frame
                ret, frame = cap.read()

                if not ret:
                    print("Erro ao capturar frame")
                    break

                # Converter para grayscale para detecção de rostos
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detectar rostos
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                # Para cada rosto detectado
                for (x, y, w, h) in faces:
                    # Extrair região do rosto
                    face_roi = gray[y:y + h, x:x + w]

                    # Predizer emoção
                    emotion, confidence, predictions = self.predict_emotion(face_roi)

                    # Desenhar resultados apenas se confiança > threshold
                    if confidence >= self.confidence_threshold:
                        frame = self.draw_results(
                            frame, x, y, w, h,
                            emotion, confidence, predictions
                        )

                # Adicionar informações na tela
                cv2.putText(
                    frame,
                    f"Rostos detectados: {len(faces)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # Exibir frame
                cv2.imshow('Detector de Emocoes em Tempo Real', frame)

                # Verificar teclas
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\nEncerrando...")
                    break
                elif key == ord('s'):
                    # Salvar screenshot
                    screenshot_path = project_root / f"screenshot_{screenshot_count}.png"
                    cv2.imwrite(str(screenshot_path), frame)
                    print(f"Screenshot salvo: {screenshot_path}")
                    screenshot_count += 1

        finally:
            # Liberar recursos
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Recursos liberados")


def main():
    """Função principal"""
    import argparse

    parser = argparse.ArgumentParser(description='Detector de Emoções em Tempo Real')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Caminho para o modelo treinado (.h5)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='ID da câmera (padrão: 0)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold de confiança (padrão: 0.5)'
    )

    args = parser.parse_args()

    # Criar detector
    detector = RealtimeEmotionDetector(
        model_path=args.model,
        camera_id=args.camera,
        confidence_threshold=args.threshold
    )

    # Executar
    detector.run()


if __name__ == "__main__":
    main()
