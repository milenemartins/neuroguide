# Próximos Passos e Roadmap do Projeto

## Status Atual ✅

O projeto está com a estrutura completa para a **Fase 1: Reconhecimento de Emoções**

**O que já está pronto:**
- ✅ Arquitetura CNN customizada
- ✅ Pipeline de treinamento completo
- ✅ Data augmentation configurável
- ✅ Sistema de configuração centralizado
- ✅ Interface de detecção em tempo real
- ✅ Notebooks de análise e exploração
- ✅ Documentação completa

## Fase 1: Melhorias Imediatas

### 1.1 Otimização do Modelo Atual
**Objetivo**: Melhorar a acurácia do modelo base

**Tarefas:**
- [ ] Experimentar diferentes learning rates (grid search)
- [ ] Testar variações de dropout (0.3, 0.5, 0.7)
- [ ] Implementar weight decay (L2 regularization)
- [ ] Testar batch normalization em diferentes posições
- [ ] Implementar early stopping mais refinado

**Skills a aprender:**
- Hyperparameter tuning
- Regularization techniques
- Cross-validation

### 1.2 Arquiteturas Alternativas
**Objetivo**: Comparar com arquiteturas conhecidas

**Tarefas:**
- [ ] Implementar MobileNetV2 com transfer learning
- [ ] Testar ResNet50 pré-treinado
- [ ] Experimentar VGG16 fine-tuned
- [ ] Criar ensemble de modelos
- [ ] Comparar performance vs. tamanho

**Skills a aprender:**
- Transfer learning
- Fine-tuning
- Model ensembling

**Código sugerido:**
```python
# src/emotion_recognition/transfer_learning.py
from keras.applications import MobileNetV2, ResNet50
from keras.layers import GlobalAveragePooling2D, Dense

def create_mobilenet_model(num_classes=7):
    base_model = MobileNetV2(
        input_shape=(48, 48, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model
```

### 1.3 Análise Avançada
**Objetivo**: Entender profundamente o comportamento do modelo

**Tarefas:**
- [ ] Implementar Grad-CAM para visualizar atenção
- [ ] Criar matriz de confusão interativa
- [ ] Análise de erro por grupo demográfico (se dados disponíveis)
- [ ] Calcular métricas por confiança (calibration)
- [ ] Analisar imagens que o modelo erra consistentemente

**Skills a aprender:**
- Explainable AI (XAI)
- Model interpretability
- Error analysis

## Fase 2: Detector de Sobrecarga Sensorial

### 2.1 Módulo de Análise de Luminosidade
**Objetivo**: Detectar ambientes muito claros ou com brilho excessivo

**Tarefas:**
- [ ] Implementar cálculo de brilho médio
- [ ] Detectar hotspots (áreas muito claras)
- [ ] Calcular distribuição de luminosidade (histograma)
- [ ] Detectar variação rápida (flicker)
- [ ] Criar thresholds personalizáveis

**Código sugerido:**
```python
# src/sensory_overload/brightness_analyzer.py
import cv2
import numpy as np

class BrightnessAnalyzer:
    def __init__(self, threshold=200):
        self.threshold = threshold

    def analyze(self, frame):
        # Converter para grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Brilho médio
        mean_brightness = np.mean(gray)

        # Detectar hotspots (áreas > threshold)
        hotspots = np.sum(gray > self.threshold) / gray.size

        # Score de sobrecarga (0-1)
        overload_score = min(mean_brightness / 255, 1.0)

        return {
            'mean_brightness': mean_brightness,
            'hotspot_ratio': hotspots,
            'overload_score': overload_score
        }
```

### 2.2 Detector de Movimento
**Objetivo**: Quantificar movimento e mudanças na cena

**Tarefas:**
- [ ] Implementar Optical Flow (Lucas-Kanade ou Farneback)
- [ ] Calcular magnitude de movimento
- [ ] Detectar movimento caótico vs. suave
- [ ] Identificar mudanças bruscas de cena
- [ ] Criar score de "movimento excessivo"

**Código sugerido:**
```python
# src/sensory_overload/motion_detector.py
import cv2

class MotionDetector:
    def __init__(self):
        self.prev_frame = None

    def analyze(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame is None:
            self.prev_frame = gray
            return {'motion_score': 0.0}

        # Optical Flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Magnitude do movimento
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        motion_score = np.mean(magnitude) / 10.0  # Normalizar

        self.prev_frame = gray

        return {'motion_score': min(motion_score, 1.0)}
```

### 2.3 Análise de Padrões Visuais
**Objetivo**: Detectar padrões complexos que podem ser desconfortáveis

**Tarefas:**
- [ ] Calcular densidade de bordas (Canny/Sobel)
- [ ] Analisar frequências espaciais (FFT)
- [ ] Detectar padrões repetitivos
- [ ] Calcular entropia visual
- [ ] Implementar GLCM para análise de textura

### 2.4 Sistema de Score Agregado
**Objetivo**: Combinar todas as métricas em um score único

**Tarefas:**
- [ ] Definir pesos para cada componente
- [ ] Criar sistema de alertas por nível
- [ ] Implementar histórico temporal (não apenas frame atual)
- [ ] Permitir personalização de thresholds
- [ ] Criar interface visual de feedback

## Fase 3: Sistema Integrado

### 3.1 Combinar Módulos
**Objetivo**: Interface única que mostra emoções + sobrecarga

**Tarefas:**
- [ ] Criar dashboard unificado
- [ ] Mostrar emoção detectada + ambiente sensorial
- [ ] Correlacionar emoções com condições ambientais
- [ ] Gerar relatórios de uso
- [ ] Implementar histórico de sessões

### 3.2 Interface Web
**Objetivo**: Tornar acessível via navegador

**Tarefas:**
- [ ] Implementar com Streamlit ou Gradio
- [ ] Adicionar upload de vídeos (além de webcam ao vivo)
- [ ] Criar modo de análise batch
- [ ] Exportar relatórios em PDF
- [ ] Adicionar configurações de usuário

**Código sugerido (Streamlit):**
```python
# app.py
import streamlit as st
from src.app.realtime_detector import RealtimeEmotionDetector

st.title("🧠 NeuroGuide - Assistente sensorial e emocional para autistas")

tab1, tab2 = st.tabs(["Emoções", "Sobrecarga Sensorial"])

with tab1:
    st.header("Reconhecimento de Emoções")
    # Implementar interface de webcam

with tab2:
    st.header("Análise de Ambiente")
    # Mostrar métricas sensoriais
```

### 3.3 Persistência de Dados
**Objetivo**: Salvar histórico para análise

**Tarefas:**
- [ ] Criar banco de dados SQLite
- [ ] Salvar detecções ao longo do tempo
- [ ] Implementar análise de padrões pessoais
- [ ] Criar gráficos de tendência
- [ ] Permitir exportação de dados

## Fase 4: Melhorias Avançadas

### 4.1 Personalização
- [ ] Sistema de perfis de usuário
- [ ] Thresholds adaptativos
- [ ] Machine learning para aprender preferências
- [ ] Alertas customizados

### 4.2 Edge Deployment
- [ ] Converter modelo para TensorFlow Lite
- [ ] Deploy em Raspberry Pi
- [ ] Criar app mobile (React Native + TF Lite)
- [ ] Otimizar para baixo consumo

### 4.3 Acessibilidade
- [ ] Adicionar suporte a texto-para-voz
- [ ] Interface simplificada
- [ ] Modo de alto contraste
- [ ] Documentação em múltiplos idiomas

## Ideias para Expansão Futura

### 1. Reconhecimento de Contexto Social
- Detectar múltiplas pessoas e suas emoções
- Identificar situações sociais complexas
- Sugerir interpretações de cenários

### 2. Treino Gamificado
- Jogos para praticar reconhecimento de emoções
- Sistema de pontuação e progresso
- Feedback adaptativo

### 3. Realidade Aumentada
- Overlay de informações sobre emoções em AR
- Alertas visuais sutis em óculos AR
- Simulações de situações sociais

### 4. Integração com Wearables
- Smartwatch para alertas discretos
- Monitorar sinais fisiológicos (frequência cardíaca)
- Correlacionar estado emocional com dados biométricos

### 5. Análise de Áudio
- Reconhecimento de emoções por voz
- Análise de prosódia e tom
- Detecção de sobrecarga auditiva (volume, frequências)

## Recursos para Cada Fase

### Para Fase 1:
- **Curso**: CS231n (Stanford) - CNNs
- **Livro**: Deep Learning (Goodfellow)
- **Papers**: FER surveys, emotion recognition

### Para Fase 2:
- **OpenCV Tutorials**: Optical flow, edge detection
- **Papers**: Visual complexity, sensory processing
- **Livro**: Computer Vision (Szeliski)

### Para Fase 3:
- **Streamlit Docs**: https://docs.streamlit.io/
- **Flask/FastAPI**: Para backend
- **React**: Se quiser web app mais sofisticado

### Para Fase 4:
- **TensorFlow Lite**: https://www.tensorflow.org/lite
- **ONNX**: Para portabilidade de modelos
- **Edge AI**: Tutoriais de deploy

## Métricas de Sucesso

**Técnicas:**
- Acurácia > 70% (FER-2013 baseline: ~65%)
- Latência < 100ms para detecção em tempo real
- Tamanho do modelo < 50MB (para mobile)

**Impacto:**
- Testes com usuários reais (ética aprovada)
- Feedback qualitativo
- Melhoria mensurável em reconhecimento social

## Considerações Éticas

- [ ] Obter consentimento para uso de câmera
- [ ] Garantir privacidade (processar localmente)
- [ ] Evitar reforçar estereótipos sobre autismo
- [ ] Consultar comunidade autista
- [ ] Transparência sobre limitações do sistema

---

## Como Usar Este Roadmap

1. **Comece pela Fase 1.1**: Otimize o modelo atual
2. **Documente tudo**: Para seu projeto da pós
3. **Faça experimentos**: Compare resultados
4. **Priorize**: Nem tudo precisa ser feito
5. **Compartilhe**: Open-source ajuda a comunidade

**Sugestão de ordem para a pós:**

1. Treinar modelo base (1-2 semanas)
2. Experimentar arquiteturas (1-2 semanas)
3. Análise profunda dos resultados (1 semana)
4. Implementar 1 funcionalidade de sobrecarga sensorial (2 semanas)
5. Documentar para o TCC/apresentação (contínuo)

Boa jornada de aprendizado! 🚀
