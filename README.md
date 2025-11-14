# NeuroGuide

**Assistente sensorial e emocional para autistas**

Sistema de Machine Learning com visão computacional para auxiliar pessoas com autismo em desafios cotidianos.

## Objetivos

### Fase 1: Reconhecimento de Emoções Faciais ✅ (Em desenvolvimento)
- Detectar e classificar emoções em tempo real via webcam
- Ajudar pessoas com autismo a interpretar expressões faciais
- Tecnologias: CNN, OpenCV, TensorFlow/PyTorch

### Fase 2: Detector de Sobrecarga Sensorial (Futuro)
- Analisar ambientes e identificar potenciais gatilhos sensoriais
- Alertar sobre luminosidade, movimento e padrões visuais complexos

## Estrutura do Projeto

```
neuroguide/
├── data/                    # Datasets (FER-2013, etc)
├── models/                  # Modelos treinados salvos (.h5, .pth)
├── src/
│   ├── emotion_recognition/ # Módulo de reconhecimento de emoções
│   ├── sensory_overload/    # Módulo de sobrecarga sensorial (futuro)
│   ├── utils/               # Utilitários compartilhados
│   └── app/                 # Interface da aplicação
├── notebooks/               # Jupyter notebooks para experimentação
├── tests/                   # Testes unitários
└── config/                  # Arquivos de configuração
```

## Setup

```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

## Uso

### Treinamento
```bash
python src/emotion_recognition/train.py
```

### Inferência em Tempo Real
```bash
python src/app/main.py
```

## Datasets

- **FER-2013**: Dataset principal para reconhecimento de emoções
- **AffectNet**: Dataset complementar (opcional)

## Tecnologias

- **Deep Learning**: TensorFlow/Keras
- **Visão Computacional**: OpenCV
- **Data Science**: NumPy, Pandas, Matplotlib
- **Interface**: Streamlit (futuro)

## Conceitos Técnicos Explicados

### Machine Learning (ML) vs Deep Learning (DL) vs Visão Computacional vs Redes Neurais

Este projeto integra múltiplas áreas da Inteligência Artificial. Entenda como cada uma funciona:

#### 🤖 Machine Learning (Aprendizado de Máquina)
**O que é**: Campo da IA onde computadores aprendem padrões a partir de dados, sem serem explicitamente programados.

**Analogia**: Como ensinar uma criança a reconhecer frutas mostrando exemplos.

**Tipos**:
- Supervisionado (com rótulos) ← **Nosso projeto usa este!**
- Não supervisionado (sem rótulos)
- Por reforço (aprendizado por tentativa e erro)

**Como usamos**: Treinamos o modelo com 35.000 imagens rotuladas de emoções para que ele aprenda os padrões faciais de cada emoção.

---

#### 🧠 Deep Learning (Aprendizado Profundo)
**O que é**: Subcampo do ML que usa redes neurais com múltiplas camadas (daí "profundo").

**Analogia**: Como um filtro de café com várias camadas, onde cada camada extrai características mais complexas.

**Relação com ML**:
```
Machine Learning (Campo amplo)
    └── Deep Learning (Especialização usando redes neurais profundas)
```

**Como usamos**: Utilizamos uma CNN (Rede Neural Convolucional) com 6+ camadas para reconhecer emoções. Cada camada aprende padrões: bordas → formas → partes do rosto → emoções completas.

---

#### 🔍 Visão Computacional (Computer Vision)
**O que é**: Campo que ensina computadores a "enxergar" e interpretar imagens/vídeos.

**Analogia**: Dar "olhos" ao computador para entender o mundo visual.

**Técnicas**:
- Clássicas: Detecção de bordas, filtros, transformações
- Modernas: Deep Learning para reconhecimento de objetos/faces

**Como usamos**:
- OpenCV para capturar vídeo da webcam e detectar rostos (técnica clássica: Haar Cascade)
- CNN para reconhecer emoções nas faces detectadas (técnica moderna)

---

#### 🕸️ Redes Neurais (Neural Networks)
**O que é**: Modelos computacionais inspirados no cérebro humano, compostos por neurônios artificiais conectados.

**Analogia**: Como neurônios no cérebro que passam sinais elétricos, mas versão digital.

**Tipos**:
- MLP (Multilayer Perceptron): Rede básica
- **CNN** (Convolutional NN): Para imagens ← **Usamos esta!**
- RNN (Recurrent NN): Para sequências temporais
- Transformer: Para linguagem natural

**Como usamos**: Nossa CNN possui:
```
Entrada (48x48 pixels)
    ↓
Conv2D → Extrai bordas e texturas
    ↓
Conv2D → Detecta formas (olhos, boca)
    ↓
Conv2D → Reconhece padrões faciais
    ↓
Dense → Combina características
    ↓
Saída → 7 emoções (Raiva, Feliz, Triste, etc.)
```

---

### Como tudo se conecta neste projeto?

```
┌────────────────────────────────────────────────┐
│         VISÃO COMPUTACIONAL                    │
│  (Objetivo: Processar imagens/vídeo)           │
│                                                │
│  ┌────────────────────────────────────────┐    │
│  │    MACHINE LEARNING                    │    │
│  │  (Método: Aprender padrões dos dados)  │    │
│  │                                        │    │
│  │  ┌──────────────────────────────────┐  │    │
│  │  │   DEEP LEARNING                  │  │    │
│  │  │(Técnica: Redes neurais profundas)│  │    │
│  │  │                                  │  │    │
│  │  │  ┌──────────────────────────┐    │  │    │
│  │  │  │  REDES NEURAIS (CNN)     │    │  │    │
│  │  │  │ (Ferramenta específica)  │    │  │    │
│  │  │  └──────────────────────────┘    │  │    │
│  │  └──────────────────────────────────┘  │    │
│  └────────────────────────────────────────┘    │
└────────────────────────────────────────────────┘
```

**Em resumo**:
1. **Visão Computacional** = O problema (processar imagens)
2. **Machine Learning** = A abordagem (aprender com dados)
3. **Deep Learning** = A técnica específica (usar redes profundas)
4. **Redes Neurais (CNN)** = A ferramenta exata (arquitetura para imagens)

---

### Pipeline do NeuroGuide

```
[Webcam]
    ↓
[OpenCV detecta rosto] ← Visão Computacional Clássica
    ↓
[Pré-processamento: 48x48, grayscale]
    ↓
[CNN (Rede Neural)] ← Deep Learning
    ↓ (Múltiplas camadas)
[Classificação] ← Machine Learning Supervisionado
    ↓
[Emoção detectada: "Feliz" (95% confiança)]
```

---

### Por que usar Deep Learning e não ML tradicional?

**ML Tradicional** (ex: SVM, Random Forest):
- Requer extração manual de características
- Você precisa dizer: "Olhe para curvatura da boca, posição das sobrancelhas..."
- Limitado em complexidade

**Deep Learning** (CNN):
- Aprende características automaticamente
- Descobre sozinho o que é importante em cada camada
- Muito melhor para dados complexos como imagens

---

### Recursos para Aprender Mais

**Machine Learning**:
- Curso: [Machine Learning - Andrew Ng (Coursera)](https://www.coursera.org/learn/machine-learning)

**Deep Learning**:
- Curso: [Deep Learning Specialization (Coursera)](https://www.coursera.org/specializations/deep-learning)
- Livro: [Deep Learning Book](https://www.deeplearningbook.org/) (gratuito)

**Visão Computacional**:
- Curso: [CS231n - Stanford](http://cs231n.stanford.edu/)
- Documentação: [OpenCV Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

**Redes Neurais**:
- Visualização: [Neural Network Playground](https://playground.tensorflow.org/)
- Vídeo: [3Blue1Brown - Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk)

## Roadmap

- [x] Estrutura base do projeto
- [ ] Implementar CNN para reconhecimento de emoções
- [ ] Pipeline de treinamento
- [ ] Interface de captura em tempo real
- [ ] Deploy do modelo
- [ ] Módulo de sobrecarga sensorial

## Autor

Projeto desenvolvido como parte dos estudos em ML/Deep Learning na pós-graduação.
