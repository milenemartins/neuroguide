# Guia Rápido - NeuroGuide

## Início Rápido (Quick Start)

### 1. Instalação

```bash
# Navegar para o projeto
cd ~/Documents/projetos-pessoais/neuroguide

# Criar e ativar ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Mac/Linux

# Instalar dependências
pip install -r requirements.txt
```

### 2. Testar Ambiente

```bash
python test_setup.py
```

### 3. Baixar Dataset

Baixe o FER-2013 de: https://www.kaggle.com/datasets/msambare/fer2013

Coloque em: `data/fer2013.csv`

### 4. Explorar Dados (Opcional)

```bash
jupyter notebook
# Abra: notebooks/01_exploracao_dataset.ipynb
```

### 5. Treinar Modelo

```bash
python src/emotion_recognition/train.py
```

**Tempo estimado**: 30 minutos a 2 horas (dependendo do hardware)

### 6. Testar em Tempo Real

```bash
python src/app/realtime_detector.py --model models/emotion_cnn_XXXXXX_best.h5
```

(Substitua XXXXXX pelo timestamp do seu modelo treinado)

---

## Estrutura do Projeto Explicada

```
neuroguide/
│
├── config/config.yaml           # ⚙️ CONFIGURAÇÕES CENTRAIS
│                                # Altere aqui: batch_size, epochs, learning_rate, etc.
│
├── data/                        # 📊 DADOS
│   └── fer2013.csv             # Dataset (você precisa baixar)
│
├── models/                      # 🧠 MODELOS TREINADOS
│   ├── *_best.h5               # Melhor modelo (maior val_accuracy)
│   ├── *_final.h5              # Modelo final após todas as epochs
│   └── *_history.npz           # Histórico de treinamento
│
├── src/
│   ├── emotion_recognition/     # 😊 MÓDULO PRINCIPAL
│   │   ├── model.py            # Arquitetura da CNN
│   │   ├── data_loader.py      # Carregamento e preprocessamento
│   │   └── train.py            # Script de treinamento
│   │
│   ├── app/                     # 📹 APLICAÇÃO EM TEMPO REAL
│   │   └── realtime_detector.py
│   │
│   ├── sensory_overload/        # 🔮 MÓDULO FUTURO
│   │   └── README.md           # Planejamento
│   │
│   └── utils/                   # 🛠️ UTILITÁRIOS
│       └── config_loader.py    # Carregador de configurações
│
├── notebooks/                   # 📓 ANÁLISES E EXPERIMENTOS
│   └── 01_exploracao_dataset.ipynb
│
└── tests/                       # ✅ TESTES (futuro)
```

---

## Comandos Importantes

### Treinar com configurações customizadas

Edite `config/config.yaml` primeiro, depois:

```bash
python src/emotion_recognition/train.py
```

### Ver progresso do treinamento (TensorBoard)

```bash
tensorboard --logdir logs/
```

Acesse: http://localhost:6006

### Testar modelo específico

```bash
python src/app/realtime_detector.py \
  --model models/emotion_cnn_20250114_143000_best.h5 \
  --threshold 0.6 \
  --camera 0
```

---

## Parâmetros Importantes (config.yaml)

### Para melhorar acurácia:
- Aumente `epochs` (50 → 100)
- Reduza `learning_rate` (0.001 → 0.0005)
- Aumente `dropout_rate` (0.5 → 0.6)

### Para treinar mais rápido:
- Aumente `batch_size` (64 → 128)
- Reduza `epochs` (50 → 30)
- Desabilite data augmentation

### Para evitar overfitting:
- Aumente `dropout_rate`
- Habilite `augmentation`
- Aumente `early_stopping.patience`

---

## Troubleshooting Rápido

**Erro de memória (OOM)?**
→ Reduza `batch_size` para 32 ou 16

**Modelo não converge?**
→ Reduza `learning_rate` para 0.0001

**Webcam não abre?**
→ Tente `--camera 1` ou verifique permissões

**Import error?**
→ Certifique-se de estar na raiz do projeto

---

## Próximos Passos Após Treinamento

1. **Avaliar performance**
   - Verificar accuracy no test set
   - Analisar matriz de confusão
   - Identificar emoções com maior erro

2. **Otimizar modelo**
   - Experimentar outras arquiteturas
   - Transfer learning (MobileNet, ResNet)
   - Ensemble de modelos

3. **Melhorar aplicação**
   - Adicionar histórico de emoções
   - Criar dashboard com gráficos
   - Salvar logs de detecções

4. **Expandir sistema**
   - Implementar detector de sobrecarga sensorial
   - Combinar ambos os módulos
   - Criar interface web com Streamlit

---

## Recursos de Aprendizado

**Deep Learning Basics:**
- https://www.tensorflow.org/tutorials
- https://keras.io/getting_started/

**CNNs para Visão Computacional:**
- https://cs231n.stanford.edu/

**Reconhecimento de Emoções:**
- FER-2013 Paper: https://arxiv.org/abs/1307.0414

**Autismo e Tecnologia:**
- Research papers sobre emotion recognition para autismo
- Guidelines de acessibilidade para neurodiversidade

---

## Dicas para o Projeto da Pós

1. **Documente o processo**: Registre cada experimento
2. **Compare abordagens**: Teste diferentes arquiteturas
3. **Analise resultados**: Não apenas a acurácia, mas quais emoções confundem o modelo
4. **Contextualize**: Explique como isso ajuda pessoas com autismo
5. **Considere ética**: Privacidade, consentimento, bias nos dados

Boa sorte no projeto! 🚀
