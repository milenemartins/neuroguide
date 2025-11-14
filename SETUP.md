# Guia de Setup - NeuroGuide

## 1. Pré-requisitos

- Python 3.9 ou superior
- pip (gerenciador de pacotes Python)
- Git (opcional, mas recomendado)
- Webcam (para testes em tempo real)

## 2. Configuração do Ambiente

### 2.1 Criar e Ativar Ambiente Virtual

```bash
# Navegar para o diretório do projeto
cd ~/Documents/projetos-pessoais/neuroguide

# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
# No Mac/Linux:
source venv/bin/activate

# No Windows:
# venv\Scripts\activate
```

### 2.2 Instalar Dependências

```bash
# Atualizar pip
pip install --upgrade pip

# Instalar todas as dependências
pip install -r requirements.txt
```

**Observação**: A instalação pode demorar alguns minutos, especialmente o TensorFlow.

## 3. Obter o Dataset FER-2013

### Opção 1: Download Manual

1. Acesse: https://www.kaggle.com/datasets/msambare/fer2013
2. Faça login no Kaggle (ou crie uma conta gratuita)
3. Clique em "Download" para baixar o `fer2013.csv`
4. Mova o arquivo para: `data/fer2013.csv`

### Opção 2: Usando a API do Kaggle

```bash
# Instalar Kaggle CLI
pip install kaggle

# Configurar credenciais (siga: https://www.kaggle.com/docs/api)
# Baixe kaggle.json e coloque em ~/.kaggle/

# Baixar o dataset
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/
```

## 4. Verificar Instalação

```bash
# Testar se o TensorFlow está funcionando
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} instalado com sucesso!')"

# Testar se o OpenCV está funcionando
python -c "import cv2; print(f'OpenCV {cv2.__version__} instalado com sucesso!')"

# Verificar se o dataset foi baixado
ls -lh data/fer2013.csv
```

## 5. Explorar o Dataset

```bash
# Iniciar Jupyter Notebook
jupyter notebook

# Abrir: notebooks/01_exploracao_dataset.ipynb
```

## 6. Treinar o Modelo

```bash
# Executar o script de treinamento
python src/emotion_recognition/train.py
```

**Observação**:
- O treinamento pode demorar bastante (30min - 2h dependendo do hardware)
- Se você tiver GPU NVIDIA, o TensorFlow irá utilizá-la automaticamente
- Os modelos treinados serão salvos em `models/`

## 7. Estrutura do Projeto

```
neuroguide/
├── config/
│   └── config.yaml          # Configurações centralizadas
├── data/
│   └── fer2013.csv          # Dataset (você precisa baixar)
├── models/                   # Modelos treinados (gerados após treino)
├── notebooks/
│   └── 01_exploracao_dataset.ipynb  # Análise exploratória
├── src/
│   ├── emotion_recognition/
│   │   ├── model.py         # Arquitetura CNN
│   │   ├── data_loader.py   # Carregamento de dados
│   │   └── train.py         # Script de treinamento
│   ├── utils/
│   │   └── config_loader.py # Utilitário de configuração
│   └── app/                  # Aplicação (futuro)
├── tests/                    # Testes unitários (futuro)
├── requirements.txt          # Dependências Python
├── README.md                 # Documentação principal
└── SETUP.md                  # Este arquivo
```

## 8. Próximos Passos

Depois que o modelo estiver treinado:

1. **Avaliar performance**: Verificar métricas no conjunto de teste
2. **Visualizar resultados**: Usar TensorBoard para ver curvas de aprendizado
3. **Implementar inferência**: Criar aplicação em tempo real com webcam
4. **Expandir sistema**: Adicionar módulo de sobrecarga sensorial

## 9. Troubleshooting

### Erro de memória durante treinamento
- Reduza o `batch_size` em `config/config.yaml`

### TensorFlow não detecta GPU
- Verifique drivers CUDA/cuDNN
- Ou use CPU mesmo (será mais lento, mas funciona)

### ImportError ao rodar scripts
- Certifique-se de estar na raiz do projeto
- Verifique se o ambiente virtual está ativado

## 10. Recursos Úteis

- **TensorFlow Docs**: https://www.tensorflow.org/tutorials
- **Keras Guide**: https://keras.io/guides/
- **OpenCV Tutorials**: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- **FER-2013 Paper**: https://arxiv.org/abs/1307.0414

---

Dúvidas ou problemas? Abra uma issue no repositório!
