# Changelog

Todas as mudanças notáveis neste projeto serão documentadas aqui.

## [0.1.0] - 2025-01-14

### NeuroGuide - Estrutura Inicial do Projeto

**Adicionado:**
- Estrutura modular do projeto com separação de responsabilidades
- Sistema de configuração centralizado (config.yaml)
- Módulo de reconhecimento de emoções faciais
  - Arquitetura CNN customizada (4 blocos convolucionais)
  - Pipeline de treinamento completo
  - Data augmentation configurável
  - Suporte ao dataset FER-2013
- Interface de detecção em tempo real via webcam
  - Detecção de rostos usando Haar Cascade
  - Visualização de emoções com cores personalizadas
  - Barra de probabilidades para cada emoção
- Notebooks Jupyter para análise
  - Exploração do dataset FER-2013
  - Análise de resultados e performance
- Documentação completa
  - README principal
  - Guia de setup detalhado
  - Guia rápido de uso
  - Recursos e referências
  - Roadmap de próximos passos
- Script de teste de ambiente (test_setup.py)
- Estrutura preparada para módulo futuro de sobrecarga sensorial

**Tecnologias:**
- Python 3.9+
- TensorFlow 2.13+
- Keras 2.13+
- OpenCV 4.8+
- NumPy, Pandas, Matplotlib, Seaborn

**Arquitetura do Modelo:**
- Input: 48x48x1 (grayscale)
- 4 blocos convolucionais com BatchNormalization
- 2 camadas densas com Dropout
- Output: 7 classes (emoções)
- Total de parâmetros: ~1M

**Features:**
- Data augmentation (rotação, shift, flip, zoom)
- Early stopping
- Learning rate scheduling
- TensorBoard logging
- Model checkpointing
- Detecção em tempo real com visualização

---

## Próximas Versões (Planejadas)

### [0.2.0] - Transfer Learning
- Implementação de MobileNetV2
- Implementação de ResNet50
- Comparação de arquiteturas
- Ensemble de modelos

### [0.3.0] - Detector de Sobrecarga Sensorial
- Análise de luminosidade
- Detecção de movimento (Optical Flow)
- Análise de padrões visuais
- Score agregado de sobrecarga

### [0.4.0] - Sistema Integrado
- Dashboard unificado (emoções + ambiente)
- Interface web com Streamlit
- Persistência de dados
- Relatórios de uso

### [1.0.0] - Release Público
- Deploy em edge devices
- App mobile
- Documentação completa
- Testes com usuários
- Publicação open-source

---

## Formato do Changelog

Este changelog segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/)
e este projeto adere ao [Semantic Versioning](https://semver.org/lang/pt-BR/).

**Tipos de mudanças:**
- `Adicionado` para novas funcionalidades
- `Modificado` para mudanças em funcionalidades existentes
- `Depreciado` para funcionalidades que serão removidas
- `Removido` para funcionalidades removidas
- `Corrigido` para correção de bugs
- `Segurança` para vulnerabilidades
