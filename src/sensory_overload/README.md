# Módulo de Sobrecarga Sensorial (Em Desenvolvimento)

## Visão Geral

Este módulo tem como objetivo detectar condições ambientais que podem causar sobrecarga sensorial em pessoas com autismo, incluindo:

- **Luminosidade excessiva**: Ambientes muito claros ou com brilho intenso
- **Movimento intenso**: Muito movimento ou mudanças rápidas na cena
- **Padrões visuais complexos**: Padrões repetitivos ou texturas que podem ser desconfortáveis
- **Contraste alto**: Diferenças bruscas de luz e sombra

## Arquitetura Planejada

```
sensory_overload/
├── __init__.py
├── brightness_analyzer.py      # Análise de luminosidade
├── motion_detector.py          # Detecção de movimento
├── pattern_analyzer.py         # Análise de padrões visuais
├── overload_scorer.py          # Score agregado de sobrecarga
└── alerts.py                   # Sistema de alertas
```

## Algoritmos a Implementar

### 1. Análise de Luminosidade
- Cálculo de brilho médio do frame
- Detecção de áreas muito claras (hotspots)
- Análise de contraste
- Detecção de flicker (variação rápida de luz)

### 2. Detecção de Movimento
- Optical Flow para quantificar movimento
- Frame differencing para detectar mudanças
- Cálculo de velocidade de movimento
- Detecção de movimento caótico vs. suave

### 3. Análise de Padrões
- Detecção de bordas (alta frequência espacial)
- Análise de textura usando GLCM
- Identificação de padrões repetitivos
- Cálculo de complexidade visual

### 4. Score de Sobrecarga
Combinar todas as métricas em um único score:
```
Sobrecarga = w1*Luminosidade + w2*Movimento + w3*Padrões + w4*Contraste
```

## Exemplo de Uso Futuro

```python
from src.sensory_overload import SensoryOverloadDetector

detector = SensoryOverloadDetector(
    brightness_threshold=200,
    motion_threshold=0.7,
    pattern_threshold=0.8
)

# Analisar frame da webcam
overload_score, details = detector.analyze_frame(frame)

if overload_score > 0.7:
    print("⚠️ Alerta: Alto risco de sobrecarga sensorial!")
    print(f"   Luminosidade: {details['brightness']}")
    print(f"   Movimento: {details['motion']}")
    print(f"   Padrões: {details['patterns']}")
```

## Datasets e Referências

- **Estudos sobre sobrecarga sensorial no autismo**
- **Métricas de qualidade de vídeo (VMAF, SSIM)**
- **Pesquisas sobre visual complexity**

## Próximos Passos

1. Implementar análise de luminosidade
2. Implementar detecção de movimento com Optical Flow
3. Implementar análise de padrões
4. Criar sistema de scoring e alertas
5. Integrar com o módulo de reconhecimento de emoções
6. Criar interface combinada

## Considerações

- Os thresholds devem ser personalizáveis por usuário
- Diferentes pessoas com autismo têm sensibilidades diferentes
- Sistema deve aprender com feedback do usuário
- Alertas devem ser sutis e não-invasivos
