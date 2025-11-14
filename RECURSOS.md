# Recursos e Referências

## Papers Científicos

### Reconhecimento de Emoções
- **FER-2013**: Goodfellow et al. (2013) - "Challenges in Representation Learning: A report on three machine learning contests"
  - https://arxiv.org/abs/1307.0414

- **EmotiNet**: Dehghan et al. (2017) - "Who To Trust: Reliable Integration of Information from Multiple Sources for Emotion Recognition"
  - Paper sobre ensemble de modelos para emoções

- **Facial Expression Recognition**: Li & Deng (2020) - "Deep Facial Expression Recognition: A Survey"
  - Survey completo sobre o estado da arte
  - https://arxiv.org/abs/1804.08348

### Autismo e Reconhecimento de Emoções
- **Autism and Facial Expressions**: Harms et al. (2010) - "Facial Emotion Recognition in Autism Spectrum Disorders"
  - Como pessoas com autismo processam emoções faciais

- **Technology for Autism**: Grynszpan et al. (2014) - "Innovative technology-based interventions for autism spectrum disorders"
  - Revisão de tecnologias assistivas para autismo

- **Emotion Recognition Training**: Golan & Baron-Cohen (2006) - "Systemizing empathy: Teaching adults with Asperger syndrome or high-functioning autism to recognize complex emotions"
  - Efetividade de treino de reconhecimento de emoções

## Datasets

### Emoções Faciais
- **FER-2013**: 35,887 imagens em grayscale 48x48
  - 7 emoções: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
  - https://www.kaggle.com/datasets/msambare/fer2013

- **AffectNet**: 450,000 imagens anotadas
  - 8 emoções + arousal/valence
  - http://mohammadmahoor.com/affectnet/

- **RAF-DB**: Real-world Affective Faces Database
  - 30,000 imagens de emoções em contextos naturais
  - http://www.whdeng.cn/RAF/model1.html

- **CK+**: Extended Cohn-Kanade Dataset
  - Sequências de vídeo de laboratório
  - http://www.consortium.ri.cmu.edu/ckagree/

### Expressões Faciais em Autismo
- **EU-Emotion**: Database específico para autismo
- Datasets clínicos (acesso limitado/ético)

## Arquiteturas de Deep Learning

### Clássicas
- **LeNet**: Primeira CNN bem-sucedida
- **AlexNet**: Revolucionou ImageNet 2012
- **VGG**: Arquitetura simples mas profunda
- **ResNet**: Residual connections para redes muito profundas

### Para Emoções
- **VGGFace**: VGG adaptada para rostos
- **DeepFace**: Facebook's facial recognition
- **EmotiNet**: Especializada em emoções
- **MobileNet**: Leve para edge devices

### Para Aplicações Móveis
- **MobileNetV2**: Eficiente para mobile
- **EfficientNet**: Escalonamento balanceado
- **SqueezeNet**: Muito compacta

## Ferramentas e Bibliotecas

### Deep Learning
- **TensorFlow**: https://www.tensorflow.org/
- **Keras**: https://keras.io/
- **PyTorch**: https://pytorch.org/ (alternativa ao TF)
- **Fast.ai**: https://www.fast.ai/ (high-level sobre PyTorch)

### Computer Vision
- **OpenCV**: https://opencv.org/
- **dlib**: http://dlib.net/ (face detection, landmarks)
- **MediaPipe**: https://google.github.io/mediapipe/ (Google's CV framework)
- **face_recognition**: https://github.com/ageitgey/face_recognition

### Data Augmentation
- **Albumentations**: https://albumentations.ai/
- **imgaug**: https://imgaug.readthedocs.io/

### Visualization
- **TensorBoard**: Visualização de treinamento
- **Grad-CAM**: Visualizar o que a CNN está "vendo"
- **LIME**: Explain predictions
- **SHAP**: SHapley Additive exPlanations

## Cursos Online

### Deep Learning
- **Deep Learning Specialization** (Coursera - Andrew Ng)
  - https://www.coursera.org/specializations/deep-learning

- **Fast.ai Practical Deep Learning**
  - https://course.fast.ai/

- **CS231n: CNNs for Visual Recognition** (Stanford)
  - http://cs231n.stanford.edu/

### TensorFlow/Keras
- **TensorFlow in Practice** (Coursera)
- **Keras Documentation** (excelente para aprender)

### Computer Vision
- **Computer Vision Nanodegree** (Udacity)
- **PyImageSearch** (blog e cursos)
  - https://pyimagesearch.com/

## Blogs e Tutoriais

### Técnicos
- **Towards Data Science**: Medium publication sobre DS/ML
- **Machine Learning Mastery**: Tutoriais práticos
- **Analytics Vidhya**: Guias e competições
- **Papers With Code**: Papers + implementações
  - https://paperswithcode.com/task/facial-expression-recognition

### Specific
- **Face Recognition Guide**: Tutoriais de OpenCV
- **Emotion Detection Tutorial**: PyImageSearch
- **Transfer Learning for Emotions**: TensorFlow tutorials

## Livros

### Deep Learning
- **Deep Learning** (Goodfellow, Bengio, Courville)
  - Gratuito online: https://www.deeplearningbook.org/

- **Hands-On Machine Learning** (Aurélien Géron)
  - Prático, com TensorFlow e Scikit-Learn

### Computer Vision
- **Computer Vision: Algorithms and Applications** (Szeliski)
  - Gratuito: http://szeliski.org/Book/

- **Deep Learning for Computer Vision** (Rajalingappaa Shanmugamani)

### Específicos
- **Python for Computer Vision with OpenCV and Deep Learning** (José Portilla)

## Comunidades e Fóruns

- **Kaggle**: Competições e datasets
  - https://www.kaggle.com/

- **Stack Overflow**: Perguntas técnicas
  - Tag: tensorflow, keras, opencv

- **Reddit**:
  - r/MachineLearning
  - r/deeplearning
  - r/computervision
  - r/autism

- **GitHub**: Repositórios open-source
  - Procure por "emotion recognition"
  - Veja implementações de papers

## Pesquisa sobre Autismo

### Organizações
- **Autism Research Institute**: https://www.autism.org/
- **INSAR**: International Society for Autism Research
- **Autism Speaks**: (controverso, mas tem recursos)

### Tecnologia Assistiva
- **AASPIRE**: Academic Autistic Spectrum Partnership in Research
- **Autism Technology**: https://www.autismtechnology.com/

### Estudos sobre Intervenção
- Papers sobre technology-based interventions
- Virtual reality para treino de habilidades sociais
- Apps e jogos para reconhecimento de emoções

## Ética e Considerações

### Papers sobre Ética em AI
- **Fairness and Bias in AI**: Research papers
- **Privacy in Computer Vision**: Considerações éticas
- **AI for Social Good**: Projetos com impacto social

### Guidelines
- **Web Accessibility for Neurodiversity**
- **Design for Autism Spectrum**: Princípios de UX/UI
- **Ethical AI Development**: Best practices

## Conferências

- **CVPR**: Computer Vision and Pattern Recognition
- **ICCV**: International Conference on Computer Vision
- **NeurIPS**: Neural Information Processing Systems
- **ICML**: International Conference on Machine Learning
- **IMFAR**: International Meeting for Autism Research

## Repositórios GitHub Úteis

```
# Emotion Recognition
https://github.com/topics/emotion-recognition
https://github.com/omar178/Emotion-recognition

# Face Detection
https://github.com/topics/face-detection

# Autism Technology
https://github.com/topics/autism

# TensorFlow Examples
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples
```

## Datasets Complementares (Futuro)

Para o módulo de sobrecarga sensorial:
- **Flicker Dataset**: Para análise de variação de luz
- **Motion Dataset**: Para optical flow
- **Texture Dataset**: Para análise de padrões
- **Environmental Videos**: Diferentes ambientes

## Ferramentas de Desenvolvimento

- **VSCode**: IDE recomendado
- **Jupyter Lab**: Para notebooks
- **Git/GitHub**: Controle de versão
- **Docker**: Para containerização (futuro)
- **MLflow**: Experiment tracking (futuro)

---

## Como Usar Este Documento

1. **Para estudar teoria**: Veja Papers Científicos e Livros
2. **Para aprender prática**: Cursos Online e Tutoriais
3. **Para implementar**: Bibliotecas e GitHub
4. **Para contextualizar**: Pesquisa sobre Autismo
5. **Para expandir**: Datasets e Ferramentas

Boa pesquisa! 📚
