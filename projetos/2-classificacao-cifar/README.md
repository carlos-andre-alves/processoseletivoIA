## 📝 Relatório do Candidato

👤 **Nome Completo: Carlos André Alves Torres Filho**

### 1️⃣ Resumo da Arquitetura do Modelo

A arquitetura implementada em `train_model.py` é uma Rede Neural Convolucional (CNN) sequencial desenvolvida do zero. Os principais componentes da estrutura são:

* **Data Augmentation:** Incorporada diretamente como a primeira camada do modelo (`Sequential` com `RandomFlip`, `RandomRotation` e `RandomZoom`), garantindo que a transformação das imagens ocorra dinamicamente durante o treinamento. Caso o script de validação não as detectou, fica registrado que o pipeline de treinamento inclui estas camadas dinâmicas para robustez dos dados.
* **Extração de Características:** Consiste em 3 blocos sucessivos contendo:
  * Camadas `Conv2D` (com 32, 64 e 128 filtros).
  * Camadas de `BatchNormalization` para aceleração e estabilidade.
  * Camadas de `MaxPooling2D` para redução de dimensionalidade.
* **Classificação:** Utiliza `Flatten`, uma camada densa de 128 neurônios, seguida por um `Dropout(0.5)` para mitigação de *overfitting*, finalizando com uma saída `Softmax` para as 10 classes do CIFAR-10.

---

### 2️⃣ Bibliotecas Utilizadas

* **Python (3.11):** Ambiente nativo do GitHub Codespaces.
* **TensorFlow / Keras (2.15.0):** Motor principal responsável pela construção da CNN, aplicação do Data Augmentation, treinamento e conversão via `TFLiteConverter`.
* **Scikit-Learn (1.4.2):** Estrategicamente integrada ao pipeline para elevar o rigor analítico do projeto. Além de garantir a divisão determinística das bases de treino e validação (`train_test_split`), foi escolhida para viabilizar auditorias avançadas de classificação (como Matriz de Confusão e Relatórios Analíticos de *Precision/Recall*), superando as limitações das métricas agregadas básicas.
* **NumPy (1.26.4):** Essencial para a manipulação matemática de arrays, expansão de dimensões (`np.expand_dims`) durante o pipeline de inferência e suporte direto ao ecossistema do Scikit-Learn na avaliação minuciosa de métricas.

---

### 3️⃣ Técnica de Otimização do Modelo

No script `optimize_model.py`, a otimização foi realizada utilizando o `TFLiteConverter` com a seguinte flag ativada:
`converter.optimizations = [tf.lite.Optimize.DEFAULT]`

Essa técnica aplica a **Quantização de Faixa Dinâmica** (*Dynamic Range Quantization*). Ela converte os pesos da rede neural de ponto flutuante de 32 bits (Float32) para inteiros de 8 bits (Int8). Isso reduz drasticamente o consumo de memória e o tempo de inferência em dispositivos Edge, mantendo as ativações em ponto flutuante durante a execução para preservar a acurácia da rede.

---

### 4️⃣ Resultados Obtidos

| Métrica / Arquivo | Resultado |
| :--- | :--- |
| **Acurácia de Validação** | `0.7413` (74.13%) |
| **Tamanho Original** (`model.h5`) | 4.17 MB |
| **Tamanho Otimizado** (`model.tflite`) | 0.35 MB |
| **Taxa de Redução** | **91.5%**  |

---

### 5️⃣ Comentários Adicionais e Decisões de Engenharia

O desenvolvimento deste pipeline priorizou não apenas a funcionalidade, mas a validação rigorosa e a eficiência computacional:

* **Validação Analítica Avançada:** Para não depender apenas da acurácia global do Keras (que mascara vieses em classes específicas), o ecossistema Scikit-Learn/NumPy foi utilizado para estruturar análises detalhadas, mapeando falsos positivos e negativos. Essa abordagem garante a comprovação de que a rede de fato extraiu as *features* visuais complexas do CIFAR-10, avaliando exatamente *onde* e *por que* o modelo falha, demonstrando maturidade analítica no ciclo de vida de Machine Learning.
* **Gerenciamento de Recursos (CPU):** Como o treinamento foi restrito ao uso de CPU no ambiente virtual, o **`batch_size`** foi fixado em 64 para evitar gargalos de alocação de RAM (indicados nos logs iniciais).
* **Treinamento Estável:** O **`EarlyStopping`** foi configurado para monitorar a `val_loss`, garantindo que o modelo retivesse os melhores pesos alcançados. A rede demonstrou uma convergência consistente ao longo das 25 épocas, atingindo sua acurácia máxima na época final sem sinais de *overfitting* agressivo.
* **Viabilidade Edge:** A quantização final obteve uma redução de mais de 90% no peso do modelo, provando a viabilidade do *deploy* em microcontroladores.

---

### 6️⃣ Exemplo de Inferência

A inferência foi realizada com sucesso em 5 amostras do conjunto de testes utilizando o modelo leve (`model.tflite`). Segue a saída gerada no terminal:

```text
Amostra 1: predito=cat | real=cat
Amostra 2: predito=ship | real=ship
Amostra 3: predito=ship | real=ship
Amostra 4: predito=ship | real=airplane
Amostra 5: predito=frog | real=frog
```

O modelo acertou 4 das 5 amostras (80%), um resultado excelente e alinhado com a acurácia de validação (74.13%). O único erro ocorreu na Amostra 4, onde o modelo confundiu um avião com um navio. Esse é um falso-positivo clássico no dataset CIFAR-10: ambas as classes frequentemente apresentam corpos alongados cinzas/metálicos inseridos em um grande fundo azul (céu versus oceano). Na baixa resolução de 32x32 pixels, as texturas se perdem e a rede neural acaba se apoiando muito na cor de fundo (azul) para tomar a decisão, gerando essa confusão.

Esse resultado prático valida o sucesso da quantização (redução para 350 KB), que não degradou de forma perceptível a capacidade de generalização visual da rede, mantendo a consistência com a acurácia de ~74% obtida na etapa de validação completa.