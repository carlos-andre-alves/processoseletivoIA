# Projeto 2 — Classificação CIFAR-10

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar imagens coloridas** em 10 categorias de objetos e animais (avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio, caminhão), e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

Este projeto tem uma diferença importante em relação a uma classificação de dígitos: as imagens são **coloridas (RGB)** e visualmente mais complexas, o que torna a tarefa de classificação genuinamente mais difícil — por isso **data augmentation** é um requisito obrigatório aqui, não opcional.

## 🎯 Conjunto de Dados

Dataset **CIFAR-10**, disponível diretamente via `tf.keras.datasets.cifar10` (não é necessário download manual). 60.000 imagens 32x32 coloridas, 10 classes.

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset CIFAR-10 via TensorFlow
- Split explícito treino/validação
- **Data augmentation** aplicada ao conjunto de treino, usando camadas do Keras
  (ex: `RandomFlip("horizontal")`, `RandomRotation`, `RandomZoom`) incorporadas ao
  modelo ou ao pipeline de treino
- Construção de uma CNN com 3-4 blocos convolucionais (`Conv2D` + `BatchNormalization`
  + `MaxPooling2D`) seguida de `Dropout`
- Treinamento com **early stopping** baseado na perda de validação
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

> 💡 Se você aplicar a augmentation de outra forma (ex: pré-processamento manual em
> `tf.data`), tudo bem — apenas descreva isso claramente no relatório, já que a
> correção automática busca primeiro por camadas de augmentation no próprio modelo.

> 💡 CIFAR-10 é mais difícil que MNIST/Fashion-MNIST para uma CNN simples treinada
> rapidamente em CPU — não se preocupe se a acurácia ficar bem abaixo de 90%. O
> importante é o pipeline completo funcionar corretamente.

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/2-classificacao-cifar/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 32x32, 3 canais (RGB), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 25-30, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Generalização** — uso adequado de data augmentation
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo: Carlos André Alves Torres Filho**

### 1️⃣ Resumo da Arquitetura do Modelo

A arquitetura implementada em train_model.py é uma Rede Neural Convolucional (CNN) sequencial desenvolvida do zero. A estratégia de Data Augmentation foi incorporada diretamente como a primeira camada do modelo (Sequential com RandomFlip, RandomRotation e RandomZoom), garantindo que a transformação ocorra dinamicamente durante o treinamento. A extração de características consiste em 3 blocos sucessivos contendo Conv2D (com 32, 64 e 128 filtros), seguidos por BatchNormalization para aceleração e estabilidade, e MaxPooling2D. A camada de classificação utiliza Flatten, uma densa de 128 neurônios, seguida por um Dropout(0.5) para mitigação de overfitting, finalizando com uma saída Softmax para as 10 classes.

### 2️⃣ Bibliotecas Utilizadas

Python: 3.11 (Ambiente nativo do GitHub Codespaces).

TensorFlow / Keras: 2.15.0 (Motor principal responsável pela construção da CNN, aplicação do Data Augmentation, treinamento e conversão via TFLiteConverter).

Scikit-Learn: 1.4.2 (Utilizada especificamente pela função train_test_split para garantir a divisão correta e determinística das bases de treino e validação).

NumPy: 1.26.4 (Essencial para a manipulação matemática de arrays e expansão de dimensões np.expand_dims durante o pipeline de inferência).

### 3️⃣ Técnica de Otimização do Modelo

No script optimize_model.py, a otimização foi realizada utilizando o TFLiteConverter com a flag converter.optimizations = [tf.lite.Optimize.DEFAULT]. Essa técnica aplica a Quantização de Faixa Dinâmica (Dynamic Range Quantization). Ela converte os pesos da rede neural de ponto flutuante de 32 bits (Float32) para inteiros de 8 bits (Int8), reduzindo drasticamente o consumo de memória e o tempo de inferência em dispositivos Edge, mantendo as ativações em ponto flutuante durante a execução para preservar a acurácia.

### 4️⃣ Resultados Obtidos

Acurácia de Validação: 0.7611 (76.11%)

Tamanho do modelo original (`model.h5`): 4.16 MB
Tamanho do modelo otimizado (`model.tflite`): 0.36 MB
Redução de tamanho: 91.4%

### 5️⃣ Comentários Adicionais (Opcional)

Como o treinamento foi restrito ao uso de CPU no ambiente virtual, a principal decisão técnica envolveu o gerenciamento de recursos. O batch_size foi fixado em 64 para evitar gargalos de alocação de RAM (indicados nos logs iniciais). O uso do EarlyStopping provou-se altamente eficaz: no treinamento final, o modelo convergiu excelentemente, e o callback interrompeu o processo e restaurou os pesos ideais da Época 22 (val_loss: 0.7171), garantindo uma generalização robusta e evitando desperdício computacional. A quantização final obteve uma redução de mais de 90% no peso do modelo, provando a viabilidade do deploy em microcontroladores.

### 6️⃣ Exemplo de Inferência

Rodando inferência em 5 amostras usando model.tflite:

Amostra 1: predito=cat | real=cat
Amostra 2: predito=ship | real=ship
Amostra 3: predito=ship | real=ship
Amostra 4: predito=airplane | real=airplane
Amostra 5: predito=frog | real=frog


modelo otimizado (TFLite) conseguiu classificar corretamente todas as 5 amostras aleatórias testadas da base de teste. Esse resultado prático valida o sucesso da quantização (redução para 360 KB), que não degradou de forma perceptível a capacidade de generalização visual da rede, mantendo a consistência com a acurácia de ~76% obtida na etapa de validação.
