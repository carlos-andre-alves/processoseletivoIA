import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Projeto 2 — Classificação CIFAR-10
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset CIFAR-10 via tf.keras.datasets.cifar10
#   2. Normalizar as imagens para [0, 1] (shape (32, 32, 3))
#   3. Separar um conjunto de validação
#   4. Incluir data augmentation (ex: layers.RandomFlip, RandomRotation, RandomZoom)
#      aplicada ao conjunto de treino
#   5. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   6. Treinar com EarlyStopping monitorando a perda de validação
#   7. Exibir a acurácia de validação final no terminal
#   8. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

print("1. Carregando o dataset CIFAR-10...")
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()

print("2. Normalizando imagens para [0, 1]...")
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print("3. Separando conjunto de validação...")
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=0.2, random_state=42
)

print("4. Configurando Data Augmentation...")
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")

print("5. Construindo a CNN (3 blocos)...")
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    data_augmentation,

    # Bloco 1
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Bloco 2
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Bloco 3
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Classificador Final com Dropout
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("6. Configurando Early Stopping...")
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True,
    verbose=1
)

print("\nIniciando o treinamento na CPU...")
history = model.fit(
    x_train, y_train,
    epochs=1, 
    batch_size=64, 
    validation_data=(x_val, y_val),
    callbacks=[early_stopping]
)

print("\n7. Calculando acurácia de validação final...")
val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
print(f"\n=========================================")
print(f"--- Acurácia de Validação Final: {val_acc:.4f} ---")
print(f"=========================================\n")

print("8. Salvando o modelo treinado...")
model.save('model.h5')
print("Sucesso: Artefato 'model.h5' gerado localmente.")