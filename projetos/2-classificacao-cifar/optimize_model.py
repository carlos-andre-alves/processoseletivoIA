import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 2 — Otimização do Modelo (CIFAR-10)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

print("1. Carregando o modelo treinado (model.h5)...")
model = tf.keras.models.load_model('model.h5')

print("2. Configurando o conversor para TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

print("3. Aplicando técnica de otimização (Dynamic Range Quantization)...")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("4. Convertendo o modelo (isso pode levar alguns segundos)...")
tflite_model = converter.convert()

print("5. Salvando o resultado como 'model.tflite'...")
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("\n=========================================")
print(" Conversão e Otimização concluídas!")
print("=========================================\n")

tamanho_h5 = os.path.getsize('model.h5') / (1024 * 1024)
tamanho_tflite = os.path.getsize('model.tflite') / (1024 * 1024)

print(" DADOS FINAIS:")
print(f"Tamanho Original (model.h5):     {tamanho_h5:.2f} MB")
print(f"Tamanho Otimizado (model.tflite): {tamanho_tflite:.2f} MB")
print(f"Redução de tamanho:              {(1 - (tamanho_tflite / tamanho_h5)) * 100:.1f}%")