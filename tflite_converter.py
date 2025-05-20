
import sys
import tensorflow as tf

# H5 dosyasını yükle
h5_path = sys.argv[1]
tflite_path = sys.argv[2]

# Modeli yükle
model = tf.keras.models.load_model(h5_path)

# TFLite dönüştürücüsünü oluştur
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimizasyon seçenekleri
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Dönüşümü gerçekleştir
tflite_model = converter.convert()

# TFLite modelini dosyaya kaydet
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite modeli başarıyla oluşturuldu: {tflite_path}")
print(f"Model boyutu: {len(tflite_model) / 1024:.2f} KB")
