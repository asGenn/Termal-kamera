"""
BASINÇ YARASI RİSK SINIFLANDIRMA SİSTEMİ - OPTİMİZE EDİLMİŞ SÜRÜM
Termal görüntülerden basınç yarası risk seviyelerini sınıflandıran gelişmiş CNN modeli

Ana İyileştirmeler:
- Geliştirilmiş model mimarisi
- Optimize edilmiş hiperparametreler
- Termal görüntülere özel veri artırma
- Detaylı hata analizi ve görselleştirme
- Kararlı eğitim için regularizasyon teknikleri
"""

# -------------------- KÜTÜPHANELER --------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# -------------------- YAPILANDIRMA --------------------
IMG_SIZE = (224, 224)        # Model giriş boyutu
BATCH_SIZE = 32              # Veri işleme boyutu
INITIAL_EPOCHS = 150         # Maksimum eğitim iterasyonu
LEARNING_RATE = 0.0001       # Başlangıç öğrenme oranı
MIN_DELTA = 0.00001          # Erken durdurma hassasiyeti
PATIENCE = 30                # Erken durdurma sabrı

# -------------------- VERİ YÜKLEME --------------------


def load_and_preprocess_data(data_dir, classes):
    """
    Termal görüntüleri yükler ve normalizasyon uygular
    
    Args:
        data_dir (str): Veri klasörü yolu
        classes (list): Sınıf etiketleri
        
    Returns:
        tuple: (images, labels, filenames)
    """
    images = []
    labels = []
    filenames = []

    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        print(f"{class_name} yükleniyor...")

        for img_file in os.listdir(class_path):
            try:
                # Görüntüyü gri tonlamalı olarak yükle
                img_path = os.path.join(class_path, img_file)
                img = tf.keras.preprocessing.image.load_img(
                    img_path,
                    color_mode='grayscale',
                    target_size=IMG_SIZE
                )

                # Numpy array'e çevir ve normalizasyon yap
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = img_array / 255.0  # [0-1] aralığına normalizasyon

                # Kanal boyutunu ekle (224,224,1)
                img_array = tf.expand_dims(img_array, axis=-1)

                images.append(img_array)
                labels.append(class_idx)
                filenames.append(img_file)

            except Exception as e:
                print(f"Hata: {img_file} yüklenemedi - {str(e)}")

    return np.array(images), np.array(labels), filenames

# -------------------- MODEL MİMARİSİ --------------------


def create_advanced_model(input_shape, num_classes):
    """
    Optimize edilmiş CNN modelini oluşturur
    
    Mimari Özellikleri:
    - 3 Konvolüsyon Bloğu
    - Global Average Pooling
    - Gelişmiş Regularizasyon
    - Entegre Veri Artırma
    
    Args:
        input_shape (tuple): Giriş görüntü boyutu
        num_classes (int): Sınıf sayısı
        
    Returns:
        tf.keras.Model: Derlenmiş model
    """
    # Giriş katmanı ve veri artırma
    inputs = layers.Input(shape=input_shape)

    # Entegre veri artırma (sadece eğitimde aktif)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomContrast(0.1)(x)  # Termal kontrast varyasyonu
    x = layers.RandomZoom(0.1)(x)

    # 1. Konvolüsyon Bloğu
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)

    # 2. Konvolüsyon Bloğu
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)

    # 3. Konvolüsyon Bloğu
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)  # Parametre optimizasyonu
    x = layers.Dropout(0.5)(x)

    # Çıkış Katmanı
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Modeli Derle
    model = models.Model(inputs, outputs)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0  # Gradyan patlamalarını önle
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# -------------------- CALLBACK'LER --------------------


def get_callbacks(model_name):
    """
    Eğitim sürecini yöneten callback'leri oluşturur
    
    Returns:
        list: Callback listesi
    """
    return [
        # Erken Durdurma: Validation loss'ta iyileşme olmazsa dur
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        ),

        # Öğrenme Oranı Optimizasyonu
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),

        # En iyi modeli kaydet
        callbacks.ModelCheckpoint(
            f'{model_name}_best.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),

        # Eğitim loglarını kaydet
        callbacks.CSVLogger(f'{model_name}_training_log.csv')
    ]

# -------------------- GÖRSELLEŞTİRME --------------------


def visualize_results(history, y_true, y_pred, classes):
    """
    Eğitim sonuçlarını ve performans metriklerini görselleştirir
    
    Args:
        history: Eğitim geçmişi
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        classes: Sınıf isimleri
    """
    # Accuracy/Loss Grafikleri
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim')
    plt.plot(history.history['val_accuracy'], label='Doğrulama')
    plt.title('Model Doğruluğu')
    plt.xlabel('Epok')
    plt.ylabel('Doğruluk')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim')
    plt.plot(history.history['val_loss'], label='Doğrulama')
    plt.title('Model Kaybı')
    plt.xlabel('Epok')
    plt.ylabel('Kayıp')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Karışıklık Matrisi
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Tahmin')
    plt.ylabel('Gerçek')
    plt.title('Sınıflandırma Performansı')
    plt.show()

# -------------------- ANA İŞLEM --------------------


def main():
    # Yapılandırma
    DATA_DIR = "Basınç Yarası Riski"
    CLASSES = ["Goreceli_Risk", "Yuksek_Risk", "Cok_Yuksek_Risk"]
    MODEL_NAME = "advanced_pressure_ulcer_model"

    # 1. Veri Yükleme
    print("\n[1/6] Veri yükleniyor...")
    X, y, filenames = load_and_preprocess_data(DATA_DIR, CLASSES)
    print(f"Toplam örnek sayısı: {len(X)}")
    print(f"Sınıf dağılımı: {dict(zip(CLASSES, np.bincount(y)))}")

    # 2. Veri Bölme
    print("\n[2/6] Veri bölünüyor...")
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test,
        test_size=0.5,
        stratify=y_val_test,
        random_state=42
    )
    print(
        f"Eğitim: {len(X_train)}, Doğrulama: {len(X_val)}, Test: {len(X_test)}")

    # 3. Sınıf Ağırlıkları
    print("\n[3/6] Sınıf ağırlıkları hesaplanıyor...")
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {i: w for i, w in enumerate(class_weights)}
    print("Sınıf Ağırlıkları:", class_weights)

    # 4. Model Oluşturma
    print("\n[4/6] Model inşa ediliyor...")
    model = create_advanced_model((*IMG_SIZE, 1), len(CLASSES))
    model.summary()

    # 5. Model Eğitimi
    print("\n[5/6] Model eğitimi başlatılıyor...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=INITIAL_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=get_callbacks(MODEL_NAME),
        verbose=1
    )

    # 6. Değerlendirme ve Raporlama
    print("\n[6/6] Performans değerlendiriliyor...")

    # Test seti değerlendirme
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Doğruluğu: {test_acc:.4f}")
    print(f"Test Kaybı: {test_loss:.4f}")

    # Tahminler
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Detaylı Rapor
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred_classes, target_names=CLASSES))

    # Sonuçları Kaydet
    results_df = pd.DataFrame({
        'Dosya': filenames[-len(y_test):],
        'Gerçek': [CLASSES[i] for i in y_test],
        'Tahmin': [CLASSES[i] for i in y_pred_classes],
        'Göreceli_Olasılık': y_pred[:, 0],
        'Yüksek_Olasılık': y_pred[:, 1],
        'Çok_Yüksek_Olasılık': y_pred[:, 2]
    })
    results_df.to_csv(f'{MODEL_NAME}_predictions.csv', index=False)
    print("\nTahminler CSV'ye kaydedildi.")

    # Görselleştirme
    visualize_results(history, y_test, y_pred_classes, CLASSES)

    # Modeli Kaydet
    model.save(f'{MODEL_NAME}_final.h5')
    print("\nFinal model kaydedildi.")


if __name__ == "__main__":
    main()
