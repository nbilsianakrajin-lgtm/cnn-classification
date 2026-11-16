import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_loader import load_and_preprocess_data

def main():
    # 1. Load Model yang sudah dilatih
    model_path = '../models/model_cnn_fashion.keras'
    if not os.path.exists(model_path):
        print("Error: Model belum ada. Silakan jalankan train.py terlebih dahulu!")
        return
    
    print("Memuat model...")
    model = tf.keras.models.load_model(model_path)

    # 2. Load Data Test
    _, (x_test, y_test), class_names = load_and_preprocess_data()

    # 3. Pilih 5 gambar secara acak
    indices = np.random.choice(len(x_test), 5, replace=False)
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        img = x_test[idx]
        
        # Prediksi
        pred_prob = model.predict(img.reshape(1, 28, 28, 1), verbose=0)[0]
        pred_class = np.argmax(pred_prob)
        confidence = pred_prob[pred_class] * 100
        
        true_class = y_test[idx]
        
        # Visualisasi
        plt.subplot(1, 5, i+1)
        plt.imshow(img.squeeze(), cmap='gray')
        
        # Warna hijau jika benar, merah jika salah
        color = 'green' if pred_class == true_class else 'red'
        plt.title(f"Pred: {class_names[pred_class]}\n({confidence:.1f}%)\nAktual: {class_names[true_class]}", 
                  color=color, fontsize=9)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()