import os
import matplotlib.pyplot as plt
from data_loader import load_and_preprocess_data
from model_builder import build_cnn_model

# Pastikan folder models dan outputs tersedia di luar folder src
os.makedirs('../models', exist_ok=True)
os.makedirs('../outputs', exist_ok=True)

def main():
    # 1. Load Data
    (x_train, y_train), (x_test, y_test), class_names = load_and_preprocess_data()

    # 2. Build & Compile Model
    model = build_cnn_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    # 3. Train Model
    print("\nMulai Training Model...")
    history = model.fit(x_train, y_train, epochs=10, batch_size=128, 
                        validation_split=0.1, verbose=1)

    # 4. Evaluasi
    print("\nMengevaluasi Model...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Akurasi Test: {test_acc*100:.2f}%")

    # 5. Simpan Model
    model_path = '../models/model_cnn_fashion.keras'
    model.save(model_path)
    print(f"Model berhasil disimpan di: {model_path}")

    # 6. Simpan Grafik Learning Curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Akurasi Model')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss Model')
    plt.legend()

    plt.suptitle('Learning Curve CNN')
    grafik_path = '../outputs/learning_curve.png'
    plt.savefig(grafik_path)
    print(f"Grafik berhasil disimpan di: {grafik_path}")
    
    # plt.show() # Hapus tanda pagar jika ingin grafik muncul di layar juga

if __name__ == "__main__":
    main()