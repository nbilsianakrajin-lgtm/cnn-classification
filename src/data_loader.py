import os
# Atur KERAS_HOME SEBELUM mengimpor tensorflow
os.environ['KERAS_HOME'] = os.path.abspath('../data')

import tensorflow as tf

def load_and_preprocess_data():
    print("Memuat data Fashion-MNIST...")
    # Sekarang tidak perlu argumen cache_dir
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test  = x_test.reshape(-1, 28, 28, 1)
    
    return (x_train, y_train), (x_test, y_test), class_names