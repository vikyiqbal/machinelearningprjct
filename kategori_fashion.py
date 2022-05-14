# Mengimpor library yang diperlukan
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import random

from keras.datasets import fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Jenis kategori
kategori = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
            'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
            'Ankle boot']
# 0 = T-shirt/top
# 1 = Trouser
# 2 = Pullover
# 3 = Dress
# 4 =  Coat
# 5 = Sandal
# 6 = Shirt
# 7 = Sneaker
# 8 = Bag
# 9 = Ankle boot
i = random.randint(1,len(X_train))
plt.figure()
plt.imshow(X_train[i,:,:], cmap = 'gray')
plt.title('Kategori = {}'.format(kategori[y_train[i]]))
plt.show()

# Melihat beberapa gambar dalam format Grid
nrow = 10
ncol = 10
fig, axes = plt.subplots(nrow, ncol)
axes = axes.ravel() # ravel digunakan untuk meratakan array (diperlukan jika ncol>1 dan nrow>1)
ntraining = len(X_train)
for i in np.arange(0, nrow*ncol): 
    index = np.random.randint(0, ntraining) # memilih angka random untuk contoh item yg ditampilkan  
    axes[i].imshow(X_train[index, :,:], cmap='gray' ) # ukuran pixel harus 28x28)
    axes[i].set_title(int(y_train[index]), fontsize = 8)
    axes[i].axis('off')
plt.subplots_adjust(hspace=0.4)

# Normalisasi value
X_train = X_train/255 # dibagi 255 agar nilainya antara 0-1
X_test = X_test/255

# Membagi dataset ke train dan validate set
from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.2, random_state = 11)

# Merubah dimensi dataset
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1)) # penulisan X_train dan X_test sama saja, tanda * untuk unpacking)
X_validate = X_validate.reshape(X_validate.shape[0], *(28, 28, 1)) # (height, width , input channels) -> 1 untuk grayscale, dan 3 untuk color

# Mengimpor library Keras
#import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
#from keras.callbacks import TensorBoard

# Memberi nama model kita dengan sebutan classifier
classifier = Sequential()

# Kita coba 32 filter
classifier.add(Conv2D(32,(3, 3), input_shape = (28,28,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25)) # dropout digunakan untuk menghindari overfitting dan meningkatkan generalisasi
classifier.add(Flatten())
classifier.add(Dense(activation="relu", units=32))
classifier.add(Dense(activation="sigmoid", units=10))
classifier.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])
classifier.summary()

# Visualisasi Model
from keras.utils.vis_utils import plot_model
plot_model(classifier, to_file='model_plot.png', show_shapes=True, show_layer_names=False)

'''
# Jalankan hanya jika Tensor Flow/ Keras gagal running (khusus versi GPU)
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # untuk membangun memory di GPU
config.log_device_placement = True  # untuk menyimpan log 
sess = tf.Session(config=config)
set_session(sess)  
'''

# Mulai melakukan training model
run_model = classifier.fit(X_train,
               y_train,
               batch_size = 500,
               nb_epoch = 30,
               verbose = 1, # matikan progress bar dengan set verbose=0
               validation_data = (X_validate, y_validate))

# Melihat parameter apa saja yang disimpan
print(run_model.history.keys())

# Proses plotting accuracy selama proses training
plt.plot(run_model.history['accuracy'])
plt.plot(run_model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()

# Proses plotting loss selama proses training
plt.plot(run_model.history['loss'])
plt.plot(run_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()

# Melakukan evaluasi model
evaluasi = classifier.evaluate(X_test, y_test)
print('Test Accuracy : {:.2f}%'.format(evaluasi[1]*100))

'''
# Menyimpan model
# classifier.save("cnn_fashion.h5")
classifier.save('cnn_fashion.hd5', include_optimizer=True)
print("Disimpan di HDD")

# Load model
from keras.models import load_model
classifier = load_model('cnn_fashion.hd5')
classifier.summary()
'''

# Mulai menguji model ke test set
hasil_prediksi = classifier.predict_classes(X_test)

# Membuat ilustrasi 5x5 hasil prediksi dan kategori aslinya
fig, axes = plt.subplots(5, 5, figsize = (12,12))
axes = axes.ravel() # 
for i in np.arange(0, 5*5):  
    axes[i].imshow(X_test[i].reshape(28,28), cmap='gray')
    axes[i].set_title("Hasil Prediksi = {:0.1f}\n Prediksi Asli = {:0.1f}".format(hasil_prediksi[i], y_test[i]))
    axes[i].axis('off')
plt.subplots_adjust(wspace=0.5)

# Membuat confusion matrix
import pandas as pd
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, hasil_prediksi)
cm_label = pd.DataFrame(cm, columns=np.unique(y_test), index = np.unique(y_test))
cm_label.index.name = 'Actual'
cm_label.columns.name = 'Predicted'
plt.figure(figsize=(14,10))
sns.heatmap(cm_label,annot=True)

# Membuat ringkasan performa model
from sklearn.metrics import classification_report
jumlah_kategori = 10
target_names = ["Class {}".format(i) for i in range(jumlah_kategori)]
print(classification_report(y_test, hasil_prediksi, target_names = target_names))
