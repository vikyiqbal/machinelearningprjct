# Mengimpor library yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Library untuk membuat model
from sklearn.linear_model import Lasso

# Library untuk mengevaluasi model
from sklearn.metrics import mean_squared_error, r2_score

# Untuk membuat model yg kita buat di sini dan yg di deploy sama hasilnya
import joblib

# Mengimpor dataset
X_train = pd.read_csv('xtrain.csv')
X_test = pd.read_csv('xtest.csv')

# Variabel dependen
y_train = X_train['SalePrice']
y_test = X_test['SalePrice']

# Mengimpor fitur terpilih
fitur = pd.read_csv('fitur_pilihan.csv')
fitur = fitur['0'].tolist()  # merubah kolom menjadi list
fitur = fitur + ['LotFrontage'] # menambahkan 'LotFrontage' karena nanti ada teknik yg ingin saya tunjukkan saat deployment

# Mengurangi training dan test set sesuai list features
X_train = X_train[fitur]
X_test = X_test[fitur]

# Menyiapkan model
model_linear = Lasso(alpha=0.005, random_state=10)

# Training model
model_linear.fit(X_train, y_train)

# Kita jalankan joblib untuk proses selanjutnya
joblib.dump(model_linear, 'regresi_lasso.pkl')

# Memprediksi model
pred_train = model_linear.predict(X_train)
pred_test = model_linear.predict(X_test)

# Menentukan mse (mean squared error) dan rmse (root mean squared error)
# np.exp menghitung nilai e^x, karena skala sebelumnya adalah dalam logaritmik
print('train mse: {:.2f}'.format(mean_squared_error(np.exp(y_train), np.exp(pred_train))))
print('train rmse: {:.2f}'.format(mean_squared_error(np.exp(y_train), np.exp(pred_train), squared=False)))
print('train r2: {:.2f}'.format(r2_score(np.exp(y_train), np.exp(pred_train))))
print()
print('test mse: {:.2f}'.format(mean_squared_error(np.exp(y_test), np.exp(pred_test))))
print('test rmse: {:.2f}'.format(mean_squared_error(np.exp(y_test), np.exp(pred_test), squared=False)))
print('test r2: {:.2f}'.format(r2_score(np.exp(y_test), np.exp(pred_test))))
print()
print('Rataan Harga Rumah Test Set: ', int(np.exp((y_test).mean())))
print('Rataan Harga Rumah Prediksi: ', int(np.exp((pred_test).mean())))

# Evaluasi hasil prediksi modle kita dengan data sesungguhnya
plt.scatter(y_test, pred_test)
plt.xlabel('Harga Jual Rumah Sesungguhnya')
plt.ylabel('Prediksi Harga Jual Rumah')
plt.title('Evaluasi Hasil Prediksi')
plt.tight_layout()

# Evaluasi distribusi dari nilai error model kita
errors = y_test - pred_test
errors.hist()

# Melihat tingkat utilitas setiap fitur
utilitas = pd.Series(np.abs(model_linear.coef_.ravel()))
utilitas.index = fitur
utilitas.sort_values(inplace = True, ascending=False)
utilitas.plot.bar()
plt.ylabel('Koefisien Lasso')
plt.xlabel('Tingkat Utilitas Fitur')
plt.tight_layout()