'''
Ini adalah script untuk tahapan Feature Engineering

Di pembelajaran kali ini kita akan memproses hal yang berkaitan dengan:
1. Nilai kosong (NA atau nan)
2. Variabel yang berhubungan dengan waktu (Tahun)
3. Variabel yang tidak berdistribusi normal
4. Menghilangkan variabel jarang untuk tipe kategori
5. Merubah format data string ke numerik untuk tipe kategori
6. Menyamakan rentang data nilai untuk beberapa variabel
'''

# Mengimpor library yang dibutuhkan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Menghilangkan warning
import warnings
warnings.simplefilter(action='ignore')

# Link download harga_rumah.csv : https://cutt.ly/ul3frWb
# Mengimpor dataset
dataku = pd.read_csv('harga_rumah.csv')

# Membagi menjadi training dan test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataku, dataku['SalePrice'], 
                                                    test_size = 0.1,
                                                    random_state = 10)

# Mengatasi data kosong untuk tipe data kategori
kolom_na_kategori = [kolom for kolom in dataku.columns if 
                     dataku[kolom].isnull().sum() > 0 and
                     dataku[kolom].dtypes == 'O']

# Menghitung persentase dari data kategori yang kosong
dataku[kolom_na_kategori].isnull().mean()*100

# Mengganti data kosong untuk tipe kategori dengan kategori 'Kosong'
X_train[kolom_na_kategori] = X_train[kolom_na_kategori].fillna('Kosong')
X_test[kolom_na_kategori] = X_test[kolom_na_kategori].fillna('Kosong')

# Memastikan tidak ada data kosong untuk tipe kategori di traning set
X_train[kolom_na_kategori].isnull().sum()

# Memastikan tidak ada data kosong untuk tipe kategori di test set
X_test[kolom_na_kategori].isnull().sum()

# Mendeteksi data kosong untuk tipe data numerik
kolom_na_numerik = [kolom for kolom in dataku.columns if 
                     dataku[kolom].isnull().sum() > 0 and
                     dataku[kolom].dtypes != 'O']

# Menghitung persentase dari data kategori yang kosong
dataku[kolom_na_numerik].isnull().mean()*100

# Mengganti data kosong numerik dengan modus/mode (nilai terbanyak)
for kolom in kolom_na_numerik:
    # menghitung modus
    hitung_modus = X_train[kolom].mode()[0]
    # Menambahkan kolom baru mendeteksi data kosong per barisnya
    X_train[kolom+'_na'] = np.where(X_train[kolom].isnull(), 1, 0)
    X_test[kolom+'_na'] = np.where(X_test[kolom].isnull(), 1, 0)
    # Memasukkan modus ke baris yang kosong
    X_train[kolom] = X_train[kolom].fillna(hitung_modus)
    X_test[kolom] = X_test[kolom].fillna(hitung_modus)

# Memastikan tidak ada data kosong numerik di training set
X_train[kolom_na_numerik].isnull().sum()

# Memastikan tidak ada data kosong numerik di test set
X_test[kolom_na_numerik].isnull().sum()

# Menghitung tahun berlalu untuk 4 variabel waktu terhadap 'YrSold'
def tahun_berlalu(data, col):
    data[col] = data['YrSold'] - data[col]
    return data

# Merubah setiap kolom 'Tahun' menjadi selisihnya terhadap 'YrSold'
for kolom in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    X_train = tahun_berlalu(X_train, kolom)
    X_test = tahun_berlalu(X_test, kolom)

# Merubah variabel yang tidak normal menjadi mendekati normal (log-transform)
for kolom in ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']:
    X_train[kolom] = np.log(X_train[kolom])
    X_test[kolom] = np.log(X_test[kolom])
# Memastikan kolom yang sudah log-transformed tidak memiliki data kosong (NA)
kolom_log = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
X_train[['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']].isnull().sum()
X_test[['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']].isnull().sum()

# Mengatasi variabel jarang
# Data kategori
kolom_kategori = [kolom for kolom in X_train.columns if X_train[kolom].dtypes == 'O']
# Membuat fungsi untuk variabel jarang
def analisis_var_jarang(data, col, persentase):
    data = data.copy()
    # Menentukan persentase setiap kategori
    isi = data.groupby(col)['SalePrice'].count() / len(data)
    return isi[isi < persentase].index

for kolom in kolom_kategori:
    # Mencari kategori yang jarang
    var_jarang = analisis_var_jarang(X_train, kolom, 0.01)
    # Mengganti nama item baris dnegan kata 'Jarang'
    X_train[kolom] = np.where(X_train[kolom].isin(var_jarang), 'Jarang', X_train[kolom])
    X_test[kolom] = np.where(X_test[kolom].isin(var_jarang), 'Jarang', X_test[kolom])

# Mengganti (encode) tipe data kategori menjadi tipe data numerik
def encode_kategori(train, test, kolom, target):
    # Mengurutkan kategori mulai dari kecil ke besar berdasarkan nilai rataan SalePrice
    data_urut = train.groupby([kolom])[target].mean().sort_values().index
    # Membuat dictionary dan enumerate untuk emndapatkan indeks dan urutannya
    data_ordinal = {k: i for i, k in enumerate(data_urut, start=0)}
    # Menggunakan dictionary di atas untuk mengganti data kategori menjadi integer
    train[kolom] = train[kolom].map(data_ordinal)
    test[kolom] = test[kolom].map(data_ordinal)
    print(kolom, data_ordinal)

# For loop untuk merubah data kategori menjadi integer
for kolom in kolom_kategori:
    encode_kategori(X_train, X_test, kolom, 'SalePrice')

# Mengecek NA untuk training dan test set
X_train.isnull().sum()
train_kosong = [kolom for kolom in X_train.columns if X_train[kolom].isnull().sum() > 0]
X_test.isnull().sum()
test_kosong = [kolom for kolom in X_test.columns if X_test[kolom].isnull().sum() > 0]

# Mengisi kolom numerik dengan modus di test set
for kolom in test_kosong:
    # Menghitung modus
    hitung_modus = X_test[kolom].mode()[0] # [0] adalah untuk mengeluarkan indeksnya
    # Menambahkan kolom baru mendeteksi data kosong per barisnya
    X_test[kolom+'_na'] = np.where(X_test[kolom].isnull(), 1, 0)
    # Memasukkan modus ke baris yang kosong
    X_test[kolom] = X_test[kolom].fillna(hitung_modus)

# Memastikan tidka ada data kosong di X_test
test_kosong2 = [kolom for kolom in X_test.columns if X_test[kolom].isnull().sum() > 0]

# Melihat plot untuk semua kolom kategori
def analisis_kategori(data, col):
    data = data.copy()
    data.groupby(col)['SalePrice'].median().plot.bar()
    plt.title(col)
    plt.ylabel('SalePrice')
    plt.tight_layout()
    plt.show()

batas = len(kolom_kategori)
i = 1
for kolom in kolom_kategori:
    i += 1
    analisis_kategori(X_train, kolom)
    if i <= batas: plt.figure()

# Membuat variabel untuk training MinMaxScaler, tidak termasuk variabel yang tidak relevan
kolom_training = [kolom for kolom in X_train.columns if kolom not in 
                  ['Id', 'SalePrice', 'LotFrontage_na',
                   'MasVnrArea_na', 'GarageYrBlt_na']]

# Proses feature scaling dengan MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
X_train[kolom_training] = sc_X.fit_transform(X_train[kolom_training])
X_test[kolom_training] = sc_X.transform(X_test[kolom_training])

# Menyimpan scaler sc_X
import joblib 
joblib.dump(sc_X, 'minmax_scaler.joblib')

# Menyimpan X_train dan X_test dalam file csv
X_train.to_csv('xtrain.csv', index=False)
X_test.to_csv('xtest.csv', index=False)
                                                