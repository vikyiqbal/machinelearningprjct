'''
Ini adalah script untuk tahapan Data Analysis
'''

# Mengimpor library yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Mengimpor dataset
# Sumber asli dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# Link download harga_rumah.csv : https://cutt.ly/ul3frWb
dataku = pd.read_csv('harga_rumah.csv')
'''
Kita akan melakukan pengecekan terhadap:
    1. Data yang hilang (NA)
    2. Kolom-kolom angka (numerik)
    3. Distribusi dari setiap kolom numerik
    4. Outlier
    5. Kolom-kolom kategori dan jumlah kategorinya
    6. Hubungan antara variabel independen dan dependen
'''

# Mendeteksi kolom yang memiliki data NA
kolom_na = [kolom for kolom in dataku.columns if dataku[kolom].isnull().sum() > 0]
''' Membedah satu demi satu
dataku.columns <- melihat kolom apa saja yang ada di variabel dataku
dataku['LotFrontage'].isnull() <- akan memberikan nilai 1 jika ada baris yang berisi NA untuk kolom tersebut
dataku['LotFrontage'].isnull().sum() <- akan menjumlahkan nilai True (1) akibat NA. 
Otomatis jika >0 maka ada NA di kolom tersebut

Penulisan di atas sama dengan di bawah ini
kolom_na = []
for kolom in dataku.columns:
    if dataku[kolom].isnull().sum() > 0:
        kolom_na.append(kolom)
'''

# Menghitung persentase dari kolom-kolom yang berisi NA
dataku[kolom_na].isnull().mean()*100


# Membuat visualisasi untuk setiap kolom berisi NA (kolom_na)
def analisis_data_na(data, col):
    data = data.copy()
    # Mengecek setiap variabel jika ada NA maka 1, jika tidak maka 0
    data[col] = np.where(data[col].isnull(), 1, 0)
    # Sekarang kita memiliki nilai binary (0=ada data, 1=NA)
    # Sekarang bandingkan nilai median (tidak sensitif terhadap outlier) dari 'SalePrice' terhadap 2 nilai binary kolom ini
    # Pengelompokan terhadap nama col, tapi perhitungan agregasi terhadap SalePrice
    data.groupby(col)['SalePrice'].median().plot.bar() # silakan mencoba menggunakan mean
    plt.title(col)
    plt.tight_layout()
    plt.show()


# Membuat for loop untuk plotting kolom_na
batas = len(kolom_na)
i = 1
for kolom in kolom_na:
    i+=1
    analisis_data_na(dataku, kolom)
    if i <= batas: plt.figure() 
# Ternyata harga SalePrice saat nilainya kosong (NA) berbeda dengan SalePrice saat nilainya tidak kosong
# Ini akan jadi pertimbangan untuk feature engineering

# Menganalisis kolom-kolom numerik
kolom_numerik = [kolom for kolom in dataku.columns if dataku[kolom].dtypes != 'O'] #'O' adalah Pandas Object = string

# Visualisasi data kolom_numerik
numerik = dataku[kolom_numerik]
# Kita lihat bahwa kolom 'Id' tidak begitu berguna bagi kita

'''
Variabel yang berhubungan dengan waktu:
    YearBuilt = kapan rumah dibangun
    YearRemodAdd = kapan rumah direnov
    GarageYrBlt = kapan garasinya dibangun
    YrSold = kapan rumahnya dijual
'''

# Mmebuat kolom_tahun yang terdiri dari variabel Tahun (Year atau Yr)
kolom_tahun = [kolom for kolom in kolom_numerik if 'Yr' in kolom or 'Year' in kolom]

# Kita visualisasikan perubahan harga dari mulai dibangun sampai terjual
dataku.groupby('YrSold')['SalePrice'].median().plot()
plt.ylabel('Nilai Median Harga Jual')
plt.title('Perubahan Harga')
plt.tight_layout()
# Ternyata harganya turun (tidak wajar), perlu investigasi lebih lanjut

# Analisis antara variabel 'Year' dan harga rumah
def analisis_data_tahun(data, col):
    data = data.copy()
    # Melihat perbedaan antara kolom tahun yang dimaksud dengan tahun penjualan rumah
    data[col] = data['YrSold'] - data[col]
    plt.scatter(data[col], data['SalePrice'])
    plt.ylabel('SalePrice')
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# Membuat for loop untuk plotting kolom_tahun
batas = len(kolom_tahun)
i = 1
for kolom in kolom_tahun:
    if kolom !='YrSold':
        i+=1
        analisis_data_tahun(dataku, kolom)
        if i < batas: plt.figure() 
# Ternyata semakin tua fitur 'YearBulit', 'YearRemodAdd', dan 'GarageYrBlt', semakin turun harganya
# Artinya semakin besar jarak antara ketiga waktu ini dengan waktu penjualan, maka semakin turun harganya
# mungkin karena tampilannya jadul atau butuh banyak biaya renovasi, sehingga harga jualnya rendah.

# Analisis variabel discrete/diskrit (skala hitung)
kolom_diskrit = [kolom for kolom in kolom_numerik if len(dataku[kolom].unique()) <= 15 and kolom not in kolom_tahun+['Id']]
diskrit = dataku[kolom_diskrit]

# Analisis data diskrit dnegan harga rumah (SalePrice)
def analisis_data_diskrit(data, col):
    data = data.copy()
    data.groupby(col)['SalePrice'].median().plot.bar()
    plt.title(col)
    plt.ylabel('Nilai Median Harga Jual')
    plt.tight_layout()
    plt.show()

# Membuat for loop untuk plotting kolom_diskrit
batas = len(kolom_diskrit)
i = 1
for kolom in kolom_diskrit:
    i+=1
    analisis_data_diskrit(dataku, kolom)
    if i <= batas: plt.figure() 
# Ternyata ada beberapa kolom yang memiliki hubungan kuat, misal kolom 'OverallQual', semakin tinggi kualitas maka semakin tinggi pula harga jualnya
# Ini kita jadikan catatan untuk proses feature engineering

# Variabel kontinu
kolom_kontinu = [kolom for kolom in kolom_numerik if kolom not in kolom_diskrit+kolom_tahun+['Id']]
kontinu = dataku[kolom_kontinu]
# Mengecek len dari var kontinu
for i in kontinu:
    print(len(dataku[i].unique()))

# Analisis data kontinu dengan harga rumah (SalePrice)
def analisis_data_kontinu(data, col):
    data = data.copy()
    data[col].hist(bins=30) # menggabungkan len(var kontinu) ke dalam 3 bins agar mudah divisualisasikan
    plt.ylabel('Jumlah rumah')
    plt.xlabel(col)
    plt.title(col)
    plt.tight_layout()
    plt.show()

# Membuat for loop untuk plotting kolom_kontinu
batas = len(kolom_kontinu)
i = 1
for kolom in kolom_kontinu:
    i+=1
    analisis_data_kontinu(dataku, kolom)
    if i <= batas: plt.figure() 
# Terlihat bahwa hampir semua datanya tidak berdistribusi normal (skewed right)
# Nanti kita akan lakukan logtransform agar datanya lebih mendekati normal

# Melakukan proses logtransform
def analisis_logtransform(data, col):
    data = data.copy()
    # CATATAN: logaritmik tidak memperhitungkan data 0 dan negatif, jadi harus di skip kolom yg memiliki 0 dan -
    if any(data[col] <= 0):
        pass
    else:
        data[col] = np.log(data[col]) # Proses logtransformation
        data[col].hist(bins=30)
        plt.ylabel('Jumlah rumah')
        plt.xlabel(col)
        plt.title(col)
        plt.tight_layout()
        plt.show()

# Menentukan batas sekaligus variabel di kolom_kontinu yang tidak memiliki 0 dan negatif       
batas = 0
kolom_kontinu_log = []
for kolom in kolom_kontinu:
    if any(dataku[kolom] <= 0):
        pass
    else:
        batas+=1
        kolom_kontinu_log.append(kolom)
kontinu_log = dataku[kolom_kontinu_log]

# Membuat for loop untuk plotting kolom_kontinu (dengan logtransform)
i = 1
for kolom in kolom_kontinu_log:
    i+=1
    analisis_logtransform(dataku, kolom)
    if i<= batas: plt.figure() 
# Sekarang sudah lebih tampak normal

# Sekarang kita analisis hubungan antara SalePrice dengan variabel yang sudha ditransformasi
def analisis_logtransform_scatter(data, col):
    data = data.copy()
    if any(data[col] <= 0):
        pass
    else:
        data[col] = np.log(data[col])
        # Logtransform SalePrice
        data['SalePrice'] = np.log(data['SalePrice'])
        
        # plot
        plt.scatter(data[col], data['SalePrice'])
        plt.ylabel('Harga Rumah')
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()

i = 1
for kolom in kolom_kontinu_log:
    if kolom != 'SalePrice':
        i+=1
        analisis_logtransform_scatter(dataku, kolom)
        if i< batas: plt.figure() 

# Analisis outlier
def analisis_outlier(data, col):
    data = data.copy()
    if any(data[col] <= 0):
        pass
    else:
        data[col] = np.log(data[col])
        data.boxplot(column=col)
        plt.title(col)
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()

# Membuat for loop untuk plotting kolom_kontinu (dengan logtransform untuk outliers)
i = 1
for kolom in kolom_kontinu_log:
    i+=1
    analisis_outlier(dataku, kolom)
    if i<= batas: plt.figure() 
# Ternyata ada banyak outliers di var kontinu kita yg sudha di logtransform. 
# Perlu dipertimbangkan apakah membuang outlier bisa meningkatkan performa modle atau tidak.

# Variabel kategori (nominal)
kolom_kategori = [kolom for kolom in dataku.columns if dataku[kolom].dtypes == 'O']
kategori = dataku[kolom_kategori]

# Mengecek berapa banyak kategori yang ada di setiap variabel
kategori.nunique()

# Analisis variabel yang jarang
def analisis_var_jarang(data, col, persentase):
    data = data.copy()
    # Menentukan persetnase setiap kategori
    isi = data.groupby(col)['SalePrice'].count() / len(data)
    # Mengembalikan variabel yang di bawah persentase kelangkaan yang kita tentukan
    return isi[isi < persentase]

# For loop untuk variabel jarang
for kolom in kolom_kategori:
    print(analisis_var_jarang(dataku, kolom, 0.01),'\n')

# For loop untuk plotting
i = 1
batas = len(kolom_kategori)
for kolom in kolom_kategori:
    i+=1
    analisis_data_diskrit(dataku, kolom)
    if i<= batas: plt.figure() 
# Ternyata setiap kategori memiliki hubungan terhadap SalePrice