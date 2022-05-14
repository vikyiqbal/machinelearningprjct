# Mengimpor library yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Untuk membuka model
import joblib

# Load dataset
dataku = pd.read_csv('test.csv')

# Mengeluarkan kolom 'Id'
dataku.drop('Id', axis=1, inplace=True)


'''
Proses Feature Engineering

Kita akan melakukan proses yang sama untuk feature engineering:
1. Nilai kosong (NA atau nan)
2. Variabel yang berhubungan dengan waktu (Tahun)
3. Variabel yang tidak berdistribusi normal
4. Menghilangkan variabel jarang untuk tipe kategori
5. Merubah format data string ke numerik untuk tipe kategori
6. Menyamakan rentang data nilai untuk beberapa variabel (feature scaling)
'''

# Data kosong (NA atau nan)
# Kita tulis secara manual kolom-kolom kategori data kosong (sesuai langkah feature engineering)
# Kolom berikut sudah diurutkan dari persentase data kosong tertinggi ke terendah
kolom_na_kategori = ['PoolQC',
                     'MiscFeature',
                     'Alley',
                     'Fence',
                     'FireplaceQu',
                     'GarageType',
                     'GarageFinish',
                     'GarageQual',
                     'GarageCond',
                     'BsmtExposure',
                     'BsmtFinType2',
                     'BsmtQual',
                     'BsmtCond',
                     'BsmtFinType1',
                     'MasVnrType',
                     'Electrical']

# Mengganti data kosong untuk tipe kategori dengan kategori 'Kosong'
dataku[kolom_na_kategori] = dataku[kolom_na_kategori].fillna('Kosong')

# Kolom NA numerik
kolom_na_numerik = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']

# Mengganti data kosong numerik dengan modus/mode (nilai terbanyak)
for kolom in kolom_na_numerik:
    # menghitung modus (bisa juga pakai mean)
    hitung_modus = dataku[kolom].mode()[0]
    # Memasukkan modus ke baris yang kosong
    dataku[kolom] = dataku[kolom].fillna(hitung_modus)

# Menghitung tahun berlalu untuk 4 variabel waktu terhadap 'YrSold'
def tahun_berlalu(data, col):
    data[col] = data['YrSold'] - data[col]
    return data

# Merubah setiap kolom 'Tahun' menjadi selisihnya terhadap 'YrSold'
for kolom in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    dataku = tahun_berlalu(dataku, kolom)

# Merubah variabel yang tidak normal menjadi mendekati normal (log-transform)
for kolom in ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']:
    dataku[kolom] = np.log(dataku[kolom])

# Mengatasi variabel jarang
''' ini adalah fungsi yang dilakukan di Feature Engineering

kolom_kategori = [kolom for kolom in dataku.columns if dataku[kolom].dtypes == 'O']

def analisis_var_jarang(data, col, persentase):
    data = data.copy()
    # Menentukan persentase setiap kategori
    isi = data.groupby(col)['SalePrice'].count() / len(data)
    return isi[isi < persentase].index

data_jarang = {} # ini tambahan untuk kategori apa saja yang jarang di setiap kolom
for kolom in kolom_kategori:
    var_jarang = analisis_var_jarang(dataku, kolom, 0.01)
    print(kolom, var_jarang, '\n')
    data_jarang[kolom] = var_jarang
    
    dataku[kolom] = np.where(dataku[kolom].isin(var_jarang), 'Jarang', dataku[kolom])
'''

data_jarang = {
    'MSZoning': ['C (all)'],
    'Street': ['Grvl'],
    'LotShape': ['IR3'],
    'Utilities': ['NoSeWa'],
    'LotConfig': ['FR3'],
    'LandSlope': ['Sev'],
    'Neighborhood': ['Blueste', 'NPkVill', 'Veenker'],
    'Condition1': ['PosA', 'RRAe', 'RRNe', 'RRNn'],
    'Condition2': ['Artery', 'Feedr', 'PosA', 'PosN', 'RRAe', 'RRAn', 'RRNn'],
    'HouseStyle': ['1.5Unf', '2.5Fin', '2.5Unf'],
    'RoofStyle': ['Flat', 'Gambrel', 'Mansard', 'Shed'],
    'RoofMatl': ['ClyTile', 'Membran', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl'],
    'Exterior1st': ['AsphShn', 'BrkComm', 'ImStucc', 'Stone'],
    'Exterior2nd': ['AsphShn', 'Brk Cmn', 'ImStucc', 'Other', 'Stone'],
    'ExterQual': ['Fa'],
    'ExterCond': ['Ex', 'Po'],
    'Foundation': ['Stone', 'Wood'],
    'BsmtCond': ['Po'],
    'BsmtFinType2': ['GLQ'],
    'Heating': ['Floor', 'Grav', 'OthW', 'Wall'],
    'HeatingQC': ['Po'],
    'Electrical': ['FuseP', 'Mix'],
    'Functional': ['Maj1', 'Maj2', 'Sev'],
    'GarageType': ['2Types', 'CarPort'],
    'GarageQual': ['Ex', 'Gd', 'Po'],
    'GarageCond': ['Ex', 'Gd', 'Po'],
    'PoolQC': ['Ex', 'Fa', 'Gd'],
    'Fence': ['MnWw'],
    'MiscFeature': ['Gar2', 'Othr', 'TenC'],
    'SaleType': ['CWD', 'Con', 'ConLD', 'ConLI', 'ConLw', 'Oth'],
    'SaleCondition': ['AdjLand', 'Alloca']
    }


for kolom in data_jarang.keys():
    dataku[kolom] = np.where(dataku[kolom].isin([x for x in data_jarang[kolom]]), 'Jarang', dataku[kolom])

# kolom_ordinal adalah hasil dari Feature Engineering sebelumnya
kolom_ordinal = {
    'MSZoning': {'Jarang': 0, 'RM': 1, 'RH': 2, 'RL': 3, 'FV': 4},
    'Street': {'Jarang': 0, 'Pave': 1},
    'Alley': {'Grvl': 0, 'Pave': 1, 'Kosong': 2},
    'LotShape': {'Reg': 0, 'Jarang': 1, 'IR1': 2, 'IR2': 3},
    'LandContour': {'Bnk': 0, 'Lvl': 1, 'Low': 2, 'HLS': 3},
    'Utilities': {'Jarang': 0, 'AllPub': 1},
    'LotConfig': {'Inside': 0, 'Corner': 1, 'FR2': 2, 'Jarang': 3, 'CulDSac': 4},
    'LandSlope': {'Gtl': 0, 'Mod': 1, 'Jarang': 2},
    'Neighborhood': {'IDOTRR': 0, 'MeadowV': 1, 'BrDale': 2, 'BrkSide': 3, 'Edwards': 4, 
                     'OldTown': 5, 'Sawyer': 6, 'SWISU': 7, 'NAmes': 8, 'Mitchel': 9, 
                     'Jarang': 10, 'SawyerW': 11, 'NWAmes': 12, 'Gilbert': 13, 'CollgCr': 14, 
                     'Blmngtn': 15, 'Crawfor': 16, 'ClearCr': 17, 'Somerst': 18, 'Timber': 19, 
                     'StoneBr': 20, 'NridgHt': 21, 'NoRidge': 22},
    'Condition1': {'Artery': 0, 'Feedr': 1, 'Jarang': 2, 'Norm': 3, 'RRAn': 4, 'PosN': 5},
    'Condition2': {'Jarang': 0, 'Norm': 1},
    'BldgType': {'2fmCon': 0, 'Twnhs': 1, 'Duplex': 2, '1Fam': 3, 'TwnhsE': 4},
    'HouseStyle': {'SFoyer': 0, '1.5Fin': 1, 'Jarang': 2, 'SLvl': 3, '1Story': 4, '2Story': 5},
    'RoofStyle': {'Gable': 0, 'Jarang': 1, 'Hip': 2},
    'RoofMatl': {'CompShg': 0, 'Jarang': 1},
    'Exterior1st': {'AsbShng': 0, 'Jarang': 1, 'WdShing': 2, 'Wd Sdng': 3, 'MetalSd': 4, 
                    'Stucco': 5, 'HdBoard': 6, 'Plywood': 7, 'BrkFace': 8, 'CemntBd': 9, 
                    'VinylSd': 10},
    'Exterior2nd': {'AsbShng': 0, 'Wd Sdng': 1, 'Stucco': 2, 'MetalSd': 3, 'Wd Shng': 4, 
                    'Jarang': 5, 'HdBoard': 6, 'Plywood': 7, 'BrkFace': 8, 'CmentBd': 9, 
                    'VinylSd': 10},
    'MasVnrType': {'BrkCmn': 0, 'None': 1, 'BrkFace': 2, 'Jarang': 3, 'Stone': 4},
    'ExterQual': {'Jarang': 0, 'TA': 1, 'Gd': 2, 'Ex': 3},
    'ExterCond': {'Fa': 0, 'Jarang': 1, 'Gd': 2, 'TA': 3},
    'Foundation': {'Slab': 0, 'BrkTil': 1, 'CBlock': 2, 'Jarang': 3, 'PConc': 4},
    'BsmtQual': {'Kosong': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
    'BsmtCond': {'Jarang': 0, 'Kosong': 1, 'Fa': 2, 'TA': 3, 'Gd': 4},
    'BsmtExposure': {'Kosong': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
    'BsmtFinType1': {'Kosong': 0, 'LwQ': 1, 'BLQ': 2, 'Rec': 3, 'ALQ': 4, 'Unf': 5, 'GLQ': 6},
    'BsmtFinType2': {'Kosong': 0, 'BLQ': 1, 'Rec': 2, 'Jarang': 3, 'LwQ': 4, 'Unf': 5, 'ALQ': 6},
    'Heating': {'Jarang': 0, 'GasW': 1, 'GasA': 2},
    'HeatingQC': {'Jarang': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
    'CentralAir': {'N': 0, 'Y': 1},
    'Electrical': {'Jarang': 0, 'FuseF': 1, 'FuseA': 2, 'SBrkr': 3},
    'KitchenQual': {'Fa': 0, 'TA': 1, 'Gd': 2, 'Ex': 3},
    'Functional': {'Jarang': 0, 'Min2': 1, 'Mod': 2, 'Min1': 3, 'Typ': 4},
    'FireplaceQu': {'Po': 0, 'Kosong': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageType': {'Kosong': 0, 'Jarang': 1, 'Detchd': 2, 'Basment': 3, 'Attchd': 4, 'BuiltIn': 5},
    'GarageFinish': {'Kosong': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
    'GarageQual': {'Kosong': 0, 'Fa': 1, 'TA': 2, 'Jarang': 3},
    'GarageCond': {'Kosong': 0, 'Fa': 1, 'Jarang': 2, 'TA': 3},
    'PavedDrive': {'N': 0, 'P': 1, 'Y': 2},
    'PoolQC': {'Kosong': 0, 'Jarang': 1},
    'Fence': {'Jarang': 0, 'GdWo': 1, 'MnPrv': 2, 'GdPrv': 3, 'Kosong': 4},
    'MiscFeature': {'Jarang': 0, 'Shed': 1, 'Kosong': 2},
    'SaleType': {'COD': 0, 'Jarang': 1, 'WD': 2, 'New': 3},
    'SaleCondition': {'Abnorml': 0, 'Jarang': 1, 'Family': 2, 'Normal': 3, 'Partial': 4}
    }

for kolom in kolom_ordinal.keys():
    label_ordinal = kolom_ordinal[kolom]
    # Merubah semua kategori menjadi numerik
    dataku[kolom] = dataku[kolom].map(label_ordinal)

kolom_kosong = [kolom for kolom in dataku.columns if dataku[kolom].isnull().sum() > 0]
# Masih banyak data kolom kosong. Dalam pembelajaran kali ini kita siasati dengan cepat
dataku.fillna(0, inplace=True)
# Dalam dunia nyata harus dicek lebih jauh lagi kenapa data kosong ini tidak ada di training set

# Feature Scaling
# Harus menggunakan feature scaling dari X_train
# Di langkah awal seharusnya kita menyimpan hasil scalernya
''' Ini adalah script di feature engineering untuk menyimpan scaler
from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
X_train[kolom_training] = sc_X.fit_transform(X_train[kolom_training])
X_test[kolom_training] = sc_X.transform(X_test[kolom_training])
import joblib
joblib.dump(sc_X, 'minmax_scaler.joblib')
'''
# Lokasi file: https://bit.ly/3wxKO0R
scaler = joblib.load('minmax_scaler.joblib') 

kolom_training = [kolom for kolom in dataku.columns if kolom not in 
                  ['Id', 'SalePrice', 'LotFrontage_na',
                   'MasVnrArea_na', 'GarageYrBlt_na']]

dataku[kolom_training] = scaler.transform(dataku[kolom_training])

# Mengimpor fitur terpilih
fitur = pd.read_csv('fitur_pilihan.csv')
fitur = fitur['0'].tolist() # merubah df menjadi list

# Menyesuaikan kolom data
dataku = dataku[fitur]

# Memprediksi dataset kita dengan model yang sebelumnya sudah disimpan
model_linear = joblib.load('regresi_lasso.pkl')
pred = model_linear.predict(dataku)
real_pred = np.exp(pred)

# Melihat distribusi prediksi model kita
pd.Series(real_pred).hist(bins=50) 
