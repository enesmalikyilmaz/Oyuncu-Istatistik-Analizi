# Gereklileri import ediyoruz
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Veriyi yüklüyoruz
data = pd.read_csv('/content/big_5_players_stats_2023_2024.csv')  # Dosya yolunuza göre düzenleyin

# İlk satırlara göz atıyoruz
print(data.head())

# Veri setinin genel özelliklerine bakış attık
print(data.info())

# Eksik veri var mı diye bakıyoruz
print("Eksik değerler:\n", data.isnull().sum())

# Sayısal özellikleri seçtik
numerical_features = ['Age', 'Playing Time_90s', 'Expected_xG', 'Expected_xAG',
                     'Progression_PrgC', 'Progression_PrgP', 'Progression_PrgR']
target = 'Performance_Gls'

# X adında özellik matrixi oluşturduk ve hedef vektörümüz y
X = data[numerical_features].astype(float)
y = data[target].astype(float)


X = X.fillna(X.mean())


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)# Data eğitme işlemi
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 1. Linear Regression algoritması
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

# 2. Random Forest algoritması
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

# 3. Support Vector Regression algoritması
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)
svr_pred = svr_model.predict(X_test)
svr_mse = mean_squared_error(y_test, svr_pred)
svr_r2 = r2_score(y_test, svr_pred)

print("Model Performans Karşılaştırması:")
print("\
Lineer Regresyon algoritması:")
print("MSE:", round(lr_mse, 4))
print("R2 Skoru:", round(lr_r2, 4))

print("\
Random Forest algoritması:")
print("MSE:", round(rf_mse, 4))
print("R2 Skoru:", round(rf_r2, 4))

print("\
Support Vector Regression algoritması:")
print("MSE:", round(svr_mse, 4))
print("R2 Skoru:", round(svr_r2, 4))

# Random Forest Özellik Önemleri
feature_importance = pd.DataFrame({
    'Özellik': numerical_features,
    'Önem Derecesi': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Önem Derecesi', ascending=False)
print("\
Random Forest Özellik Önemleri:")
print(feature_importance)

# Görselleştirme işlemleri

plt.figure(figsize=(10, 6))
sns.barplot(x='Önem Derecesi', y='Özellik', data=feature_importance)
plt.title('Random Forest Özellik Önem Dereceleri')
plt.tight_layout()
plt.show()

# Expected_xG ve Expected_xAG arasındaki korelasyonu hesapladım
correlation = data['Expected_xG'].corr(data['Expected_xAG'])
print(f"Expected_xG ile Expected_xAG arasındaki korelasyon: {correlation:.4f}")

# Scatter plot ile ilişkiyi görselleştirdim

plt.figure(figsize=(8, 5))
sns.scatterplot(x=data['Expected_xG'], y=data['Expected_xAG'], alpha=0.7)
plt.title("Expected_xG ve Expected_xAG Arasındaki İlişki", fontsize=14)
plt.xlabel("Expected_xG (Gol Beklentisi)")
plt.ylabel("Expected_xAG (Asist Beklentisi)")
plt.grid(True)
plt.show()

# Oynama süresini sayıya dönüştürdüm
data['Playing Time_90s'] = pd.to_numeric(data['Playing Time_90s'], errors='coerce')
data['Performance_Gls'] = pd.to_numeric(data['Performance_Gls'], errors='coerce')

# Eksik değerler varsa kaldırdım
data = data.dropna(subset=['Performance_Gls', 'Playing Time_90s'])

# Oynama süresi grupları böyle yaptım
bins = [0, 10, 20, 30, 35, float('inf')]
labels = ['0-10', '10-20', '20-30', '30-35', '35+']
data['PlayingTime_Group'] = pd.cut(data['Playing Time_90s'], bins=bins, labels=labels)

# Gruplar için ortalama performansı hesaplama işlemi
group_performance = data.groupby('PlayingTime_Group')['Performance_Gls'].mean()
print(group_performance)


data_sorted = data.sort_values(by='Age')

# Scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=data_sorted['Age'], y=data_sorted['Performance_Gls'], alpha=0.5)
plt.title("Yaş ve Gol Performansı Arasındaki İlişki ", fontsize=14)
plt.xlabel("Yaş")
plt.ylabel("Gol Performansı (Performance_Gls)")
plt.grid(True)
plt.show()
bins = [0, 10, 20, 30, 35, float('inf')]
labels = ['0-10', '10-20', '20-30', '30-35', '35+']
data['PlayingTime_Group'] = pd.cut(data['Playing Time_90s'], bins=bins, labels=labels)

# Gruplar için ortalama gol performansı
group_performance = data.groupby('PlayingTime_Group')['Performance_Gls'].mean()

print("Oynama Süresi Gruplarına Göre Ortalama Performans:")
print(group_performance)

# Oynama süresi ile performans arasındaki korelasyon
correlation = data['Playing Time_90s'].corr(data['Performance_Gls'])
print(f"Oynama süresi ile performans arasındaki korelasyon: {correlation:.4f}")

# Sütun grafiği
plt.figure(figsize=(8, 5))
group_performance.plot(kind='bar', color='skyblue', alpha=0.8)
plt.title("Oynama Süresi Gruplarına Göre Ortalama Gol Performansı", fontsize=14)
plt.xlabel("Oynama Süresi Grupları (90 dk)", fontsize=12)
plt.ylabel("Ortalama Performans (Gol)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Scatter plot ve regresyon çizgisi
plt.figure(figsize=(8, 5))
sns.scatterplot(x=data_sorted['Age'], y=data_sorted['Performance_Gls'], alpha=0.5)
plt.title("Oynama Süresi ile Performans İlişkisi", fontsize=14)
plt.xlabel("Oynama Süresi (90 dk)", fontsize=12)
plt.ylabel("Gol Performansı", fontsize=12)
plt.grid(True)
plt.show()

# Yaş aralıklarını belirle
bins = [20, 25, 30, 35, 40]
labels = ['20-24', '25-29', '30-34', '35-39']
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
data['Performance_Gls'] = pd.to_numeric(data['Performance_Gls'], errors='coerce')

# Yaş grubu sütunu ekle
data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)

# Yaş gruplarına göre ortalama gol performansını hesapla
average_performance = data.groupby('Age_Group')['Performance_Gls'].mean().reset_index()

# Sonuçları görselleştir
plt.figure(figsize=(10, 6))
sns.barplot(x='Age_Group', y='Performance_Gls', data=average_performance, palette='viridis')
plt.title("Yaş Gruplarına Göre Ortalama Gol Performansı", fontsize=14)
plt.xlabel("Yaş Grupları")
plt.ylabel("Ortalama Gol Performansı")
plt.grid(axis='y')
plt.show()