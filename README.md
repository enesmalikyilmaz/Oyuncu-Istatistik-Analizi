##             OYUNCU İSTATİSTİK ANALİZİ

## Veri Hazırlama: DataFrame’i Kullanılabilir Hale Getirme

Ham veri üzerinde çeşitli veri hazırlama teknikleri uygulanarak makine öğrenimi modelleri için kullanılabilir hale getirilmiştir. Veri hazırlama sürecinde izlenen adımlar:

### 1. Eksik Değerlerin Analizi ve İşlenmesi
Eksik değerleri analiz sürecini olumsuz etkileyebilir. Bu değerleri aşağıdaki yöntemlerle işledim:
- **Eksik Değerlerin Belirlenmesi**:
  ```python
  print(data.isnull().sum())
  ```
-**Eksik Değerlerin Doldurulması**: Sayısal sütunlardaki eksik değerler sütun ortalaması ile doldurulmuştur.
  ```python
  data = data.fillna(data.mean())
  ```
### 2. Veri Türlerinin Dönüştürülmesi
Veri türlerindeki hatalar düzeltilmiştir:
-**Sayısal Değerlere Dönüştürme**:
  ```python
  data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
  data['Performance_Gls'] = pd.to_numeric(data['Performance_Gls'], errors='coerce')
  ```
-**Hatalı Değerlerin Kaldırılması**:
  ```python
  data = data.dropna(subset=['Age', 'Performance_Gls'])
  ```
### 3. Özelliklerin Ölçeklendirilmesi
Farklı ölçeklerdeki sayısal değerler standart bir ölçeğe getirilmiştir:

  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  scaled_features = scaler.fit_transform(data[['Age', 'Playing Time_90s', 'Expected_xG', 'Expected_xAG']])
  data[['Age', 'Playing Time_90s', 'Expected_xG', 'Expected_xAG']] = scaled_features
  ```





# Kapsamlı Sonuç Analizi ve Değerlendirme

Bu çalışmada, oyuncu performansını tahmin etmek için **Linear Regression (Doğrusal Regresyon)**, **Random Forest (Rastgele Orman)** ve **Support Vector Regression (Destek Vektör Regresyonu)** olmak üzere üç farklı makine öğrenimi modeli uyguladım. Modellerin sonuçları **MSE (Ortalama Kare Hatası)** ve **R² (Açıklama Katsayısı)** parametreleri ile karşılaştırıp değerlendirdim.

## Sonuçların Karşılaştırılması

| Model                      | MSE      | R²       |
|----------------------------|----------|----------|
| **Linear Regression**      | 1.1611   | 0.862    |
| **Random Forest**          | 1.4317   | 0.8299   |
| **Support Vector Regression** | 1.5877 | 0.8113   |

---

## 1. Model Performanslarının Değerlendirilmesi

### Linear Regression (Doğrusal Regresyon)
- **Avantajlar**: Basit ve hızlı bir modeldir. Özellikle özellikler arasındaki ilişki doğrusal olduğunda iyi performans gösterir.
- **Sonuçlar**:
  - **MSE**: 1.1611 ile en düşük hata değerine sahiptir, tahminlerde en az yanılan modeldir.
  - **R²**: 0.862, modelin hedef değişkendeki varyansın %86.2’sini açıkladığını ifade eder.
- **Yorum**: Bu model, doğrusal regresyonun hedef değişkenle bağımsız değişkenler arasındaki doğrusal ilişkiyi iyi bir şekilde yakalayabildiğini göstermektedir.

### Random Forest (Rastgele Orman)
- **Avantajlar**: Karmaşık, doğrusal olmayan ilişkileri etkili bir şekilde modelleyebilir ve özelliklerin önem derecesini çıkarabilir.
- **Sonuçlar**:
  - **MSE**: 1.4317 ile orta seviyede bir hata oranına sahiptir.
  - **R²**: 0.8299, hedef değişkendeki varyansın %82.99’unu açıklamaktadır.
- **Yorum**: Rastgele Orman modeli, doğrusal olmayan ilişkileri yakalamakta başarılıdır ancak bu veri setinde Linear Regression kadar iyi performans göstermemiştir.

### Support Vector Regression (Destek Vektör Regresyonu)
- **Avantajlar**: Karmaşık, doğrusal olmayan ilişkileri modellemek için uygundur.
- **Sonuçlar**:
  - **MSE**: 1.5877 ile en yüksek hata oranına sahiptir.
  - **R²**: 0.8113 ile hedef değişkendeki varyansın %81.13’ünü açıklamaktadır.
- **Yorum**: SVR, doğrusal olmayan ilişkileri modelleyebilmiş olsa da, hem doğrusal regresyon hem de Rastgele Orman modeline kıyasla daha düşük performans gösterdi.

---

## 2. Özellik Önemleri (Random Forest)

Rastgele Orman modelinden elde edilen özellik önemleri şu şekildedir:

| Özellik                  | Önem Derecesi |
|--------------------------|---------------|
| **Expected_xG**          | 0.8832        |
| **Playing Time_90s**     | 0.0224        |
| **Progression_PrgP**     | 0.0205        |
| **Progression_PrgR**     | 0.0193        |
| **Progression_PrgC**     | 0.0190        |
| **Expected_xAG**         | 0.0187        |
| **Age**                  | 0.0170        |

### Yorum:
- **Expected_xG** (beklenen gol), açık ara en yüksek önem derecesine sahiptir (0.8832). Bu, oyuncu performansı tahmininde en kritik faktör olduğunu göstermektedir.
- Diğer özelliklerin önem dereceleri oldukça düşüktür. Bu, modelin tahmin gücünün büyük ölçüde **Expected_xG** değişkenine dayandığını ifade eder.

---

## 3. Genel Değerlendirme

### Model Karşılaştırması
- **Linear Regression**, hem en düşük MSE’ye (1.1611) hem de en yüksek R²’ye (0.862) sahip olduğu için bu veri seti üzerinde en iyi performansı göstermiştir.
- **Random Forest**, özellik önemlerini değerlendirme avantajına sahiptir ve kabul edilebilir bir doğruluk seviyesi sağlar. Ancak, bu veri seti için doğrusal regresyon kadar etkili değildir.
- **Support Vector Regression**, doğrusal olmayan ilişkileri modellemesine rağmen, diğer modellere kıyasla daha düşük performans sergilemiştir.

### İş Problemi ve Uygulama Alanı
Bu analiz, oyuncu performansını tahmin etmek için kullanılabilir. Özellikle takımlar, oyuncuların hücumdaki etkinliğini değerlendirmek ve transfer kararlarını desteklemek için **Beklenen gol(Expected_xG)** gibi önemli değişkenlere odaklanabilir.



Bu çalışmamın sonucunda, Linear Regression modeli bu veri seti üzerinde en iyi performansı gösterdiğini tespit etdim. Ancak, daha geniş bir veri seti ve hiperparametre optimizasyonuyla diğer modellerin de performansı artırılabilir.
## 📜 Dosyanın İndirilmesi

Proje kapsamında oluşturulan Python kodlarıma aşağıdaki bağlantıdan ulaşabilirsiniz:

**[Makine Öğrenmesi Final_Proje 22360859004.py Dosyasını İndir](Makine%20%C3%96%C4%9Frenmesi%20Final_Proje%2022360859004.py)** linkine tıklayarak indirme simgesine bastığınızda dosya cihazınıza inmiş olacaktır.

---
## YouTube linki

