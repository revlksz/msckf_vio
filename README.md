# Indoor Localization for Autonomous Quadcopters Operating in GPS-Denied Environments

**GPS sinyali olmayan kapalı ortamlarda otonom drone'lar için görsel-ataleti tabanlı konumlandırma sistemi**

## 🎯 Proje Özeti

Bu proje, GPS olmayan ortamlarda dronların yüksek doğrulukla konum ve yön tespiti yapabilen **MSCKF (Multi-State Constraint Kalman Filter)** tabanlı Görsel-İnertial Odometre (VIO) sistemidir.

Sistem şu bileşenleri birleştirmektedir:
- **Görsel Odometre (VO)**: Kamera görüntülerinden özellik takibi ve konum tahmini
- **İnertial Measurement Unit (IMU)**: Gyro ve hızlandırma sensörleri ile hareket tahmini
- **MSCKF Filtresi**: Kamera ve IMU verilerinin optimal sensor fusion'ı
- **CUDA Akselerasyon**: GPU destekli gerçek zamanlı işleme (Jetson uyumlu)

---

## 🔧 Gerekli Kütüphaneler

### Temel Kütüphaneler

| Kütüphane | Amaç | Sürüm |
|-----------|------|-------|
| `opencv-contrib-python` | Görüntü işleme ve CUDA desteği | >=4.5.0 |
| `numpy` | Sayısal hesaplamalar | >=1.19.0 |
| `scipy` | Bilimsel hesaplamalar (Rotation, Interpolation) | >=1.5.0 |
| `pandas` | CSV veri okuma ve işleme | >=1.1.0 |
| `matplotlib` | Görselleştirme ve grafik çizimi | >=3.3.0 |
| `opencv-python-headless` | Sunucu ortamı için (isteğe bağlı) | >=4.5.0 |

### Manuel Kurulum

```bash
pip install opencv-contrib-python numpy scipy pandas matplotlib
```

---

## 📦 Proje Yapısı

```
├── README.md                      # Bu dosya
├── requirements.txt               # Python bağımlılıkları
├── main.py                        # Ana program
├── config.py                      # Konfigürasyon dosyası
├── camera.py                      # Kamera kalibrasyon ve yönetim
├── feature_tracker.py             # CUDA ORB özellik takibi
├── imu.py                         # IMU sensör verisi işleme
├── initializer.py                 # VINS-Mono sistem başlatması
├── msckf_filter.py               # MSCKF filtresi (ana algoritma)
├── data_loader.py                # EuRoC veri seti yükleme
├── utils.py                       # Yardımcı fonksiyonlar
├── visualizer.py                 # 2D görselleştirme
├── visualizer3d.py               # 3D görselleştirme
└── old_version/                   # Eski implementasyonlar
```

---

## 📊 Ana Modüller

### 1. **main.py** - Ana Program
- Tüm bileşenleri koordine eder
- VIO döngüsünü çalıştırır
- Ground truth ile karşılaştırma yapar
- Logları dosyaya kaydeder

```bash
python main.py
```

### 2. **config.py** - Sistem Konfigürasyonu
- **IMU Parametreleri**
  - IMU frekansı: 200 Hz
  - Kamera frekansı: 20 Hz
  
- **Gürültü Parametreleri**
  - Gyro gürültüsü: 1.6968e-04
  - Hızlandırma gürültüsü: 2.0000e-3
  
- **MSCKF Ayarları**
  - Maksimum kamera durumları: 20
  - Minimum parallaks: 0.005
  - Minimum izlek uzunluğu: 3

- **ORB Parametreleri**
  - Özellik sayısı: 150
  - Ölçek faktörü: 1.2
  - Seviye sayısı: 4

### 3. **feature_tracker.py** - Görsel Özellik Takibi
- CUDA-ORB ile GPU hızlandırmalı özellik tespiti
- KNN eşleştirme ve Lowe's ratio test
- RANSAC ile geometrik doğrulama
- Özellik izleklerinin yönetimi

**CUDA Avantajları:**
- Real-time işleme (Jetson uyumlu)
- CPU'da 100-150 ms alacak işlem GPU'da 10-15 ms

### 4. **imu.py** - İnertial Measurement Unit
- IMU sensör verisi parse etme
- Gyro bias hesaplama
- Hızlandırma dünya çerçevesine dönüştürme

### 5. **msckf_filter.py** - Kalman Filtresi
- **State Vector**: `[pozisyon, yönelim(quaternion), hız, gyro_bias, accel_bias]`
- **Prediction**: IMU verileri ile hareket modeli
- **Update**: Kamera ölçümleri ile konum doğrulaması
- **Prune**: Eski kamera durumlarını kaldırma

### 6. **initializer.py** - Sistem Başlatması
- VINS-Mono metodu ile 5 kare başlatma
- IMU ile bootstrap
- Monokular ölçek belirleme

### 7. **camera.py** - Kamera Kalibrasyon
EuRoC kamerası için varsayılan parametreler:
```
K = [[458.654, 0, 367.215],
     [0, 457.296, 248.375],
     [0, 0, 1]]

D = [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]
```

### 8. **data_loader.py** - Veri Seti Yönetimi
- **EuRoC Format** desteği
- **Dummy Data** üretimi test için

### 9. **visualizer.py** & **visualizer3d.py** - Görselleştirme
- 2D yörünge çizimi
- 3D trajectory animasyonu
- Ground truth karşılaştırması

---

## 🚀 Kullanım

### 1. Veri Setini Hazırlama

Proje [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualodometrymonozoom) formatını kullanır.

Veri yapısı:
```
MH_02_easy/mav0/
├── cam0/
│   └── data/        # *.png görüntüleri
├── imu0/
│   └── data.csv     # IMU sensör verileri
└── state_groundtruth_estimate0/
    └── data.csv     # Gerçek konum verileri (opsiyonel)
```

### 2. Sistem Başlatma

```bash
# Ana programı çalıştırma
python main.py

# Kurulum talimatları:
# 1. Veri seti yüklenir (EuRoC veya dummy)
# 2. İlk 5 kare ile sistem initialize olur
# 3. MSCKF filtresi çalışmaya başlar
# 4. Her 30 karede istatistikler yazdırılır
```

### 3. Çıktılar

Program şu çıktıları üretir:

**Terminal Çıktısı:**
```
Frame 30 | FPS: 15.2 | Features: 45 | Pos: [0.12345 0.54321 -1.23456] | GT: [0.12340 0.54320 -1.23450]
   [Zamanlar] ORB: 8.5ms | Match: 4.2ms | MSCKF: 12.3ms
```

**Dosya Çıktısı:**
- `vio_estimation_logs.txt` - Tüm frame istatistikleri

---

## ⚙️ Konfigürasyon Ayarları

`config.py` dosyasında değiştirebileceğiniz önemli parametreler:

```python
# IMU Parametreleri
self.imu_freq = 200.0              # Hz
self.cam_freq = 20.0               # Hz

# Gürültü (Continuous Time)
self.gyro_noise_density = 1.6968e-04
self.accel_noise_density = 2.0000e-3

# MSCKF
self.max_cam_states = 20           # Maksimum kamera durumu
self.min_parallax = 0.005          # Minimum parallaks
self.max_track_horizon = 10        # Unseen kalma süresi

# ORB
self.orb_nfeatures = 150           # Özellik sayısı
self.orb_fastThreshold = 20        # Hassasiyet
```

---

## 📈 Veri Seti

### EuRoC MAV Dataset
- **Çözünürlük**: 752 × 480 piksel
- **Kamera**: Monoküler (tek kamera)
- **IMU**: 6-DOF (Gyro + Accelerometer)
- **Kamera Frekansı**: 20 Hz
- **IMU Frekansı**: 200 Hz
- **Ortam**: Kapalı alanlar (makine odası, odalar vb.)

### Uyumlu Veri Setleri
1. EuRoC MAV Dataset

---

## 📊 Performans Metrikleri

Sistem performansını değerlendirmek için:

```bash
# Terminal çıktılarından
- FPS: Frame per second (hedef: 15+ FPS)
- Features: Takip edilen özellik sayısı (hedef: 30+)
- Pos Error: Tahmin ve ground truth farkı
- Zamanlar: Her modülün işlem süresi
```

**Hedef Performans:**
- **FPS**: ≥15 (Jetson Nano'da ≥10)
- **Özellik Sayısı**: 40-100
- **Pozisyon Hatası**: <5% yolu
- **ORB Süresi**: <15 ms
- **MSCKF Süresi**: <20 ms

---

## ⚠️ Bilinen Limitasyonlar

1. **Monokular Ölçek Belirsizliği**
   - Absolute ölçek sistem başlatmasında ground truth veya IMU ile belirlenir
   
2. **Hızlı Hareket**
   - Çok hızlı hareket halinde motion blur oluşabilir
   
3. **Dinamik Ortamlar**
   - Hareket eden objeler takibi olumsuz etkiler
   
4. **IMU Drift**
   - Uzun süreli IMU integrasyonunda hata birikir
   - Kamera ölçümleri ile düzeltilir
   
5. **Tekrarlayan Dokular**
   - Benzer dokulu ortamlarda özellik eşleştirmesi başarısız olabilir

---

## 🔍 Sorun Giderme

### Problem: "Dataset not found"
```bash
# Çözüm: config.py veya main.py'da veri yolu kontrol edin
dataset_path = os.path.join(os.path.dirname(__file__), 'MH_02_easy')
```

### Problem: CUDA Hatası
```bash
# Jetson cihazlarda OpenCV CUDA desteğini kontrol edin
python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
# Çıktı ≥1 ise CUDA aktif
```

### Problem: Düşük FPS
1. ORB özellik sayısını azaltın: `orb_nfeatures = 100`
2. Kamera çözünürlüğünü düşürün
3. Visualizer'ı kapatın (main.py'da comment out edin)


---

**Son Güncelleme**: 25.06.2026  
**Durum**: Aktif Geliştirme  
**Python Versiyonu**: 3.7+  
**Platform**: Linux, Jetson Series