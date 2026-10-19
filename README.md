# Sürücü Uyuşukluk Tespit Sistemi

Kamera görüntüsünden göz kapanması ve esneme belirtilerini gerçek zamanlı olarak analiz eden, sürücünün uyuşukluk riskini hesaplayan bir bitirme projesidir. Uygulama; OpenCV, MediaPipe ve TensorFlow/Keras modellerini kullanır, risk yükseldiğinde görsel ve sesli uyarı verir.

> [!WARNING]
> Bu proje akademik ve deneysel amaçlıdır. Gerçek bir sürüş güvenliği sisteminin veya sürücünün dikkatinin yerini alamaz.

## Özellikler

- Kameradan gerçek zamanlı yüz, göz ve ağız takibi
- Göz kapanma oranına dayalı PERCLOS hesabı
- Esneme sayısının izlenmesi
- Birleşik risk skoru ve sesli uyarı
- Kamera seçimi ve masaüstü arayüzü
- Oturum sırasında olay kaydı

## Proje yapısı

```text
.
├── realtime app/
│   ├── app.py                         # Masaüstü uygulaması
│   ├── realtime_drowsiness.py         # Basit OpenCV sürümü
│   ├── eye_model_compat.h5            # Göz sınıflandırma modeli
│   ├── yawn_model_compat.h5           # Esneme sınıflandırma modeli
│   ├── face_landmarker.task           # MediaPipe yüz işaretleyici
│   ├── requirements.txt               # Python bağımlılıkları
│   ├── DrowsinessDetection.spec       # PyInstaller yapılandırması
│   ├── KURULUM.bat                    # Windows kurulum yardımcısı
│   └── NASIL_KULLANILIR.txt           # Kısa kullanım rehberi
├── eye_model_mrl.ipynb / .py          # Göz modeli eğitim çalışması
└── finalyawn.ipynb / .py              # Esneme modeli eğitim çalışması
```

`build/`, `dist/`, günlükler ve Python önbellekleri kaynak depoya dahil edilmez.

## Gereksinimler

- Windows 10 veya 11
- Python 3.10
- Çalışan bir kamera
- Git LFS (depoyu klonlayacak kullanıcılar için)

## Kurulum

Önce projeyi klonlayın:

```powershell
git clone <DEPO-ADRESINIZ>
cd <DEPO-KLASORU>\realtime app
```

Sanal ortam oluşturup bağımlılıkları kurun:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Alternatif olarak `realtime app` klasöründeki `KURULUM.bat` dosyasını çalıştırabilirsiniz.

## Çalıştırma

Masaüstü arayüzü:

```powershell
cd "realtime app"
python app.py
```

Basit OpenCV arayüzü:

```powershell
cd "realtime app"
python realtime_drowsiness.py
```

Kamera izni istenirse izin verin. Basit arayüzde çıkış için `q`, istatistikleri sıfırlamak için `r` tuşuna basın.

## Windows uygulaması oluşturma

PyInstaller yüklendikten sonra aşağıdaki komut `realtime app/dist/` altında uygulama paketini oluşturur:

```powershell
cd "realtime app"
pip install pyinstaller
pyinstaller --clean DrowsinessDetection.spec
```

Derlenmiş paket büyük olduğu için GitHub kaynak deposuna eklenmez. Dağıtılacaksa ZIP dosyası olarak bir GitHub Release'e eklenmesi önerilir.

## Modeller ve Git LFS

Çalışma modelleri Git LFS ile izlenir. Depoyu ilk kez klonlamadan önce Git LFS kurulmalı, ardından gerekirse modeller şu komutla indirilmelidir:

```powershell
git lfs install
git lfs pull
```

## Eğitim dosyaları

Kök dizindeki notebook ve Python dosyaları model eğitim çalışmalarını içerir. Eğitim veri kümeleri boyutları ve kullanım lisansları nedeniyle bu depoya dahil edilmemiştir. Projeyi herkese açık yayımlamadan önce kullanılan veri kümelerinin ve üçüncü taraf model dosyalarının lisans koşullarını kontrol edin.

## Lisans

Bu depoda henüz bir açık kaynak lisansı tanımlanmamıştır. Kullanım ve dağıtım koşullarını belirtmek için yayımlamadan önce uygun bir `LICENSE` dosyası ekleyin.
