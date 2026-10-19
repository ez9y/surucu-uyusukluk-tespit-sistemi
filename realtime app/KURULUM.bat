@echo off
chcp 65001 >nul
title Driver Drowsiness Detection - Kurulum

echo ============================================
echo   Driver Drowsiness Detection - Kurulum
echo ============================================
echo.

:: Python kontrolu
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [HATA] Python bulunamadi!
    echo.
    echo Lutfen once Python 3.10 yukleyin:
    echo https://www.python.org/downloads/release/python-31011/
    echo.
    echo Kurulum sirasinda "Add Python to PATH" secenegini isaretleyin!
    echo.
    pause
    exit /b 1
)

echo [OK] Python bulundu:
python --version
echo.

echo Kutuphaneler yukleniyor, bu islem 5-10 dakika surebilir...
echo.

pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [HATA] Kurulum basarisiz. Lutfen internet baglantinizi kontrol edin.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Kurulum tamamlandi!
echo   Uygulamayi baslatmak icin:
echo   "Drowsiness Detection" kisayolunu kullanin
echo   veya DrowsinessDetection.vbs dosyasina
echo   cift tiklayin.
echo ============================================
echo.
pause
