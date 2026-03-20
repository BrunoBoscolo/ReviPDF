@echo off
color 0B

echo ========================================================
echo   [1/3] Compilando o backend em Python...
echo ========================================================
python -m PyInstaller --name pdf_processor --onefile --collect-data spacy --collect-data pt_core_news_lg --collect-data langdetect --collect-data wordfreq pdf_processor.py

:: Verifica se o PyInstaller deu erro
if %errorlevel% neq 0 (
    echo.
    color 0C
    echo [ERRO] Falha na compilacao do Python! Abortando.
    pause
    exit /b %errorlevel%
)

echo.
echo ========================================================
echo   [2/3] Copiando o executavel para a pasta do Tauri...
echo ========================================================
:: O /Y forca a substituicao sem perguntar
copy /Y ".\dist\pdf_processor.exe" ".\revipdf\src-tauri\binaries\pdf_processor-x86_64-pc-windows-msvc.exe"

echo.
echo ========================================================
echo   [3/3] Iniciando o Tauri (Interface Grafica)...
echo ========================================================
cd revipdf
npm run tauri dev