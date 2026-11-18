@echo off
REM Builds Tolk.exe into dist\Tolk\
REM Run this from the folder containing tolk.py, inside your activated venv.

pip install pyinstaller

pyinstaller --noconfirm --windowed --name Tolk ^
  --collect-all faster_whisper ^
  --collect-all ctranslate2 ^
  --collect-all pyaudiowpatch ^
  --collect-all tokenizers ^
  --collect-all onnxruntime ^
  --collect-all av ^
  --collect-all soxr ^
  tolk.py

if exist models (
  echo Copying models folder...
  xcopy /E /I /Y models dist\Tolk\models
)

echo.
echo Done. Your app is in dist\Tolk\Tolk.exe
echo Copy the whole dist\Tolk folder if you move it to another machine.
pause
