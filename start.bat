@echo off
chcp 65001 >nul
title 🤖 DeepSeek (v0.70 + Images) — Auto Restart

echo.
echo ╔══════════════════════════════════════════════╗
echo ║       🚀 DeepSeek бот v0.70 (Image Gen)      ║
echo ║     Автоперезапуск при ошибках сети          ║
echo ╚══════════════════════════════════════════════╝
echo.

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python не найден. Устанавливаю 3.12.6...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$v='3.12.6'; $u='https://www.python.org/ftp/python/'+$v+'/python-'+$v+'-amd64.exe'; ^
       $p=$env:TEMP+'\python-installer.exe'; ^
       try { Invoke-WebRequest $u -OutFile $p } catch { [Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest $u -OutFile $p }; ^
       Start-Process $p -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1 Include_launcher=1 Include_pip=1' -Wait"
)

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

:loop
echo.
echo ▶️  Запуск бота...
python bot.py
echo ⏳ Перезапуск через 5 секунд...
timeout /t 5 >nul
goto loop
