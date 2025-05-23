@echo off
REM Executa o digitador por voz sem abrir o terminal (usando pythonw.exe)

set VENV_DIR=.venv
set SCRIPT=gui_voice_typing.py

REM Ativa o ambiente virtual
call "%VENV_DIR%\Scripts\activate.bat"

REM Executa o script Python sem terminal
start "" "%VENV_DIR%\Scripts\pythonw.exe" "%SCRIPT%"