@echo off
REM Ativa o ambiente virtual e executa o programa de digitação por voz

REM Caminho para o ambiente virtual (ajuste se necessário)
set VENV_DIR=.venv

REM Caminho para o script Python
set SCRIPT=gui_voice_typing.py

REM Ativa o ambiente virtual
call "%VENV_DIR%\Scripts\activate.bat"

REM Executa o script Python
python "%SCRIPT%"

REM Mantém a janela aberta após execução para ver mensagens
echo.
echo Pressione qualquer tecla para sair...
pause >nul