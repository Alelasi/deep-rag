@echo off
REM ================================================================
REM  DeepRAG v2.9 多功能菜单批处理脚本
REM ================================================================
chcp 65001 >nul 2>&1
setlocal
color 0A
title DeepRAG v2.9 控制台
cd /d "%~dp0"

echo.
echo    ____  _  _   _  __  ____   ___  ____
echo   ^|  _ \^| ^|^| ^| ^|/ / ^| ___^| / _ \^|  _ \
echo   ^| ^| ^| ^|^| ^|^|_ ' /  ^|___ \^| ^| ^| ^| ^|_) ^|
echo   ^| ^|^|_ ^|__   _^| . \   ___) ^| ^|^|_ ^|  _ ^< 
echo   ^|____/   ^|_^| ^|^|\_\ ^|____/ \___/^|_^| \_\
echo.
echo                    v2.9
echo   ================================================
echo    Enterprise Agentic RAG System
echo   ================================================
echo.

if not exist "app.py" (
    echo [ERROR] 未找到 app.py 文件!
    echo [ERROR] 请确认在 deep-rag 根目录下运行此脚本。
    pause
    exit /b 1
)
echo [OK] 目录检查通过

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到 Python!
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version 2^^>^&1') do set "PYVER=%%i"
echo [OK] %PYVER%

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [OK] 已激活虚拟环境
) else (
    echo [WARN] 未找到 venv，使用系统 Python
)
echo.

:MENU
cls
echo.
echo   ================================================
echo            DeepRAG v2.9 多功能菜单
echo   ================================================
echo.
echo    [1] 启动所有服务 (Qdrant + Streamlit)
echo    [2] 构建知识库
echo    [3] 运行测试
echo    [4] 部署种子数据到 Qdrant Cloud
echo    [5] 退出
echo.
set "choice="
set /p "choice=请选择操作 [1-5]: "

if "%choice%"=="1" goto START_SERVICES
if "%choice%"=="2" goto BUILD_KB
if "%choice%"=="3" goto RUN_TESTS
if "%choice%"=="4" goto SEED_QDRANT
if "%choice%"=="5" goto EXIT_MENU
echo [WARN] 无效输入
timeout /t 2 >nul
goto :MENU

:START_SERVICES
cls
netstat -ano | findstr ":6333 " | findstr "LISTENING" >nul 2>&1
if errorlevel 1 (
    echo [OK] 端口 6333 空闲
    where qdrant >nul 2>&1
    if not errorlevel 1 (
        start "Qdrant Server" cmd /k qdrant --config-dir .qdrant
        echo [OK] Qdrant 已启动
        timeout /t 3 >nul
    ) else (
        where chroma >nul 2>&1
        if not errorlevel 1 (
            start "ChromaDB" cmd /k chroma run --path chroma_data --port 8000
            echo [OK] ChromaDB 已启动
            timeout /t 3 >nul
        ) else (
            echo [ERROR] 未找到 qdrant 或 chroma
        )
    )
) else (
    echo [WARN] 端口 6333 已被占用
)
netstat -ano | findstr ":8501 " | findstr "LISTENING" >nul 2>&1
if errorlevel 1 (
    start "Streamlit App" cmd /k streamlit run app.py
    echo [OK] Streamlit 已启动 http://localhost:8501
) else (
    echo [WARN] 端口 8501 已被占用
)
echo.
pause
goto :MENU

:BUILD_KB
cls
echo [INFO] python scripts/build_all_kb_v2.py
python scripts/build_all_kb_v2.py
if errorlevel 1 (echo [ERROR] 构建失败) else (echo [OK] 构建完成)
pause
goto :MENU

:RUN_TESTS
cls
echo [INFO] python scripts/run_pyramid_tests.py --level all
python scripts/run_pyramid_tests.py --level all
if errorlevel 1 (echo [ERROR] 测试失败) else (echo [OK] 测试完成)
pause
goto :MENU

:SEED_QDRANT
cls
echo [WARN] 需要 QDRANT_CLOUD_URL 和 QDRANT_CLOUD_KEY
echo [INFO] python scripts/seed_qdrant_cloud.py
python scripts/seed_qdrant_cloud.py
if errorlevel 1 (echo [ERROR] 部署失败) else (echo [OK] 部署完成)
pause
goto :MENU

:EXIT_MENU
echo 感谢使用 DeepRAG v2.9
timeout /t 2 >nul
exit /b 0
