@echo off
REM ================================================================
REM  DeepRAG v2.9 多功能菜单批处理脚本
REM  功能：启动服务 / 构建知识库 / 运行测试 / 部署种子数据
REM ================================================================
chcp 65001 >nul 2>&1
setlocal
color 0A
title DeepRAG v2.9 控制台
cd /d "%~dp0"

REM ===== ASCII Art 标题 =====
echo.
echo    ____  _  _   _  __  ____   ___  ____
echo   ^|  _ \^| ^|^| ^| ^|/ / ^| ___^| / _ \^|  _ \
echo   ^| ^| ^| ^|^| ^|^|_ ' /  ^|___ \^| ^| ^| ^| ^|_) ^|
echo   ^| ^|^|_ ^|__   _^| . \   ___) ^| ^|^|_ ^|  _ ^< 
echo   ^|____/   ^|_^| ^|^|\_\ ^|____/ \___/^|_^| \_\
echo.
echo                    v2.9
echo.
echo   ================================================
echo    Enterprise Agentic RAG System
echo   ================================================
echo.

REM ===== 环境检查 =====

REM 检查是否在正确的目录（检查 app.py 是否存在）
if not exist "app.py" (
    echo [ERROR] 未找到 app.py 文件!
    echo [ERROR] 请确认在 deep-rag 根目录下运行此脚本。
    echo.
    pause
    exit /b 1
)
echo [OK] 目录检查通过 - 已找到 app.py

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到 Python! 请先安装 Python 并添加到 PATH。
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set "PYVER=%%i"
echo [OK] Python 检测通过 - %PYVER%

REM 激活虚拟环境（优先 .venv，兼容 venv）
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [OK] 已激活虚拟环境 (.venv)
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [OK] 已激活虚拟环境 (venv)
) else (
    echo [WARN] 未找到虚拟环境，使用系统 Python
)
echo.

REM ===== 主菜单 =====
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
echo   ================================================
echo.
set "choice="
set /p "choice=请选择操作 [1-5]: "

if "%choice%"=="1" goto START_SERVICES
if "%choice%"=="2" goto BUILD_KB
if "%choice%"=="3" goto RUN_TESTS
if "%choice%"=="4" goto SEED_QDRANT
if "%choice%"=="5" goto EXIT_MENU
echo.
echo [WARN] 无效输入，请选择 1-5 之间的数字
timeout /t 2 >nul
goto :MENU

REM ===== [1] 启动所有服务 =====
:START_SERVICES
cls
echo.
echo   ================================================
echo    [1] 启动所有服务
echo   ================================================
echo.

REM 使用 netstat 检查 Qdrant 端口 6333 是否被占用
set "QDRANT_RUNNING=0"
netstat -ano | findstr ":6333 " | findstr "LISTENING" >nul 2>&1
if errorlevel 1 (
    echo [OK] 端口 6333 空闲 - 准备启动 Qdrant 服务器
) else (
    set "QDRANT_RUNNING=1"
    echo [WARN] 端口 6333 已被占用 - Qdrant 可能已在运行
)

REM 使用 netstat 检查 Streamlit 端口 8501 是否被占用
set "STREAMLIT_RUNNING=0"
netstat -ano | findstr ":8501 " | findstr "LISTENING" >nul 2>&1
if errorlevel 1 (
    echo [OK] 端口 8501 空闲 - 准备启动 Streamlit 应用
) else (
    set "STREAMLIT_RUNNING=1"
    echo [WARN] 端口 8501 已被占用 - Streamlit 可能已在运行
)
echo.

REM 启动 Qdrant 服务器（或回退到 chroma run）
if "%QDRANT_RUNNING%"=="1" goto :STREAMLIT_START

echo [INFO] 正在启动向量数据库服务器...
if not exist ".qdrant" goto :TRY_CHROMA
where qdrant >nul 2>&1
if errorlevel 1 goto :QDRANT_NOT_FOUND
start "Qdrant Server" cmd /k qdrant --config-dir .qdrant
echo [OK] Qdrant 服务器已在新窗口启动 (端口 6333)
timeout /t 3 >nul
goto :STREAMLIT_START

:TRY_CHROMA
where chroma >nul 2>&1
if errorlevel 1 goto :NO_VECTOR_DB
start "ChromaDB Server" cmd /k chroma run --path chroma_data --port 8000
echo [OK] ChromaDB 服务器已在新窗口启动 (端口 8000)
timeout /t 3 >nul
goto :STREAMLIT_START

:QDRANT_NOT_FOUND
echo [ERROR] 未找到 qdrant 命令，请先安装 Qdrant
goto :STREAMLIT_START

:NO_VECTOR_DB
echo [ERROR] 未找到 qdrant 或 chroma 命令，请先安装向量数据库
goto :STREAMLIT_START

:STREAMLIT_START
echo.
if "%STREAMLIT_RUNNING%"=="1" (
    echo [INFO] Streamlit 应用已在运行，跳过启动
    goto :SERVICES_DONE
)
echo [INFO] 正在启动 Streamlit 应用...
start "Streamlit App" cmd /k streamlit run app.py
echo [OK] Streamlit 应用已在新窗口启动
echo [INFO] 浏览器访问: http://localhost:8501

:SERVICES_DONE
echo.
echo [OK] 服务启动流程完成
echo.
pause
goto :MENU

REM ===== [2] 构建知识库 =====
:BUILD_KB
cls
echo.
echo   ================================================
echo    [2] 构建知识库
echo   ================================================
echo.
echo [INFO] 正在执行: python scripts/build_all_projects_qdrant.py
echo.
python scripts/build_all_projects_qdrant.py
if errorlevel 1 (
    echo.
    echo [ERROR] 构建知识库失败! 请检查上方错误信息
) else (
    echo.
    echo [OK] 知识库构建完成
)
echo.
pause
goto :MENU

REM ===== [3] 运行测试 =====
:RUN_TESTS
cls
echo.
echo   ================================================
echo    [3] 运行测试
echo   ================================================
echo.
echo [INFO] 正在执行: python scripts/run_pyramid_tests.py --level all
echo.
python scripts/run_pyramid_tests.py --level all
if errorlevel 1 (
    echo.
    echo [ERROR] 测试运行失败! 请检查上方错误信息
) else (
    echo.
    echo [OK] 所有测试运行完成
)
echo.
pause
goto :MENU

REM ===== [4] 部署种子数据到 Qdrant Cloud =====
:SEED_QDRANT
cls
echo.
echo   ================================================
echo    [4] 部署种子数据到 Qdrant Cloud
echo   ================================================
echo.
echo [INFO] 正在执行: python scripts/seed_qdrant_cloud.py
echo [WARN] 此操作需要配置 QDRANT_CLOUD_URL 和 QDRANT_CLOUD_KEY 环境变量
echo.
python scripts/seed_qdrant_cloud.py
if errorlevel 1 (
    echo.
    echo [ERROR] 种子数据部署失败! 请检查上方错误信息
) else (
    echo.
    echo [OK] 种子数据部署完成
)
echo.
pause
goto :MENU

REM ===== [5] 退出 =====
:EXIT_MENU
cls
echo.
echo   ================================================
echo    感谢使用 DeepRAG v2.9
echo    再见!
echo   ================================================
echo.
timeout /t 2 >nul
exit /b 0