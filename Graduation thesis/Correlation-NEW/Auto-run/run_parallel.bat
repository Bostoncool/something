@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

if not exist logs mkdir logs

echo =============== 开始并行执行 9 个脚本 ===============
echo.

start /b cmd /c "python BTH-Spearman.py > logs\BTH-Spearman_log.txt 2>&1 & echo. > logs\BTH-Spearman.done"
start /b cmd /c "python BTH-Mutual-Info.py > logs\BTH-Mutual-Info_log.txt 2>&1 & echo. > logs\BTH-Mutual-Info.done"
start /b cmd /c "python BTH-Geo-detector.py > logs\BTH-Geo-detector_log.txt 2>&1 & echo. > logs\BTH-Geo-detector.done"
start /b cmd /c "python YRD-Spearman.py > logs\YRD-Spearman_log.txt 2>&1 & echo. > logs\YRD-Spearman.done"
start /b cmd /c "python YRD-Mutual-Info.py > logs\YRD-Mutual-Info_log.txt 2>&1 & echo. > logs\YRD-Mutual-Info.done"
start /b cmd /c "python YRD-Geo-detector.py > logs\YRD-Geo-detector_log.txt 2>&1 & echo. > logs\YRD-Geo-detector.done"
start /b cmd /c "python PRD-Spearman.py > logs\PRD-Spearman_log.txt 2>&1 & echo. > logs\PRD-Spearman.done"
start /b cmd /c "python PRD-Mutual-Info.py > logs\PRD-Mutual-Info_log.txt 2>&1 & echo. > logs\PRD-Mutual-Info.done"
start /b cmd /c "python PRD-Geo-detector.py > logs\PRD-Geo-detector_log.txt 2>&1 & echo. > logs\PRD-Geo-detector.done"

echo 已启动 9 个脚本，等待全部完成...
echo.

:wait_loop
set count=0
for %%f in (logs\*.done) do set /a count+=1
if !count! lss 9 (
    timeout /t 2 /nobreak >nul
    goto wait_loop
)

del logs\*.done 2>nul
echo.
echo =============== 所有 9 个脚本并行执行完成 ===============
echo 日志已保存至 logs 文件夹
echo.
pause
