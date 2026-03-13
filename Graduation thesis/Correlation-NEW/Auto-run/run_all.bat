@echo off
cd /d "%~dp0"

echo 开始执行第1个脚本...
python BTH-Spearman.py
echo 第1个脚本执行完成，开始第2个...

python BTH-Mutual-Info.py
echo 第2个脚本执行完成，开始第3个...

python BTH-Geo-detector.py
echo 第3个脚本执行完成，开始第4个...

python YRD-Spearman.py
echo 第4个脚本执行完成，开始第5个...

python YRD-Mutual-Info.py
echo 第5个脚本执行完成，开始第6个...

python YRD-Geo-detector.py
echo 第6个脚本执行完成，开始第7个...

python PRD-Spearman.py
echo 第7个脚本执行完成，开始第8个...

python PRD-Mutual-Info.py
echo 第8个脚本执行完成，开始第9个...

python PRD-Geo-detector.py
echo 第9个脚本执行完成！

echo.
echo 所有9个脚本执行完成！
pause
