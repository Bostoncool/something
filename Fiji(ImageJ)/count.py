# 导入 ImageJ 和相关的库
from ij import IJ, Prefs
from ij.plugin.frame import RoiManager
from ij.measure import Measurements
from ij.plugin.filter import ParticleAnalyzer

# 获取当前活动的图像
imp = IJ.getImage()

# 1. 将图像转换为8位
# run("8-bit");
IJ.run(imp, "8-bit", "")

# 2. 设置自动阈值
# setAutoThreshold("Default dark no-reset");
# 在Jython中，这通常通过直接调用 IJ.setAutoThreshold 来完成
# "Default" 是标准方法, "dark" 指的是暗背景。
# "no-reset" 在脚本环境中通常是默认行为，所以不需要特别设置。
IJ.setAutoThreshold(imp, "Default dark")

# 3. 关闭操作 (形态学)
# run("Close");
IJ.run(imp, "Close-", "")

# 4. 设置黑色背景选项
# setOption("BlackBackground", true);
Prefs.blackBackground = True

# 5. 转换为蒙版 (Mask)
# run("Convert to Mask");
# 这个命令会基于当前的阈值创建一个二值图像
IJ.run(imp, "Convert to Mask", "")

# 注意：原始脚本中有两次 "Convert to Mask"。
# 这通常是多余的，因为第一次转换后图像已经是蒙版了。
# 如果确实需要特定效果，可以保留第二次调用，但这里我只保留了一次。
# IJ.run(imp, "Convert to Mask", "")

# 6. 分水岭算法
# run("Watershed");
IJ.run(imp, "Watershed", "")

# 7. 分析颗粒
# run("Analyze Particles...", "size=2000-Infinity show=Outlines display clear include summarize");

# 获取结果表
rt = ResultsTable.getResultsTable()
if rt is None:
    rt = ResultsTable()

# 设置颗粒分析器的参数
# 对应 "size=2000-Infinity"
min_size = 2000
max_size = float('inf')

# 对应 "show=Outlines" "display" "clear" "include" "summarize"
# SHOW_OUTLINES = 8, DISPLAY_RESULTS = 32, CLEAR_WORKSHEET = 64, INCLUDE_HOLES = 128, SUMMARIZE = 1
options = (ParticleAnalyzer.SHOW_OUTLINES |
           ParticleAnalyzer.DISPLAY_RESULTS |
           ParticleAnalyzer.CLEAR_WORKSHEET |
           ParticleAnalyzer.INCLUDE_HOLES |
           ParticleAnalyzer.SUMMARIZE)

# 创建并运行颗粒分析器
pa = ParticleAnalyzer(options, Measurements.ALL_STATS, rt, min_size, max_size)
pa.analyze(imp)

# 显示结果
rt.show("Results")