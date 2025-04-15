# Python绘图颜色选择指南
Python的绘图库，如Matplotlib、Seaborn、Plotly，等的颜色选择非常丰富，包括： 

🌟【CSS Colors】 将CSS颜色值（如十六进制代码）转换为RGB或其他格式，在Python绘图中使用。比如，（也可以直接输入颜色名）非常便捷！ 
🌟【Cyclical Color scales】循环颜色尺度： 对于周期性⏳数据，如季节或时间，循环颜色尺度可以形成一个闭环，首尾相接 
🌟【Tableau Palette】 除了基础颜色，Tableau调色板还提供了更多常用颜色选择，如橙色、粉色、棕色等可以从常用颜色中调配出来的单色 
🌟【Base Colors】 红色🔴、绿色🟢、蓝色🔵…等常用的基础颜色。在Matplotlib等库中，可以直接通过颜色名称或RGB元组来指定 
🌟【Sequential Color scales】顺序颜色尺度 'viridis'🌈、'plasma'🌌...这些内置的顺序颜色尺度非常适合表示连续变量的变化，如热图或等高线图 
🌟【Diverging Color scales】分歧颜色尺度： 'coolwarm'🌡️、'PiYG'🌈...这些颜色尺度用于表示数据从中心值向两个相反方向的变化，非常直观 🎨此外，还可以自定义颜色！通过指定RGB、RGBA或HEX颜色代码，实现个性化的颜色搭配 ✨快来试试这些丰富的颜色选择，为你的Python图表增添更多魅力吧！

让我为您介绍Python主要的绘图工具库：

1. **Matplotlib**
- 最基础和使用最广泛的绘图库
- 提供了类似MATLAB的绘图API
- 适合创建出版质量的图表
- 高度可定制化，但学习曲线相对较陡
- 适用场景：基础统计图表、科学绘图、自定制化要求高的场景

2. **Seaborn**
- 基于Matplotlib的统计绘图库
- 提供了更现代和美观的默认样式
- 专注于统计可视化
- 使用更简单，代码更简洁
- 适用场景：统计分析、数据探索、快速绘制美观的统计图表

3. **Plotly**
- 创建交互式图表的强大工具
- 支持网页端交互
- 可以生成HTML格式的图表
- 适合数据仪表板开发
- 适用场景：交互式数据可视化、Web应用、数据仪表板

4. **Bokeh**
- 专注于Web浏览器中的交互式可视化
- 可以创建复杂的图表和仪表板
- 支持大数据集的可视化
- 适用场景：Web应用中的交互式图表、大数据可视化

5. **Altair**
- 基于Vega和Vega-Lite的声明式可视化库
- 语法简洁，表达力强
- 适合快速创建统计图表
- 适用场景：数据探索、统计分析

6. **PyViz**
- 一个可视化工具的综合生态系统
- 包含多个专门的可视化库（如HoloViews、GeoViews等）
- 适用场景：复杂的数据可视化需求，特别是地理空间数据

图例，配色参考地址：

https://matplotlib.org/

https://seaborn.pydata.org/

https://plotly.com/graphing-libraries/

https://bokeh.org/

https://altair-viz.github.io/

https://pyviz.org/overviews/

