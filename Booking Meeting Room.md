我的目标：
    1. 完成一个会议室预定项目
    2. 打开网页：http://hjxy-meeting.env.bnu.edu.cn
    3. 使用Chrome浏览器完成
    4. 完成自动输入账号，密码，选择会议室，选择时间，提交

工具要求：
    Python  selenium ChromeBrowser 
    操作已经打开的浏览器

步骤：
    1. 打开浏览器，并访问：http://hjxy-meeting.env.bnu.edu.cn
    2. 找到工号输入框，输入工号:07199
    3. 找到密码输入框，输入密码:hj2021
    4. 找到登录按钮，点击登录
    5. 找到发起会议按钮选择框，点击发起会议
    6. 找到日期选择按钮，点击选择日期，选择日期可以在程序内设定，每次运行前，手动输入
    7. 找到会议室选择按钮，点击选择会议室，选择会议室可以在程序内设定，只选择这几个选项，其余的不选：
       “本院-102会议室”
       “本院-103会议室”
       “本院-105会议室”
       “本院-106会议室”
       
       依次遍历
    8. 找到查询按钮，点击查询
    9. 选择对应日期的开始时间点和结束时间点，这个需要程序内设定，每次运行前，手动输入
    10. 如果没有查询到对应的时间段，向下依次遍历
    11. 如果查询到对应的时间段，点击预定
    12. 找到确定按钮，点击确定

备注：
    1. 预约之后不可以取消
    2.  通过滑块登录的步骤
    获取验证码图片
        识别图片，计算轨迹距离
        寻找滑块，控制滑动