import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from datetime import datetime
import time

class MeetingRoomBooker:
    def __init__(self):
        self.url = "http://hjxy-meeting.env.bnu.edu.cn"
        self.username = "07199"
        self.password = "hj2021"
        self.meeting_rooms = [
            "本院-102会议室",
            "本院-103会议室",
            "本院-105会议室",
            "本院-106会议室"
        ]
        self.setup_driver()

    def setup_driver(self):
        chrome_options = Options()
        chrome_options.add_experimental_option("detach", True)
        # 添加一些额外的选项以提高稳定性
        chrome_options.add_argument('--start-maximized')  # 最大化窗口
        chrome_options.add_argument('--disable-gpu')  # 禁用GPU加速
        chrome_options.add_argument('--no-sandbox')  # 禁用沙箱模式
        chrome_options.add_argument('--disable-dev-shm-usage')  # 禁用/dev/shm使用
        
        self.driver = webdriver.Chrome(options=chrome_options)
        # 增加隐式等待时间
        self.driver.implicitly_wait(10)
        # 增加显式等待时间
        self.wait = WebDriverWait(self.driver, 20)

    def login(self):
        try:
            self.driver.get(self.url)
            time.sleep(2)
            
            # 使用placeholder属性定位用户名输入框
            username_input = self.wait.until(
                EC.presence_of_element_located((By.XPATH, "//input[@placeholder='请输入工号']"))
            )
            
            # 使用placeholder属性定位密码输入框
            password_input = self.wait.until(
                EC.presence_of_element_located((By.XPATH, "//input[@placeholder='请输入密码']"))
            )
            
            # 清除并输入账号密码
            username_input.clear()
            password_input.clear()
            
            username_input.send_keys(self.username)
            time.sleep(0.5)
            password_input.send_keys(self.password)
            time.sleep(0.5)
            
            # 尝试多种方式定位登录按钮
            try:
                # 方式1：通过文本内容精确匹配
                login_button = self.wait.until(
                    EC.element_to_be_clickable((By.XPATH, "//*[text()='登录']"))
                )
            except:
                try:
                    # 方式2：通过包含文本内容匹配
                    login_button = self.wait.until(
                        EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), '登录')]"))
                    )
                except:
                    # 方式3：通过value属性匹配
                    login_button = self.wait.until(
                        EC.element_to_be_clickable((By.XPATH, "//button[@value='登录']"))
                    )
            
            # 确保元素可点击后再点击
            self.driver.execute_script("arguments[0].scrollIntoView();", login_button)
            time.sleep(0.5)
            login_button.click()
            
        except Exception as e:
            print(f"登录过程出现错误: {str(e)}")
            print("页面源代码:")
            print(self.driver.page_source)
            raise e

    def handle_slider_verification(self):
        try:
            # 等待验证码加载
            time.sleep(2)
            
            # 获取验证码图片元素
            slider = self.wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "verify-move-block"))
            )
            background = self.wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "verify-img-block"))
            )
            
            # 保存验证码图片
            background.screenshot("background.png")
            slider.screenshot("slider.png")
            
            # 图片处理和缺口识别
            gap_position = self.find_gap()
            
            # 模拟滑动
            self.simulate_drag(slider, gap_position)
            
            # 等待验证结果
            time.sleep(1)
            
        except Exception as e:
            print(f"滑块验证过程出现错误: {str(e)}")
            raise e

    def find_gap(self):
        # 读取图片
        background = cv2.imread("background.png")
        slider = cv2.imread("slider.png")
        
        # 转换为灰度图
        gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        gray_slider = cv2.cvtColor(slider, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊处理
        blur_background = cv2.GaussianBlur(gray_background, (5, 5), 0)
        blur_slider = cv2.GaussianBlur(gray_slider, (5, 5), 0)
        
        # Canny边缘检测
        edges_background = cv2.Canny(blur_background, 100, 200)
        edges_slider = cv2.Canny(blur_slider, 100, 200)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges_slider, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 获取最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # 在背景图中匹配滑块
        result = cv2.matchTemplate(edges_background, edges_slider, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # 计算缺口位置（需要加上滑块的偏移）
        gap_position = max_loc[0] + w//2
        
        return gap_position

    def simulate_drag(self, slider, distance):
        try:
            action = ActionChains(self.driver)
            
            # 先移动到滑块元素
            action.move_to_element(slider)
            # 先点击
            action.click()
            # 再次点击并按住,准备拖动
            action.click_and_hold().perform()
            # 短暂停顿,模拟人类行为
            time.sleep(0.2)
            
            # 更真实的轨迹模拟
            # 先快后慢，并加入微小的Y轴移动来模拟人手抖动
            current = 0
            while current < distance:
                # 随机位移
                x_offset = min(10, distance - current)
                y_offset = np.random.randint(-2, 2)  # 微小的垂直抖动
                
                action.move_by_offset(x_offset, y_offset).perform()
                current += x_offset
                
                # 随机停顿
                time.sleep(np.random.randint(10, 50) / 1000)
            
            # 模拟人在终点附近的微调
            action.move_by_offset(distance - current, 0).perform()
            time.sleep(0.5)
            
            # 释放滑块
            action.release().perform()
            time.sleep(0.5)
            
        except Exception as e:
            print(f"模拟滑动过程出现错误: {str(e)}")
            raise e

    def generate_tracks(self, distance):
        """生成移动轨迹"""
        tracks = []
        current = 0
        # 减速阈值
        mid = distance * 4 / 5
        # 计算间隔
        t = 0.2
        # 初速度
        v = 0
        
        while current < distance:
            if current < mid:
                # 加速度为正2
                a = 2
            else:
                # 加速度为负3
                a = -3
            # 初速度v0
            v0 = v
            # 当前速度v = v0 + at
            v = v0 + a * t
            # 移动距离x = v0t + 1/2 * a * t^2
            move = v0 * t + 1/2 * a * t * t
            # 当前位移
            current += move
            # 加入轨迹
            tracks.append(round(move))
        
        # 返回轨迹数组
        return tracks

    def book_meeting_room(self, date, start_time, end_time):
        # 点击发起会议
        create_meeting_btn = self.wait.until(
            EC.element_to_be_clickable((By.CLASS_NAME, "create-meeting"))
        )
        create_meeting_btn.click()

        # 选择日期
        date_picker = self.wait.until(
            EC.element_to_be_clickable((By.CLASS_NAME, "date-picker"))
        )
        date_picker.click()
        self.select_date(date)

        # 遍历会议室
        for room in self.meeting_rooms:
            self.select_meeting_room(room)
            if self.check_availability(start_time, end_time):
                self.submit_booking()
                return True
                
        return False

    def select_date(self, date):
        # 实现日期选择逻辑
        pass

    def select_meeting_room(self, room):
        # 实现会议室选择逻辑
        pass

    def check_availability(self, start_time, end_time):
        # 实现时间段检查逻辑
        pass

    def submit_booking(self):
        # 实现预定提交逻辑
        pass

def main():
    # 获取用户输入
    date_str = input("请输入预定日期 (YYYY-MM-DD): ")
    start_time = input("请输入开始时间 (HH:MM): ")
    end_time = input("请输入结束时间 (HH:MM): ")

    # 创建预定实例
    booker = MeetingRoomBooker()
    
    try:
        # 登录
        booker.login()
        
        # 处理滑块验证
        booker.handle_slider_verification()
        
        # 预定会议室
        if booker.book_meeting_room(date_str, start_time, end_time):
            print("预定成功！")
        else:
            print("所有会议室在指定时间段都已被预定。")
            
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        # 保持浏览器打开
        input("按回车键关闭浏览器...")
        booker.driver.quit()

if __name__ == "__main__":
    main() 