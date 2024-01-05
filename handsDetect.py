"""
_*_ coding utf-8 _*_
 @Author: 水煮蛋
 @Email: 3220064177@qq.com
 @FileName: handsDetect.py
 @DateTime: 2024-1-5 下午 07:38
 @SoftWare: PyCharm
"""
import cv2
import mediapipe as mp

class HandDetector():
    def __init__(self):
        self.hand_detector = mp.solutions.hands.Hands()
        self.drawer = mp.solutions.drawing_utils

    def process(self,img,draw = True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转变色彩模式，opencv的色彩模式是GBR，转换成RGB
        self.hands_data = self.hand_detector.process(img_rgb)
        if draw:
            if self.hands_data.multi_hand_landmarks:  # 当有手时，multi_hand_landmarks是一个包含20个关节点的数组
                for handlms in self.hands_data.multi_hand_landmarks:
                    self.drawer.draw_landmarks(img, handlms, mp.solutions.hands.HAND_CONNECTIONS)

    def find_position(self,img):
        h,w,c=img.shape
        position = {'Left':{},'Right':{}} #建立一个字典存储手指点的位置
        if self.hands_data.multi_hand_landmarks:  #如果有手
            i = 0
            for point in self.hands_data.multi_handedness: #关节点在multi_hand_landmarks中
                score = point.classification[0].score #是左手或者右手的可能性
                if score >= 0.8:
                    label = point.classification[0].label #label就是左手或者右手
                    hand_lms = self.hands_data.multi_hand_landmarks[i].landmark #landmark包括一只手的20个标记点，id就是第几个关节点
                    for id,lm in enumerate(hand_lms):
                        x,y = int(lm.x*w),int (lm.y*h)
                        position[label][id] = (x,y)
                i=i+1
        return position

