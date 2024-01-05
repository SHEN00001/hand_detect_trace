"""
_*_ coding utf-8 _*_
 @Author: 水煮蛋
 @Email: 3220064177@qq.com
 @FileName: handsDetect.py
 @DateTime: 2024-1-5 下午 07:38
 @SoftWare: PyCharm
"""
import cv2
from collections import defaultdict
from handsDetect import HandDetector

# 创建轨迹队列和跟踪轨迹的帧数
trajectory = defaultdict(list)
max_frames = 50  # 跟踪的帧数
fade_time = 15  # 轨迹消失时间

camera = cv2.VideoCapture(0)
hand_detector = HandDetector()

while True:
    success, img = camera.read() # read()函数会返回两个值，一个是是否成功，一个是获取的图像
    if success:
        img = cv2.flip(img, 1)

        # 处理手部并找到指尖位置
        hand_detector.process(img, draw=False) #要画全手的关节点的话，False改成True
        position = hand_detector.find_position(img)
        left_finger = position['Left'].get(8, None) #食指的关节点序号是8
        right_finger = position['Right'].get(8, None)

        # 清除轨迹队列
        if not left_finger and not right_finger:
            trajectory.clear()

        # 绘制指尖圆圈并更新轨迹队列
        if left_finger:
            cv2.circle(img, (left_finger[0], left_finger[1]), 10, (0, 0, 255), cv2.FILLED)
            trajectory['Left'].append(left_finger)
        if right_finger:
            cv2.circle(img, (right_finger[0], right_finger[1]), 10, (0, 255, 0), cv2.FILLED)
            trajectory['Right'].append(right_finger)

        # 更新轨迹队列
        for hand in trajectory.keys():
            if len(trajectory[hand]) > max_frames:
                trajectory[hand].pop(0)

        # 绘制和消失轨迹
        for hand in trajectory.keys():
            points = trajectory[hand]
            for i in range(1, len(points)):
                start = points[i - 1]
                end = points[i]
                alpha = int(255 * (len(points) - i) / fade_time)
                if alpha < 0:
                    alpha = 0
                cv2.line(img, start, end, (0, 255, 255), 3)
                overlay = img.copy()
                cv2.line(overlay, start, end, (0, 255, 255, alpha), 3)
                img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

        cv2.imshow("camera", img)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
