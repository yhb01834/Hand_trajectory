# Hand_trajectory
<img src="Screenshot from 2023-06-19 20-08-21.png"  width="700" height="370">
손가락으로 1을 가리키면 궤적 그리기를 시작하고 5를 펼치면 멈추게 됩니다.

[영상 속 코드](#ros_hand_trajectory.py)

'''
def vector_2d_angle(v1, v2):
    d_v1_v2 = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos = v1.dot(v2) / (d_v1_v2)
    sin = np.cross(v1, v2) / (d_v1_v2)
    angle = np.degrees(np.arctan2(sin, cos))
    return angle

def distance(point_1, point_2):
    return math.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)
'''

'''
#!/usr/bin/python3
# coding=utf8
import cv2
import time

class FPS:
    def __init__(self):
        self.fps = 0.0
        self.last_time = 0
        self.current_time = 0
        self.confidence = 0.1

    def update(self):
        self.last_time = self.current_time
        self.current_time = time.time()
        new_fps = 1.0 / (self.current_time - self.last_time)
        if self.fps == 0.0:
            self.fps = new_fps if self.last_time != 0 else 0.0
        else:
            self.fps = new_fps * self.confidence + self.fps * (1.0 - self.confidence)
        return float(self.fps)

    def show_fps(self, img):
        line = cv2.LINE_AA
        font = cv2.FONT_HERSHEY_PLAIN
        fps_text = 'FPS: {:.2f}'.format(self.fps)
        cv2.putText(img, fps_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
        cv2.putText(img, fps_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)
        return img
'''
