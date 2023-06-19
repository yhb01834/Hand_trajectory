# Hand_trajectory
<img src="Screenshot from 2023-06-19 20-08-21.png"  width="700" height="370">
손가락으로 1을 가리키면 궤적 그리기를 시작하고 5를 펼치면 멈추게 됩니다.


[영상 속 코드](ros_hand_trajectory.py)
해당 코드를 돌리려면 ros 관련 파일들이 필요해 바로 확인하기는 어려울 것입니다.
</br>
Colab에서 mediapipe의 hand_landmarks를 직접 확인하실 수 있습니다. 
[go_to_ipynb](Hand_Detection.ipynb)


다음은 추가적으로 필요한 라이브러리 내용입니다.

#### from jetauto_sdk.common import vector_2d_angle, distance 
```
def vector_2d_angle(v1, v2):
    d_v1_v2 = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos = v1.dot(v2) / (d_v1_v2)
    sin = np.cross(v1, v2) / (d_v1_v2)
    angle = np.degrees(np.arctan2(sin, cos))
    return angle

def distance(point_1, point_2):
    return math.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)
```

#### import jetauto_sdk.fps as fps
```
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
```



