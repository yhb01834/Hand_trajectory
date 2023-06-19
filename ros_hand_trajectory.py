#!/usr/bin/env python3
# encoding: utf-8


### 출처 : JetAuto
import cv2
import enum
import rospy
import numpy as np
import faulthandler
import mediapipe as mp
import jetauto_sdk.fps as fps
from sensor_msgs.msg import Image
from jetauto_interfaces.msg import Points, PixelPosition
from jetauto_sdk.common import vector_2d_angle, distance

faulthandler.enable()

def vector_2d_angle(v1, v2):
    d_v1_v2 = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos = v1.dot(v2) / (d_v1_v2)
    sin = np.cross(v1, v2) / (d_v1_v2)
    angle = np.degrees(np.arctan2(sin, cos))
    return angle

def distance(point_1, point_2):
    return math.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)

def get_hand_landmarks(img, landmarks):
    """
    medipipe 출력에서 랜드마크를 픽셀 좌표로 변환
    :param img: 픽셀 좌표에 해당하는 이미지
    :return:
    """
    h, w, _ = img.shape
    landmarks = [(lm.x * w, lm.y * h) for lm in landmarks]
    return np.array(landmarks)

def hand_angle(landmarks):
    """
    각 손가락의 굽어진 각도 계산
    :param landmarks: 손 키 포인트 21개
    :return: 각 손가락의 각도
    """
    angle_list = []
    # thumb 엄지
    angle_ = vector_2d_angle(landmarks[3] - landmarks[4], landmarks[0] - landmarks[2])
    angle_list.append(angle_)
    # index 검지
    angle_ = vector_2d_angle(landmarks[0] - landmarks[6], landmarks[7] - landmarks[8])
    angle_list.append(angle_)
    # middle 중지
    angle_ = vector_2d_angle(landmarks[0] - landmarks[10], landmarks[11] - landmarks[12])
    angle_list.append(angle_)
    # ring 약지
    angle_ = vector_2d_angle(landmarks[0] - landmarks[14], landmarks[15] - landmarks[16])
    angle_list.append(angle_)
    # pink 소지
    angle_ = vector_2d_angle(landmarks[0] - landmarks[18], landmarks[19] - landmarks[20])
    angle_list.append(angle_)
    angle_list = [abs(a) for a in angle_list]
    return angle_list

def h_gesture(angle_list):
    """
    2D 기능에서 손가락 제스처 결정
    :param angle_list: 각 손가락이 구부러지는 각도
    :return : 제스처 이름
    """
    thr_angle = 65.
    thr_angle_thumb = 53.
    thr_angle_s = 49.
    gesture_str = "none"
    if (angle_list[0] > thr_angle_thumb) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (
            angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
        gesture_str = "fist" 
    elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (
            angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
        gesture_str = "hand_heart"
    elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (
            angle_list[3] > thr_angle) and (angle_list[4] < thr_angle_s):
        gesture_str = "nico-nico-ni"
    elif (angle_list[0] < thr_angle_s) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (
            angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
        gesture_str = "hand_heart"
    elif (angle_list[0] > 5) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (
            angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
        gesture_str = "one"
    elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
            angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
        gesture_str = "two"
    elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
            angle_list[3] < thr_angle_s) and (angle_list[4] > thr_angle):
        gesture_str = "three"
    elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] > thr_angle) and (angle_list[2] < thr_angle_s) and (
            angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle_s):
        gesture_str = "OK"
    elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
            angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle_s):
        gesture_str = "four"
    elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
            angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle_s):
        gesture_str = "five"
    elif (angle_list[0] < thr_angle_s) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (
            angle_list[3] > thr_angle) and (angle_list[4] < thr_angle_s):
        gesture_str = "six"
    else:
        "none"
    return gesture_str

class State(enum.Enum):
    NULL = 0
    START = 1
    TRACKING = 2
    RUNNING = 3

def draw_points(img, points, thickness=4, color=(0, 0, 255)):
    points = np.array(points).astype(dtype=np.int)
    if len(points) > 2:
        for i, p in enumerate(points):
            if i + 1 >= len(points):
                break
            cv2.line(img, p, points[i + 1], color, thickness)

class HandTrajectoryNode:
    def __init__(self, name):
        rospy.init_node(name, anonymous=True)

        self.drawing = mp.solutions.drawing_utils

        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_tracking_confidence=0.05,
            min_detection_confidence=0.6
        )

        self.name = name
        self.running = True
        self.image = None
        self.fps = fps.FPS()  # fps
        self.state = State.NULL
        self.points = []
        self.count = 0

        use_depth_cam = rospy.get_param('~use_depth_cam', False)
        if use_depth_cam:
            camera = rospy.get_param('/depth_camera/camera_name', 'camera')  
            rospy.Subscriber('/%s/rgb/image_raw' % camera, Image, self.image_callback)
        else:
            camera = rospy.get_param('/usb_cam_name', 'usb_cam')  
            rospy.Subscriber('/%s/image_raw' % camera, Image, self.image_callback)  
        self.point_publisher = rospy.Publisher(self.name + '/points', Points, queue_size=1)

        self.image_proc()

    def image_proc(self):
        points_list = []
        while self.running:
            if self.image is not None:
                image_flip = cv2.flip(self.image, 1)
                self.image = None
                bgr_image = cv2.cvtColor(image_flip, cv2.COLOR_RGB2BGR)
                try:
                    cv2.waitKey(10)
                    results = self.hand_detector.process(image_flip)
                    if results is not None and results.multi_hand_landmarks:
                        gesture = "none"
                        index_finger_tip = [0, 0]
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.drawing.draw_landmarks(
                                bgr_image,
                                hand_landmarks,
                                mp.solutions.hands.HAND_CONNECTIONS)
                            landmarks = get_hand_landmarks(image_flip, hand_landmarks.landmark)
                            angle_list = (hand_angle(landmarks))
                            gesture = (h_gesture(angle_list))
                            index_finger_tip = landmarks[8].tolist()
                        if self.state != State.TRACKING:
                            if gesture == "one":  #검지 손가락 제스처 감지, 손가락 끝 추적 시작
                                self.count += 1
                                if self.count > 5:
                                    self.count = 0
                                    self.state = State.TRACKING
                                    self.points = []
                                    points_list = []
                            else:
                                self.count = 0

                        if gesture == "five":
                            self.state = State.NULL
                            if points_list:
                                points = Points()
                                points.points = points_list
                                self.point_publisher.publish(points)
                            self.points = []
                            points_list = []
                            draw_points(bgr_image, self.points)


                        if gesture == "ok":
                            print("ok")
                        if gesture == "four":
                            print("four")
                        if gesture =="nico-nico-ni":
                            print("nico-nico-ni")
                except Exception as e:
                    print(e)
                self.fps.update()
                result_image = self.fps.show_fps(bgr_image)
                cv2.imshow(self.name, cv2.resize(result_image, (640, 480)))
                key = cv2.waitKey(1)
                if key != -1:
                    break

    def image_callback(self, ros_image):
        rgb_image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
        self.image = rgb_image

if __name__ == "__main__":
    
    HandTrajectoryNode('hand_trajectory')
