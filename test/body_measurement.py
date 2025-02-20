from math import sqrt
import cv2
from matplotlib import pyplot as plt
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True)

def detect_keypoints(image_path):
    landmarks = []

    image = cv2.imread(image_path)
    image_width = image.shape[0]
    image_hight = image.shape[1]
    print(f"image width: {image_width}")
    print(f"image height type: {type(image_hight)}")
    resized_image = cv2.resize(image, (int((image_width / (image_width + image_hight))*300), int((image_hight / (image_width + image_hight))*300)))
    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]

    return landmarks, resized_image.shape


def get_conversion_factor(landmarks, user_height_cm, image_shape):
    height = image_shape[0]
    width = image_shape[1]

    print(f"landmarks type {type(landmarks)}")
    print(f"width {type(width)}")
    print(f"nose landmark: {landmarks[mp_pose.PoseLandmark.NOSE]}")
    
    nose_px = (float(landmarks[mp_pose.PoseLandmark.NOSE][0] * width), float(landmarks[mp_pose.PoseLandmark.NOSE][1] * height))
    left_ear_px = (float(landmarks[mp_pose.PoseLandmark.LEFT_EAR][0] * width), float(landmarks[mp_pose.PoseLandmark.LEFT_EAR][1] * height))
    right_ear_px = (float(landmarks[mp_pose.PoseLandmark.RIGHT_EAR][0] * width), float(landmarks[mp_pose.PoseLandmark.RIGHT_EAR][1] * height))
    left_ankle_px = (float(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE][0] * width), float(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE][1] * height))
    right_ankle_px = (float(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE][0] * width), float(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE][1] * height))

    print(f"nose_px: {nose_px}")
    print(f"left_ear_px: {left_ear_px}")
    print(f"right_ear_px: {right_ear_px}")
    print(f"left_ankle_px: {left_ankle_px}")
    print(f"right_ankle_px: {right_ankle_px}")

    # Estimate forehead height as ~1.5x the nose-to-ear distance
    ear_distance = abs(left_ear_px[0] - right_ear_px[0])
    print(f"ear_distance: {ear_distance}")
    forehead_height_from_nose = float(1.5 * (ear_distance/2))
    print(f"forehead_height_from_nose: {forehead_height_from_nose}")

    # Estimate top of head position by moving up from the nose
    top_of_head_px = (nose_px[0], max(nose_px[1] + forehead_height_from_nose, 0))
    print(f"top_of_head_px: {top_of_head_px}")
    ankle_avg = ((left_ankle_px[0] + right_ankle_px[0]) / 2, (left_ankle_px[1] + right_ankle_px[1]) / 2)
    print(f"ankle_avg: {ankle_avg}")
    
    x1, y1 = top_of_head_px
    x2, y2 = ankle_avg
    print(f"x1: {x1}, y1: {y1}")
    print(f"x2: {x2}, y2: {y2}")
    user_height_px =  abs(y2 - y1)
    print(f"user_height_px: {user_height_px}")

    return user_height_cm / user_height_px


def draw_keypoints(image_path, landmarks):
    image = cv2.imread(image_path)

    for landmark in landmarks:
        x, y, _ = landmark
        cv2.circle(image, (int(x * image.shape[1]), int(y * image.shape[0])), 5, (0, 255, 0), -1)

    return image


def visualize_keypoints_3d(keypoints, title="3D Keypoints"):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], c='b', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    ax.view_init(elev=20, azim=45)
    plt.show()


def measure_size_in_cm(point_1, point_2, image_shape, conversion_factor):
    height = image_shape[0]
    width = image_shape[1]

    x_ls = point_1[0] * width
    y_ls = point_1[1] * height

    x_rs = point_2[0] * width
    y_rs = point_2[1] * height

    return sqrt((x_ls - x_rs) ** 2 + (y_ls - y_rs) ** 2) * conversion_factor


image_path = "C:\PROJECTS\FYP\\body-measurement-model\\images\\user-pose-2.jpg"
user_height_cm = 172.71

landmarks, image_shape = detect_keypoints(image_path)
print(f"\nLandmarks: {landmarks}\n")
conversion_factor = get_conversion_factor(landmarks, user_height_cm, image_shape)
print(f"Conversion Factor: {conversion_factor}")

soulder_width_cm = measure_size_in_cm(
    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], 
    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], 
    image_shape, 
    conversion_factor
)

print(f"Shoulder Width: {soulder_width_cm} cm")

image = draw_keypoints(image_path, landmarks)

fixed_height = 800
fixed_width = 600
resized_image = cv2.resize(image, (fixed_width, fixed_height))

cv2.imshow("Image", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

visualize_keypoints_3d(np.array(landmarks), title="3D Keypoints")

