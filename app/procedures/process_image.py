import argparse
from matplotlib import pyplot as plt
import mediapipe as mp
import cv2
import numpy as np

# ===== Load MediaPipe Pose model =====
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True)

# ===== Main function =====
def process_image(image_path):
    """
    Process an image and extract keypoints using MediaPipe Pose model
    """
    # ===== Ensure portrait orientation of the image =====
    def ensure_portrait(frame):
        frame_np = np.array(frame)
        if frame_np.shape[1] > frame_np.shape[0]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame

    frame = cv2.imread(image_path)
    frame = ensure_portrait(frame)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        keypoints = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]

    return frame, keypoints


# ===================================
# ========= Test main ===============
# ===================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing function of image processing.")
    parser.add_argument(
        "--test_process_image",
        action="store_true",
        help="Measure a mean shape smpl model.",
    )
    args = parser.parse_args()

    if args.test_process_image:
        """
        Testing fuction for process_image
        """

        # ===== Function implementation =====
        image_path = "C:\\PROJECTS\FYP\\body-measurement-model\\images\\user-pose-1.jpg"
        frame, keypoints = process_image(image_path)
        print(f"Data type of keypoints: {type(keypoints)}")


        # ===== list to numpy array =====
        keypoints_np = np.array(keypoints)
        print(f"shape of keypoints: {keypoints_np.shape}")


        # ===== plot keypoints to a 3d graph =====
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(keypoints_np[:, 0], keypoints_np[:, 1], keypoints_np[:, 2], c='b', marker='o')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("3D Keypoints")

        ax.view_init(elev=20, azim=45)
        plt.show()