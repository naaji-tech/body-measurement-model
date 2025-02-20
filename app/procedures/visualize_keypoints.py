import cv2

def visualize_keypoints(video_path, keypoints_list, output_video="./output/output_keypoints.mp4"):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(keypoints_list):
            keypoints = keypoints_list[frame_idx]
            for (x, y, z) in keypoints:
                x_px, y_px = int(x * frame_width), int(y * frame_height)  # Scale back to pixel coordinates
                cv2.circle(frame, (x_px, y_px), 5, (0, 255, 0), -1)  # Draw keypoint

            out.write(frame)  # Save frame to video
            cv2.imshow("Keypoints", frame)

        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Keypoints visualization saved as {output_video}")