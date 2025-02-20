import cv2
import torch

# Load MiDaS for Depth Estimation
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

# Function to estimate depth from a frame
def estimate_depth(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (384, 384))
    frame = torch.tensor(frame).float().unsqueeze(0).permute(0, 3, 1, 2) / 255.0
    with torch.no_grad():
        depth_map = midas(frame)
    return depth_map.squeeze().numpy()