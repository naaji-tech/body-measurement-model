import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import json


from lib.spin.models import hmr
from lib.spin.utils.imutils import crop
from lib.spin import config
from lib.spin import constants


def _bbox_from_openpose(openpose_file, rescale=1.2, detection_thresh=0.2):
    """Get center and scale for bounding box from openpose detections."""
    with open(openpose_file, "r") as f:
        keypoints = json.load(f)["people"][0]["pose_keypoints_2d"]
    keypoints = np.reshape(np.array(keypoints), (-1, 3))
    valid = keypoints[:, -1] > detection_thresh
    valid_keypoints = keypoints[valid][:, :-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale


def _bbox_from_json(bbox_file):
    """Get center and scale of bounding box from bounding box annotations.
    The expected format is [top_left(x), top_left(y), width, height].
    """
    with open(bbox_file, "r") as f:
        bbox = np.array(json.load(f)["bbox"]).astype(np.float32)
    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    # make sure the bounding box is rectangular
    return center, scale


def process_image(img_file, bbox_file, openpose_file, input_res=224):
    """
    Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img = cv2.imread(img_file)[
        :, :, ::-1
    ].copy()  # PyTorch does not support negative stride at the moment
    if bbox_file is None and openpose_file is None:
        # Assume that the person is centerered in the image
        height = img.shape[0]
        width = img.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width) / 200
    else:
        if bbox_file is not None:
            center, scale = _bbox_from_json(bbox_file)
        elif openpose_file is not None:
            center, scale = _bbox_from_openpose(openpose_file)
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img


def extract_beta(
    img_path: str,
    checkpoint_path="data/model_checkpoint.pt",
    bbox=None,
    openpose=None,
) -> torch.Tensor:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Running on device: %s" % device)

    # Load pretrained model
    model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    print("Loading pretrained model from %s" % checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    # Preprocess input image and generate predictions
    img, norm_img = process_image(img_path, bbox, openpose, input_res=constants.IMG_RES)
    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))

    print()
    print("pred_rotmat from the model")
    print(pred_rotmat)
    print()
    print("pred_camera from the model")
    print(pred_camera)
    print()
    print("pred_betas from the model")
    print(pred_betas)
    print()

    return pred_betas
