import numpy as np
import cv2


class Renderer:
    def __init__(self, focal_length=5000, img_res=224, faces=None):
        self.focal_length = focal_length
        self.img_res = img_res
        self.faces = faces

    def __call__(self, vertices, camera_translation, image=None):
        """
        Simple rendering of vertices onto image plane
        """
        vertices = vertices.copy()

        # Project 3D points to 2D
        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]

        camera_translation = camera_translation.reshape(3, 1)

        # Perspective projection
        x = (
            self.focal_length
            * (x + camera_translation[0])
            / (z + camera_translation[2])
        )
        y = (
            self.focal_length
            * (y + camera_translation[1])
            / (z + camera_translation[2])
        )

        # Center the image
        x = x + self.img_res / 2.0
        y = y + self.img_res / 2.0

        # Create output image
        if image is None:
            image = np.ones((self.img_res, self.img_res, 3))

        # Stack all points
        points = np.stack([x, y], axis=-1)

        # Draw mesh if faces are provided
        if self.faces is not None:
            for face in self.faces:
                # Get points for this face
                pts = points[face]

                # Check if all points are valid (within image bounds)
                valid = (pts >= 0) & (pts < self.img_res)
                if valid.all():
                    # Convert to integer coordinates
                    pts = pts.astype(np.int32).reshape((-1, 1, 2))
                    if len(pts) > 2:  # Need at least 3 points for a triangle
                        cv2.fillPoly(image, [pts], color=(0.7, 0.7, 0.9))

        return image
