
import glob
import yaml
import numpy as np
import cv2
import os

class GenericDataset:

    def __init__(self, root, img_path, config_path, size) -> None:
        self.root_path = root
        self.path = img_path
        _, self.name = os.path.split(root)
        self.size = size

        self.image_paths = sorted(glob.glob(img_path))
        self.map1 = None
        self.map2 = None

        if os.path.exists(config_path):
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)

            calibration = np.array(config["intrinsics"], dtype=np.float32)

            distortion = np.array(
                config["distortion_coefficients"],  dtype=np.float32)

            camera_matrix = np.eye(3, 3, dtype=np.float32)

            camera_matrix[0, 0] = calibration[0]
            camera_matrix[1, 1] = calibration[1]
            camera_matrix[0, 2] = calibration[2]
            camera_matrix[1, 2] = calibration[3]

            resolution = np.array(config["resolution"])

            newcameramatrix, _ = cv2.getOptimalNewCameraMatrix(
                camera_matrix, distortion, resolution, 0, resolution)

            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                camera_matrix, distortion, None, newcameramatrix, resolution, 5)

    def __len__(self):
        return len(self.image_paths)

    def get_path(self):
        return self.root_path

    def get_name(self):
        return ""  # self.name

    def get_image(self, idx):

        image = cv2.imread(self.image_paths[idx])

        if self.map1 is not None and self.map2 is not None:

            image = cv2.remap(image,
                              self.map1,
                              self.map2,
                              cv2.INTER_CUBIC)

        return cv2.resize(image, self.size)
