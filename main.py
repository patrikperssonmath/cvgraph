from cv_graph.trace_generator import TraceGenerator
from cv_graph.covisibility_graph_generator_multi_core import CovisibilityGraphGenerator
import glob
import cv2
import os
import yaml
import numpy as np


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


def main():

    paths = "/database/data/EuRoCMavDatasets/*/"

    for path in glob.glob(paths):

        dataset = GenericDataset(os.path.join(path, "cam0"),
                                 os.path.join(path, "mav0/cam0/data/*.png"),
                                 os.path.join(path, "mav0/cam0/sensor.yaml"),
                                 [480, 480])

        trace_gen = TraceGenerator(False)

        result_path = trace_gen.calculate(dataset)

        cv_graph_gen = CovisibilityGraphGenerator()

        cv_graph_gen.calculate(result_path)


if __name__ == "__main__":
    main()
