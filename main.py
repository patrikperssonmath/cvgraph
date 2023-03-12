from cv_graph.trace_generator import TraceGenerator
from cv_graph.covisibility_graph_generator_multi_core import CovisibilityGraphGenerator
import glob
import cv2
import os


class GenericDataset:

    def __init__(self, root, img_path, size) -> None:
        self.root_path = root
        self.path = img_path
        _, self.name = os.path.split(root)
        self.size = size

        self.image_paths = sorted(glob.glob(img_path))

    def __len__(self):
        return len(self.image_paths)

    def get_path(self):
        return self.root_path

    def get_name(self):
        return self.name

    def get_image(self, idx):

        image = cv2.imread(self.image_paths[idx])

        return cv2.resize(image, self.size)


def main():

    dataset = GenericDataset("/database/data/abandonedfactory_sample_P001/P001",
                             "/database/data/abandonedfactory_sample_P001/P001/image_left/*.png",
                             [480, 480])

    trace_gen = TraceGenerator(True)

    result_path = trace_gen.calculate(dataset)

    cv_graph_gen = CovisibilityGraphGenerator()

    cv_graph_gen.calculate(result_path)


if __name__ == "__main__":
    main()
