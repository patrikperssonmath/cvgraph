from cv_graph.trace_generator import TraceGenerator
from cv_graph.covisibility_graph_generator_multi_core import CovisibilityGraphGenerator
import glob
import os

from generic_dataset import GenericDataset

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
