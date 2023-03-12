import json
import os

import cv2
import numpy as np

import pickle
from cv_graph.fps_counter import FPS
import multiprocessing


def run_job(data):

    p0, p1, edge, min_points, min_point_flow, scale, ransac_th, max_homography_agreement = data

    success, distance = check_edge(
        p0, p1, min_points, min_point_flow, scale, ransac_th, max_homography_agreement)

    return (edge, success, distance)


def check_edge(p0, p1, min_points, min_point_flow, scale, ransac_th, max_homography_agreement):

    p0 = np.stack(p0, axis=0)
    p1 = np.stack(p1, axis=0)

    if len(p0) < min_points:
        return False, 0

    distance = np.mean(np.linalg.norm((p1-p0)/scale, axis=-1))

    if distance < min_point_flow:
        return False, 0

    _, mask = cv2.findFundamentalMat(p0, p1, cv2.RANSAC, ransac_th)

    if mask is None:
        return False, 0

    inliers = mask.ravel() == 1

    p0 = p0[inliers]
    p1 = p1[inliers]

    if len(p0) < min_points:
        return False, 0

    _, mask = cv2.findHomography(p0, p1, cv2.RANSAC, ransac_th)

    if mask is None:
        return False, 0

    mask_mean = mask.astype(np.float32).mean()

    if mask_mean > max_homography_agreement:
        return False, 0

    distance = np.mean(np.linalg.norm((p1-p0)/scale, axis=-1))

    return True, distance


class CovisibilityGraphGenerator:
    def __init__(self, min_points=20, min_point_flow=0.01, max_homography_agreement=0.75, ransac_th=3.0, graph_name="graph") -> None:

        self.min_points = min_points
        self.min_point_flow = min_point_flow
        self.max_homography_agreement = max_homography_agreement
        self.ransac_th = ransac_th
        self.fps = FPS()
        self.success_count = 0
        self.fail_count = 0
        self.graph_name = graph_name

        self.scale = np.array([320, 320], dtype=np.float32).reshape(1, 1, 2)

    def check_if_processed(self, flow_path):

        if not os.path.exists(flow_path):
            os.makedirs(flow_path)

            return False

        finished_file_marker = os.path.join(flow_path, "finished_graph")

        if os.path.exists(finished_file_marker):
            print(f"file processed, skipping {flow_path}")
            return True

        return False

    def load_traces(self, result_path):

        json_path = os.path.join(result_path, "traces.json")

        print(f"\nReading traces from {json_path}!")

        with open(json_path, "r", encoding="utf8") as file:

            traces = json.load(file)

        for trace in traces:

            trace["points"] = [np.expand_dims(
                np.array(point), 0) for point in trace["points"]]

        return traces

    def store_flow_graph(self, result_path, edges, covisibility_graph):

        pkl_path = os.path.join(result_path, "edges.pkl")

        print(f"\nWriting edges to {pkl_path}!")

        with open(pkl_path, "wb") as file:

            pickle.dump(edges, file)

        pkl_path = os.path.join(result_path, "covis.pkl")

        print(f"\nWriting covisibility graph to {pkl_path}!")

        with open(pkl_path, "wb") as file:

            pickle.dump(covisibility_graph, file)

        finished_file_marker = os.path.join(result_path, "finished_graph")

        with open(finished_file_marker, "w", encoding="utf8") as file:
            file.write("finished")

    def calculate(self, trace_path):

        path, _ = os.path.split(trace_path)

        graph_path = os.path.join(path, self.graph_name)

        if self.check_if_processed(graph_path):
            return None

        traces = self.load_traces(trace_path)

        frames, traces = self.extract_frames(traces)

        edges, covisibility_graph = self.generate_flow_graph(
            frames, traces, trace_path)

        self.store_flow_graph(graph_path, edges, covisibility_graph)

        print(f"\n{trace_path} done! :)")

    def extract_frames(self, traces):

        frames = {}

        for idx, trace in enumerate(traces):

            trace["map"] = {k: v for k, v in zip(
                trace["keys"], trace["points"])}

            for key in trace["keys"]:

                if key not in frames:

                    frames[key] = {"coframe": set(
                        trace["keys"]), "trace": [idx]}
                else:
                    frames[key]["trace"].append(idx)
                    frames[key]["coframe"].update(trace["keys"])

            if (idx % 100) == 0:
                print(
                    f"frames: trace {idx} of {len(traces)}. total frames {len(frames)}")

        return frames, traces

    def generate_flow_graph(self, frames, traces, name):

        visited_edges = set()

        edges = {}

        with multiprocessing.Pool() as pool:

            jobs = []

            for frame_idx, (k1, v) in enumerate(frames.items()):

                for k2 in v["coframe"]:

                    edge = (k1, k2)

                    if k1 > k2:
                        edge = (k2, k1)

                    if edge in visited_edges:
                        continue

                    visited_edges.add(edge)

                    p0 = []
                    p1 = []

                    for trace_idx in v["trace"]:

                        trace = traces[trace_idx]
                        point_map = trace["map"]

                        if k2 in point_map:

                            p0.append(point_map[k1])
                            p1.append(point_map[k2])

                    jobs.append((p0, p1, edge, self.min_points, self.min_point_flow,
                                self.scale, self.ransac_th, self.max_homography_agreement))

                jobs, edges = self.process_jobs(jobs, edges, pool)

                if (frame_idx % 100) == 0:
                    print(f"{name} processig {frame_idx} of {len(frames)}. edges {len(edges)}. total visited edges: {len(visited_edges)} fps {self.fps.get_fps():.2f}, time left: {self.fps.time_left(frame_idx, len(frames))}. S: {self.success_count} F:{self.fail_count}", end="\n")

                self.fps.tic()

            jobs, edges = self.process_jobs(jobs, edges, pool, True)

        return edges, {key: v["coframe"] for key, v in frames.items()}

    def process_jobs(self, jobs, edges, pool, last=False):

        if len(jobs) > 1000 or last:

            for res in pool.map(run_job, jobs):

                edge, success, distance = res

                if success:
                    edges[edge] = distance

                    self.success_count += 1
                else:
                    self.fail_count += 1

            jobs = []

        return jobs, edges
