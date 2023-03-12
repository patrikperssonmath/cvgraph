import json
import os

import cv2
import numpy as np
import copy
import pickle

import csv


def query_graph(edges, k1, k2):

    edge = (k1, k2)

    if k1 > k2:
        edge = (k2, k1)

    if edge in edges:

        return edges[edge]

    return None


def exists_in(graph, k1, k2):
    edge = (k1, k2)

    if k1 > k2:
        edge = (k2, k1)

    if edge in graph:
        return True

    return False


def filter_covisible(k1, co, edges):
    return [k2 for k2 in co if exists_in(edges, k1, k2)]


def filter_edges(flow, min_flow, max_flow):

    return (flow > min_flow) and (flow < max_flow)


def filter_missing_keys(co, covis_frames):

    return [k2 for k2 in co if k2 in covis_frames]


class ImageSetGeneration:
    def __init__(self, min_flow=0.05, max_flow=0.30, img_nbr=4, graph_name="graph") -> None:
        self.min_flow = min_flow
        self.max_flow = max_flow
        self.img_nbr = img_nbr
        self.graph_name = graph_name

    def check_if_processed(self, flow_path):

        if not os.path.exists(flow_path):
            os.makedirs(flow_path)

            return False

        finished_file_marker = os.path.join(
            flow_path, "finished_filtered_graph")

        if os.path.exists(finished_file_marker):
            print(f"file processed, skipping {flow_path}")
            return True

        return False

    def load_flow_graph(self, result_path):

        pkl_path = os.path.join(result_path, "edges.pkl")

        print(f"\nLoading edges from {pkl_path}!")

        with open(pkl_path, "rb") as file:

            edges = pickle.load(file)

        pkl_path = os.path.join(result_path, "covis.pkl")

        print(f"\nLoading covisibility graph from {pkl_path}!")

        with open(pkl_path, "rb") as file:

            covisibility_graph = pickle.load(file)

        return edges, covisibility_graph

    def store_filtered_graph(self, result_path, edges, covisibility_graph):

        pkl_path = os.path.join(result_path, "edges_filtered.pkl")

        print(f"\nWriting edges to {pkl_path}!")

        with open(pkl_path, "wb") as file:

            pickle.dump(edges, file)

        pkl_path = os.path.join(result_path, "covis_filtered.pkl")

        print(f"\nWriting covisibility graph to {pkl_path}!")

        with open(pkl_path, "wb") as file:

            pickle.dump(covisibility_graph, file)

        finished_file_marker = os.path.join(
            result_path, "finished_filtered_graph")

        with open(finished_file_marker, "w", encoding="utf8") as file:
            file.write("finished")

    def filter_bad_sets(self, path):

        _, name = os.path.split(path)

        graph_path = os.path.join(path, self.graph_name)

        """
        if self.check_if_processed(graph_path):
            return None 
        """

        edges, covisibility_graph = self.load_flow_graph(graph_path)

        covisibility_graph, edges = self.prefilter(covisibility_graph, edges)

        covisibility_graph = self.filter_images_with_sufficient_covisible_set(
            covisibility_graph)

        covisibility_graph = {k1: filter_missing_keys(
            co, covisibility_graph) for k1, co in covisibility_graph.items()}

        self.store_filtered_graph(graph_path, edges, covisibility_graph)

        print(f"\n{name} done! :)")

    def prefilter(self, covis_frames, edges):

        edges = {key: val for key, val in edges.items() if filter_edges(
            val, self.min_flow, self.max_flow)}

        covis_frames = {k1: filter_covisible(
            k1, co, edges) for k1, co in covis_frames.items()}

        change = True
        while change:
            change = False
            # remove keys with too few connections
            covis_frames_new = {
                k1: co for k1, co in covis_frames.items() if len(co) >= self.img_nbr}

            if len(covis_frames_new) < len(covis_frames):
                change = True

            # remove keys in covisible sets that were removed in previous step
            covis_frames_new = {k1: filter_missing_keys(
                co, covis_frames_new) for k1, co in covis_frames_new.items()}

            covis_frames = covis_frames_new

        covis_frames = {k1: set(co) for k1, co in covis_frames.items()}

        return covis_frames, edges

    def filter_images_with_sufficient_covisible_set(self, covisibility_graph):

        good_covisible_images = {}

        for idx, (k1, co) in enumerate(covisibility_graph.items()):

            if self.good_set_exists(k1, set({k1}), co, covisibility_graph):

                good_covisible_images[k1] = co

            if idx % 100 == 0:
                print(
                    f" idx {idx} of {len(covisibility_graph)} img set len {len(good_covisible_images)}", end="\n")

        return good_covisible_images

    def good_set_exists(self, k1, img_set, candidates, covisibility_graph):

        if len(img_set) >= self.img_nbr:
            return True

        cand_int = candidates.intersection(covisibility_graph[k1])

        for c in cand_int:

            img_set_cpy = copy.deepcopy(img_set)
            candidates_cpy = copy.deepcopy(cand_int)

            img_set_cpy.add(c)
            candidates_cpy.remove(c)

            if self.good_set_exists(c, img_set_cpy, candidates_cpy, covisibility_graph):
                return True

        return False