import json
import os

import cv2
import numpy as np
from cv_graph.fps_counter import FPS


class TraceGenerator:
    def __init__(self, debug=False, max_trace_len=-1) -> None:
        self.last_frame = None
        self.min_features = 60
        self.debug = debug
        self.point_window = 30
        self.chunks = 4
        self.corner_density = 300
        self.max_trace_len = max_trace_len
        self.min_frame_motion = 0.5  # pixels max
        self.min_points = 20
        self.ransac_th = 3.0
        self.fps = FPS()

        self.feature_params = dict(maxCorners=80,
                                   qualityLevel=0.01,
                                   minDistance=10
                                   )

        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def check_if_processed(self, flow_path):

        if not os.path.exists(flow_path):
            os.makedirs(flow_path)

            return False

        finished_file_marker = os.path.join(flow_path, "finished_traces")

        if os.path.exists(finished_file_marker):
            print(f"file processed, skipping {flow_path}")
            return True

        return False

    def store_traces(self, result_path, traces):

        for trace in traces:

            trace["points"] = [np.squeeze(point).tolist()
                               for point in trace["points"]]

        json_path = os.path.join(result_path, "traces.json")

        print(f"\nWriting traces to {json_path}!")

        with open(json_path, "w", encoding="utf8") as file:

            json.dump(traces, file)

        finished_file_marker = os.path.join(result_path, "finished_traces")

        with open(finished_file_marker, "w", encoding="utf8") as file:
            file.write("finished")

    def get_max_frame_len(self, dataset, max_len):

        if max_len == -1:
            max_len = len(dataset)

        return min(len(dataset), max_len)

    def calculate(self, dataset, max_len=-1, skip=0):

        result_path = os.path.join(
            dataset.get_path(), dataset.get_name(), "traces")

        if self.check_if_processed(result_path):
            return result_path

        stored_traces = []
        trace_list = []

        max_len = self.get_max_frame_len(dataset, max_len)

        for index in range(skip, max_len):

            image = dataset.get_image(index)

            trace_list, finised_traces = self.track(trace_list, image, index)

            stored_traces.extend(finised_traces)

            if (index % 100) == 0:

                print(f"{dataset.get_name()} processig {index} of {max_len}. fps {self.fps.get_fps():.2f}, time left: {self.fps.time_left(index, self.get_max_frame_len(dataset, max_len))}. tracking {len(trace_list)} traces. stored traces {len(stored_traces)}", end="\n")

            self.fps.tic()

        stored_traces.extend(trace_list)

        trace_list = []

        self.store_traces(result_path, stored_traces)

        print(f"\n{dataset.get_name()} done! :)")

        return result_path

    def fill_mask(self, traces, mask):

        for trace in traces:

            point = trace["points"][-1].round().astype(np.int32)

            mask = cv2.rectangle(
                mask, point[0]-self.point_window//2, point[0]+self.point_window//2, 0, -1)

        H, W = mask.shape

        cw = W//self.chunks
        ch = H//self.chunks

        for x in range(self.chunks):
            for y in range(self.chunks):

                mask_part = mask[y*ch:(y+1)*ch, x*cw:(x+1)*cw]

                val = np.mean(mask_part)

                if val > 0.8:

                    #print(f"empty area {x}, {y}, val {val}")
                    return mask, True

        return mask, False

    def initialize(self, traces, img, idx):

        feature_points = cv2.goodFeaturesToTrack(
            img, mask=None, **self.feature_params)

        for point in feature_points:

            traces.append({"keys": [idx], "points": [point]})

        self.last_frame = img

        return traces, []

    def track(self, traces, img, idx):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.last_frame is None:
            return self.initialize(traces, img, idx)

        if len(traces) < self.min_features:
            new_traces = []
            new_traces, _ = self.initialize(new_traces, img, idx)

            return new_traces, traces

        st, p1, skip = self.optical_tracker(traces, img)

        if skip:
            return traces, []

        if st is None:

            return [], traces

        active_traces = [trace for ind, trace in zip(st, traces) if ind == 1]
        finished_traces = [trace for ind, trace in zip(st, traces) if ind == 0]

        for i, point in enumerate(p1):
            active_traces[i]["keys"].append(idx)
            active_traces[i]["points"].append(point)

        active_traces, removed_traces = self.remove_long_traces(active_traces)

        finished_traces.extend(removed_traces)

        active_traces, removed_traces = self.filter_traces(active_traces)

        finished_traces.extend(removed_traces)

        mask, active_traces = self.extract_corners(active_traces, img, idx)

        if self.debug:
            self.debug_output(p1, img, mask)

        return active_traces, finished_traces

    def remove_long_traces(self, traces):

        if self.max_trace_len > -1:

            active_traces = [trace for trace in traces if len(
                trace["keys"]) <= self.max_trace_len]
            finished_traces = [trace for trace in traces if len(
                trace["keys"]) > self.max_trace_len]

            return active_traces, finished_traces

        return traces, []

    def optical_tracker(self, traces, img):

        p0 = []

        for trace in traces:

            p0.append(trace["points"][-1])

        p0 = np.stack(p0, axis=0)

        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            self.last_frame, img, p0, None, **self.lk_params)

        self.last_frame = img

        if p1 is None:

            return None, None, False

        st = np.squeeze(st)

        p0 = p0[st == 1]
        p1 = p1[st == 1]

        st = st.tolist()

        skip = False

        max_dist = np.max(np.linalg.norm(p0-p1, axis=-1))

        if max_dist < self.min_frame_motion:
            skip = True

        return st, p1, skip

    def find_long_traces(self, traces):

        long_traces = traces
        short_traces = []

        index = -1

        while len(long_traces) > (self.min_points + 10):
            tmp_long = []
            tmp_short = []

            for trace in traces:

                if -len(trace["points"]) > index:
                    tmp_short.append(trace)
                else:
                    tmp_long.append(trace)

            if len(tmp_long) > (self.min_points + 10):

                long_traces = tmp_long
                short_traces = tmp_short

                index -= 1

            else:

                index += 1

                break

        return long_traces, short_traces, index

    def filter_traces(self, traces):

        long_traces, short_traces, index = self.find_long_traces(traces)

        active_traces, finished_traces = self.filter_traces_(
            long_traces, index)

        active_traces.extend(short_traces)

        return active_traces, finished_traces

    def filter_traces_(self, traces, start_idx):

        if len(traces) < self.min_points:
            return traces, []

        p0 = []
        p1 = []

        start_idx_check = -1

        for trace in traces:

            if start_idx_check == -1:
                start_idx_check = trace["keys"][start_idx]

            elif start_idx_check != trace["keys"][start_idx]:
                Exception("filter_traces_: different start keys ")

            p0.append(trace["points"][start_idx])
            p1.append(trace["points"][-1])

        p0 = np.stack(p0, axis=0)
        p1 = np.stack(p1, axis=0)

        if len(p0) < self.min_points:
            return traces, []

        _, mask = cv2.findFundamentalMat(p0, p1, cv2.RANSAC, self.ransac_th)

        if mask is None:
            return [], traces

        mask = mask.ravel().tolist()

        active_traces = [trace for ind, trace in zip(mask, traces) if ind == 1]
        finished_traces = [trace for ind,
                           trace in zip(mask, traces) if ind == 0]

        return active_traces, finished_traces

    def extract_corners(self, traces, img, idx):

        mask = np.ones_like(img)

        mask, extract_corners = self.fill_mask(traces, mask)

        if len(traces) < self.min_features or extract_corners:

            H, W = mask.shape

            nbr_corners = int((self.corner_density/(H*W))*np.sum(mask))

            self.feature_params["maxCorners"] = nbr_corners

            feature_points = cv2.goodFeaturesToTrack(
                img, mask=mask, **self.feature_params)

            if feature_points is not None:
                for point in feature_points:

                    traces.append({"keys": [idx], "points": [point]})

        return mask, traces

    def debug_output(self, p1, img, mask):

        disp = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)

        for point in p1.astype(np.int32).tolist():

            disp = cv2.circle(disp, point[0], 2, (0, 255, 0), 1)

        cv2.imshow('features', disp)

        cv2.imshow('mask', mask*255)

        cv2.waitKey(1)
