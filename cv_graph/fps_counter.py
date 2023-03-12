
import time
from datetime import timedelta


class Timer:
    """ time measuring class  """

    def __init__(self) -> None:
        self.time = time.perf_counter()

    def duration(self):
        """ returns time between creation of object and call to duration  """
        return time.perf_counter() - self.time


class FPS:
    def __init__(self) -> None:
        self.count = 0
        self.timer = None

        self.fps = -1
        self.fps_current = -1

    def tic(self):

        if self.timer == None:
            self.count = 0

            self.timer = Timer()

            self.fps = 1

        else:

            self.count += 1

            self.fps = self.count/self.timer.duration()

        if self.count > 100:

            self.fps_current = self.fps

            self.count = 0
            self.timer = None

    def get_fps(self):
        return self.fps_current

    def time_left(self, idx, total_len):

        return str(timedelta(seconds=(total_len-idx)/self.fps_current))
