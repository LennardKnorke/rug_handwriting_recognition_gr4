import numpy as np
import torch
from typing import Callable, Dict


class Pipeline:

    def __init__(self, start: str, processes: Dict={}):
        self.processes = processes
        self.links = {}
        self.links.setdefault([])
        self.start = start

    def add_process(self, key: str, func: Callable):
        self.processes[key] = func

    def add_link(self, key_src: str, key_dest: str):
        self.links[key_src].append(key_dest)

    def __call__(self, args: dict=None):
        vals = [[]]
        queue = [(self.start, 0)]
        idx = 0

        while queue:
            process, val_idx = queue.pop(0)
            if val_idx > idx:
                idx += 1

            queue.append(
                zip(
                    *self.links[process],
                    [idx for _ in range(len(self.links[process]))]
                )
            )

            vals.append(
                self.processes
            )


