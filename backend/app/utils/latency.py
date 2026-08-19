import time
from contextlib import contextmanager
from typing import Generator

class StageTimer:
    def __init__(self):
        self.elapsed_ms: float = 0.0

    @contextmanager
    def measure(self) -> Generator[None, None, None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self.elapsed_ms = (end - start) * 1000.0
