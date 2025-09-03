# go_odif/utils.py
import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    t0 = time.time()
    yield
    t1 = time.time()
    print(f"[TIMER] {name}: {t1 - t0:.3f}s")
