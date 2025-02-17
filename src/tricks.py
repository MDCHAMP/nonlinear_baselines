import contextlib
import time


@contextlib.contextmanager
def timer(time_func=time.perf_counter, message="Complete in {:.4g}"):
    t0 = time_func()
    try:
        yield
    finally:
        t1 = time_func()
        elapsed = t1 - t0
        print(message.format(elapsed))


def flatten(dic, prefix=""):
    if prefix != "":
        prefix = prefix + "."
    result = {}
    for k, v in dic.items():
        if isinstance(v, dict):
            for k1, v1 in flatten(v, prefix + k).items():
                result[k1] = v1
        else:
            result[prefix + k] = v
    return result
