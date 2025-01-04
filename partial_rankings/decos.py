"""Module containing useful decorators"""

import time
from datetime import timedelta
import functools


def timeit(func):
    """Decorator that times execution of a function"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        scores_df = func(*args, **kwargs)
        exec_time = time.time() - start
        exec_time = timedelta(seconds=exec_time)
        # print(f"Execution time: {exec_time}\n")

        return scores_df, exec_time

    return wrapper


def count_calls(func):
    """Decorator that counts number of times a function is called"""

    def wrapper(*args, **kwargs):
        wrapper.num_calls += 1
        return func(*args, **kwargs)

    wrapper.num_calls = 0
    return wrapper
