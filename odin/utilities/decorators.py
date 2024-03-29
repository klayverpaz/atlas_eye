import functools
import time
from typing import Callable, Optional


def timer(func: Optional[Callable] = None, *, prefix: str = "", return_execution_time: bool = False):
    """
    Decorator to measure the execution time of a function.

    Parameters:
        func (Optional[Callable]):
            The function to be decorated. Defaults to None.
        prefix (str):
            A prefix to be added to the output message. Defaults to ''.
        return_execution_time (bool):
            Whether to return the execution time. Defaults to False.

    Returns:
        Callable:
            The decorated function.
    """

    if func is None:
        return functools.partial(timer, prefix=prefix, return_execution_time=return_execution_time)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        if return_execution_time:
            return result, execution_time
        else:
            print(f"{prefix}{func.__name__} took {execution_time:.2f} seconds to complete.")
            return result

    return wrapper
