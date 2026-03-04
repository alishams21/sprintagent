"""Utilities for parallel execution in isolated processes."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable

# Type for result: (success, return_value or error_message)
ResultT = tuple[bool, Any]


def run_parallel_isolated(
    tasks: list[tuple[str, Callable, dict]],
    max_workers: int,
    return_values: bool = False,
) -> dict[str, ResultT]:
    """Run tasks in isolated processes with fault tolerance.

    Each task runs in a separate process. If one fails, others continue.
    All task arguments must be picklable (no non-picklable objects).

    Args:
        tasks: List of (task_id, func, kwargs). func(**kwargs) is called in worker.
        max_workers: Maximum number of concurrent processes.
        return_values: If True, successful results are returned; if False,
            only (True, None) or (False, error_message) per task.

    Returns:
        Dict mapping task_id to (success: bool, result_or_error).
        If return_values is True and success, result_or_error is the return value;
        otherwise it is the exception string on failure.
    """
    results: dict[str, ResultT] = {}
    task_ids = [t[0] for t in tasks]
    funcs = [t[1] for t in tasks]
    kwargs_list = [t[2] for t in tasks]

    def run_one(idx: int) -> tuple[str, bool, Any]:
        task_id = task_ids[idx]
        func = funcs[idx]
        kwargs = kwargs_list[idx]
        try:
            out = func(**kwargs)
            return (task_id, True, out if return_values else None)
        except Exception as e:
            return (task_id, False, str(e))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_one, i): task_ids[i] for i in range(len(tasks))}
        for future in as_completed(futures):
            task_id, success, value = future.result()
            results[task_id] = (success, value)

    return results
