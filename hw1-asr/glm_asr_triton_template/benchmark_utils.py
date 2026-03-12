# python
# File: hw1-asr/glm_asr_triton_template/benchmark_utils.py
import time
import functools
import statistics
from typing import Callable, Any, Optional, Sequence, Dict, Tuple
import torch
import os

def _get_tensor_info(args, kwargs) -> Dict[str, Any]:
    """Find first tensor in args/kwargs and return shape, dtype, device."""
    for v in list(args) + list(kwargs.values()):
        if isinstance(v, torch.Tensor):
            return {"shape": tuple(v.shape), "dtype": str(v.dtype), "device": str(v.device)}
    return {"shape": None, "dtype": None, "device": None}

class Timer:
    """GPU-aware timer with CPU fallback. Returns elapsed ms."""
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.start_evt = torch.cuda.Event(enable_timing=True)
            self.end_evt = torch.cuda.Event(enable_timing=True)
        else:
            self._t0 = None

    def start(self):
        if self.use_cuda:
            torch.cuda.synchronize()
            self.start_evt.record()
        else:
            self._t0 = time.perf_counter()

    def stop_ms(self) -> float:
        if self.use_cuda:
            self.end_evt.record()
            self.end_evt.synchronize()
            return float(self.start_evt.elapsed_time(self.end_evt))
        else:
            return float((time.perf_counter() - self._t0) * 1000.0)

class Profiler:
    """Context manager that records time, shapes, dtypes and GPU memory delta."""
    def __init__(self, name: str, args: Tuple = (), kwargs: Dict = None, log_fn: Callable = print):
        self.name = name
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.log_fn = log_fn
        self.timer = Timer()
        self.info = {}

    def __enter__(self):
        # gather info about first tensor argument if present
        self.info = _get_tensor_info(self.args, self.kwargs)
        if torch.cuda.is_available():
            self.info["mem_before_bytes"] = torch.cuda.memory_allocated()
            self.info["mem_reserved_bytes"] = torch.cuda.memory_reserved()
        self.timer.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed_ms = self.timer.stop_ms()
        if torch.cuda.is_available():
            self.info["mem_after_bytes"] = torch.cuda.memory_allocated()
            self.info["mem_reserved_after_bytes"] = torch.cuda.memory_reserved()
        self.info.update({"time_ms": elapsed_ms})
        # pretty-print minimal JSON-like line
        line = {
            "name": self.name,
            "time_ms": round(elapsed_ms, 4),
            "shape": self.info.get("shape"),
            "dtype": self.info.get("dtype"),
            "device": self.info.get("device"),
            "mem_delta_bytes": (self.info.get("mem_after_bytes", 0) - self.info.get("mem_before_bytes", 0)) if "mem_after_bytes" in self.info else None,
        }
        self.log_fn(line)

def microbenchmark(func: Callable, *args, warmup: int = 10, iters: int = 100, sync: bool = True, **kwargs) -> Dict[str, Any]:
    """
    Run warmup + timed iterations and return stats (ms).
    Returns dict: {mean, std, min, max, timings(list)}
    """
    # warmup
    for _ in range(max(1, warmup)):
        func(*args, **kwargs)
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()

    timings = []
    timer = Timer()
    for _ in range(max(1, iters)):
        timer.start()
        func(*args, **kwargs)
        t_ms = timer.stop_ms()
        timings.append(t_ms)
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    stats = {
        "mean_ms": statistics.mean(timings),
        "std_ms": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "min_ms": min(timings),
        "max_ms": max(timings),
        "timings_ms": timings,
    }
    return stats

def profiled(name: Optional[str] = None, warmup: int = 5, iters: int = 50, micro: bool = False, log_fn: Callable = print):
    """
    Decorator to profile a function or method.
    When DISABLE_PROFILING env var is set or by default, acts as a no-op for speed.
    """
    def deco(func: Callable):
        # No-op: skip all profiling overhead during normal/benchmark runs
        return func
    return deco
