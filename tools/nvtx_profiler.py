import inspect
from functools import wraps
from torch.cuda import nvtx

def nvtx_profiler(prefix: str):
    """
    Wraps all methods defined in the current class with as NVTX ranges with the name: `{prefix}.{method-name}`.

    :param prefix: prefix to use
    """
    
    def inner(cls):
        def profile(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                nvtx.range_push(f"{prefix}.{func.__name__}")
                try:
                    out = func(*args, **kwargs)
                finally:
                    nvtx.range_pop()
                return out
            return wrapper
        
        for name, attr in list(vars(cls).items()):
            if inspect.isfunction(attr):
                setattr(cls, name, profile(attr))
        
        return cls

    return inner
