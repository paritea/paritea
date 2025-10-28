from functools import wraps
from typing import TypeVar, ParamSpec, Callable

T = TypeVar("T")
P = ParamSpec("P")


def canonicalize_input(**arg_converters):
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            import inspect

            bound = inspect.signature(func).bind(*args, **kwargs)
            bound.apply_defaults()
            for arg, converter in arg_converters.items():
                if arg in bound.arguments:
                    bound.arguments[arg] = converter(bound.arguments[arg])
            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator
