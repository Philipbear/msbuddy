import inspect
from typing import Callable

global dependencies
dependencies = dict()


# This is to create a dependency injection system
# The dependencies are stored in a global variable, and can be set by calling set_dependency

def set_dependency(**kwargs):
    globals.dependencies.update(kwargs)


# The idea is to use the decorator @with_dependency to inject dependencies into functions
def with_dependency(fn: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        kwargs.update(globals.dependencies)
        sigs = [p.name for p in inspect.signature(fn).parameters.values()]
        params = {k: v for k, v in kwargs.items() if k in sigs}
        remains = set(sigs) - set(params.keys())
        if remains and len(args) != len(remains):
            raise KeyError("{} hasn't been set previously!".format(list(remains)[len(args):]))
        return fn(*args, **params)

    wrapper.__name__ = fn.__name__
    return wrapper
