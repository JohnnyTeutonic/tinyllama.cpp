try:
    from .tinyllama_bindings import *

    _all_symbols = [s for s in dir(tinyllama_bindings) if not s.startswith('_')]
    __all__ = _all_symbols

except ImportError as e:
    import sys
    print("Could not import 'tinyllama_bindings' C++ extension. "
          "If you are building from source, please make sure the C++ code is compiled.", file=sys.stderr)
    print(f"ImportError: {e}", file=sys.stderr)
