try:
    from ._version import __version__
except ImportError:
    # Fallback version if _version.py doesn't exist (development mode)
    __version__ = "0.1.0+dev"

try:
    from .tinyllama_bindings import *

    _all_symbols = [s for s in dir(tinyllama_bindings) if not s.startswith('_')]
    __all__ = _all_symbols + ['__version__']

except ImportError as e:
    import sys
    print("Could not import 'tinyllama_bindings' C++ extension. "
          "If you are building from source, please make sure the C++ code is compiled.", file=sys.stderr)
    print(f"ImportError: {e}", file=sys.stderr)
    
    # Still export __version__ even if bindings fail to import
    __all__ = ['__version__']
