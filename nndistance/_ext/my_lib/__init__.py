from torch.utils.ffi import _wrap_function

from ._my_lib import ffi as _ffi
from ._my_lib import lib as _lib

__all__ = []


def _import_symbols(locals_vars):
    for symbol in dir(_lib):
        fn = getattr(_lib, symbol)
        locals_vars[symbol] = _wrap_function(fn, _ffi)
        __all__.append(symbol)


_import_symbols(locals())
