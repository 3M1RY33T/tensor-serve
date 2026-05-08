"""Compatibility entry point for the FastAPI app.

The canonical API application now lives in :mod:`api.main`. This module keeps
existing commands such as ``uvicorn main:app`` and older tests working.
"""

from importlib import import_module as _import_module
import sys as _sys

_module = _import_module("api.main")
_sys.modules[__name__] = _module
