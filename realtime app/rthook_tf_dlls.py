import ctypes
import os
import sys

if sys.platform == 'win32' and getattr(sys, 'frozen', False):
    exe_dir = os.path.dirname(sys.executable)
    _internal = os.path.join(exe_dir, '_internal')

    # Add directories to search path
    for d in [
        _internal,
        os.path.join(_internal, 'tensorflow'),
        os.path.join(_internal, 'tensorflow', 'python'),
        os.path.join(_internal, 'tensorflow', 'lite', 'python'),
    ]:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass

    # Pre-load TF DLLs with full path so Windows caches them in process
    tf_dlls = [
        os.path.join(_internal, 'tensorflow', 'tensorflow_framework.2.dll'),
        os.path.join(_internal, 'tensorflow', 'tensorflow_cc.2.dll'),
        os.path.join(_internal, 'tensorflow', 'python', '_pywrap_tensorflow_common.dll'),
    ]
    for dll in tf_dlls:
        if os.path.exists(dll):
            try:
                ctypes.CDLL(dll)
            except Exception:
                pass
