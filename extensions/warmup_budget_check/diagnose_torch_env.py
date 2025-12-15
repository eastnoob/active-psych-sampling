#!/usr/bin/env python
import sys, subprocess, traceback
from pathlib import Path
import platform

print('Python executable:', sys.executable)
print('Python version:', sys.version)
print('Platform:', platform.platform())
import struct
print('Pointer size (bits):', struct.calcsize('P') * 8)

# pip show torch
print('\n--- pip show torch ---')
try:
    import pkgutil
    import importlib
    torch_spec = pkgutil.find_loader('torch')
    print('torch spec:', torch_spec)
except Exception as e:
    print('pkgutil error:', e)

try:
    import importlib.metadata as importlib_metadata
except Exception:
    try:
        import importlib_metadata
    except Exception:
        importlib_metadata = None

if importlib_metadata:
    try:
        info = importlib_metadata.metadata('torch')
        print('torch metadata:')
        for k in ('Name','Version','Summary','Home-page'):
            print(' ', k, ':', info.get(k))
    except Exception as e:
        print('importlib metadata error:', e)

# locate torch package
print('\n--- locating torch package files ---')
try:
    import torch
    print('imported torch from', getattr(torch, '__file__', None))
    tpath = Path(torch.__file__).resolve().parent
    print('torch package dir:', tpath)
    libdir = tpath / 'lib'
    print('torch lib dir exists:', libdir.exists())
    if libdir.exists():
        dlls = list(libdir.glob('*.dll'))
        print('DLL count in lib:', len(dlls))
        for d in dlls[:20]:
            print(' -', d.name, d.stat().st_size)
        # check for shm.dll
        shm = libdir / 'shm.dll'
        print('shm.dll exists:', shm.exists())
        if shm.exists():
            print('shm.dll size:', shm.stat().st_size)
except Exception as e:
    print('Import torch failed:')
    traceback.print_exc()

# attempt import to capture full exception
print('\n--- try import torch to capture error details ---')
try:
    import torch
    print('torch version:', torch.__version__)
    print('cuda available:', torch.cuda.is_available())
except Exception as e:
    print('Exception during import:')
    traceback.print_exc()

print('\n--- ldd-like check of torch lib shm.dll existence only (Windows) ---')
# On Windows we cannot run ldd; just check DLL deps not available. Recommend next steps.
print('Done.')
