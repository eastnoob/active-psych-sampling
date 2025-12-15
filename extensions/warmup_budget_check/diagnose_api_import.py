#!/usr/bin/env python
# diagnostic helper: check core path and try importing config_models / warmup_api
import sys, traceback
from pathlib import Path
base = Path(__file__).parent
core = base / 'core'
print('BASE:', base)
print('CORE DIR:', core)
print('CORE EXISTS:', core.exists())
if core.exists():
    print('CORE LISTING:')
    for p in sorted(core.iterdir()):
        print(' -', p.name)

# ensure core is on sys.path
sys.path.insert(0, str(core))
print('\nSYS.PATH[0]=', sys.path[0])

for name in ('config_models', 'warmup_api'):
    print('\n--- Trying import:', name)
    try:
        mod = __import__(name)
        print('Imported', name, 'from', getattr(mod, '__file__', None))
        # print first 10 lines of file for quick sanity
        try:
            p = Path(mod.__file__)
            print('File exists:', p.exists())
            if p.exists():
                print('---- File head ----')
                for i, line in enumerate(p.read_text(encoding='utf-8').splitlines()):
                    print(line)
                    if i >= 9:
                        break
                print('---- End head ----')
        except Exception as e:
            print('Could not read file:', e)
    except Exception:
        print('Import failed for', name)
        traceback.print_exc()

print('\nDone diagnostic.')
