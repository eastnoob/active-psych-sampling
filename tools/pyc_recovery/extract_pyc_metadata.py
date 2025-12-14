#!/usr/bin/env python3
"""Extract all available metadata from .pyc bytecode files.

Even when decompilation fails, we can extract:
- Function and class names
- Variable names
- String constants
- Import names
- Code structure
"""

import sys
import marshal
import dis
from pathlib import Path
from types import CodeType

def extract_code_info(code_obj, indent=0):
    """Recursively extract information from a code object."""
    prefix = "  " * indent
    info = []

    # Basic code object info
    info.append(f"{prefix}=== {code_obj.co_name} ===")
    info.append(f"{prefix}Filename: {code_obj.co_filename}")
    info.append(f"{prefix}Line: {code_obj.co_firstlineno}")
    info.append(f"{prefix}Arguments: {code_obj.co_argcount}")
    info.append(f"{prefix}Local vars: {code_obj.co_nlocals}")

    # Variables
    if code_obj.co_varnames:
        info.append(f"{prefix}Variables: {', '.join(code_obj.co_varnames[:20])}")
        if len(code_obj.co_varnames) > 20:
            info.append(f"{prefix}... and {len(code_obj.co_varnames) - 20} more")

    # Names (imported/referenced)
    if code_obj.co_names:
        info.append(f"{prefix}Names: {', '.join(code_obj.co_names[:30])}")
        if len(code_obj.co_names) > 30:
            info.append(f"{prefix}... and {len(code_obj.co_names) - 30} more")

    # Constants (literals, nested functions)
    if code_obj.co_consts:
        info.append(f"{prefix}Constants ({len(code_obj.co_consts)} total):")
        for i, const in enumerate(code_obj.co_consts[:10]):
            if isinstance(const, CodeType):
                info.append(f"{prefix}  [{i}] <code object {const.co_name}>")
            elif isinstance(const, str):
                preview = const[:60] + "..." if len(const) > 60 else const
                info.append(f"{prefix}  [{i}] str: {repr(preview)}")
            elif const is not None:
                info.append(f"{prefix}  [{i}] {type(const).__name__}: {repr(const)[:60]}")

    info.append("")

    # Recursively process nested code objects (functions/classes)
    nested_codes = [c for c in code_obj.co_consts if isinstance(c, CodeType)]
    if nested_codes:
        info.append(f"{prefix}--- Nested functions/classes ---")
        for nested in nested_codes:
            info.extend(extract_code_info(nested, indent + 1))

    return info

def extract_from_pyc(pyc_path):
    """Extract information from a .pyc file."""
    pyc_path = Path(pyc_path)
    print(f"\n{'='*80}")
    print(f"Analyzing: {pyc_path.name}")
    print(f"{'='*80}\n")

    try:
        with open(pyc_path, 'rb') as f:
            # Skip the .pyc header (16 bytes in Python 3.7+)
            f.read(16)

            # Load the marshalled code object
            code = marshal.load(f)

            # Extract information
            info = extract_code_info(code)
            for line in info:
                print(line)

            # Also print a disassembly of the main module
            print("\n" + "="*80)
            print(f"Disassembly of {code.co_name} (first 100 instructions):")
            print("="*80 + "\n")

            # Capture disassembly
            import io
            buf = io.StringIO()
            dis.dis(code, file=buf)
            disasm = buf.getvalue()

            # Print first 100 lines
            lines = disasm.split('\n')
            for line in lines[:100]:
                print(line)
            if len(lines) > 100:
                print(f"\n... and {len(lines) - 100} more lines of bytecode")

    except Exception as e:
        print(f"Error processing {pyc_path}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_pyc_metadata.py <pyc_file> [<pyc_file2> ...]")
        sys.exit(1)

    for pyc_file in sys.argv[1:]:
        extract_from_pyc(pyc_file)
