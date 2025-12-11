# PyTest Closed Stream Issue - Fix Plan

## Problem Summary
Pytest fails with `ValueError: I/O operation on closed file` during session teardown. The error occurs in `_pytest/capture.py:591` when pytest tries to read captured output.

## Root Cause
**The issue is NOT about sys.stdout/sys.stderr being closed directly.** Instead:

1. Many test scripts rebind `sys.stdout`/`sys.stderr` with new `io.TextIOWrapper` instances:
   ```python
   sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
   sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
   ```

2. When pytest runs with capture enabled, it replaces `sys.stdout`/`sys.stderr` with special capture objects that have internal `tmpfile` attributes for capturing output.

3. When these scripts rebind stdout/stderr, they create NEW TextIOWrapper instances wrapping the SAME buffer (`sys.stdout.buffer`).

4. The OLD TextIOWrapper (pytest's capture object) is dereferenced and gets garbage collected.

5. When the old TextIOWrapper's `__del__` finalizer runs, it closes the underlying buffer/file descriptor, which ALSO closes pytest's internal `tmpfile`.

6. Later, when pytest tries to read captured output via `self.tmpfile.seek(0)`, it fails with the closed file error.

## Evidence
The following files all contain the problematic pattern (grep result shows 27+ instances):
- `tests/is_EUR_work/archive/*.py`
- `tests/is_EUR_work/*.py`
- `tests/is_EUR_work/00_plans/251206/scripts/*.py`

Stack trace confirms the issue is in pytest's capture mechanism, not terminal writer:
```
File "_pytest/capture.py", line 591, in snap
    self.tmpfile.seek(0)
ValueError: I/O operation on closed file.
```

## Solution Strategy

### Option 1: Conditional Encoding Fix (Recommended)
Modify all files to only rebind stdout/stderr when NOT running under pytest:

```python
import sys
import io

# Only rebind stdout/stderr when not running under pytest
if sys.platform == 'win32' and not hasattr(sys, '_called_from_test'):
    # Check if we're running under pytest
    import sys
    if 'pytest' not in sys.modules:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
```

### Option 2: Use Non-Closing Wrapper (Alternative)
Create a custom wrapper that doesn't close the underlying stream:

```python
class NonClosingTextIOWrapper(io.TextIOWrapper):
    """TextIOWrapper that doesn't close the underlying buffer when finalized."""
    def close(self):
        # Flush but don't close the underlying buffer
        try:
            self.flush()
        except Exception:
            pass

    def __del__(self):
        # Override __del__ to prevent closing on garbage collection
        try:
            self.flush()
        except Exception:
            pass

if sys.platform == 'win32':
    sys.stdout = NonClosingTextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = NonClosingTextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
```

### Option 3: Global conftest.py Fix (Complementary)
Add a pytest fixture in `conftest.py` that prevents the rebinding entirely:

```python
@pytest.fixture(scope="session", autouse=True)
def prevent_stdout_rebinding():
    """Prevent test modules from rebinding stdout/stderr during pytest runs."""
    # Mark that we're running under pytest
    sys._pytest_active = True
    yield
    # Cleanup marker
    if hasattr(sys, '_pytest_active'):
        delattr(sys, '_pytest_active')
```

## Implementation Steps

1. **Remove the complex detection code from conftest.py** (lines 14-510)
   - The current detection/wrapping code is trying to catch the wrong issue
   - Keep only the simple `pytest_unconfigure` hook to ensure streams aren't closed

2. **Create a utility module for safe encoding setup**
   - Location: `tests/is_EUR_work/utils/encoding_utils.py`
   - Provide `setup_utf8_encoding()` function that safely handles pytest context

3. **Update all affected files** (27+ files identified)
   - Replace direct `io.TextIOWrapper` rebinding with call to utility function
   - Or add pytest detection logic to each file

4. **Verify the fix**
   - Run: `pytest -k "test_config_validation" -v`
   - Run: `pytest -s` (full test suite with output capture disabled)
   - Ensure exit code is 0

## Files to Modify

### Priority 1: Create Utility Module
```
tests/is_EUR_work/utils/encoding_utils.py (NEW FILE)
```

### Priority 2: Simplify conftest.py
```
conftest.py (SIMPLIFY - remove lines 14-510, keep only basic pytest hooks)
```

### Priority 3: Update All Test Scripts
All files containing the pattern `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, ...)`:
- tests/is_EUR_work/archive/*.py (multiple files)
- tests/is_EUR_work/*.py (multiple files)
- tests/is_EUR_work/00_plans/251206/scripts/*.py (multiple files)

## Testing Plan

1. **Unit test the fix**: Create a minimal test that reproduces the issue
2. **Integration test**: Run pytest on actual test suite
3. **Regression test**: Ensure encoding still works correctly when NOT under pytest

## Expected Outcome

- Pytest runs complete with exit code 0
- No `ValueError: I/O operation on closed file` errors
- Output capture works correctly during tests
- UTF-8 encoding still works when scripts run outside pytest

## Timeline Estimate

- Utility module creation: 15 minutes
- conftest.py cleanup: 10 minutes
- Update all affected files: 30-45 minutes (can be scripted)
- Testing and verification: 20 minutes

**Total: ~90 minutes**

---

**Next Action**: Implement Option 1 (conditional encoding fix) with the utility module approach for cleaner, maintainable code.
