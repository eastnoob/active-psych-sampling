import sys
import os

# Record that conftest was imported and where debug files will be written.
try:
    _repo_debug_dir = os.path.join(os.path.dirname(__file__), "debug")
    os.makedirs(_repo_debug_dir, exist_ok=True)
    with open(
        os.path.join(_repo_debug_dir, "conftest-loaded.txt"), "w", encoding="utf-8"
    ) as _cf:
        _cf.write("conftest imported\n")
except Exception:
    pass


def pytest_unconfigure(config):
    """Ensure stdout/stderr are not closed when pytest finishes.

    Some tests or modules may close the global streams; when pytest tries
    to write its final output this can raise ValueError: I/O operation on
    closed file. As a pragmatic mitigation we replace any closed stream
    with a devnull file so pytest can finish cleanly.
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is not None and getattr(stream, "closed", False):
            try:
                setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
            except Exception:
                pass
