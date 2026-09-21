"""Confirm the serving environment can actually launch a data-parallel server.

Run by setup_env.sh. Kept as a file rather than inline in the shell script so
that it can be read, linted and run on its own after a partial install.
"""

import sys

try:
    import sglang
except ImportError as error:
    raise SystemExit(f"sglang did not install: {error}")

print("sglang", sglang.__version__)

try:
    import sglang_router
except ImportError:
    raise SystemExit(
        "sglang-router did not install, so --dp-size has no launcher; "
        "the serving topology this package uses needs it")

print("sglang-router", getattr(sglang_router, "__version__", "installed"))
print("python", sys.version.split()[0])
