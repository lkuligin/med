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

# Data parallelism is SGLang's own, so the check is that the launcher exists
# and takes --dp-size, not that a separate router package is present.
from sglang.srt.server_args import prepare_server_args  # noqa: E402

args = prepare_server_args(
    ["--model-path", "/nonexistent", "--dp-size", "4", "--tp-size", "1"])
if args.dp_size != 4:
    raise SystemExit("this sglang does not accept --dp-size 4")

print("dp-size accepted by", sglang.__version__)
print("python", sys.version.split()[0])
