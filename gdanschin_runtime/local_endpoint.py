"""Check the locally served endpoint before a run commits to it.

    from gdanschin_runtime.local_endpoint import require_ready
    require_ready(BASE_MODELS["gemma-4-26b-local"])

Without this, a run against a server that is down or serving a different model
fails deep inside the agent, one question at a time, after the dataset has
loaded - and a run against the WRONG model does not fail at all. It produces a
directory of plausible answers attributed to a model that never saw them,
which is the failure this whole split of entries exists to prevent.

Nothing here imports gpu_serving. The endpoint's URL and the name it answers to
are the entire contract, so the serving package stays separable and this check
works just as well against a tunnel, another port, or someone else's server.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

TIMEOUT = 10


class EndpointNotReady(RuntimeError):
    """The local endpoint cannot serve this run, and the message says why."""


def served_names(base_url: str, timeout: float = TIMEOUT) -> list[str]:
    """What the endpoint says it is serving. Raises EndpointNotReady if silent."""
    url = f"{base_url.rstrip('/')}/models"
    request = urllib.request.Request(url, headers={"Authorization": "Bearer local"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            listing = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise EndpointNotReady(
            f"{url} answered {error.code}. Is something else on that port?"
        ) from None
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
        raise EndpointNotReady(
            f"nothing answering at {url} ({error}).\n"
            f"Start a server:  ./gpu_serving/serve.sh up <model>\n"
            f"Or point elsewhere with MEDQA_LOCAL_BASE_URL."
        ) from None
    return [entry["id"] for entry in listing.get("data", [])]


def require_ready(entry, timeout: float = TIMEOUT) -> None:
    """Raise unless the endpoint for `entry` is up and serving that model.

    A no-op for gateway-served entries, so a caller can apply it to whatever
    model it was given without asking where that model lives.
    """
    if not getattr(entry, "base_url", ""):
        return

    names = served_names(entry.base_url, timeout)
    if entry.gateway_model not in names:
        raise EndpointNotReady(
            f"{entry.base_url} serves {names or '(nothing)'}, "
            f"not {entry.gateway_model!r}.\n"
            f"Either the wrong model is up, or this entry's served name is "
            f"stale - check ./gpu_serving/serve.sh status."
        )


__all__ = ["require_ready", "served_names", "EndpointNotReady"]
