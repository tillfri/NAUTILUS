"""Minimal BIIGLE API client with resumable video download.

Replaces ``old_scripts/biigle.py`` + ``old_scripts/helper_biigle.py``, which
hardcoded placeholder credentials (``helper_biigle.py:5``) and looked for env
vars (``BIIGLE_API_EMAIL`` / ``BIIGLE_API_TOKEN``) that the shipped ``.env``
does not define -- it only defines ``API_TOKEN``.

``GET videos/{id}/file`` answers with a 302 to the object store. ``requests``
drops the Authorization header on that cross-host redirect (which the store
requires -- forwarding it yields HTTP 400) and the store honours Range
requests, so partial downloads can be resumed.
"""

import os
import time

import requests
from requests.auth import HTTPBasicAuth

DEFAULT_BASE_URL = "https://biigle.de/api/v1"
DEFAULT_ENV_FILE = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "old_scripts", ".env")
)


def load_env_file(path=DEFAULT_ENV_FILE):
    """Parse a simple KEY=VALUE .env file.

    Written by hand because python-dotenv is installed neither on the host nor
    in the nautilus-qwen container.

    Args:
        path: Path to the .env file.

    Returns:
        Dict of key/value pairs, empty if the file does not exist.
    """
    values = {}
    if not os.path.isfile(path):
        return values
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


class BiigleApi(object):
    """Authenticated BIIGLE API session."""

    def __init__(self, email=None, token=None, base_url=DEFAULT_BASE_URL, env_file=DEFAULT_ENV_FILE):
        """Create a client, resolving credentials from args, then env, then .env file.

        Args:
            email: BIIGLE account email (the Basic auth user).
            token: BIIGLE API token.
            base_url: API root.
            env_file: .env file to fall back to.

        Raises:
            ValueError: If no email/token could be resolved.
        """
        env = load_env_file(env_file)
        email = email or os.getenv("BIIGLE_API_EMAIL") or env.get("BIIGLE_API_EMAIL")
        token = (
            token
            or os.getenv("BIIGLE_API_TOKEN")
            or env.get("BIIGLE_API_TOKEN")
            or env.get("API_TOKEN")
        )
        if not email or not token:
            raise ValueError(
                "Missing BIIGLE credentials. Set BIIGLE_API_EMAIL and API_TOKEN "
                "in {} or pass --email/--token.".format(env_file)
            )
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.auth = HTTPBasicAuth(email, token)
        self.session.headers.update({"Accept": "application/json"})

    def get(self, path, **kwargs):
        """Perform a GET against the API, raising on non-ok responses."""
        response = self.session.get("{}/{}".format(self.base_url, path.lstrip("/")), **kwargs)
        if response.status_code == 422:
            body = response.json()
            raise Exception(body.get("message"), body.get("errors"))
        response.raise_for_status()
        return response

    def whoami(self):
        """Return the authenticated user object (cheap credential check)."""
        return self.get("users/my").json()

    def project_volumes(self, project_id):
        """Return the volume objects of a project."""
        return self.get("projects/{}/volumes".format(project_id)).json()

    def volume_files(self, volume_id):
        """Return the file IDs of a volume (the API returns bare integer IDs)."""
        return self.get("volumes/{}/files".format(volume_id)).json()

    def download_video(self, video_id, dest, expected_size=None, chunk_size=1 << 20,
                       retries=5, progress=None, min_rate=256 << 10, stall_window=20.0,
                       max_reconnects=500):
        """Download a video file, resuming a previous partial download if present.

        The de.NBI object store degrades badly under load: a connection can wedge
        at ~0.1 MB/s for a minute while a freshly opened one bursts at >200 MB/s.
        So throughput is sampled over ``stall_window``; a connection that falls
        below ``min_rate`` is dropped and resumed from the current offset. Those
        reconnects are budgeted separately from ``retries`` (which covers real
        errors) so a slow file cannot exhaust the error budget.

        Args:
            video_id: BIIGLE video ID.
            dest: Final output path. A sibling ``<dest>.part`` holds the partial file.
            expected_size: Known size in bytes; used to skip completed files and to
                verify the result. ``None`` disables both checks.
            chunk_size: Streaming chunk size.
            retries: Number of attempts after a hard error before giving up.
            progress: Optional callable ``(bytes_done, total_or_None)``.
            min_rate: Bytes/s below which a connection counts as stalled.
            stall_window: Seconds of throughput to average before judging a stall.
            max_reconnects: Cap on stall-triggered reconnects, so a permanently
                slow store fails loudly instead of looping forever.

        Returns:
            "skipped" if the file was already complete, "downloaded" otherwise.

        Raises:
            IOError: If the finished file does not match ``expected_size``.
            requests.HTTPError: If every attempt failed.
        """
        if os.path.exists(dest):
            if expected_size is None or os.path.getsize(dest) == expected_size:
                return "skipped"
            os.remove(dest)

        part = dest + ".part"
        url = "{}/videos/{}/file".format(self.base_url, video_id)
        last_error = None
        errors = 0
        reconnects = 0
        complete = False

        while not complete:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            if expected_size is not None and have > expected_size:
                os.remove(part)
                have = 0

            headers = {"Range": "bytes={}-".format(have)} if have else {}
            stalled = False
            try:
                with self.session.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
                    if response.status_code == 416 and expected_size is not None and have == expected_size:
                        break  # already complete
                    response.raise_for_status()
                    # A server that ignores Range answers 200 with the whole file.
                    mode = "ab" if (have and response.status_code == 206) else "wb"
                    if mode == "wb":
                        have = 0
                    total = expected_size
                    if total is None:
                        length = response.headers.get("Content-Length")
                        total = have + int(length) if length else None
                    window_start = time.time()
                    window_bytes = 0
                    with open(part, mode) as out:
                        for block in response.iter_content(chunk_size=chunk_size):
                            if not block:
                                continue
                            out.write(block)
                            have += len(block)
                            if progress is not None:
                                progress(have, total)
                            window_bytes += len(block)
                            elapsed = time.time() - window_start
                            if elapsed >= stall_window:
                                if window_bytes / elapsed < min_rate:
                                    stalled = True
                                    break
                                window_start = time.time()
                                window_bytes = 0
                if not stalled:
                    if expected_size is None or os.path.getsize(part) == expected_size:
                        complete = True
                        break
                    last_error = IOError(
                        "size mismatch: got {}, expected {}".format(
                            os.path.getsize(part), expected_size)
                    )
            except (requests.RequestException, IOError) as error:
                last_error = error
                stalled = False

            # A stall is the store misbehaving, not our error: reconnect and resume
            # without spending the error budget.
            if stalled:
                reconnects += 1
                if reconnects > max_reconnects:
                    raise IOError(
                        "video {}: gave up after {} stall reconnects at {} bytes; "
                        "the object store is not delivering".format(video_id, reconnects, have)
                    )
                continue

            errors += 1
            if errors >= retries:
                raise last_error if last_error else IOError("download failed")
            time.sleep(min(2 ** errors, 30))

        if expected_size is not None and os.path.getsize(part) != expected_size:
            raise last_error if last_error else IOError(
                "size mismatch for video {}: {} != {}".format(
                    video_id, os.path.getsize(part), expected_size
                )
            )
        os.replace(part, dest)
        return "downloaded"
