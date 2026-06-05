"""
Robust subprocess management with safe cleanup handling.
Ensures process termination does not leave dangling resources.
"""
import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def managed_subprocess(
    cmd: list,
    timeout: float = 30.0,
    kill_timeout: float = 2.0,
    **kwargs,
) -> subprocess.CompletedProcess:
    """
    Run a subprocess with guaranteed resource cleanup even if termination times out.

    The second wait() after SIGKILL is wrapped so that a TimeoutExpired there
    does not prevent stream handles from being closed.
    """
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    kwargs.setdefault("check", False)

    proc: Optional[subprocess.Popen] = None
    try:
        proc = subprocess.Popen(cmd, **kwargs)
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
        )
    except subprocess.TimeoutExpired:
        if proc is None:
            raise
        _force_kill(proc, kill_timeout)
        stdout, stderr = proc.communicate()
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout or "",
            stderr=stderr or "",
        )


def _force_kill(proc: subprocess.Popen, kill_timeout: float) -> None:
    """Kill a subprocess and wait for it to exit. Handles secondary timeout."""
    try:
        proc.kill()
        proc.wait(timeout=kill_timeout)
    except subprocess.TimeoutExpired:
        logger.warning(
            "Subprocess did not exit within %.1fs after kill; continuing cleanup",
            kill_timeout,
        )
    except Exception:
        logger.exception("Unexpected error while killing subprocess")
