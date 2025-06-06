"""Version information for :mod:`timsimaging`.

Run with ``python -m timsimaging.version``
"""

import os
from subprocess import CalledProcessError, check_output

__all__ = [
    "VERSION",
    "get_version",
    "get_git_hash",
]

VERSION = "0.0.1-dev"


def get_git_hash() -> str:
    """Get the :mod:`timsimaging` git hash."""
    with open(os.devnull, "w") as devnull:
        try:
            ret = check_output(
                ["git", "rev-parse", "HEAD"],  # noqa: S603,S607
                cwd=os.path.dirname(__file__),
                stderr=devnull,
            )
        except CalledProcessError:
            return "UNHASHED"
        else:
            return ret.strip().decode("utf-8")[:8]


def get_version(with_git_hash: bool = False):
    """Get the :mod:`timsimaging` version string, including a git hash."""
    return f"{VERSION}-{get_git_hash()}" if with_git_hash else VERSION


if __name__ == "__main__":
    print(get_version(with_git_hash=True))  # noqa:T201
