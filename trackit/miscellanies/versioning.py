import os
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GitAppVersion:
    sha: str
    has_diff: bool
    branch: str


_cached_git_status = None


def generate_app_version_from_git() -> GitAppVersion:
    global _cached_git_status
    if _cached_git_status is not None:
        return _cached_git_status
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('utf-8').strip()
    sha = 'N/A'
    diff = False
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = len(_run(['git', 'diff-index', 'HEAD'])) > 0
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    git_state = GitAppVersion(sha, diff, branch)
    _cached_git_status = git_state
    return git_state


def get_app_version_string():
    version = generate_app_version_from_git()
    version_string = f"{version.sha}-{version.branch}"
    if version.has_diff:
        version_string += "-dirty"
    return version_string
