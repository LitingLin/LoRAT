import sys
import os
from typing import List

from vot.utilities.cli import main

def _run_vot_command(workspace_path: str, command_args: List[str]) -> None:
    """
    Generic function to run VOT CLI commands in-process

    Args:
        workspace_path: Path to the VOT workspace
        command_args: List of command arguments to pass to the VOT CLI

    Raises:
        SystemExit: If the VOT command exits with a non-zero status code
    """
    old_sys_argv = sys.argv
    old_wd = os.getcwd()

    sys.argv = [old_sys_argv[0]] + command_args
    os.chdir(workspace_path)

    try:
        main()
    except SystemExit as e:
        if e.code != 0:
            raise e
    finally:
        sys.argv = old_sys_argv
        os.chdir(old_wd)


def launch_vot_initialize(workspace_path: str, stack_name: str):
    _run_vot_command(workspace_path, ['initialize', stack_name])


def launch_vot_evaluation(workspace_path: str, name: str):
    _run_vot_command(workspace_path, ['evaluate', name])


def launch_vot_analysis(workspace_path: str):
    _run_vot_command(workspace_path, ['analyze'])


def launch_vot_report(workspace_path: str):
    _run_vot_command(workspace_path, ['report'])


def launch_vot_pack(workspace_path: str, name: str):
    _run_vot_command(workspace_path, ['pack', name])
