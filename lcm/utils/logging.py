# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
#

import os
import subprocess
from pathlib import Path
from typing import Dict

import torch.distributed as dist
from fairseq2.gang import get_rank
from fairseq2.logging import get_log_writer
from fairseq2.recipes.logging import _setup_aten_logging, _setup_nccl_logging
from fairseq2.recipes.utils.log import log_environment_info
from fairseq2.typing import Device

logger = get_log_writer(__name__)

LCM_REPOS = ["lcm", "fairseq2", "sonar", "stopes"]


def setup_additional_logging(log_folder: Path):
    slurm_job_id: str = os.environ.get("SLURM_JOB_ID", "local")
    base_log_file = log_folder / f"{slurm_job_id}_{get_rank()}.log"
    _setup_aten_logging(base_log_file, force=False)
    _setup_nccl_logging(base_log_file, force=False)


def log_git_status(
    repo: str = "lcm",
    tolerate_uncommitted: bool = False,
) -> str:
    assert repo in LCM_REPOS, (
        f"Only the LCM core repos ({LCM_REPOS}) are supported in `log_git_status`"
    )

    repo_path = os.path.dirname(globals()[repo].__file__)

    try:
        # check for modifications
        mod_output = subprocess.run(
            f"cd {repo_path}; git status --porcelain", capture_output=True, shell=True
        )
        modifications = mod_output.stdout.decode("utf-8").split("\n")
        uncommitted = len(
            [
                m
                for m in modifications
                if m.startswith(" M") or m.startswith(("M ", "A ", "D ", "R "))
            ]
        )
        if uncommitted > 0:
            if tolerate_uncommitted:
                logger.warning(
                    (
                        "Changes to {} should be committed before running a job "
                        "- found {} change(s)."
                        " We will continue regardless, but the git commit hashes are unreliable!"
                    ).format(repo, uncommitted)
                )
            else:
                raise AssertionError(
                    f"Changes to {repo} should be committed before running a job - found {uncommitted} change(s). If runing tests try adding `--debug-training`"
                )

        # get commit hash
        output = subprocess.run(
            f"cd {repo_path}; git rev-parse HEAD", capture_output=True, shell=True
        )
        commit_hash = output.stdout.decode("ascii").strip()
        logger.info(f"{repo} ({repo_path}) commit hash: {commit_hash}")

        return commit_hash

    except AssertionError:
        raise

    except BaseException:
        raise ValueError(
            f"Could not check the git revision hash, make sure you can run `git status` in {repo} ({repo_path})"
        )


def log_lcm_environment(tolerate_uncommitted: bool = False) -> Dict:
    """
    For traceability and reproducibility, get the latest commit hash for the four key repos
    """

    commit_hashes = {
        repo: log_git_status(repo, tolerate_uncommitted) for repo in LCM_REPOS
    }

    return commit_hashes


def log_env_variables(device: Device) -> None:
    """Log environment variables useful for debugging, including
    fs2's `log_environment_info` to dump Fairseq2, torch, nccl and other relevant metadata
    """
    for key in sorted(os.environ.keys()):
        if not (
            key.startswith(
                ("SLURM_", "SUBMITIT_", "NCCL_", "FI_", "CUDA_", "FAIRSEQ2_", "TORCH_")
            )
            or key
            in (
                "MASTER_ADDR",
                "MASTER_PORT",
                "RANK",
                "WORLD_SIZE",
                "LOCAL_RANK",
                "LOCAL_WORLD_SIZE",
            )
        ):
            continue
        value = os.environ[key]
        logger.info(f"R{dist.get_rank()} -- {key}={value}")

    # For Fairseq2, torch and devices
    log_environment_info(logger, device)
