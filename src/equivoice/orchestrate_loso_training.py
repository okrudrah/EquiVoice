"""Plan or explicitly execute the four frozen CUDA fine-tuning folds."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from equivoice.check_cuda_environment import artifact_hashes


FOLD_CODES = ("aba", "ska", "ybaa", "zhaa")


def selected_folds(selection: str) -> tuple[str, ...]:
    if selection == "all":
        return FOLD_CODES
    if selection not in FOLD_CODES:
        raise ValueError(f"unknown fold: {selection}")
    return (selection,)


def build_commands(
    folds: Sequence[str],
    output_root: Path,
    python_executable: str,
    resume_from_checkpoint: Path | None = None,
) -> list[list[str]]:
    if resume_from_checkpoint is not None and len(folds) != 1:
        raise ValueError("checkpoint resume is allowed for exactly one fold")
    commands: list[list[str]] = []
    for code in folds:
        command = [
            python_executable,
            "-m",
            "equivoice.train_whisper_loso",
            "--fold-manifest",
            f"results/manifests/l2_arctic_v5_loso/fold_{code}.csv",
            "--output-dir",
            str(output_root / f"held_out_{code}"),
            "--device",
            "cuda",
        ]
        if resume_from_checkpoint is not None:
            command.extend(
                ["--resume-from-checkpoint", str(resume_from_checkpoint)]
            )
        commands.append(command)
    return commands


def verify_preflight_report(report_path: Path, project_root: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise ValueError("CUDA preflight report did not pass")
    if not report.get("cuda", {}).get("available"):
        raise ValueError("CUDA preflight report does not confirm a GPU")
    expected_hashes = artifact_hashes(project_root)
    if report.get("project_artifact_hashes") != expected_hashes:
        raise ValueError("project artifacts changed after CUDA preflight")
    return report


def validate_output_targets(
    folds: Sequence[str],
    output_root: Path,
    resume_from_checkpoint: Path | None,
) -> None:
    for code in folds:
        output_dir = output_root / f"held_out_{code}"
        if resume_from_checkpoint is None and output_dir.exists() and any(
            output_dir.iterdir()
        ):
            raise ValueError(
                f"refusing to overwrite nonempty training output: {output_dir}"
            )
    if resume_from_checkpoint is not None:
        code = folds[0]
        output_dir = (output_root / f"held_out_{code}").resolve()
        checkpoint = resume_from_checkpoint.resolve()
        if not checkpoint.is_dir() or checkpoint.parent != output_dir:
            raise ValueError(
                "resume checkpoint must be an existing checkpoint directory "
                "directly inside the selected fold output"
            )


def execute_commands(
    commands: Sequence[Sequence[str]], project_root: Path
) -> None:
    environment = os.environ.copy()
    source_path = str(project_root / "src")
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path
        if not existing_python_path
        else os.pathsep.join((source_path, existing_python_path))
    )
    environment["TOKENIZERS_PARALLELISM"] = "false"
    for command in commands:
        subprocess.run(
            list(command),
            cwd=project_root,
            env=environment,
            check=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=("all", *FOLD_CODES), default="all")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/experiments/fine_tuning"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the commands; omission is always a dry run",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    folds = selected_folds(args.fold)
    commands = build_commands(
        folds,
        args.output_root,
        sys.executable,
        args.resume_from_checkpoint,
    )
    print("Frozen LOSO training plan:")
    for command in commands:
        print(shlex.join(command))
    if not args.execute:
        print("Dry run only. No training was started.")
        return
    if args.preflight_report is None:
        raise ValueError("--execute requires --preflight-report")
    verify_preflight_report(args.preflight_report, project_root)
    validate_output_targets(folds, args.output_root, args.resume_from_checkpoint)
    execute_commands(commands, project_root)


if __name__ == "__main__":
    main()
