"""Run the pinned pretrained Whisper model on LibriSpeech test-clean."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from equivoice.run_whisper_baseline import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_SEED,
    run_baseline,
)
from equivoice.validate_l2_arctic import sha256_file


def require_prepared_control(path: Path, manifest: Path) -> tuple[str, ...]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise ValueError("LibriSpeech preparation report did not pass")
    if report.get("dataset") != "LibriSpeech" or report.get("subset") != "test-clean":
        raise ValueError("unexpected control dataset in preparation report")
    if report.get("scope") != "complete test-clean":
        raise ValueError("control evaluation requires the complete test-clean split")
    if report.get("manifest_sha256") != sha256_file(manifest):
        raise ValueError("control manifest does not match preparation report")
    speakers = tuple(report.get("speakers", []))
    if len(speakers) != 40:
        raise ValueError("control preparation report does not contain 40 speakers")
    return speakers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the pinned Whisper baseline on LibriSpeech test-clean."
    )
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--processed-manifest", type=Path, required=True)
    parser.add_argument("--preparation-report", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--public-metrics", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    speakers = require_prepared_control(
        args.preparation_report, args.processed_manifest
    )
    report = run_baseline(
        processed_root=args.prepared_root,
        raw_root=args.prepared_root,
        processed_manifest=args.processed_manifest,
        model_cache=args.model_cache,
        predictions_path=args.predictions,
        public_metrics_path=args.public_metrics,
        report_path=args.report,
        device_request=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        limit=None,
        local_files_only=args.local_files_only,
        expected_speakers=speakers,
        dataset="LibriSpeech",
        release="SLR12/test-clean",
        full_scope="complete test-clean native control",
        evaluation="pretrained native-English control baseline",
        run_config_overrides={
            "control_subset": "test-clean",
            "preparation_report": args.preparation_report.name,
            "preparation_report_sha256": sha256_file(args.preparation_report),
        },
    )
    print(
        f"Native control {report['status']}: "
        f"{report['aggregate']['utterances']} utterances, "
        f"WER={report['aggregate']['wer']:.6f}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
