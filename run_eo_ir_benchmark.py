#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import eo_ir_benchmark as benchmark
import model_registry


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT_DIR / "eo_ir_benchmark_config.example.json"
VALID_STEPS = {
    "prepare_dataset",
    "prepare_clips",
    "export_requests",
    "export_model_requests",
    "run_random_baseline",
    "score_predictions",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Config-driven wrapper around eo_ir_benchmark.py. "
            "The main benchmark protocol uses full-video input; "
            "prepare_clips is optional and only for debug/oracle analysis."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to a JSON config file. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=sorted(VALID_STEPS),
        default=None,
        help=(
            "Override config['steps'] for this run. "
            "prepare_clips is optional and not part of the main benchmark protocol."
        ),
    )
    parser.add_argument(
        "--print-effective-config",
        action="store_true",
        help="Print the resolved config before running.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    if args.steps is not None:
        config["steps"] = args.steps
    if args.print_effective_config:
        print(json.dumps(config, ensure_ascii=False, indent=2))

    steps = config["steps"]
    output_dir = Path(config["output_dir"]).resolve()
    questions_path = output_dir / "questions.jsonl"
    predictions_dir = Path(config["score_predictions"]["predictions_dir"]).resolve()
    results_dir = Path(config["score_predictions"]["results_dir"]).resolve()
    model_specs = model_registry.parse_model_specs(config["models"])

    for step in steps:
        if step == "prepare_dataset":
            benchmark.prepare_dataset_command(
                SimpleNamespace(
                    annotations_dir=Path(config["annotations_dir"]),
                    metadata_csv=Path(config["metadata_csv"]),
                    output_dir=output_dir,
                    data_root=Path(config["data_root"]) if config.get("data_root") else None,
                    disable_option_shuffle=bool(config["prepare_dataset"]["disable_option_shuffle"]),
                    option_shuffle_salt=str(config["prepare_dataset"]["option_shuffle_salt"]),
                    disable_media_aliases=bool(config["prepare_dataset"]["disable_media_aliases"]),
                    media_alias_salt=str(config["prepare_dataset"]["media_alias_salt"]),
                )
            )
        elif step == "prepare_clips":
            clip_cfg = config["prepare_clips"]
            benchmark.prepare_clips_command(
                SimpleNamespace(
                    questions=questions_path,
                    force=bool(clip_cfg["force"]),
                    limit=clip_cfg["limit"],
                )
            )
        elif step == "export_requests":
            benchmark.export_requests_command(
                SimpleNamespace(
                    questions=questions_path,
                    output_dir=output_dir,
                    settings=config["settings"],
                )
            )
        elif step == "export_model_requests":
            result = model_registry.export_model_requests(
                questions_path=questions_path,
                output_dir=output_dir,
                settings=config["settings"],
                model_specs=model_specs,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif step == "run_random_baseline":
            random_cfg = config["random_baseline"]
            benchmark.random_baseline_command(
                SimpleNamespace(
                    questions=questions_path,
                    output_dir=output_dir,
                    model_id=random_cfg["model_id"],
                    seed=int(random_cfg["seed"]),
                    settings=config["settings"],
                )
            )
        elif step == "score_predictions":
            score_cfg = config["score_predictions"]
            benchmark.score_predictions_command(
                SimpleNamespace(
                    questions=questions_path,
                    predictions_dir=predictions_dir,
                    results_dir=results_dir,
                    allow_partial=bool(score_cfg["allow_partial"]),
                )
            )
        else:  # pragma: no cover
            raise benchmark.BenchmarkError(f"Unsupported step: {step}")


def load_config(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text())
    config = apply_defaults(raw, root_dir=path.resolve().parent)
    validate_config(config, source=path)
    return config


def apply_defaults(raw: Dict[str, Any], root_dir: Path) -> Dict[str, Any]:
    output_dir = str(resolve_config_path(root_dir, raw.get("output_dir"), ROOT_DIR / "eval_outputs"))
    data_root = raw.get("data_root")
    if data_root is None:
        data_root = os.environ.get(benchmark.DEFAULT_DATA_ROOT_ENV)

    config = {
        "annotations_dir": str(resolve_config_path(root_dir, raw.get("annotations_dir"), ROOT_DIR / "annotations")),
        "metadata_csv": str(resolve_config_path(root_dir, raw.get("metadata_csv"), ROOT_DIR / "maritime_video_metadata.csv")),
        "output_dir": output_dir,
        "data_root": str(resolve_config_path(root_dir, data_root, Path(data_root))) if data_root else None,
        "settings": raw.get("settings", list(benchmark.VALID_SETTINGS)),
        "steps": raw.get("steps", ["prepare_dataset", "export_model_requests"]),
        "models": raw.get("models", model_registry.default_model_configs()),
        "prepare_dataset": {
            "disable_option_shuffle": bool(raw.get("prepare_dataset", {}).get("disable_option_shuffle", False)),
            "option_shuffle_salt": raw.get("prepare_dataset", {}).get(
                "option_shuffle_salt", benchmark.DEFAULT_OPTION_SHUFFLE_SALT
            ),
            "disable_media_aliases": bool(raw.get("prepare_dataset", {}).get("disable_media_aliases", False)),
            "media_alias_salt": raw.get("prepare_dataset", {}).get(
                "media_alias_salt", benchmark.DEFAULT_MEDIA_ALIAS_SALT
            ),
        },
        "prepare_clips": {
            "force": bool(raw.get("prepare_clips", {}).get("force", False)),
            "limit": raw.get("prepare_clips", {}).get("limit"),
        },
        "random_baseline": {
            "model_id": raw.get("random_baseline", {}).get("model_id", "random_mcq"),
            "seed": int(raw.get("random_baseline", {}).get("seed", 7)),
        },
        "score_predictions": {
            "predictions_dir": raw.get("score_predictions", {}).get(
                "predictions_dir", str(Path(output_dir) / "predictions")
            ),
            "results_dir": raw.get("score_predictions", {}).get(
                "results_dir", str(Path(output_dir) / "results")
            ),
            "allow_partial": bool(
                raw.get("score_predictions", {}).get("allow_partial", False)
            ),
        },
    }

    config["score_predictions"]["predictions_dir"] = str(
        resolve_config_path(root_dir, config["score_predictions"]["predictions_dir"], Path(output_dir) / "predictions")
    )
    config["score_predictions"]["results_dir"] = str(
        resolve_config_path(root_dir, config["score_predictions"]["results_dir"], Path(output_dir) / "results")
    )
    return config


def resolve_config_path(root_dir: Path, value: Any, default: Path) -> Path:
    path_text = os.path.expandvars(os.path.expanduser(str(value))) if value is not None else str(default)
    path = Path(path_text)
    if not path.is_absolute():
        path = root_dir / path
    return path.resolve()


def validate_config(config: Dict[str, Any], source: Path) -> None:
    unknown_steps = [step for step in config["steps"] if step not in VALID_STEPS]
    if unknown_steps:
        raise benchmark.BenchmarkError(
            f"{source} contains unsupported steps: {unknown_steps}"
        )
    invalid_settings = [
        setting for setting in config["settings"] if setting not in benchmark.VALID_SETTINGS
    ]
    if invalid_settings:
        raise benchmark.BenchmarkError(
            f"{source} contains unsupported settings: {invalid_settings}"
        )
    if not config["settings"]:
        raise benchmark.BenchmarkError(f"{source} must contain at least one setting")
    model_registry.parse_model_specs(config["models"])


if __name__ == "__main__":
    main()
