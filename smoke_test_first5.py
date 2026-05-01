#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import eo_ir_benchmark as benchmark
import model_registry


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "eval_outputs" / "smoke_first5"
FIRST_FIVE_ANNOTATIONS = (
    "5th.Wheel.json",
    "Airplane.json",
    "Air_Canada_Airliner.json",
    "Automobile.json",
    "Cargo.Ship_Horizon.json",
)
DEFAULT_MODELS = [
    {
        "model_id": "gemini_2_5_flash",
        "provider": "gemini",
        "model_name": "gemini-2.5-flash",
        "enabled": True,
        "generation_config": {"temperature": 0.0, "max_output_tokens": 32},
        "notes": "Smoke-test export target.",
    }
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a smoke test on the first five release-schema annotations using the "
            "main full-video benchmark protocol."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for smoke-test artifacts. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate the smoke-test output directory before running.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    benchmark.ensure_dir(output_dir)

    subset_dir = output_dir / "annotations_subset"
    benchmark.ensure_dir(subset_dir)
    for name in FIRST_FIVE_ANNOTATIONS:
        source = ROOT_DIR / "annotations" / name
        if not source.exists():
            raise benchmark.BenchmarkError(f"Missing smoke-test annotation: {source}")
        shutil.copy2(source, subset_dir / name)

    benchmark.prepare_dataset_command(
        SimpleNamespace(
            annotations_dir=subset_dir,
            metadata_csv=ROOT_DIR / "maritime_video_metadata.csv",
            output_dir=output_dir,
        )
    )

    questions_path = output_dir / "questions.jsonl"
    model_safe_questions_path = output_dir / "model_safe_questions.jsonl"
    benchmark.export_requests_command(
        SimpleNamespace(
            questions=model_safe_questions_path,
            output_dir=output_dir,
            settings=list(benchmark.VALID_SETTINGS),
        )
    )

    model_specs = model_registry.parse_model_specs(DEFAULT_MODELS)
    model_request_summary = model_registry.export_model_requests(
        questions_path=model_safe_questions_path,
        output_dir=output_dir,
        settings=list(benchmark.VALID_SETTINGS),
        model_specs=model_specs,
    )

    benchmark.random_baseline_command(
        SimpleNamespace(
            questions=questions_path,
            output_dir=output_dir,
            model_id="random_mcq_smoke",
            seed=7,
            settings=list(benchmark.VALID_SETTINGS),
        )
    )

    benchmark.score_predictions_command(
        SimpleNamespace(
            questions=questions_path,
            predictions_dir=output_dir / "predictions",
            results_dir=output_dir / "results",
            allow_partial=False,
        )
    )

    dataset_summary = json.loads((output_dir / "dataset_summary.json").read_text())
    results_summary = json.loads((output_dir / "results" / "results_summary.json").read_text())
    annotation_check = summarize_annotations(subset_dir)

    summary = {
        "status": "ok",
        "output_dir": str(output_dir),
        "annotations_subset_dir": str(subset_dir),
        "smoke_annotations": list(FIRST_FIVE_ANNOTATIONS),
        "dataset_summary": dataset_summary,
        "model_request_summary": model_request_summary,
        "results_summary": results_summary,
        "annotation_check": annotation_check,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def summarize_annotations(subset_dir: Path) -> dict:
    payload = {}
    for path in sorted(subset_dir.glob("*.json")):
        annotation = json.loads(path.read_text())
        qas = annotation["qa"]
        payload[path.name] = {
            "sample_id": annotation["sample_id"],
            "scenario_type": annotation["scenario_type"],
            "question_count": len(qas),
            "group_ids": sorted({qa["group_id"] for qa in qas}),
            "all_have_group_family": all("group_family" in qa for qa in qas),
            "all_have_group_focus": all("group_focus" in qa for qa in qas),
            "all_have_capability_level": all("capability_level" in qa for qa in qas),
        }
    return payload


if __name__ == "__main__":
    try:
        main()
    except benchmark.BenchmarkError as exc:
        print(f"error: {exc}")
        raise SystemExit(1)
