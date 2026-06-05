#!/usr/bin/env python3
"""
Run the processed-result scripts that generate the public paper figures.

Prerequisites:
  - ltm_processed/<model-folder>/<scenario>/processed_data.parquet exists.

Most paper scripts still define MODEL_FOLDER as a module constant. This runner
keeps those scripts unchanged by executing temporary same-directory copies with
MODEL_FOLDER overridden.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from time import perf_counter
from uuid import uuid4

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
DEFAULT_OUTPUT_DIR = "visualizations"


@dataclass(frozen=True)
class Task:
    name: str
    script: str
    stage: str
    args: tuple[str, ...] = ()
    transform: bool = True
    override_model_folder: bool = True
    rewrite_visualizations_paper: bool = False
    optional: bool = False
    description: str = ""
    default_enabled: bool = True


TASKS: tuple[Task, ...] = (
    Task(name="de_price_histogram", script="scripts/plot_de_price_histogram.py", stage="static", transform=False),
    Task(name="reservoir_trajectory", script="scripts/paper/plot_reservoir_trajectory.py", stage="figures"),
    Task(name="price_mean_std_uprate", script="scripts/paper/plot_price_mean_std_uprate.py", stage="figures"),
    Task(name="hydro_uprate_areas", script="scripts/paper/plot_hydro_uprate_areas.py", stage="figures_tables"),
    Task(
        name="hydro_uprate_value_factor",
        script="scripts/paper/calculate_hydro_uprate_value_factor.py",
        stage="figures",
    ),
    Task(
        name="uprated_hydro_duration_ba_selected",
        script="scripts/paper/plot_uprated_hydro_plant_duration.py",
        stage="figures",
        args=(
            "--model-folder",
            "{model_folder}",
            "--scenario-set",
            "ba",
            "--plants",
            "roeldal",
            "mauranger",
            "kvanndal",
        ),
        transform=False,
    ),
    Task(name="price_duration_smr_lmr", script="scripts/paper/plot_price_duration_smr_lmr.py", stage="figures"),
    Task(name="smr_lmr_surplus", script="scripts/paper/visualize_smr_lmr_surplus.py", stage="surplus"),
    Task(
        name="nuclear_offshore_revenue",
        script="scripts/paper/calculate_nuclear_offshore_revenue.py",
        stage="revenue",
    ),
    Task(name="smr_lmr_revenue", script="scripts/paper/plot_revenue.py", stage="revenue"),
)


def read_config_model_folder() -> str:
    path = PROJECT_ROOT / "config" / "visualization_config.yaml"
    if not path.exists():
        return DEFAULT_MODEL_FOLDER
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    return str(config.get("model_folder") or DEFAULT_MODEL_FOLDER)


def discover_scenarios(model_folder: str) -> dict[str, str]:
    root = PROJECT_ROOT / "ltm_processed" / model_folder
    if not root.exists():
        raise FileNotFoundError(f"Processed model folder not found: {root}")
    scenarios = {}
    for path in sorted(root.iterdir()):
        has_processed_result = (path / "processed_data.parquet").exists()
        if path.is_dir() and has_processed_result:
            scenarios[path.name] = path.name
    if not scenarios:
        raise RuntimeError(f"No scenario folders found under {root}")
    return scenarios


def replace_assignment(source: str, name: str, value: object) -> str:
    lines = source.splitlines(keepends=True)
    start = None
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if line == stripped and stripped.startswith(f"{name} ="):
            start = idx
            break
    if start is None:
        return source

    rhs = lines[start].split("=", 1)[1]
    bracket_depth = rhs.count("{") + rhs.count("[") + rhs.count("(") - rhs.count("}") - rhs.count("]") - rhs.count(")")
    end = start
    while bracket_depth > 0 and end + 1 < len(lines):
        end += 1
        segment = lines[end]
        bracket_depth += segment.count("{") + segment.count("[") + segment.count("(")
        bracket_depth -= segment.count("}") + segment.count("]") + segment.count(")")

    value_repr = repr(value)
    if isinstance(value, (dict, list, tuple)):
        value_repr = json.dumps(value, indent=4)
    lines[start : end + 1] = [f"{name} = {value_repr}\n"]
    return "".join(lines)


def transformed_script(task: Task, model_folder: str, output_dir: str) -> Path:
    source_path = PROJECT_ROOT / task.script
    source = source_path.read_text()
    if task.override_model_folder:
        source = replace_assignment(source, "MODEL_FOLDER", model_folder)
    source = replace_assignment(source, "OUTPUT_DIR", output_dir)
    if task.rewrite_visualizations_paper:
        replacement = f'Path("{output_dir}") / "{model_folder}" / "paper"'
        source = source.replace('Path("visualizations/paper")', replacement)
        source = source.replace("Path('visualizations/paper')", replacement)

    temp_name = f".__paper_runner_{source_path.stem}_{uuid4().hex}.py"
    temp_path = source_path.with_name(temp_name)
    temp_path.write_text(
        "# Auto-generated temporary script from scripts/paper/run_all_paper_outputs.py\n"
        + source
    )
    return temp_path


def format_args(args: tuple[str, ...], *, model_folder: str, output_dir: str, workers: int) -> list[str]:
    paper_output_dir = str(PROJECT_ROOT / output_dir / model_folder / "paper")
    return [
        item.format(
            model_folder=model_folder,
            output_dir=output_dir,
            paper_output_dir=paper_output_dir,
            workers=workers,
        )
        for item in args
    ]


def stream_command(command: list[str], *, env: dict[str, str], log_file: Path, dry_run: bool) -> int:
    command_text = " ".join(command)
    with open(log_file, "a") as log:
        log.write(f"\n$ {command_text}\n")
    print(f"$ {command_text}")
    if dry_run:
        return 0

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    with open(log_file, "a") as log:
        for line in process.stdout:
            print(line, end="")
            log.write(line)
    return process.wait()


def selected_tasks(args: argparse.Namespace) -> list[Task]:
    tasks = [task for task in TASKS if task.default_enabled or args.include_optional]
    if args.only:
        wanted = set(args.only)
        tasks = [task for task in tasks if task.name in wanted or task.stage in wanted]
    if args.skip:
        skipped = set(args.skip)
        tasks = [task for task in tasks if task.name not in skipped and task.stage not in skipped]
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the scripts that generate paper figures and tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python scripts/paper/run_all_paper_outputs.py --model-folder PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load_imp_nuke
              python scripts/paper/run_all_paper_outputs.py --list
              python scripts/paper/run_all_paper_outputs.py --only revenue surplus --workers 4
              python scripts/paper/run_all_paper_outputs.py --only figures
            """
        ),
    )
    parser.add_argument("--model-folder", default=os.environ.get("PAPER_MODEL_FOLDER", read_config_model_folder()))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=4, help="Workers passed to scripts that support worker args.")
    parser.add_argument("--include-optional", action="store_true", help="Include optional Plotly/browser-dependent scripts.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after a task fails.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--list", action="store_true", help="List available tasks and exit.")
    parser.add_argument("--only", nargs="+", help="Run only task names or stage names.")
    parser.add_argument("--skip", nargs="+", help="Skip task names or stage names.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = selected_tasks(args)

    if args.list:
        for task in TASKS:
            marker = "optional" if task.optional else "default"
            enabled = "yes" if task in tasks else "no"
            print(f"{task.name:34s} stage={task.stage:14s} {marker:8s} selected={enabled}")
            if task.description:
                print(f"  {task.description}")
        return 0

    all_scenarios = discover_scenarios(args.model_folder)
    paper_output_dir = PROJECT_ROOT / args.output_dir / args.model_folder / "paper"
    paper_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = paper_output_dir / "run_all_paper_outputs.log"
    with open(log_file, "a") as log:
        log.write(
            "\n"
            f"model_folder={args.model_folder}\n"
            f"output_dir={args.output_dir}\n"
            f"scenarios={len(all_scenarios)}\n"
        )

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env["PAPER_MODEL_FOLDER"] = args.model_folder
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not env.get("PYTHONPATH")
        else f"{PROJECT_ROOT}{os.pathsep}{env['PYTHONPATH']}"
    )

    failures: list[tuple[str, int]] = []
    for task in tasks:
        start = perf_counter()
        script_path: Path | None = None
        temp_path: Path | None = None
        try:
            if task.transform:
                temp_path = transformed_script(task, args.model_folder, args.output_dir)
                script_path = temp_path
            else:
                script_path = PROJECT_ROOT / task.script
            command = [
                sys.executable,
                str(script_path),
                *format_args(task.args, model_folder=args.model_folder, output_dir=args.output_dir, workers=args.workers),
            ]

            print(f"\n=== {task.name} ({task.stage}) ===")
            return_code = stream_command(command, env=env, log_file=log_file, dry_run=args.dry_run)
            elapsed = perf_counter() - start
            with open(log_file, "a") as log:
                log.write(f"task={task.name} return_code={return_code} elapsed_s={elapsed:.1f}\n")
            if return_code != 0:
                failures.append((task.name, return_code))
                if not args.continue_on_error:
                    break
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass

    if failures:
        print("\nFailed tasks:")
        for name, code in failures:
            print(f"  {name}: exit {code}")
        print(f"Log: {log_file}")
        return 1

    print(f"\nAll selected paper tasks completed. Log: {log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
