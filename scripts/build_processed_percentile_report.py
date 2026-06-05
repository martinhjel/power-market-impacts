"""
Build a single HTML report from processed EMPS scenario outputs.

The expected folder structure is:

    processed/
      PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load/
        <scenario>/
          <area>/
            price.parquet
            reservoir_agg.parquet

Each parquet file is expected to contain a time index and one column per weather year
or realization. The report summarizes each timestamp/week across columns using
percentiles and produces Plotly subplot grids for:

- power prices in all areas with `price.parquet`
- aggregated reservoirs in the areas with `reservoir_agg.parquet`

Usage:
    python scripts/build_processed_percentile_report.py
    python scripts/build_processed_percentile_report.py --output report.html
    python scripts/build_processed_percentile_report.py --scenario-pattern "BASELINE*"
"""

from __future__ import annotations

import html
import math
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


DEFAULT_PROCESSED_ROOT = Path(
    "/Users/martihj/Library/CloudStorage/OneDrive-NTNU/Postdoc/Papers/Nuclear hydrodominated/processed"
)
HISTORICAL_RESERVOIR_PATH = Path(__file__).resolve().parents[1] / "app" / "data" / "historic_reservoir_nve.parquet"
NORDIC_AREA_ORDER = [
    "NO1",
    "NO2",
    "NO3",
    "NO4",
    "NO5",
    "SE1",
    "SE2",
    "SE3",
    "SE4",
    "DK1",
    "DK2",
    "FI",
]
NORDIC_AREAS = set(NORDIC_AREA_ORDER)
NORWEGIAN_AREAS = {"NO1", "NO2", "NO3", "NO4", "NO5"}
PERCENTILE_BAND_OUTER = "rgba(29, 78, 216, 0.12)"
PERCENTILE_BAND_INNER = "rgba(29, 78, 216, 0.24)"
PERCENTILE_BAND_FULL = "rgba(29, 78, 216, 0.06)"
MEDIAN_COLOR = "#0f172a"
MEAN_COLOR = "#dc2626"
HISTORICAL_COLOR = "#059669"


def _parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Build an HTML percentile report from processed scenario results.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help="Root processed folder, or the direct scenario root under it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "images" / "processed_percentile_report.html",
        help="Output HTML file.",
    )
    parser.add_argument(
        "--scenario-pattern",
        type=str,
        default="*",
        help="Glob pattern for filtering scenario folders.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["auto", "pyarrow", "fastparquet"],
        default="auto",
        help="Preferred parquet engine.",
    )
    return parser


def _looks_like_area_dir(path: Path) -> bool:
    return (path / "price.parquet").exists() or (path / "reservoir_agg.parquet").exists()


def _looks_like_scenario_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(child.is_dir() and _looks_like_area_dir(child) for child in path.iterdir())


def _scenario_children(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return [child for child in path.iterdir() if child.is_dir() and _looks_like_scenario_dir(child)]


def _resolve_scenario_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input root not found: {root}")

    if _looks_like_scenario_dir(root):
        return root

    # If the supplied directory contains scenario folders directly, it is the scenario root.
    candidates = _scenario_children(root)
    if candidates:
        return root

    # Otherwise allow one wrapper directory, e.g. processed/<dataset-name>/<scenario>/...
    wrapper_candidates = [child for child in root.iterdir() if child.is_dir() and _scenario_children(child)]
    if len(wrapper_candidates) == 1:
        return wrapper_candidates[0]
    if len(wrapper_candidates) > 1:
        names = ", ".join(candidate.name for candidate in wrapper_candidates)
        raise ValueError(
            f"Multiple possible dataset roots found under {root}: {names}. "
            "Point --input-root to the specific dataset directory."
        )

    if not candidates:
        raise FileNotFoundError(
            f"Could not find a scenario root under {root}. Expected either scenario folders directly "
            "or a single dataset directory containing scenario folders."
        )
    return root


def _ordered_areas(areas: Iterable[str]) -> list[str]:
    return [area for area in NORDIC_AREA_ORDER if area in areas]


def _read_with_fastparquet_direct(path: Path) -> pd.DataFrame:
    from fastparquet import ParquetFile

    pf = ParquetFile(path)
    df = pf.to_pandas(index=False)

    if "__index_level_0__" in df.columns:
        df = df.set_index("__index_level_0__")

    return df


def _read_parquet_with_fallback(path: Path, engine: str) -> pd.DataFrame:
    attempts: list[str | None]
    if engine == "auto":
        attempts = [None, "pyarrow", "fastparquet"]
    else:
        attempts = [engine]

    errors: list[str] = []
    for candidate in attempts:
        try:
            if candidate == "fastparquet":
                return _read_with_fastparquet_direct(path)

            kwargs = {} if candidate is None else {"engine": candidate}
            return pd.read_parquet(path, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on local parquet engine/data
            label = "default" if candidate is None else candidate
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    error_block = "\n".join(errors)
    raise RuntimeError(
        f"Failed to read parquet file {path}.\n"
        f"Tried engines:\n{error_block}\n"
        "If pyarrow fails with these processed files, install and retry with fastparquet:\n"
        "  pip install fastparquet\n"
        "  python scripts/build_processed_percentile_report.py --engine fastparquet"
    )


def _ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.sort_index()
        if out.index.tz is not None:
            out.index = out.index.tz_convert("Europe/Oslo").tz_localize(None)
        return out

    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()].sort_index()
    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_convert("Europe/Oslo").tz_localize(None)
    return out


def _numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError("No numeric columns found in parquet data.")
    return numeric


def _summarize_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    numeric = _numeric_columns(_ensure_time_index(df))
    summary = pd.DataFrame(
        {
            "min": numeric.min(axis=1),
            "p10": numeric.quantile(0.10, axis=1),
            "p25": numeric.quantile(0.25, axis=1),
            "median": numeric.quantile(0.50, axis=1),
            "mean": numeric.mean(axis=1),
            "p75": numeric.quantile(0.75, axis=1),
            "p90": numeric.quantile(0.90, axis=1),
            "max": numeric.max(axis=1),
        },
        index=numeric.index,
    )
    return summary.sort_index()


def _load_historical_reservoir_means(engine: str) -> dict[str, pd.Series]:
    if not HISTORICAL_RESERVOIR_PATH.exists():
        return {}

    df = _read_parquet_with_fallback(HISTORICAL_RESERVOIR_PATH, engine)
    required_cols = {"omrType", "omrnr", "iso_uke", "fyllingsgrad"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"Historical reservoir file is missing required columns: {sorted(missing)}"
        )

    df = df.loc[df["omrType"] == "EL"].copy()
    df["iso_uke"] = pd.to_numeric(df["iso_uke"], errors="coerce")
    df["fyllingsgrad"] = pd.to_numeric(df["fyllingsgrad"], errors="coerce")
    df = df.dropna(subset=["omrnr", "iso_uke", "fyllingsgrad"])

    historical: dict[str, pd.Series] = {}
    for area in sorted(NORWEGIAN_AREAS):
        area_number = int(area[2:])
        area_df = df.loc[df["omrnr"] == area_number]
        if area_df.empty:
            continue
        weekly_mean = area_df.groupby("iso_uke")["fyllingsgrad"].mean().sort_index()
        historical[area] = weekly_mean

    return historical


def _build_historical_overlay(
    summary: pd.DataFrame,
    historical_weekly_mean: pd.Series,
    area_max_volume: float,
) -> pd.Series:
    if summary.empty or historical_weekly_mean.empty or area_max_volume <= 0:
        return pd.Series(dtype=float)

    weeks = summary.index.isocalendar().week.astype(int)
    values = weeks.map(historical_weekly_mean.to_dict()).astype(float)
    return pd.Series(values.to_numpy() * area_max_volume, index=summary.index, name="historical_mean")


def _add_percentile_traces(
    fig: go.Figure,
    series: pd.DataFrame,
    row: int,
    col: int,
    show_legend: bool,
    historical_overlay: pd.Series | None = None,
    include_full_range: bool = False,
) -> None:
    x = series.index

    if include_full_range:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=series["max"],
                mode="lines",
                line=dict(color="rgba(0,0,0,0)", width=0),
                showlegend=False,
                hoverinfo="skip",
                legendgroup="p0_p100",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=series["min"],
                mode="lines",
                line=dict(color="rgba(0,0,0,0)", width=0),
                fill="tonexty",
                fillcolor=PERCENTILE_BAND_FULL,
                name="P0-P100",
                showlegend=show_legend,
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>P0-P100 band<extra></extra>",
                legendgroup="p0_p100",
            ),
            row=row,
            col=col,
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=series["p90"],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)", width=0),
            showlegend=False,
            hoverinfo="skip",
            legendgroup="p10_p90",
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=series["p10"],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)", width=0),
            fill="tonexty",
            fillcolor=PERCENTILE_BAND_OUTER,
            name="P10-P90",
            showlegend=show_legend,
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>P10-P90 band<extra></extra>",
            legendgroup="p10_p90",
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=series["p75"],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)", width=0),
            showlegend=False,
            hoverinfo="skip",
            legendgroup="p25_p75",
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=series["p25"],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)", width=0),
            fill="tonexty",
            fillcolor=PERCENTILE_BAND_INNER,
            name="P25-P75",
            showlegend=show_legend,
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>P25-P75 band<extra></extra>",
            legendgroup="p25_p75",
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=series["median"],
            mode="lines",
            line=dict(color=MEDIAN_COLOR, width=2),
            name="Median",
            showlegend=show_legend,
            legendgroup="median",
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=series["mean"],
            mode="lines",
            line=dict(color=MEAN_COLOR, width=1.6, dash="dash"),
            name="Mean",
            showlegend=show_legend,
            legendgroup="mean",
        ),
        row=row,
        col=col,
    )

    if historical_overlay is not None and not historical_overlay.empty:
        fig.add_trace(
            go.Scatter(
                x=historical_overlay.index,
                y=historical_overlay.values,
                mode="lines",
                line=dict(color=HISTORICAL_COLOR, width=1.8, dash="dot"),
                name="Historical mean",
                showlegend=show_legend,
                legendgroup="historical_mean",
            ),
            row=row,
            col=col,
        )


def _build_figure(
    area_series: dict[str, pd.DataFrame],
    *,
    title: str,
    yaxis_title: str,
    ncols: int,
    yaxis_ranges: dict[str, tuple[float, float]] | None = None,
    historical_overlays: dict[str, pd.Series] | None = None,
    include_full_range: bool = False,
) -> go.Figure:
    areas = _ordered_areas(area_series)
    rows = max(1, math.ceil(len(areas) / ncols))
    fig = make_subplots(rows=rows, cols=ncols, subplot_titles=areas, shared_xaxes=False)

    for idx, area in enumerate(areas):
        row = idx // ncols + 1
        col = idx % ncols + 1
        _add_percentile_traces(
            fig,
            area_series[area],
            row=row,
            col=col,
            show_legend=idx == 0,
            historical_overlay=None if historical_overlays is None else historical_overlays.get(area),
            include_full_range=include_full_range,
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.20)", row=row, col=col)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.20)", row=row, col=col)
        if yaxis_ranges is not None and area in yaxis_ranges:
            ymin, ymax = yaxis_ranges[area]
            fig.update_yaxes(range=[ymin, ymax], row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text=yaxis_title, row=row, col=col)

    fig.update_layout(
        title=title,
        height=max(380, rows * 290),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=50, r=20, t=90, b=40),
    )
    return fig


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _scenario_html(
    scenario_dir: Path,
    *,
    engine: str,
    include_plotlyjs: bool,
    historical_means: dict[str, pd.Series],
) -> str:
    price_areas: dict[str, pd.DataFrame] = {}
    reservoir_areas: dict[str, pd.DataFrame] = {}
    reservoir_y_ranges: dict[str, tuple[float, float]] = {}
    reservoir_historical_overlays: dict[str, pd.Series] = {}

    for area_dir in sorted(child for child in scenario_dir.iterdir() if child.is_dir()):
        area_name = area_dir.name
        if area_name not in NORDIC_AREAS:
            continue

        price_path = area_dir / "price.parquet"
        if price_path.exists():
            price_areas[area_name] = _summarize_percentiles(_read_parquet_with_fallback(price_path, engine))

        reservoir_path = area_dir / "reservoir_agg.parquet"
        if reservoir_path.exists():
            reservoir_raw = _read_parquet_with_fallback(reservoir_path, engine)
            reservoir_summary = _summarize_percentiles(reservoir_raw)
            reservoir_areas[area_name] = reservoir_summary

            reservoir_numeric = _numeric_columns(_ensure_time_index(reservoir_raw))
            area_max_volume = float(reservoir_numeric.max().max())
            reservoir_y_ranges[area_name] = (0.0, area_max_volume)

            if area_name in historical_means:
                historical_overlay = _build_historical_overlay(
                    reservoir_summary,
                    historical_means[area_name],
                    area_max_volume,
                )
                if not historical_overlay.empty:
                    reservoir_historical_overlays[area_name] = historical_overlay

    if not price_areas:
        raise ValueError(f"No price.parquet files found in scenario {scenario_dir}")

    price_fig = _build_figure(
        price_areas,
        title=f"{scenario_dir.name}: Power price percentiles",
        yaxis_title="EUR/MWh",
        ncols=4,
    )

    html_parts = [
        f'<section class="scenario-section" id="{_slugify(scenario_dir.name)}">',
        f"<h2>{html.escape(scenario_dir.name)}</h2>",
        pio.to_html(price_fig, full_html=False, include_plotlyjs="cdn" if include_plotlyjs else False),
    ]

    if reservoir_areas:
        reservoir_fig = _build_figure(
            reservoir_areas,
            title=f"{scenario_dir.name}: Aggregated hydro reservoir percentiles",
            yaxis_title="Reservoir volume",
            ncols=3,
            yaxis_ranges=reservoir_y_ranges,
            historical_overlays=reservoir_historical_overlays,
            include_full_range=True,
        )
        html_parts.append(pio.to_html(reservoir_fig, full_html=False, include_plotlyjs=False))
    else:
        html_parts.append('<p class="note">No <code>reservoir_agg.parquet</code> files found for this scenario.</p>')

    html_parts.append("</section>")
    return "\n".join(html_parts)


def _build_report_html(sections: list[str], scenario_names: list[str], scenario_root: Path) -> str:
    toc_items = "\n".join(
        f'<li><a href="#{_slugify(name)}">{html.escape(name)}</a></li>' for name in scenario_names
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Processed Scenario Percentile Report</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --panel: #ffffff;
      --text: #0f172a;
      --muted: #475569;
      --border: #cbd5e1;
      --accent: #1d4ed8;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Helvetica, Arial, sans-serif;
      color: var(--text);
      background: linear-gradient(180deg, #eff6ff 0%, var(--bg) 220px);
    }}
    .page {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 28px 24px 64px;
    }}
    .hero, .scenario-section {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }}
    .hero {{
      padding: 24px 28px;
      margin-bottom: 24px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 2rem;
    }}
    .hero p {{
      margin: 6px 0;
      color: var(--muted);
    }}
    .toc {{
      margin-top: 18px;
      columns: 2 320px;
      column-gap: 32px;
      padding-left: 18px;
    }}
    .toc li {{
      margin-bottom: 8px;
      break-inside: avoid;
    }}
    .toc a {{
      color: var(--accent);
      text-decoration: none;
    }}
    .toc a:hover {{
      text-decoration: underline;
    }}
    .scenario-section {{
      padding: 20px 22px 14px;
      margin-bottom: 24px;
    }}
    .scenario-section h2 {{
      margin: 0 0 16px;
      font-size: 1.2rem;
    }}
    .note {{
      color: var(--muted);
      font-size: 0.95rem;
    }}
    code {{
      background: #eff6ff;
      border-radius: 6px;
      padding: 0.15rem 0.35rem;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Processed Scenario Percentile Report</h1>
      <p><strong>Scenario root:</strong> {html.escape(str(scenario_root))}</p>
      <p><strong>Scenarios:</strong> {len(scenario_names)}</p>
      <p>Each subplot shows row-wise percentiles across the realization columns stored in the parquet files.</p>
      <ol class="toc">
        {toc_items}
      </ol>
    </section>
    {' '.join(sections)}
  </div>
</body>
</html>
"""


def main() -> None:
    args = _parse_args().parse_args()
    scenario_root = _resolve_scenario_root(args.input_root)
    scenario_dirs = sorted(path for path in scenario_root.glob(args.scenario_pattern) if _looks_like_scenario_dir(path))
    historical_means = _load_historical_reservoir_means(args.engine)

    if not scenario_dirs:
        raise FileNotFoundError(
            f"No scenario directories matched pattern {args.scenario_pattern!r} under {scenario_root}"
        )

    sections: list[str] = []
    for idx, scenario_dir in enumerate(scenario_dirs):
        print(f"[{idx + 1}/{len(scenario_dirs)}] Building report section for {scenario_dir.name}")
        sections.append(
            _scenario_html(
                scenario_dir,
                engine=args.engine,
                include_plotlyjs=idx == 0,
                historical_means=historical_means,
            )
        )

    report_html = _build_report_html(sections, [scenario.name for scenario in scenario_dirs], scenario_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_html, encoding="utf-8")
    print(f"Saved HTML report to {args.output}")


if __name__ == "__main__":
    main()
