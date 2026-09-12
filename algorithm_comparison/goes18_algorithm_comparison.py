#!/usr/bin/env python3
"""Run and compare the GOES-18 Horizons and TLE/SGP4 algorithms.

The program prompts once for a UTC start date, end date, and Sun exclusion
angle. It then launches both existing visible-sky programs with the same
settings and forces both CSV outputs to a uniform five-minute cadence.

The signed computational difference is defined as:

    Horizons visible sky (%) - TLE/SGP4 visible sky (%)

One PNG is written beside this script. Its upper panel is a time-domain dot
plot of the signed differences. Its lower panel is a probability-density
histogram with a fitted Gaussian distribution. The reported Gaussian RMS
computational error is sqrt(mu**2 + sigma**2), where mu is the mean bias and
sigma is the population standard deviation of the differences.

Expected repository structure:

    survey-area-calculator/
    |-- horizons_api_algorithm/
    |   `-- goes18_visible_sky.py
    |-- tle_sgp4_algorithm/
    |   `-- goes18_visible_sky_tle.py
    `-- algorithm_comparison/
        `-- compare_algorithms.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

# Avoid warnings on systems whose default Matplotlib directory is read-only.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "goes18-comparison-matplotlib-cache"),
)
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


UTC = timezone.utc
FIVE_MINUTES = timedelta(minutes=5)
MAX_HORIZONS_FIVE_MINUTE_SPAN = timedelta(days=34)
REQUIRED_COLUMNS = {
    "utc",
    "sun_exclusion_radius_deg",
    "visible_sky_percent",
}


@dataclass(frozen=True)
class VisibilityData:
    """Visible-sky percentages indexed by UTC timestamp."""

    label: str
    path: Path
    values_by_time: dict[datetime, float]
    sun_angles_deg: np.ndarray

    @property
    def times(self) -> tuple[datetime, ...]:
        return tuple(sorted(self.values_by_time))


@dataclass(frozen=True)
class ErrorStatistics:
    """Summary of the Horizons-minus-TLE differences."""

    mean_bias_pp: float
    standard_deviation_pp: float
    standard_error_pp: float
    mean_absolute_error_pp: float
    direct_rms_error_pp: float
    gaussian_rms_error_pp: float
    gaussian_fit_rmse_density: float


def parse_arguments() -> argparse.Namespace:
    """Parse optional arguments; missing experiment settings are prompted."""
    script_directory = Path(__file__).resolve().parent
    project_root = script_directory.parent
    horizons_csv = (
        project_root
        / "horizons_api_algorithm"
        / "visible_sky_data"
        / "goes18_visible_sky.csv"
    )
    tle_csv = (
        project_root
        / "tle_sgp4_algorithm"
        / "visible_sky_data"
        / "goes18_visible_sky_tle.csv"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Run the Horizons and TLE/SGP4 GOES-18 algorithms and compare "
            "their visible-sky percentages."
        )
    )
    parser.add_argument(
        "--start",
        help="UTC start date in YYYY-MM-DD format; prompted if omitted",
    )
    parser.add_argument(
        "--stop",
        help="UTC end date in YYYY-MM-DD format; prompted if omitted",
    )
    parser.add_argument(
        "--sun-exclusion",
        type=float,
        choices=(30.0, 45.0),
        help="Sun-centered exclusion radius; prompted if omitted",
    )
    parser.add_argument(
        "--horizons-script",
        type=Path,
        default=(
            project_root
            / "horizons_api_algorithm"
            / "goes18_visible_sky.py"
        ),
        help="Path to the Horizons algorithm script",
    )
    parser.add_argument(
        "--tle-script",
        type=Path,
        default=(
            project_root
            / "tle_sgp4_algorithm"
            / "goes18_visible_sky_tle.py"
        ),
        help="Path to the TLE/SGP4 algorithm script",
    )
    parser.add_argument(
        "--horizons-csv",
        type=Path,
        default=horizons_csv,
        help="Horizons result CSV path",
    )
    parser.add_argument(
        "--tle-csv",
        type=Path,
        default=tle_csv,
        help="TLE/SGP4 result CSV path",
    )
    parser.add_argument(
        "--tle-file",
        type=Path,
        help=(
            "Optional saved GOES-18 TLE passed to the TLE algorithm; when "
            "omitted, that script uses its online-download/fallback behavior"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_directory / "goes18_algorithm_comparison.png",
        help="Comparison PNG path",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Compare existing CSVs without rerunning either source algorithm",
    )
    return parser.parse_args()


def parse_date(value: str) -> date:
    """Parse one YYYY-MM-DD calendar date."""
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            f"Invalid date {value!r}; use YYYY-MM-DD, for example 2026-08-01."
        ) from exc


def prompt_for_date(label: str) -> date:
    """Prompt repeatedly until a valid date is entered."""
    while True:
        try:
            return parse_date(input(f"{label} (YYYY-MM-DD): "))
        except ValueError as exc:
            print(exc)


def prompt_for_sun_exclusion() -> float:
    """Prompt repeatedly until 30 or 45 degrees is selected."""
    while True:
        value = input("Choose Sun exclusion angle (30 or 45 degrees): ").strip()
        if value in {"30", "30.0"}:
            return 30.0
        if value in {"45", "45.0"}:
            return 45.0
        print("Please enter either 30 or 45.")


def collect_settings(args: argparse.Namespace) -> tuple[date, date, float]:
    """Collect and validate the shared settings for both algorithms."""
    start = parse_date(args.start) if args.start else prompt_for_date("Start date")
    stop = parse_date(args.stop) if args.stop else prompt_for_date("End date")
    sun_exclusion = (
        args.sun_exclusion
        if args.sun_exclusion is not None
        else prompt_for_sun_exclusion()
    )

    if stop <= start:
        raise ValueError("The end date must be later than the start date.")
    if not args.skip_run and datetime.combine(
        stop, datetime.min.time()
    ) - datetime.combine(start, datetime.min.time()) > MAX_HORIZONS_FIVE_MINUTE_SPAN:
        raise ValueError(
            "A uniform five-minute Horizons request must span 34 days or less "
            "to remain below the API row limit. Choose a shorter comparison range."
        )
    return start, stop, sun_exclusion


def require_file(path: Path, description: str) -> Path:
    """Resolve a required file and provide a useful error if it is absent."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} was not found at:\n  {resolved}")
    return resolved


def run_command(command: list[str], project_root: Path, label: str) -> None:
    """Run one source algorithm while streaming its normal terminal output."""
    print(f"\nRunning {label} algorithm...")
    try:
        subprocess.run(command, cwd=project_root, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Could not launch {label}. Python executable: {sys.executable}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{label} algorithm exited with status {exc.returncode}. "
            "Review its terminal output above for the underlying error."
        ) from exc


def run_source_algorithms(
    args: argparse.Namespace,
    start: date,
    stop: date,
    sun_exclusion: float,
) -> None:
    """Run both algorithms with identical settings and five-minute CSVs."""
    horizons_script = require_file(args.horizons_script, "Horizons script")
    tle_script = require_file(args.tle_script, "TLE/SGP4 script")
    project_root = Path(__file__).resolve().parent.parent

    common_arguments = [
        "--start",
        start.isoformat(),
        "--stop",
        stop.isoformat(),
        "--step",
        "5 min",
        "--sun-exclusion",
        f"{sun_exclusion:g}",
        # This only reduces the cost of the source scripts' unrelated
        # Sun-exposure plot; it does not alter their visibility CSV values.
        "--exposure-sky-points",
        "1000",
    ]

    horizons_prefix = args.horizons_csv.expanduser().resolve().with_suffix("")
    horizons_command = [
        sys.executable,
        str(horizons_script),
        *common_arguments,
        "--output-prefix",
        str(horizons_prefix),
    ]
    run_command(horizons_command, project_root, "JPL Horizons")

    tle_prefix = args.tle_csv.expanduser().resolve().with_suffix("")
    tle_command = [
        sys.executable,
        str(tle_script),
        *common_arguments,
        "--output-prefix",
        str(tle_prefix),
    ]
    if args.tle_file is not None:
        tle_command.extend(["--tle-file", str(args.tle_file.expanduser().resolve())])
    run_command(tle_command, project_root, "TLE/SGP4")


def parse_utc(value: str) -> datetime:
    """Parse an ISO timestamp and normalize it to timezone-aware UTC."""
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        timestamp = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid UTC timestamp {value!r}.") from exc
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def read_visibility_csv(path: Path, label: str) -> VisibilityData:
    """Read UTC, Sun angle, and visible percentage from one source CSV."""
    resolved = require_file(path, f"{label} result CSV")
    values_by_time: dict[datetime, float] = {}
    sun_angles: list[float] = []

    with resolved.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(
                f"{resolved} is missing required column(s): "
                + ", ".join(sorted(missing))
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp = parse_utc(row["utc"])
                visible_percent = float(row["visible_sky_percent"])
                sun_angle = float(row["sun_exclusion_radius_deg"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid data in {resolved} on CSV line {line_number}."
                ) from exc

            if not math.isfinite(visible_percent) or not 0.0 <= visible_percent <= 100.0:
                raise ValueError(
                    f"Invalid visible-sky percentage in {resolved} on line "
                    f"{line_number}."
                )
            if not math.isfinite(sun_angle):
                raise ValueError(
                    f"Invalid Sun exclusion angle in {resolved} on line {line_number}."
                )
            if timestamp in values_by_time:
                raise ValueError(f"Duplicate timestamp {timestamp} in {resolved}.")

            values_by_time[timestamp] = visible_percent
            sun_angles.append(sun_angle)

    if len(values_by_time) < 2:
        raise ValueError(f"{resolved} contains fewer than two result rows.")
    return VisibilityData(
        label=label,
        path=resolved,
        values_by_time=values_by_time,
        sun_angles_deg=np.asarray(sun_angles, dtype=float),
    )


def verify_sun_angle(data: VisibilityData, requested_angle: float) -> None:
    """Require all source rows to use the requested Sun exclusion angle."""
    if np.allclose(
        data.sun_angles_deg,
        requested_angle,
        rtol=0.0,
        atol=1.0e-6,
    ):
        return
    raise ValueError(
        f"{data.label} CSV Sun angle does not match {requested_angle:g} degrees. "
        f"Observed range: {np.min(data.sun_angles_deg):g} to "
        f"{np.max(data.sun_angles_deg):g} degrees."
    )


def largest_time_step(times: tuple[datetime, ...]) -> timedelta:
    """Return the largest interval between consecutive timestamps."""
    return max(
        (later - earlier for earlier, later in zip(times, times[1:])),
        default=timedelta(0),
    )


def align_results(
    horizons: VisibilityData,
    tle: VisibilityData,
    require_uniform_five_minutes: bool,
) -> tuple[list[datetime], np.ndarray, np.ndarray]:
    """Align both result series using their exact common UTC timestamps."""
    horizons_times = set(horizons.values_by_time)
    tle_times = set(tle.values_by_time)
    common_times = sorted(horizons_times & tle_times)
    if len(common_times) < 2:
        raise ValueError("The two CSVs have fewer than two matching UTC timestamps.")

    if require_uniform_five_minutes:
        for data in (horizons, tle):
            times = data.times
            intervals = [
                later - earlier for earlier, later in zip(times, times[1:])
            ]
            if any(interval != FIVE_MINUTES for interval in intervals):
                raise ValueError(
                    f"{data.label} CSV is not uniformly sampled every five minutes. "
                    "Rerun without --skip-run to regenerate comparable data."
                )
        if horizons_times != tle_times:
            raise ValueError(
                "The newly generated CSVs do not contain identical timestamps."
            )
    else:
        if horizons_times != tle_times:
            print(
                "Warning: existing CSV timestamps differ; only their exact "
                f"intersection ({len(common_times)} rows) will be compared."
            )
        for data in (horizons, tle):
            maximum_step = largest_time_step(data.times)
            if maximum_step != FIVE_MINUTES:
                print(
                    f"Warning: {data.label} existing CSV has sampling intervals "
                    f"as large as {maximum_step}; histogram points are not "
                    "uniformly time-weighted."
                )

    horizons_percent = np.asarray(
        [horizons.values_by_time[timestamp] for timestamp in common_times],
        dtype=float,
    )
    tle_percent = np.asarray(
        [tle.values_by_time[timestamp] for timestamp in common_times],
        dtype=float,
    )
    return common_times, horizons_percent, tle_percent


def histogram_bin_edges(values: np.ndarray) -> np.ndarray:
    """Choose stable histogram bins using the Freedman-Diaconis rule."""
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    if math.isclose(value_min, value_max, rel_tol=0.0, abs_tol=1.0e-12):
        half_width = max(0.001, 0.05 * max(1.0, abs(value_min)))
        return np.linspace(value_min - half_width, value_max + half_width, 13)

    first_quartile, third_quartile = np.percentile(values, (25.0, 75.0))
    interquartile_range = float(third_quartile - first_quartile)
    if interquartile_range > 0.0:
        bin_width = 2.0 * interquartile_range / np.cbrt(len(values))
        bin_count = int(math.ceil((value_max - value_min) / bin_width))
    else:
        bin_count = int(math.ceil(math.sqrt(len(values))))
    bin_count = max(12, min(80, bin_count))
    return np.linspace(value_min, value_max, bin_count + 1)


def gaussian_density(
    values: np.ndarray,
    mean: float,
    standard_deviation: float,
) -> np.ndarray:
    """Evaluate a normal probability density function."""
    exponent = -0.5 * np.square((values - mean) / standard_deviation)
    return np.exp(exponent) / (standard_deviation * math.sqrt(2.0 * math.pi))


def calculate_statistics(
    differences: np.ndarray,
    bin_edges: np.ndarray,
) -> ErrorStatistics:
    """Calculate algorithm-difference and Gaussian-fit error statistics."""
    mean_bias = float(np.mean(differences))
    standard_deviation = float(np.std(differences, ddof=0))
    standard_error = standard_deviation / math.sqrt(len(differences))
    mean_absolute_error = float(np.mean(np.abs(differences)))
    direct_rms_error = float(np.sqrt(np.mean(np.square(differences))))
    gaussian_rms_error = math.hypot(mean_bias, standard_deviation)

    if standard_deviation > 1.0e-12:
        observed_density, _ = np.histogram(
            differences,
            bins=bin_edges,
            density=True,
        )
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        fitted_density = gaussian_density(
            bin_centers,
            mean_bias,
            standard_deviation,
        )
        fit_rmse = float(
            np.sqrt(np.mean(np.square(observed_density - fitted_density)))
        )
    else:
        fit_rmse = 0.0

    return ErrorStatistics(
        mean_bias_pp=mean_bias,
        standard_deviation_pp=standard_deviation,
        standard_error_pp=standard_error,
        mean_absolute_error_pp=mean_absolute_error,
        direct_rms_error_pp=direct_rms_error,
        gaussian_rms_error_pp=gaussian_rms_error,
        gaussian_fit_rmse_density=fit_rmse,
    )


def write_comparison_plot(
    path: Path,
    times: list[datetime],
    horizons_percent: np.ndarray,
    tle_percent: np.ndarray,
    sun_exclusion: float,
) -> ErrorStatistics:
    """Write the time-domain dot plot and Gaussian-highlighted histogram."""
    differences = horizons_percent - tle_percent
    bin_edges = histogram_bin_edges(differences)
    statistics = calculate_statistics(differences, bin_edges)

    figure, (time_axis, histogram_axis) = plt.subplots(
        2,
        1,
        figsize=(14.0, 9.0),
        gridspec_kw={"height_ratios": (1.55, 1.0)},
        constrained_layout=True,
    )
    dot_color = "#176B87"
    gaussian_color = "#D97706"

    time_axis.axhline(0.0, color="#475569", linewidth=0.9, zorder=1)
    if statistics.standard_deviation_pp > 0.0:
        time_axis.axhspan(
            statistics.mean_bias_pp - statistics.standard_deviation_pp,
            statistics.mean_bias_pp + statistics.standard_deviation_pp,
            color=gaussian_color,
            alpha=0.13,
            label="Gaussian mean +/- 1 standard deviation",
            zorder=0,
        )
    time_axis.scatter(
        times,
        differences,
        s=9,
        color=dot_color,
        alpha=0.62,
        linewidths=0.0,
        label="Horizons - TLE/SGP4",
        zorder=2,
    )
    time_axis.axhline(
        statistics.mean_bias_pp,
        color=gaussian_color,
        linewidth=1.5,
        linestyle="--",
        label=f"Gaussian mean bias = {statistics.mean_bias_pp:+.5f} pp",
        zorder=3,
    )
    time_axis.set_ylabel("Difference (percentage points)")
    time_axis.set_xlabel("Time (UTC)")
    time_axis.set_title(
        "Time-domain computational differences",
        loc="left",
        fontsize=11.5,
        weight="bold",
    )
    time_axis.grid(True, which="major", color="#D7DEE5", linewidth=0.8)
    time_axis.grid(True, which="minor", color="#EDF1F4", linewidth=0.5)
    time_axis.spines[["top", "right"]].set_visible(False)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10, tz=UTC)
    time_axis.xaxis.set_major_locator(locator)
    time_axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=UTC))
    time_axis.legend(loc="best", frameon=True, fontsize=8.5)

    histogram_axis.hist(
        differences,
        bins=bin_edges,
        density=True,
        color=dot_color,
        alpha=0.58,
        edgecolor="white",
        linewidth=0.7,
        label="Difference histogram",
    )
    if statistics.standard_deviation_pp > 1.0e-12:
        gaussian_min = min(
            float(bin_edges[0]),
            statistics.mean_bias_pp - 4.0 * statistics.standard_deviation_pp,
        )
        gaussian_max = max(
            float(bin_edges[-1]),
            statistics.mean_bias_pp + 4.0 * statistics.standard_deviation_pp,
        )
        gaussian_x = np.linspace(gaussian_min, gaussian_max, 600)
        gaussian_y = gaussian_density(
            gaussian_x,
            statistics.mean_bias_pp,
            statistics.standard_deviation_pp,
        )
        histogram_axis.plot(
            gaussian_x,
            gaussian_y,
            color=gaussian_color,
            linewidth=2.2,
            label="Fitted Gaussian",
        )
        histogram_axis.fill_between(
            gaussian_x,
            0.0,
            gaussian_y,
            color=gaussian_color,
            alpha=0.12,
        )
        within_one_sigma = (
            np.abs(gaussian_x - statistics.mean_bias_pp)
            <= statistics.standard_deviation_pp
        )
        histogram_axis.fill_between(
            gaussian_x,
            0.0,
            gaussian_y,
            where=within_one_sigma,
            color=gaussian_color,
            alpha=0.25,
            interpolate=True,
            label="Gaussian central 1-sigma region",
        )
    else:
        histogram_axis.axvline(
            statistics.mean_bias_pp,
            color=gaussian_color,
            linewidth=2.2,
            label="Degenerate Gaussian (sigma = 0)",
        )

    histogram_axis.axvline(
        statistics.mean_bias_pp,
        color=gaussian_color,
        linewidth=1.2,
        linestyle="--",
    )
    histogram_axis.set_xlabel(
        "Horizons - TLE/SGP4 visible sky (percentage points)"
    )
    histogram_axis.set_ylabel("Probability density (1/pp)")
    histogram_axis.set_title(
        "Difference distribution with Gaussian fit",
        loc="left",
        fontsize=11.5,
        weight="bold",
    )
    histogram_axis.grid(True, axis="y", color="#D7DEE5", linewidth=0.8)
    histogram_axis.spines[["top", "right"]].set_visible(False)
    histogram_axis.legend(loc="best", frameon=True, fontsize=8.5)

    statistics_text = (
        f"Samples: {len(differences):,}\n"
        f"Mean bias, mu: {statistics.mean_bias_pp:+.6f} pp\n"
        f"Gaussian spread, sigma: {statistics.standard_deviation_pp:.6f} pp\n"
        f"Standard error of mu: {statistics.standard_error_pp:.6f} pp\n"
        f"Mean absolute error: {statistics.mean_absolute_error_pp:.6f} pp\n"
        f"Gaussian RMS computational error: "
        f"{statistics.gaussian_rms_error_pp:.6f} pp\n"
        f"Gaussian-fit density RMSE: "
        f"{statistics.gaussian_fit_rmse_density:.6f} 1/pp"
    )
    histogram_axis.text(
        0.995,
        0.97,
        statistics_text,
        transform=histogram_axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.8,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#CBD5E1",
            "alpha": 0.94,
        },
    )

    figure.suptitle(
        "GOES-18 visible-sky algorithm comparison\n"
        f"JPL Horizons minus TLE/SGP4 | {times[0]:%Y-%m-%d %H:%M} to "
        f"{times[-1]:%Y-%m-%d %H:%M} UTC | Sun exclusion: "
        f"{sun_exclusion:g} degrees",
        x=0.07,
        ha="left",
        fontsize=14,
        weight="bold",
    )

    resolved_path = path.expanduser().resolve()
    if resolved_path.suffix.lower() != ".png":
        resolved_path = resolved_path.with_suffix(".png")
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(resolved_path, dpi=180, facecolor="white")
    plt.close(figure)
    return statistics


def main() -> int:
    """Run both methods, align their CSV results, and create the comparison."""
    args = parse_arguments()
    try:
        start, stop, sun_exclusion = collect_settings(args)
        if not args.skip_run:
            run_source_algorithms(args, start, stop, sun_exclusion)
        else:
            print("Skipping source algorithm runs; reading existing CSV files.")

        horizons = read_visibility_csv(args.horizons_csv, "JPL Horizons")
        tle = read_visibility_csv(args.tle_csv, "TLE/SGP4")
        verify_sun_angle(horizons, sun_exclusion)
        verify_sun_angle(tle, sun_exclusion)
        times, horizons_percent, tle_percent = align_results(
            horizons,
            tle,
            require_uniform_five_minutes=not args.skip_run,
        )
        statistics = write_comparison_plot(
            args.output,
            times,
            horizons_percent,
            tle_percent,
            sun_exclusion,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".png":
        output = output.with_suffix(".png")
    print("\nAlgorithm comparison complete")
    print(f"Matched samples: {len(times):,}")
    print("Difference definition: Horizons - TLE/SGP4")
    print(f"Mean bias: {statistics.mean_bias_pp:+.9f} percentage points")
    print(
        "Gaussian standard deviation: "
        f"{statistics.standard_deviation_pp:.9f} percentage points"
    )
    print(
        "Gaussian RMS computational error, sqrt(mu^2 + sigma^2): "
        f"{statistics.gaussian_rms_error_pp:.9f} percentage points"
    )
    print(
        "Direct RMS difference check: "
        f"{statistics.direct_rms_error_pp:.9f} percentage points"
    )
    print(
        "Gaussian-fit density RMSE: "
        f"{statistics.gaussian_fit_rmse_density:.9f} 1/percentage-point"
    )
    print(f"Wrote comparison PNG: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
