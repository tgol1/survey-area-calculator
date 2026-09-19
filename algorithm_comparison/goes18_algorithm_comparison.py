#!/usr/bin/env python3
"""Compare GOES-18 positions from Horizons and TLE/SGP4 ephemerides.

The program prompts once for a UTC start date, end date, and Sun exclusion
angle. It then launches both existing visible-sky programs with the same
settings and forces both CSV outputs to a uniform five-minute cadence. Each
source CSV supplies the geocentric right ascension and declination of GOES-18.
The comparison converts those directions to unit vectors and calculates their
great-circle angular separation in arcseconds.

One PNG is written beside this script. Its upper panel shows angular separation
versus time, and its lower panel is an ordinary histogram of those separations.

Expected repository structure:

    survey-area-calculator/
    |-- horizons_api_algorithm/
    |   `-- goes18_visible_sky.py
    |-- tle_sgp4_algorithm/
    |   `-- goes18_visible_sky_tle.py
    `-- algorithm_comparison/
        `-- goes18_algorithm_comparison.py
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
    "goes18_ra_deg",
    "goes18_dec_deg",
}


@dataclass(frozen=True)
class EphemerisData:
    """GOES-18 geocentric right ascension and declination by timestamp."""

    label: str
    path: Path
    directions_by_time: dict[datetime, tuple[float, float]]
    sun_angles_deg: np.ndarray

    @property
    def times(self) -> tuple[datetime, ...]:
        return tuple(sorted(self.directions_by_time))


@dataclass(frozen=True)
class AngularSeparationStatistics:
    """Empirical summary of the Horizons-to-TLE angular separations."""

    mean_arcsec: float
    median_arcsec: float
    rms_arcsec: float
    percentile_95_arcsec: float
    maximum_arcsec: float


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
            "their geocentric right ascension and declination."
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


def read_ephemeris_csv(path: Path, label: str) -> EphemerisData:
    """Read UTC, Sun angle, and GOES-18 RA/Dec from one source CSV."""
    resolved = require_file(path, f"{label} result CSV")
    directions_by_time: dict[datetime, tuple[float, float]] = {}
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
                right_ascension = float(row["goes18_ra_deg"])
                declination = float(row["goes18_dec_deg"])
                sun_angle = float(row["sun_exclusion_radius_deg"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid data in {resolved} on CSV line {line_number}."
                ) from exc

            if not math.isfinite(right_ascension) or not (
                0.0 <= right_ascension < 360.0
            ):
                raise ValueError(
                    f"Invalid GOES-18 right ascension in {resolved} on line "
                    f"{line_number}."
                )
            if not math.isfinite(declination) or not -90.0 <= declination <= 90.0:
                raise ValueError(
                    f"Invalid GOES-18 declination in {resolved} on line "
                    f"{line_number}."
                )
            if not math.isfinite(sun_angle):
                raise ValueError(
                    f"Invalid Sun exclusion angle in {resolved} on line {line_number}."
                )
            if timestamp in directions_by_time:
                raise ValueError(f"Duplicate timestamp {timestamp} in {resolved}.")

            directions_by_time[timestamp] = (right_ascension, declination)
            sun_angles.append(sun_angle)

    if len(directions_by_time) < 2:
        raise ValueError(f"{resolved} contains fewer than two result rows.")
    return EphemerisData(
        label=label,
        path=resolved,
        directions_by_time=directions_by_time,
        sun_angles_deg=np.asarray(sun_angles, dtype=float),
    )


def verify_sun_angle(data: EphemerisData, requested_angle: float) -> None:
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
    horizons: EphemerisData,
    tle: EphemerisData,
    require_uniform_five_minutes: bool,
) -> tuple[list[datetime], np.ndarray, np.ndarray]:
    """Align both result series using their exact common UTC timestamps."""
    horizons_times = set(horizons.directions_by_time)
    tle_times = set(tle.directions_by_time)
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

    horizons_radec = np.asarray(
        [horizons.directions_by_time[timestamp] for timestamp in common_times],
        dtype=float,
    )
    tle_radec = np.asarray(
        [tle.directions_by_time[timestamp] for timestamp in common_times],
        dtype=float,
    )
    return common_times, horizons_radec, tle_radec


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


def radec_to_unit_vectors(radec_deg: np.ndarray) -> np.ndarray:
    """Convert an N-by-2 RA/Dec array in degrees to Cartesian unit vectors."""
    coordinates = np.asarray(radec_deg, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("RA/Dec coordinates must have shape (samples, 2).")
    right_ascension = np.radians(coordinates[:, 0])
    declination = np.radians(coordinates[:, 1])
    cosine_declination = np.cos(declination)
    return np.column_stack(
        (
            cosine_declination * np.cos(right_ascension),
            cosine_declination * np.sin(right_ascension),
            np.sin(declination),
        )
    )


def angular_separation_arcseconds(
    horizons_radec_deg: np.ndarray,
    tle_radec_deg: np.ndarray,
) -> np.ndarray:
    """Return stable great-circle separations between paired RA/Dec samples."""
    horizons_unit = radec_to_unit_vectors(horizons_radec_deg)
    tle_unit = radec_to_unit_vectors(tle_radec_deg)
    if horizons_unit.shape != tle_unit.shape:
        raise ValueError("Horizons and TLE RA/Dec arrays have different shapes.")

    cross_magnitude = np.linalg.norm(
        np.cross(horizons_unit, tle_unit),
        axis=1,
    )
    dot_product = np.clip(
        np.einsum("ij,ij->i", horizons_unit, tle_unit),
        -1.0,
        1.0,
    )
    separation_radians = np.arctan2(cross_magnitude, dot_product)
    return np.degrees(separation_radians) * 3600.0


def calculate_statistics(
    separations_arcsec: np.ndarray,
) -> AngularSeparationStatistics:
    """Calculate empirical angular-separation summary statistics."""
    return AngularSeparationStatistics(
        mean_arcsec=float(np.mean(separations_arcsec)),
        median_arcsec=float(np.median(separations_arcsec)),
        rms_arcsec=float(np.sqrt(np.mean(np.square(separations_arcsec)))),
        percentile_95_arcsec=float(np.percentile(separations_arcsec, 95.0)),
        maximum_arcsec=float(np.max(separations_arcsec)),
    )


def write_comparison_plot(
    path: Path,
    times: list[datetime],
    horizons_radec_deg: np.ndarray,
    tle_radec_deg: np.ndarray,
    sun_exclusion: float,
) -> AngularSeparationStatistics:
    """Write time-domain and histogram views of angular separation."""
    separations_arcsec = angular_separation_arcseconds(
        horizons_radec_deg,
        tle_radec_deg,
    )
    bin_edges = histogram_bin_edges(separations_arcsec)
    statistics = calculate_statistics(separations_arcsec)

    figure, (time_axis, histogram_axis) = plt.subplots(
        2,
        1,
        figsize=(14.0, 9.0),
        gridspec_kw={"height_ratios": (1.55, 1.0)},
        constrained_layout=True,
    )
    dot_color = "#176B87"
    reference_color = "#D97706"

    time_axis.plot(
        times,
        separations_arcsec,
        color=dot_color,
        linewidth=1.0,
        alpha=0.82,
        zorder=1,
    )
    time_axis.scatter(
        times,
        separations_arcsec,
        s=7,
        color=dot_color,
        alpha=0.55,
        linewidths=0.0,
        label="Great-circle position separation",
        zorder=2,
    )
    time_axis.axhline(
        statistics.mean_arcsec,
        color=reference_color,
        linewidth=1.3,
        linestyle="--",
        label=f"Mean = {statistics.mean_arcsec:.3f} arcsec",
        zorder=3,
    )
    time_axis.set_ylabel("Angular separation (arcsec)")
    time_axis.set_xlabel("Time (UTC)")
    time_axis.set_title(
        "GOES-18 position difference over time",
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
    time_axis.set_ylim(bottom=0.0)
    time_axis.legend(loc="best", frameon=True, fontsize=8.5)

    histogram_axis.hist(
        separations_arcsec,
        bins=bin_edges,
        color=dot_color,
        alpha=0.65,
        edgecolor="white",
        linewidth=0.7,
        label="Angular-separation samples",
    )
    histogram_axis.axvline(
        statistics.mean_arcsec,
        color=reference_color,
        linewidth=1.2,
        linestyle="--",
        label=f"Mean = {statistics.mean_arcsec:.3f} arcsec",
    )
    histogram_axis.set_xlabel("Great-circle angular separation (arcsec)")
    histogram_axis.set_ylabel("Sample count")
    histogram_axis.set_title(
        "Distribution of GOES-18 position differences",
        loc="left",
        fontsize=11.5,
        weight="bold",
    )
    histogram_axis.grid(True, axis="y", color="#D7DEE5", linewidth=0.8)
    histogram_axis.spines[["top", "right"]].set_visible(False)
    histogram_axis.legend(loc="best", frameon=True, fontsize=8.5)

    statistics_text = (
        f"Samples: {len(separations_arcsec):,}\n"
        f"Mean: {statistics.mean_arcsec:.6f} arcsec\n"
        f"Median: {statistics.median_arcsec:.6f} arcsec\n"
        f"RMS: {statistics.rms_arcsec:.6f} arcsec\n"
        f"95th percentile: {statistics.percentile_95_arcsec:.6f} arcsec\n"
        f"Maximum: {statistics.maximum_arcsec:.6f} arcsec"
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
        "GOES-18 ephemeris algorithm comparison\n"
        "Great-circle separation from geocentric ICRF/GCRS RA and Dec | "
        f"{times[0]:%Y-%m-%d %H:%M} to {times[-1]:%Y-%m-%d %H:%M} UTC",
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

        horizons = read_ephemeris_csv(args.horizons_csv, "JPL Horizons")
        tle = read_ephemeris_csv(args.tle_csv, "TLE/SGP4")
        verify_sun_angle(horizons, sun_exclusion)
        verify_sun_angle(tle, sun_exclusion)
        times, horizons_radec, tle_radec = align_results(
            horizons,
            tle,
            require_uniform_five_minutes=not args.skip_run,
        )
        statistics = write_comparison_plot(
            args.output,
            times,
            horizons_radec,
            tle_radec,
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
    print(
        "Difference definition: great-circle separation between the "
        "JPL Horizons and TLE/SGP4 GOES-18 RA/Dec directions"
    )
    print(f"Mean angular separation: {statistics.mean_arcsec:.9f} arcsec")
    print(
        f"Median angular separation: {statistics.median_arcsec:.9f} arcsec"
    )
    print(
        f"RMS angular separation: {statistics.rms_arcsec:.9f} arcsec"
    )
    print(
        "95th-percentile angular separation: "
        f"{statistics.percentile_95_arcsec:.9f} arcsec"
    )
    print(
        f"Maximum angular separation: {statistics.maximum_arcsec:.9f} arcsec"
    )
    print(f"Wrote comparison PNG: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
