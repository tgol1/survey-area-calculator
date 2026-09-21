#!/usr/bin/env python3
"""Compare GOES-18 positions from Horizons and TLE/SGP4 ephemerides.

The program prompts once for a UTC start date, end date, and Sun exclusion
angle. It then launches both existing visible-sky programs with the same
settings and forces both CSV outputs to a uniform five-minute cadence. Each
source CSV supplies the geocentric right ascension and declination of GOES-18.
The comparison calculates signed coordinate residuals in arcseconds using the
convention JPL Horizons minus TLE/SGP4.  Right-ascension differences are wrapped
across 0/360 degrees so a coordinate-boundary crossing cannot create a false
360-degree jump.

One PNG is written beside this script. Its upper panel shows the signed
right-ascension residual versus time, and its lower panel shows the signed
declination residual versus time.  Each panel includes zero and mean-bias
reference lines so systematic or periodic structure is easy to identify.

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
class ResidualStatistics:
    """Summary of signed Horizons-minus-TLE RA and declination residuals."""

    ra_mean_arcsec: float
    ra_standard_deviation_arcsec: float
    ra_rms_arcsec: float
    dec_mean_arcsec: float
    dec_standard_deviation_arcsec: float
    dec_rms_arcsec: float


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
            "omitted, that script selects online TLE data by requested date"
        ),
    )
    parser.add_argument(
        "--tle-source",
        choices=("auto", "space-track", "celestrak"),
        default="auto",
        help=(
            "Online source passed to the TLE algorithm when --tle-file is "
            "omitted (default: auto)"
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
        # Match the Horizons solar-system ephemeris so the residuals isolate
        # the GOES-18 TLE/SGP4 spacecraft trajectory as closely as possible.
        "--body-ephemeris",
        "horizons",
        "--output-prefix",
        str(tle_prefix),
    ]
    if args.tle_file is not None:
        tle_command.extend(["--tle-file", str(args.tle_file.expanduser().resolve())])
    else:
        tle_command.extend(["--tle-source", args.tle_source])
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
                    f"as large as {maximum_step}; the residual time series is "
                    "not uniformly sampled."
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


def coordinate_residuals_arcseconds(
    horizons_radec_deg: np.ndarray,
    tle_radec_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return signed JPL-minus-TLE RA and declination coordinate residuals."""
    horizons = np.asarray(horizons_radec_deg, dtype=float)
    tle = np.asarray(tle_radec_deg, dtype=float)
    if horizons.shape != tle.shape:
        raise ValueError("Horizons and TLE RA/Dec arrays have different shapes.")
    if horizons.ndim != 2 or horizons.shape[1] != 2:
        raise ValueError("RA/Dec coordinates must have shape (samples, 2).")

    # Wrap the RA coordinate difference into [-180, 180) degrees.  Without
    # this, samples straddling 0/360 degrees would show a false full-circle
    # residual instead of the small signed difference requested.
    ra_difference_deg = (
        (horizons[:, 0] - tle[:, 0] + 180.0) % 360.0
    ) - 180.0
    dec_difference_deg = horizons[:, 1] - tle[:, 1]
    return ra_difference_deg * 3600.0, dec_difference_deg * 3600.0


def calculate_statistics(
    ra_residual_arcsec: np.ndarray,
    dec_residual_arcsec: np.ndarray,
) -> ResidualStatistics:
    """Calculate signed residual bias, spread, and RMS for each coordinate."""
    return ResidualStatistics(
        ra_mean_arcsec=float(np.mean(ra_residual_arcsec)),
        ra_standard_deviation_arcsec=float(np.std(ra_residual_arcsec)),
        ra_rms_arcsec=float(np.sqrt(np.mean(np.square(ra_residual_arcsec)))),
        dec_mean_arcsec=float(np.mean(dec_residual_arcsec)),
        dec_standard_deviation_arcsec=float(np.std(dec_residual_arcsec)),
        dec_rms_arcsec=float(np.sqrt(np.mean(np.square(dec_residual_arcsec)))),
    )


def write_comparison_plot(
    path: Path,
    times: list[datetime],
    horizons_radec_deg: np.ndarray,
    tle_radec_deg: np.ndarray,
    sun_exclusion: float,
) -> ResidualStatistics:
    """Write signed RA and declination residuals versus time."""
    ra_residual_arcsec, dec_residual_arcsec = coordinate_residuals_arcseconds(
        horizons_radec_deg,
        tle_radec_deg,
    )
    statistics = calculate_statistics(
        ra_residual_arcsec,
        dec_residual_arcsec,
    )

    figure, (ra_axis, dec_axis) = plt.subplots(
        2,
        1,
        figsize=(14.0, 8.5),
        sharex=True,
        constrained_layout=True,
    )
    panels = (
        (
            ra_axis,
            ra_residual_arcsec,
            "#176B87",
            statistics.ra_mean_arcsec,
            statistics.ra_standard_deviation_arcsec,
            statistics.ra_rms_arcsec,
            "Right ascension residual",
            r"$RA_{JPL}-RA_{TLE}$ (arcsec)",
        ),
        (
            dec_axis,
            dec_residual_arcsec,
            "#7C3AED",
            statistics.dec_mean_arcsec,
            statistics.dec_standard_deviation_arcsec,
            statistics.dec_rms_arcsec,
            "Declination residual",
            r"$Dec_{JPL}-Dec_{TLE}$ (arcsec)",
        ),
    )
    for axis, residuals, color, mean, spread, rms, title, ylabel in panels:
        axis.plot(
            times,
            residuals,
            color=color,
            linewidth=1.0,
            alpha=0.88,
            zorder=1,
        )
        axis.scatter(
            times,
            residuals,
            s=7,
            color=color,
            alpha=0.48,
            linewidths=0.0,
            zorder=2,
        )
        axis.axhline(
            0.0,
            color="#111827",
            linewidth=1.0,
            label="Zero residual",
            zorder=3,
        )
        axis.axhline(
            mean,
            color="#D97706",
            linewidth=1.3,
            linestyle="--",
            label=f"Mean bias = {mean:.3f} arcsec",
            zorder=3,
        )
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontsize=11.5, weight="bold")
        axis.grid(True, which="major", color="#D7DEE5", linewidth=0.8)
        axis.grid(True, which="minor", color="#EDF1F4", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="best", frameon=True, fontsize=8.5)
        axis.text(
            0.995,
            0.03,
            f"Mean: {mean:.6f} arcsec\n"
            f"Standard deviation: {spread:.6f} arcsec\n"
            f"RMS: {rms:.6f} arcsec",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.6,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": "#CBD5E1",
                "alpha": 0.94,
            },
        )

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10, tz=UTC)
    dec_axis.xaxis.set_major_locator(locator)
    dec_axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=UTC))
    dec_axis.set_xlabel("Time (UTC)")

    figure.suptitle(
        "GOES-18 ephemeris algorithm comparison\n"
        "Signed coordinate residuals: JPL Horizons minus TLE/SGP4 | "
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
        "Difference convention: JPL Horizons minus TLE/SGP4; RA is wrapped "
        "across the 0/360-degree boundary"
    )
    print(
        f"RA mean bias: {statistics.ra_mean_arcsec:.9f} arcsec"
    )
    print(
        "RA standard deviation: "
        f"{statistics.ra_standard_deviation_arcsec:.9f} arcsec"
    )
    print(
        f"RA RMS residual: {statistics.ra_rms_arcsec:.9f} arcsec"
    )
    print(
        f"Declination mean bias: {statistics.dec_mean_arcsec:.9f} arcsec"
    )
    print(
        "Declination standard deviation: "
        f"{statistics.dec_standard_deviation_arcsec:.9f} arcsec"
    )
    print(
        f"Declination RMS residual: {statistics.dec_rms_arcsec:.9f} arcsec"
    )
    print(f"Wrote comparison PNG: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
