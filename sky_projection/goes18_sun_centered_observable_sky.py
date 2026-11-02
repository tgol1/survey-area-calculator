#!/usr/bin/env python3
"""Plot continuous sky access in the hemisphere centered on the Sun.

For every five-minute-aligned 10-minute, 1-hour, and 24-hour exposure,
the calculation defines the solar hemisphere as all celestial directions
within 90 degrees of the Sun at the exposure midpoint. A direction is
observable only if it remains outside the Earth, Moon, and Sun exclusion
regions for the entire exposure. Results are averaged over every possible
exposure start in the selected 24-hour planning interval.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from horizons_api_algorithm.goes18_visible_sky import (  # noqa: E402
    EARTH_MEAN_RADIUS_KM,
    GOES18_SPK_ID,
    MOON_MEAN_RADIUS_KM,
)
from sky_projection.goes18_daily_camera_projection import (  # noqa: E402
    UTC,
    fetch_ephemeris,
    parse_clock_time,
    parse_date,
    prompt_for_date,
    prompt_for_start_time,
    prompt_for_sun_exclusion,
)


EXPOSURE_DURATIONS_MINUTES = (10, 60, 24 * 60)
TIME_STEP_MINUTES = 5
PLANNING_HOURS = 24
DEFAULT_SKY_POINT_COUNT = 8_192
DEFAULT_OUTPUT_PREFIX = (
    SCRIPT_DIRECTORY
    / "visible_sky_data"
    / "goes18_sun_centered_observable_sky"
)
DURATION_LABELS = {
    10: "10-minute continuous exposure",
    60: "1-hour continuous exposure",
    24 * 60: "24-hour continuous exposure",
}
DURATION_COLUMN_NAMES = {
    10: "10min",
    60: "1hr",
    24 * 60: "24hr",
}


def fibonacci_sphere_points(point_count: int) -> np.ndarray:
    """Return a deterministic, approximately equal-area full-sky grid."""
    indices = np.arange(point_count, dtype=float) + 0.5
    z = 1.0 - 2.0 * indices / point_count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    longitude = golden_angle * indices
    return np.column_stack(
        (radius * np.cos(longitude), radius * np.sin(longitude), z)
    )


def sliding_maximum(values: np.ndarray, window_size: int) -> np.ndarray:
    """Calculate row-wise sliding maxima with a vectorized block method."""
    array = np.asarray(values)
    row_count, column_count = array.shape
    if not 1 <= window_size <= row_count:
        raise ValueError("Exposure window is outside the available time grid.")

    padding = (-row_count) % window_size
    if padding:
        padded = np.pad(
            array,
            ((0, padding), (0, 0)),
            mode="constant",
            constant_values=-np.inf,
        )
    else:
        padded = array

    blocks = padded.reshape(-1, window_size, column_count)
    prefix = np.maximum.accumulate(blocks, axis=1).reshape(-1, column_count)
    suffix = np.maximum.accumulate(
        blocks[:, ::-1, :], axis=1
    )[:, ::-1, :].reshape(-1, column_count)
    output_rows = row_count - window_size + 1
    starts = np.arange(output_rows)
    return np.maximum(suffix[starts], prefix[starts + window_size - 1])


def sun_centered_exposure_sweep(
    earth_vectors_km: np.ndarray,
    moon_vectors_km: np.ndarray,
    sun_vectors_km: np.ndarray,
    earth_clearance_deg: float,
    moon_clearance_deg: float,
    moon_reference: str,
    sky_point_count: int = DEFAULT_SKY_POINT_COUNT,
    sky_block_size: int = 128,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, int]]:
    """Return observable fractions of the moving Sun-centered hemisphere."""
    earth_vectors = np.asarray(earth_vectors_km, dtype=float)
    moon_vectors = np.asarray(moon_vectors_km, dtype=float)
    sun_vectors = np.asarray(sun_vectors_km, dtype=float)
    if not (
        earth_vectors.shape == moon_vectors.shape == sun_vectors.shape
        and earth_vectors.ndim == 2
        and earth_vectors.shape[1] == 3
    ):
        raise ValueError("Earth, Moon, and Sun vectors must be matching N-by-3 arrays.")
    if sky_point_count < 1_000:
        raise ValueError("Use at least 1,000 sky points.")
    if sky_block_size < 1:
        raise ValueError("Sky-grid block size must be positive.")

    sample_count = len(earth_vectors)
    window_sizes = {
        duration: duration // TIME_STEP_MINUTES + 1
        for duration in EXPOSURE_DURATIONS_MINUTES
    }
    if sample_count < max(window_sizes.values()):
        raise ValueError("The ephemeris must span at least 24 hours.")

    earth_distance = np.linalg.norm(earth_vectors, axis=1)
    moon_distance = np.linalg.norm(moon_vectors, axis=1)
    sun_distance = np.linalg.norm(sun_vectors, axis=1)
    if np.any(earth_distance <= EARTH_MEAN_RADIUS_KM):
        raise ValueError("Invalid observer-to-Earth distance in the ephemeris.")
    if np.any(moon_distance <= MOON_MEAN_RADIUS_KM):
        raise ValueError("Invalid observer-to-Moon distance in the ephemeris.")
    if np.any(sun_distance <= 0.0):
        raise ValueError("Invalid observer-to-Sun distance in the ephemeris.")

    earth_unit = earth_vectors / earth_distance[:, None]
    moon_unit = moon_vectors / moon_distance[:, None]
    sun_unit = sun_vectors / sun_distance[:, None]
    earth_radius = (
        np.arcsin(EARTH_MEAN_RADIUS_KM / earth_distance)
        + math.radians(earth_clearance_deg)
    )
    if moon_reference == "limb":
        moon_radius = (
            np.arcsin(MOON_MEAN_RADIUS_KM / moon_distance)
            + math.radians(moon_clearance_deg)
        )
    elif moon_reference == "center":
        moon_radius = np.full(sample_count, math.radians(moon_clearance_deg))
    else:
        raise ValueError("moon_reference must be 'center' or 'limb'.")

    earth_threshold = np.cos(earth_radius)
    moon_threshold = np.cos(moon_radius)
    sun_angles_deg = np.arange(0.0, 91.0, 1.0)
    sun_thresholds = np.cos(np.radians(sun_angles_deg))
    safe_counts = {
        duration: np.zeros(len(sun_angles_deg), dtype=np.int64)
        for duration in EXPOSURE_DURATIONS_MINUTES
    }
    hemisphere_counts = {
        duration: 0 for duration in EXPOSURE_DURATIONS_MINUTES
    }
    midpoint_sun_units: dict[int, np.ndarray] = {}
    window_counts: dict[int, int] = {}
    for duration, window_size in window_sizes.items():
        count = sample_count - window_size + 1
        window_counts[duration] = count
        midpoint_indices = (
            np.arange(count, dtype=int) + (window_size - 1) // 2
        )
        midpoint_sun_units[duration] = sun_unit[midpoint_indices]

    sky_points = fibonacci_sphere_points(sky_point_count)
    for block_start in range(0, sky_point_count, sky_block_size):
        block = sky_points[block_start : block_start + sky_block_size]
        earth_dot = np.asarray(earth_unit @ block.T, dtype=np.float32)
        moon_dot = np.asarray(moon_unit @ block.T, dtype=np.float32)
        sun_dot = np.asarray(sun_unit @ block.T, dtype=np.float32)
        earth_moon_excluded = (
            (earth_dot >= earth_threshold[:, None])
            | (moon_dot >= moon_threshold[:, None])
        )
        cumulative_excluded = np.zeros(
            (sample_count + 1, len(block)), dtype=np.int32
        )
        np.cumsum(
            earth_moon_excluded,
            axis=0,
            dtype=np.int32,
            out=cumulative_excluded[1:],
        )

        for duration, window_size in window_sizes.items():
            earth_moon_safe = (
                cumulative_excluded[window_size:]
                - cumulative_excluded[:-window_size]
                == 0
            )
            solar_hemisphere = (
                midpoint_sun_units[duration] @ block.T >= 0.0
            )
            hemisphere_counts[duration] += int(np.count_nonzero(solar_hemisphere))
            eligible = earth_moon_safe & solar_hemisphere
            closest_sun_dot = sliding_maximum(sun_dot, window_size)
            eligible_sun_dots = np.sort(closest_sun_dot[eligible])
            safe_counts[duration] += np.searchsorted(
                eligible_sun_dots,
                sun_thresholds,
                side="left",
            )

    observable_percent = {
        duration: 100.0 * safe_counts[duration] / hemisphere_counts[duration]
        for duration in EXPOSURE_DURATIONS_MINUTES
    }
    return sun_angles_deg, observable_percent, window_counts


def write_results_csv(
    path: Path,
    sun_angles_deg: np.ndarray,
    curves: dict[int, np.ndarray],
    start_datetime: datetime,
    observer_name: str,
    ephemeris_source: str,
    selected_sun_exclusion_deg: float,
    earth_clearance_deg: float,
    moon_clearance_deg: float,
    moon_reference: str,
    sky_point_count: int,
) -> None:
    """Write every Sun-angle curve and its calculation metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stop_datetime = start_datetime + timedelta(hours=PLANNING_HOURS)
    fieldnames = [
        "sun_exclusion_deg",
        "selected_sun_exclusion",
        "observable_fraction_10min",
        "observable_percent_10min",
        "observable_fraction_1hr",
        "observable_percent_1hr",
        "observable_fraction_24hr",
        "observable_percent_24hr",
        "start_utc",
        "stop_utc",
        "observer",
        "ephemeris_source",
        "hemisphere_definition",
        "earth_clearance_deg",
        "moon_clearance_deg",
        "moon_reference",
        "sky_point_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, angle in enumerate(sun_angles_deg):
            row: dict[str, str | int] = {
                "sun_exclusion_deg": f"{angle:.1f}",
                "selected_sun_exclusion": int(
                    math.isclose(angle, selected_sun_exclusion_deg)
                ),
                "start_utc": start_datetime.isoformat(),
                "stop_utc": stop_datetime.isoformat(),
                "observer": observer_name,
                "ephemeris_source": ephemeris_source,
                "hemisphere_definition": (
                    "directions within 90 deg of Sun at exposure midpoint"
                ),
                "earth_clearance_deg": f"{earth_clearance_deg:.6f}",
                "moon_clearance_deg": f"{moon_clearance_deg:.6f}",
                "moon_reference": moon_reference,
                "sky_point_count": sky_point_count,
            }
            for duration in EXPOSURE_DURATIONS_MINUTES:
                column = DURATION_COLUMN_NAMES[duration]
                percent = float(curves[duration][index])
                row[f"observable_fraction_{column}"] = f"{percent / 100.0:.12f}"
                row[f"observable_percent_{column}"] = f"{percent:.9f}"
            writer.writerow(row)


def write_figure(
    path: Path,
    sun_angles_deg: np.ndarray,
    curves: dict[int, np.ndarray],
    start_datetime: datetime,
    observer_name: str,
    ephemeris_source: str,
    selected_sun_exclusion_deg: float,
    earth_clearance_deg: float,
    moon_clearance_deg: float,
    moon_reference: str,
    sky_point_count: int,
    window_counts: dict[int, int],
) -> None:
    """Create three solar-hemisphere exposure-duration panels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_index = int(round(selected_sun_exclusion_deg))
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 12.2), sharex=True)
    for axis, duration in zip(axes, EXPOSURE_DURATIONS_MINUTES):
        curve = curves[duration]
        axis.plot(sun_angles_deg, curve, color="#176B87", linewidth=2.1)
        axis.fill_between(
            sun_angles_deg,
            curve,
            0.0,
            color="#64CCC5",
            alpha=0.17,
        )
        axis.axvline(
            selected_sun_exclusion_deg,
            color="#F59E0B",
            linestyle="--",
            linewidth=1.5,
        )
        axis.scatter(
            [selected_sun_exclusion_deg],
            [curve[selected_index]],
            color="#F59E0B",
            edgecolor="#111827",
            linewidth=0.5,
            s=38,
            zorder=4,
        )
        axis.set_title(
            f"{DURATION_LABELS[duration]} · "
            f"{window_counts[duration]:,} start time(s)",
            loc="left",
            fontsize=11,
            weight="bold",
        )
        axis.set_ylabel("Observable solar hemisphere (%)")
        axis.set_ylim(0.0, 100.0)
        axis.set_xlim(0.0, 90.0)
        axis.set_xticks(np.arange(0.0, 91.0, 10.0))
        axis.grid(True, color="#D7DEE5", linewidth=0.8, alpha=0.9)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            0.99,
            0.94,
            f"At {selected_sun_exclusion_deg:g}°: "
            f"{curve[selected_index]:.3f}% "
            f"({curve[selected_index] / 100.0:.5f})",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#4B5563",
        )

    axes[-1].set_xlabel("Sun exclusion angle from center (degrees)")
    stop_datetime = start_datetime + timedelta(hours=PLANNING_HOURS)
    moon_wording = "Moon limb" if moon_reference == "limb" else "Moon center"
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.07, top=0.80, hspace=0.40)
    fig.suptitle(
        "Continuously observable sky in the Sun-centered hemisphere",
        x=0.11,
        y=0.98,
        ha="left",
        fontsize=15,
        weight="bold",
    )
    fig.text(
        0.11,
        0.94,
        f"{start_datetime:%Y-%m-%d %H:%M} to "
        f"{stop_datetime:%Y-%m-%d %H:%M} UTC · {observer_name}\n"
        "Hemisphere = directions within 90° of the Sun at each exposure "
        f"midpoint; {sky_point_count:,}-point equal-area grid\n"
        f"Earth limb + {earth_clearance_deg:g}°; "
        f"{moon_wording} + {moon_clearance_deg:g}°; "
        f"five-minute samples; source = {ephemeris_source}",
        ha="left",
        va="top",
        fontsize=9.1,
        color="#4B5563",
        linespacing=1.35,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate continuously observable fractions of the celestial "
            "hemisphere centered on the Sun."
        )
    )
    parser.add_argument(
        "--date",
        dest="start_date",
        type=parse_date,
        help="UTC start date, YYYY-MM-DD; prompts when omitted",
    )
    parser.add_argument(
        "--start-time",
        type=parse_clock_time,
        help="UTC start time, HH:MM; prompts when omitted",
    )
    parser.add_argument(
        "--sun-exclusion",
        type=float,
        choices=(30.0, 45.0),
        help="Sun-center exclusion radius to highlight; prompts when omitted",
    )
    parser.add_argument(
        "--earth-clearance",
        type=float,
        default=20.0,
        help="Clearance beyond the apparent Earth limb (default: 20 deg)",
    )
    parser.add_argument(
        "--moon-clearance",
        type=float,
        default=20.0,
        help="Moon exclusion clearance (default: 20 deg)",
    )
    parser.add_argument(
        "--moon-reference",
        choices=("center", "limb"),
        default="center",
        help="Measure Moon clearance from its center or limb (default: center)",
    )
    parser.add_argument(
        "--ephemeris-source",
        choices=("tle", "horizons"),
        default="tle",
        help="GOES-18 ephemeris source (default: tle)",
    )
    parser.add_argument(
        "--observer-spk",
        type=int,
        default=GOES18_SPK_ID,
        help=f"Horizons observer SPK ID (default: {GOES18_SPK_ID})",
    )
    parser.add_argument(
        "--tle-file",
        type=Path,
        help="Optional local TLE file used with --ephemeris-source tle",
    )
    parser.add_argument(
        "--tle-source",
        choices=("auto", "space-track", "celestrak"),
        default="auto",
        help="Online TLE source when --tle-file is omitted (default: auto)",
    )
    parser.add_argument(
        "--allow-stale-tle",
        action="store_true",
        help="Allow stale historical TLEs; future dates continue automatically",
    )
    parser.add_argument(
        "--sky-points",
        type=int,
        default=DEFAULT_SKY_POINT_COUNT,
        help="Deterministic full-sky grid size (default: 8192)",
    )
    parser.add_argument(
        "--sky-block-size",
        type=int,
        default=128,
        help="Sky directions processed per memory block (default: 128)",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output path without extension",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        start_date = args.start_date or prompt_for_date()
        start_time = args.start_time or prompt_for_start_time()
        selected_sun_exclusion = (
            args.sun_exclusion
            if args.sun_exclusion is not None
            else prompt_for_sun_exclusion()
        )
        if not 0.0 <= args.earth_clearance <= 60.0:
            raise ValueError("--earth-clearance must be between 0 and 60 degrees.")
        if not 0.0 <= args.moon_clearance <= 60.0:
            raise ValueError("--moon-clearance must be between 0 and 60 degrees.")
        if args.sky_points < 1_000:
            raise ValueError("--sky-points must be at least 1000.")
        if args.sky_block_size < 1:
            raise ValueError("--sky-block-size must be positive.")

        start_datetime = datetime.combine(start_date, start_time, tzinfo=UTC)
        print(
            "GOES-18 Sun-centered observable-sky calculation\n"
            f"Start: {start_datetime:%Y-%m-%d %H:%M} UTC\n"
            f"Interval: {PLANNING_HOURS} hours\n"
            f"Selected Sun exclusion: {selected_sun_exclusion:g} degrees\n"
            f"Ephemeris source: {args.ephemeris_source}\n"
            f"Sky grid: {args.sky_points:,} points",
            flush=True,
        )
        ephemeris = fetch_ephemeris(
            start_datetime,
            window_hours=PLANNING_HOURS,
            step_minutes=TIME_STEP_MINUTES,
            observer_spk=args.observer_spk,
            ephemeris_source=args.ephemeris_source,
            tle_file=args.tle_file,
            tle_source=args.tle_source,
            allow_stale_tle=args.allow_stale_tle,
        )
        print(
            "Calculating 10-minute, 1-hour, and 24-hour continuous "
            "exposure curves...",
            flush=True,
        )
        sun_angles, curves, window_counts = sun_centered_exposure_sweep(
            ephemeris.earth_vectors_km,
            ephemeris.moon_vectors_km,
            ephemeris.sun_vectors_km,
            args.earth_clearance,
            args.moon_clearance,
            args.moon_reference,
            args.sky_points,
            args.sky_block_size,
        )

        output_prefix = args.output_prefix.expanduser().resolve()
        plot_path = output_prefix.with_suffix(".png")
        csv_path = output_prefix.with_suffix(".csv")
        write_figure(
            plot_path,
            sun_angles,
            curves,
            start_datetime,
            ephemeris.observer_name,
            args.ephemeris_source,
            selected_sun_exclusion,
            args.earth_clearance,
            args.moon_clearance,
            args.moon_reference,
            args.sky_points,
            window_counts,
        )
        write_results_csv(
            csv_path,
            sun_angles,
            curves,
            start_datetime,
            ephemeris.observer_name,
            args.ephemeris_source,
            selected_sun_exclusion,
            args.earth_clearance,
            args.moon_clearance,
            args.moon_reference,
            args.sky_points,
        )

        selected_index = int(round(selected_sun_exclusion))
        print(
            "\nObservable fraction of the Sun-centered hemisphere at "
            f"{selected_sun_exclusion:g} degrees:",
            flush=True,
        )
        for duration in EXPOSURE_DURATIONS_MINUTES:
            percent = float(curves[duration][selected_index])
            print(
                f"  {DURATION_LABELS[duration]}: "
                f"fraction={percent / 100.0:.9f}, percent={percent:.6f}%",
                flush=True,
            )
        print(f"Wrote plot: {plot_path}", flush=True)
        print(f"Wrote CSV: {csv_path}", flush=True)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
