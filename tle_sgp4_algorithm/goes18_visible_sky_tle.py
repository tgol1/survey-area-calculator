## TLE Algorithm
# Uses SGP4 (Simplified General Perturbations Model 4) to compute satellite positions and velocities based on Two-Line Element (TLE) sets. 
# This algorithm is widely used in satellite tracking and orbital mechanics.

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import warnings
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from astropy import units as u
from astropy.coordinates import (
    CartesianRepresentation,
    GCRS,
    TEME,
    get_body,
    solar_system_ephemeris,
)
from astropy.time import Time
from astropy.utils import iers
from sgp4.api import SGP4_ERRORS, Satrec, WGS72

from horizons_api_algorithm.goes18_visible_sky import (
    ANGULAR_FINE_LIMITS_DEG,
    adaptive_sample_indices,
    compute_visible_sky,
    prompt_for_date,
    prompt_for_sun_exclusion,
    step_to_minutes,
    subset_results,
    validate_date_range,
    write_plot,
    write_results_csv,
)


GOES18_NORAD_ID = 51850
CELESTRAK_TLE_URL = (
    "https://celestrak.org/NORAD/elements/"
    "gp.php?CATNR=51850&FORMAT=TLE"
)
FINE_STEP_MINUTES = 5
DEFAULT_MAX_TLE_AGE_DAYS = 14.0
DEFAULT_FALLBACK_TLE_FILE = Path(__file__).with_name("goes18_2026-08-27.tle")


def parse_tle_text(text: str, source: str) -> tuple[str, str, str]:
    """Extract a satellite name and two TLE element lines."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line1_index = next(
        (index for index, line in enumerate(lines) if line.startswith("1 ")),
        None,
    )
    if line1_index is None or line1_index + 1 >= len(lines):
        raise ValueError(f"No two-line element set was found in {source}.")

    line1 = lines[line1_index]
    line2 = lines[line1_index + 1]
    if not line2.startswith("2 "):
        raise ValueError(f"The second element line is missing from {source}.")

    line1_satellite = line1[2:7].strip()
    line2_satellite = line2[2:7].strip()
    if line1_satellite != line2_satellite:
        raise ValueError(f"The two element lines in {source} describe different objects.")
    if int(line1_satellite) != GOES18_NORAD_ID:
        raise ValueError(
            f"Expected GOES-18 NORAD ID {GOES18_NORAD_ID}, "
            f"but {source} contains {line1_satellite}."
        )

    if line1_index > 0 and not lines[line1_index - 1].startswith(("1 ", "2 ")):
        name = lines[line1_index - 1].removeprefix("0 ").strip()
    else:
        name = "GOES 18"
    return name, line1, line2


def fetch_current_tle(url: str) -> tuple[str, str, str]:
    """Download the current GOES-18 TLE from CelesTrak."""
    request = Request(url, headers={"User-Agent": "GOES18-TLE-sky/1.0"})
    try:
        with urlopen(request, timeout=30) as response:
            text = response.read().decode("ascii")
    except (HTTPError, URLError, TimeoutError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            "Could not download the GOES-18 TLE from CelesTrak: "
            f"{exc}"
        ) from exc
    return parse_tle_text(text, url)


def read_tle_file(path: Path) -> tuple[str, str, str]:
    """Read and validate a local GOES-18 TLE file."""
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Could not read TLE file {path}: {exc}") from exc
    return parse_tle_text(text, str(path))


def load_tle(
    path: Path | None,
    url: str,
    fallback_path: Path,
) -> tuple[str, str, str, str]:
    """Load an explicit TLE, or try CelesTrak then the bundled fallback."""
    if path is not None:
        name, line1, line2 = read_tle_file(path)
        return name, line1, line2, str(path.resolve())

    try:
        name, line1, line2 = fetch_current_tle(url)
        return name, line1, line2, url
    except (RuntimeError, ValueError) as online_error:
        print(f"Online TLE download failed: {online_error}")
        print(f"Using fallback GOES-18 TLE: {fallback_path.resolve()}")
        try:
            name, line1, line2 = read_tle_file(fallback_path)
        except (RuntimeError, ValueError) as fallback_error:
            raise RuntimeError(
                "The online TLE download failed, and the fallback TLE could "
                f"not be loaded from {fallback_path}: {fallback_error}"
            ) from fallback_error
        return (
            name,
            line1,
            line2,
            f"{fallback_path.resolve()} (offline fallback)",
        )


def save_used_tle(path: Path, name: str, line1: str, line2: str) -> None:
    """Save the exact TLE used so a run can be reproduced later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{name}\n{line1}\n{line2}\n", encoding="ascii")


def build_sample_datetimes(
    start: str,
    stop: str,
    step_minutes: int = FINE_STEP_MINUTES,
) -> list[datetime]:
    """Create an inclusive, uniformly spaced UTC time grid."""
    start_time = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    stop_time = datetime.strptime(stop, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    step = timedelta(minutes=step_minutes)
    count = int((stop_time - start_time) // step) + 1
    return [start_time + index * step for index in range(count)]


def tle_epoch(satellite: Satrec) -> Time:
    """Return the epoch stored in an SGP4 satellite record."""
    return Time(
        satellite.jdsatepoch + satellite.jdsatepochF,
        format="jd",
        scale="utc",
    )


def warn_if_tle_is_stale(
    times: Time,
    satellite: Satrec,
    max_age_days: float,
) -> float:
    """Warn when the propagation interval is far from the TLE epoch."""
    age_days = np.abs(times.utc.jd - tle_epoch(satellite).utc.jd)
    maximum_age = float(np.max(age_days))
    if maximum_age > max_age_days:
        warnings.warn(
            "The requested range extends "
            f"{maximum_age:.1f} days from the TLE epoch. TLE/SGP4 accuracy "
            "degrades away from the epoch and across station-keeping "
            "maneuvers. Use a historical GOES-18 TLE near the requested "
            "dates with --tle-file.",
            RuntimeWarning,
            stacklevel=2,
        )
    return maximum_age


def propagate_tle_ephemeris(
    datetimes: list[datetime],
    satellite: Satrec,
) -> dict[str, np.ndarray | Time]:
    """Generate GOES-18 and Earth/Moon/Sun vectors on the UTC time grid.

    SGP4 returns GOES-18 coordinates in TEME.  Astropy converts the spacecraft
    position to GCRS, and its built-in ephemeris supplies GCRS positions for
    the Moon and Sun relative to Earth's center.  Subtracting the GOES-18 GCRS
    position produces the three observer-relative vectors used by the sky
    geometry calculation.
    """
    times = Time(datetimes, scale="utc")
    errors, teme_position, teme_velocity = satellite.sgp4_array(
        np.asarray(times.utc.jd1, dtype=float),
        np.asarray(times.utc.jd2, dtype=float),
    )
    if np.any(errors != 0):
        failures = sorted({int(code) for code in errors if code != 0})
        explanations = ", ".join(
            f"{code}: {SGP4_ERRORS.get(code, 'unknown error')}" for code in failures
        )
        raise RuntimeError(f"SGP4 propagation failed ({explanations}).")

    # Keep the calculation deterministic and usable offline.  Astropy ships
    # sufficient Earth-orientation data for this transformation; it will not
    # attempt to update those tables over the network.
    iers.conf.auto_download = False
    teme_coordinates = TEME(
        CartesianRepresentation(teme_position.T * u.km),
        obstime=times,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="astropy.utils.iers")
        goes_gcrs = teme_coordinates.transform_to(GCRS(obstime=times))
    goes_gcrs_km = goes_gcrs.cartesian.xyz.to_value(u.km).T

    # The built-in ephemeris uses analytical solar-system models and requires
    # no Horizons call and no downloaded JPL BSP/SPK kernel.
    with solar_system_ephemeris.set("builtin"):
        moon_geocentric_km = get_body("moon", times).cartesian.xyz.to_value(u.km).T
        sun_geocentric_km = get_body("sun", times).cartesian.xyz.to_value(u.km).T

    earth_from_goes_km = -goes_gcrs_km
    moon_from_goes_km = moon_geocentric_km - goes_gcrs_km
    sun_from_goes_km = sun_geocentric_km - goes_gcrs_km

    return {
        "times": times,
        "goes_teme_position_km": np.asarray(teme_position, dtype=float),
        "goes_teme_velocity_km_s": np.asarray(teme_velocity, dtype=float),
        "goes_gcrs_position_km": goes_gcrs_km,
        "earth_from_goes_km": earth_from_goes_km,
        "moon_from_goes_km": moon_from_goes_km,
        "sun_from_goes_km": sun_from_goes_km,
    }


def calendar_labels(datetimes: list[datetime]) -> list[str]:
    """Format UTC dates like the labels used by the existing output code."""
    return [timestamp.strftime("A.D. %Y-%b-%d %H:%M:%S.%f") for timestamp in datetimes]


def write_ephemeris_csv(
    path: Path,
    datetimes: list[datetime],
    ephemeris: dict[str, np.ndarray | Time],
) -> None:
    """Write the complete five-minute TLE-generated vector table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    times = ephemeris["times"]
    if not isinstance(times, Time):
        raise TypeError("Ephemeris times must be an Astropy Time array.")

    position_teme = np.asarray(ephemeris["goes_teme_position_km"])
    velocity_teme = np.asarray(ephemeris["goes_teme_velocity_km_s"])
    position_gcrs = np.asarray(ephemeris["goes_gcrs_position_km"])
    earth_vectors = np.asarray(ephemeris["earth_from_goes_km"])
    moon_vectors = np.asarray(ephemeris["moon_from_goes_km"])
    sun_vectors = np.asarray(ephemeris["sun_from_goes_km"])

    header = [
        "utc",
        "julian_date_ut",
        "goes18_teme_x_km",
        "goes18_teme_y_km",
        "goes18_teme_z_km",
        "goes18_teme_vx_km_s",
        "goes18_teme_vy_km_s",
        "goes18_teme_vz_km_s",
        "goes18_gcrs_x_km",
        "goes18_gcrs_y_km",
        "goes18_gcrs_z_km",
        "earth_from_goes_x_km",
        "earth_from_goes_y_km",
        "earth_from_goes_z_km",
        "moon_from_goes_x_km",
        "moon_from_goes_y_km",
        "moon_from_goes_z_km",
        "sun_from_goes_x_km",
        "sun_from_goes_y_km",
        "sun_from_goes_z_km",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, timestamp in enumerate(datetimes):
            values = np.concatenate(
                (
                    position_teme[index],
                    velocity_teme[index],
                    position_gcrs[index],
                    earth_vectors[index],
                    moon_vectors[index],
                    sun_vectors[index],
                )
            )
            writer.writerow(
                [
                    timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    f"{times.utc.jd[index]:.9f}",
                    *(f"{value:.9f}" for value in values),
                ]
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate GOES-18 visible sky using a TLE, SGP4, and Astropy's "
            "built-in solar-system ephemeris; JPL Horizons is not queried."
        )
    )
    parser.add_argument(
        "--start",
        help="Start date in YYYY-MM-DD format; prompted for if omitted",
    )
    parser.add_argument(
        "--stop",
        help="End date in YYYY-MM-DD format; prompted for if omitted",
    )
    parser.add_argument(
        "--step",
        default="1 h",
        help="Coarse output step outside angular-alignment regions (default: '1 h')",
    )
    parser.add_argument(
        "--tle-file",
        type=Path,
        help=(
            "Use this local GOES-18 TLE instead of attempting an online download"
        ),
    )
    parser.add_argument(
        "--tle-url",
        default=CELESTRAK_TLE_URL,
        help="Current GOES-18 TLE URL used when --tle-file is omitted",
    )
    parser.add_argument(
        "--fallback-tle-file",
        "--fallback-tle",
        dest="fallback_tle_file",
        type=Path,
        default=DEFAULT_FALLBACK_TLE_FILE,
        help=(
            "Local TLE used automatically if the CelesTrak download fails "
            "(default: goes18_2026-08-27.tle beside this script)"
        ),
    )
    parser.add_argument(
        "--max-tle-age",
        type=float,
        default=DEFAULT_MAX_TLE_AGE_DAYS,
        help=(
            "Warn when a requested time is more than this many days from the "
            "TLE epoch (default: 14)"
        ),
    )
    parser.add_argument(
        "--earth-clearance",
        type=float,
        default=20.0,
        help="Required clearance beyond the Earth limb, in degrees",
    )
    parser.add_argument(
        "--moon-clearance",
        type=float,
        default=20.0,
        help="Required clearance from the selected Moon reference, in degrees",
    )
    parser.add_argument(
        "--moon-reference",
        choices=("center", "limb"),
        default="center",
        help="Interpret Moon clearance from its center (default) or limb",
    )
    parser.add_argument(
        "--sun-exclusion",
        type=float,
        choices=(30.0, 45.0),
        help="Sun-centered exclusion radius (30 or 45 degrees); prompted if omitted",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("goes18_visible_sky_tle"),
        help="Output path without extension",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    start = args.start or prompt_for_date("Enter start date")
    stop = args.stop or prompt_for_date("Enter end date")
    start, stop = validate_date_range(start, stop)
    sun_exclusion = args.sun_exclusion or prompt_for_sun_exclusion()
    coarse_step_minutes = step_to_minutes(args.step)

    for label, clearance in (
        ("Earth", args.earth_clearance),
        ("Moon", args.moon_clearance),
    ):
        if not 0.0 <= clearance <= 60.0:
            raise SystemExit(f"{label} clearance must be between 0 and 60 degrees.")
    if args.max_tle_age <= 0.0:
        raise SystemExit("--max-tle-age must be positive.")

    if args.tle_file is None:
        print("Downloading current GOES-18 TLE from CelesTrak...")
    else:
        print(f"Reading GOES-18 TLE from {args.tle_file}...")
    name, line1, line2, tle_source = load_tle(
        args.tle_file,
        args.tle_url,
        args.fallback_tle_file,
    )
    satellite = Satrec.twoline2rv(line1, line2, WGS72)
    if satellite.satnum != GOES18_NORAD_ID:
        raise RuntimeError(
            f"TLE resolved to NORAD {satellite.satnum}, not GOES-18 "
            f"({GOES18_NORAD_ID})."
        )

    datetimes = build_sample_datetimes(start, stop)
    astropy_times = Time(datetimes, scale="utc")
    maximum_tle_age = warn_if_tle_is_stale(
        astropy_times,
        satellite,
        args.max_tle_age,
    )

    print("Propagating GOES-18 with SGP4 and generating Moon/Sun ephemerides...")
    ephemeris = propagate_tle_ephemeris(datetimes, satellite)
    earth_vectors = np.asarray(ephemeris["earth_from_goes_km"])
    moon_vectors = np.asarray(ephemeris["moon_from_goes_km"])
    sun_vectors = np.asarray(ephemeris["sun_from_goes_km"])
    results = compute_visible_sky(
        earth_vectors,
        moon_vectors,
        sun_vectors,
        args.earth_clearance,
        args.moon_clearance,
        args.moon_reference,
        sun_exclusion,
    )

    labels = calendar_labels(datetimes)
    angular_limits_deg = ANGULAR_FINE_LIMITS_DEG[sun_exclusion]
    selected_indices = adaptive_sample_indices(
        labels,
        results["earth_moon_separation_rad"],
        results["earth_sun_separation_rad"],
        results["moon_sun_separation_rad"],
        coarse_step_minutes,
        angular_limits_deg,
    )
    selected_labels = [labels[index] for index in selected_indices]
    selected_results = subset_results(results, selected_indices)
    selected_jd = astropy_times.utc.jd[selected_indices]

    csv_path = args.output_prefix.with_suffix(".csv")
    plot_path = args.output_prefix.with_suffix(".png")
    ephemeris_path = args.output_prefix.parent / (
        args.output_prefix.name + "_ephemeris.csv"
    )
    tle_path = args.output_prefix.parent / (args.output_prefix.name + "_used.tle")
    save_used_tle(tle_path, name, line1, line2)
    write_ephemeris_csv(ephemeris_path, datetimes, ephemeris)
    write_results_csv(csv_path, selected_jd, selected_labels, selected_results)

    epoch_datetime = tle_epoch(satellite).to_datetime(timezone=timezone.utc)
    epoch_label = epoch_datetime.strftime("%Y-%m-%d %H:%M UTC")
    observer_name = "GOES-18 TLE/SGP4"
    sampling_description = f"5 min near angular alignments; {args.step} elsewhere"
    write_plot(
        plot_path,
        selected_labels,
        selected_results["visible_fraction"],
        observer_name,
        args.earth_clearance,
        args.moon_clearance,
        args.moon_reference,
        sun_exclusion,
        sampling_description,
        args.step,
    )

    em_limit, es_limit, ms_limit = angular_limits_deg
    fraction = results["visible_fraction"]
    percent = 100.0 * fraction
    print(f"TLE source: {tle_source}")
    print(f"TLE object: {name} (NORAD {satellite.satnum})")
    print(f"TLE epoch: {epoch_label}")
    print(f"Maximum distance from TLE epoch: {maximum_tle_age:.2f} days")
    print(f"SGP4 mode: {'deep-space' if satellite.method == 'd' else 'near-Earth'}")
    print(f"Date range: {start} through {stop} UTC")
    print(f"Sun exclusion angle: {sun_exclusion:g} degrees from Sun center")
    print(
        "Five-minute angular limits: "
        f"Earth-Moon <= {em_limit:g} deg, "
        f"Earth-Sun <= {es_limit:g} deg, "
        f"Moon-Sun <= {ms_limit:g} deg"
    )
    print(f"Full ephemeris samples (5 min): {len(datetimes)}")
    print(
        "Visibility CSV/plot samples (5 min near angular alignments; "
        f"{args.step} elsewhere): {len(selected_indices)}"
    )
    print(
        "Visible sky fraction (unitless): "
        f"min={np.min(fraction):.9f}, "
        f"mean={np.mean(fraction):.9f}, "
        f"max={np.max(fraction):.9f}"
    )
    print(
        "Visible sky percent: "
        f"min={np.min(percent):.6f}%, "
        f"mean={np.mean(percent):.6f}%, "
        f"max={np.max(percent):.6f}%"
    )
    print(f"Wrote TLE: {tle_path.resolve()}")
    print(f"Wrote ephemeris table: {ephemeris_path.resolve()}")
    print(f"Wrote visibility CSV: {csv_path.resolve()}")
    print(f"Wrote plot: {plot_path.resolve()}")


if __name__ == "__main__":
    main()
