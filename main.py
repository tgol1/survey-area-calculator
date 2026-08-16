# Main Script for GOES-18 visible-sky fraction calculation using JPL Horizons ephemerides.
# George Tolis
# 14 Aug 26

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

# Avoid a warning on systems whose default Matplotlib config folder is read-only.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "goes18-matplotlib-cache")
)
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"
GOES18_SPK_ID = -151850
EARTH_ID = 399
MOON_ID = 301
EARTH_MEAN_RADIUS_KM = 6371.0084
MOON_MEAN_RADIUS_KM = 1737.4
FULL_SKY_SR = 4.0 * math.pi

# Nodes for deterministic integration of a partially overlapping spherical lens.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(128)


def _horizons_value(value: object) -> str:
    """Return a value quoted in the form expected by Horizons."""
    return f"'{value}'"


def fetch_vectors(
    target_id: int,
    observer_spk_id: int,
    start: str,
    stop: str,
    step: str,
) -> tuple[np.ndarray, list[str], np.ndarray, str]:
    """Download geometric ICRF position vectors from JPL Horizons.

    Returns Julian dates, calendar labels, XYZ vectors in km, and the center
    description from the Horizons response.
    """
    params = {
        "format": "json",
        "COMMAND": _horizons_value(target_id),
        "OBJ_DATA": _horizons_value("NO"),
        "MAKE_EPHEM": _horizons_value("YES"),
        "EPHEM_TYPE": _horizons_value("VECTORS"),
        "CENTER": _horizons_value(f"@{observer_spk_id}"),
        "START_TIME": _horizons_value(start),
        "STOP_TIME": _horizons_value(stop),
        "STEP_SIZE": _horizons_value(step),
        "TIME_TYPE": _horizons_value("UT"),
        "OUT_UNITS": _horizons_value("KM-S"),
        "REF_SYSTEM": _horizons_value("ICRF"),
        "REF_PLANE": _horizons_value("FRAME"),
        "VEC_CORR": _horizons_value("NONE"),
        "VEC_TABLE": _horizons_value("1"),
        "CSV_FORMAT": _horizons_value("YES"),
    }
    url = HORIZONS_API + "?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "GOES18-visible-sky/1.0"})

    try:
        with urlopen(request, timeout=90) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Horizons request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Horizons returned a response that was not JSON.") from exc

    if "error" in payload:
        raise RuntimeError(f"Horizons error: {payload['error']}")

    result = payload.get("result", "")
    if "$$SOE" not in result or "$$EOE" not in result:
        # Horizons sometimes reports input/time-span errors only in result text.
        raise RuntimeError("No ephemeris table was returned by Horizons:\n" + result[:2500])

    center_match = re.search(r"^Center body name:\s*(.+)$", result, re.MULTILINE)
    center_description = (
        center_match.group(1).split("{", 1)[0].strip()
        if center_match
        else "unknown"
    )
    if f"({observer_spk_id})" not in center_description:
        raise RuntimeError(
            "Horizons did not use the requested observer. "
            f"Returned center: {center_description}"
        )

    table = result.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    julian_dates: list[float] = []
    calendar_dates: list[str] = []
    vectors: list[list[float]] = []

    for row in csv.reader(io.StringIO(table.strip())):
        if len(row) < 5:
            continue
        try:
            jd = float(row[0])
            xyz = [float(row[2]), float(row[3]), float(row[4])]
        except ValueError as exc:
            raise RuntimeError(f"Could not parse Horizons row: {row}") from exc
        julian_dates.append(jd)
        calendar_dates.append(row[1].strip())
        vectors.append(xyz)

    if not vectors:
        raise RuntimeError("Horizons returned an empty ephemeris table.")

    return (
        np.asarray(julian_dates, dtype=float),
        calendar_dates,
        np.asarray(vectors, dtype=float),
        center_description,
    )


def spherical_cap_area(radius_rad: float) -> float:
    """Solid angle of a spherical cap of angular radius ``radius_rad``."""
    return 2.0 * math.pi * (1.0 - math.cos(radius_rad))


def spherical_cap_intersection_area(
    radius_1: float, radius_2: float, separation: float
) -> float:
    """Solid angle shared by two spherical caps, in steradians.

    The partial-overlap case is evaluated by Gauss-Legendre quadrature in a
    polar coordinate system centered on cap 1.  The result is deterministic,
    unlike a Monte Carlo sky sampling calculation.
    """
    radius_1 = float(radius_1)
    radius_2 = float(radius_2)
    separation = float(np.clip(separation, 0.0, math.pi))

    if radius_1 < 0.0 or radius_2 < 0.0:
        raise ValueError("Spherical-cap radii cannot be negative.")
    if radius_1 + radius_2 >= math.pi:
        raise ValueError(
            "This implementation requires the two cap radii to sum to less "
            "than 180 degrees. Reduce the avoidance angles."
        )

    # Disjoint caps.
    if separation >= radius_1 + radius_2:
        return 0.0

    # One cap is fully inside the other.
    if separation <= abs(radius_1 - radius_2):
        return spherical_cap_area(min(radius_1, radius_2))

    # In the partial-overlap case, central latitude rings of cap 1 may be
    # completely within cap 2.
    full_ring_limit = max(0.0, min(radius_1, radius_2 - separation))
    area = spherical_cap_area(full_ring_limit)

    partial_lower = abs(separation - radius_2)
    partial_upper = min(radius_1, separation + radius_2)
    if partial_upper <= partial_lower:
        return area

    # Map standard Gauss-Legendre nodes from [-1, 1] to the partial interval.
    half_width = 0.5 * (partial_upper - partial_lower)
    midpoint = 0.5 * (partial_upper + partial_lower)
    theta = midpoint + half_width * _GL_NODES

    denominator = np.sin(theta) * math.sin(separation)
    q = (
        math.cos(radius_2)
        - np.cos(theta) * math.cos(separation)
    ) / denominator
    half_longitude = np.arccos(np.clip(q, -1.0, 1.0))
    integrand = 2.0 * half_longitude * np.sin(theta)
    area += half_width * float(np.dot(_GL_WEIGHTS, integrand))
    return area


def compute_visible_sky(
    earth_vectors_km: np.ndarray,
    moon_vectors_km: np.ndarray,
    earth_clearance_deg: float,
    moon_clearance_deg: float,
    moon_reference: str,
) -> dict[str, np.ndarray]:
    """Calculate distances, cap geometry, and visible sky fraction."""
    earth_distance = np.linalg.norm(earth_vectors_km, axis=1)
    moon_distance = np.linalg.norm(moon_vectors_km, axis=1)

    if np.any(earth_distance <= EARTH_MEAN_RADIUS_KM):
        raise ValueError("Observer is not outside the Earth.")
    if np.any(moon_distance <= MOON_MEAN_RADIUS_KM):
        raise ValueError("Observer is not outside the Moon.")

    earth_apparent_radius = np.arcsin(EARTH_MEAN_RADIUS_KM / earth_distance)
    moon_apparent_radius = np.arcsin(MOON_MEAN_RADIUS_KM / moon_distance)

    earth_cap_radius = earth_apparent_radius + math.radians(earth_clearance_deg)
    if moon_reference == "limb":
        moon_cap_radius = moon_apparent_radius + math.radians(moon_clearance_deg)
    else:
        moon_cap_radius = np.full_like(
            moon_apparent_radius, math.radians(moon_clearance_deg)
        )

    cos_separation = np.einsum("ij,ij->i", earth_vectors_km, moon_vectors_km)
    cos_separation /= earth_distance * moon_distance
    separation = np.arccos(np.clip(cos_separation, -1.0, 1.0))

    overlap_area = np.empty_like(separation)
    visible_fraction = np.empty_like(separation)
    for index, (earth_cap, moon_cap, angle) in enumerate(
        zip(earth_cap_radius, moon_cap_radius, separation)
    ):
        earth_area = spherical_cap_area(float(earth_cap))
        moon_area = spherical_cap_area(float(moon_cap))
        overlap = spherical_cap_intersection_area(earth_cap, moon_cap, angle)
        excluded_area = earth_area + moon_area - overlap
        overlap_area[index] = overlap
        visible_fraction[index] = np.clip(1.0 - excluded_area / FULL_SKY_SR, 0.0, 1.0)

    return {
        "earth_distance_km": earth_distance,
        "moon_distance_km": moon_distance,
        "earth_moon_separation_rad": separation,
        "earth_apparent_radius_rad": earth_apparent_radius,
        "moon_apparent_radius_rad": moon_apparent_radius,
        "earth_cap_radius_rad": earth_cap_radius,
        "moon_cap_radius_rad": moon_cap_radius,
        "overlap_area_sr": overlap_area,
        "visible_fraction": visible_fraction,
    }


def horizons_calendar_to_utc(calendar_date: str) -> datetime:
    """Parse a modern A.D. calendar label returned by Horizons."""
    value = calendar_date.removeprefix("A.D. ")
    for date_format in ("%Y-%b-%d %H:%M:%S.%f", "%Y-%b-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, date_format).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"Unsupported Horizons calendar date: {calendar_date}")


def utc_isoformat(timestamp: datetime) -> str:
    """Format a UTC timestamp without spurious Julian-date roundoff."""
    value = timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return value.replace(".000Z", "Z")


def write_results_csv(
    path: Path,
    julian_dates: np.ndarray,
    calendar_dates: list[str],
    results: dict[str, np.ndarray],
) -> None:
    """Write the full calculation to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    degrees = np.degrees
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "utc",
                "julian_date_ut",
                "earth_distance_km",
                "moon_distance_km",
                "earth_moon_separation_deg",
                "earth_apparent_radius_deg",
                "earth_exclusion_radius_deg",
                "moon_apparent_radius_deg",
                "moon_exclusion_radius_deg",
                "earth_moon_cap_overlap_sr",
                "visible_sky_fraction",
                "visible_sky_percent",
            ]
        )
        for index, jd in enumerate(julian_dates):
            writer.writerow(
                [
                    utc_isoformat(horizons_calendar_to_utc(calendar_dates[index])),
                    f"{jd:.9f}",
                    f"{results['earth_distance_km'][index]:.6f}",
                    f"{results['moon_distance_km'][index]:.6f}",
                    f"{degrees(results['earth_moon_separation_rad'][index]):.9f}",
                    f"{degrees(results['earth_apparent_radius_rad'][index]):.9f}",
                    f"{degrees(results['earth_cap_radius_rad'][index]):.9f}",
                    f"{degrees(results['moon_apparent_radius_rad'][index]):.9f}",
                    f"{degrees(results['moon_cap_radius_rad'][index]):.9f}",
                    f"{results['overlap_area_sr'][index]:.12f}",
                    f"{results['visible_fraction'][index]:.12f}",
                    f"{100.0 * results['visible_fraction'][index]:.9f}",
                ]
            )


def write_plot(
    path: Path,
    calendar_dates: list[str],
    visible_fraction: np.ndarray,
    observer_name: str,
    earth_clearance_deg: float,
    moon_clearance_deg: float,
    moon_reference: str,
) -> None:
    """Create the requested visible-fraction-versus-time plot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    times = [horizons_calendar_to_utc(value) for value in calendar_dates]
    percent = 100.0 * visible_fraction

    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    ax.plot(times, percent, color="#176B87", linewidth=2.0)
    ax.fill_between(times, percent, np.min(percent) - 1.0, color="#64CCC5", alpha=0.16)

    spread = float(np.max(percent) - np.min(percent))
    padding = max(0.15, 0.12 * spread)
    ax.set_ylim(float(np.min(percent) - padding), float(np.max(percent) + padding))
    ax.set_ylabel("Visible sky (%)")
    ax.set_xlabel("Time (UTC)")
    ax.grid(True, color="#D7DEE5", linewidth=0.8, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    moon_wording = "Moon limb" if moon_reference == "limb" else "Moon center"
    ax.set_title(
        "Instantaneous visible fraction of sky from GOES-18",
        loc="left",
        fontsize=14,
        weight="bold",
        pad=22,
    )
    ax.text(
        0.0,
        1.02,
        f"Earth limb + {earth_clearance_deg:g} deg; "
        f"{moon_wording} + {moon_clearance_deg:g} deg | {observer_name}",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#4B5563",
        va="bottom",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate visible sky fraction from GOES-18 using JPL Horizons "
            "Earth and Moon ephemerides."
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
        help="Horizons time step, for example '1 h', '30 min', or '1 d'",
    )
    parser.add_argument(
        "--observer-spk",
        type=int,
        default=GOES18_SPK_ID,
        help="Observer spacecraft SPK ID (default: GOES-18, -151850)",
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
        help=(
            "Interpret Moon clearance from its center (default) or limb. "
            "The assignment wording is ambiguous, so both are supported."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("goes18_visible_sky"),
        help="Output path without extension; .csv and .png are added",
    )
    return parser.parse_args()


def prompt_for_date(label: str) -> str:
    """Prompt until the user enters a valid YYYY-MM-DD calendar date."""
    while True:
        try:
            value = input(f"{label} (YYYY-MM-DD): ").strip()
        except EOFError as exc:
            raise SystemExit(
                "No interactive input was available. Use --start and --stop instead."
            ) from exc

        try:
            parsed = datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            print("Invalid date. Enter it in YYYY-MM-DD format, for example 2026-08-01.")
            continue

        # strptime accepts some non-zero-padded inputs, so normalize the result.
        return parsed.strftime("%Y-%m-%d")


def validate_date_range(start: str, stop: str) -> tuple[str, str]:
    """Validate and normalize dates supplied through command-line options."""
    try:
        start_date = datetime.strptime(start, "%Y-%m-%d")
        stop_date = datetime.strptime(stop, "%Y-%m-%d")
    except ValueError as exc:
        raise SystemExit(
            "Start and end dates must use YYYY-MM-DD format, for example 2026-08-01."
        ) from exc

    if stop_date <= start_date:
        raise SystemExit("The end date must be later than the start date.")

    return start_date.strftime("%Y-%m-%d"), stop_date.strftime("%Y-%m-%d")


def main() -> None:
    args = parse_arguments()
    start = args.start or prompt_for_date("Enter start date")
    stop = args.stop or prompt_for_date("Enter end date")
    start, stop = validate_date_range(start, stop)

    for label, clearance in (
        ("Earth", args.earth_clearance),
        ("Moon", args.moon_clearance),
    ):
        if not 0.0 <= clearance <= 60.0:
            raise SystemExit(f"{label} clearance must be between 0 and 60 degrees.")

    print("Downloading Earth vectors from JPL Horizons...")
    earth_jd, earth_calendar, earth_vectors, observer_name = fetch_vectors(
        EARTH_ID, args.observer_spk, start, stop, args.step
    )
    print("Downloading Moon vectors from JPL Horizons...")
    moon_jd, moon_calendar, moon_vectors, moon_observer_name = fetch_vectors(
        MOON_ID, args.observer_spk, start, stop, args.step
    )

    if observer_name != moon_observer_name:
        raise RuntimeError("Earth and Moon queries returned different observers.")
    if earth_jd.shape != moon_jd.shape or not np.allclose(
        earth_jd, moon_jd, rtol=0.0, atol=1e-9
    ):
        raise RuntimeError("Earth and Moon ephemeris times do not match.")
    if earth_calendar != moon_calendar:
        raise RuntimeError("Earth and Moon calendar labels do not match.")

    if args.observer_spk == GOES18_SPK_ID and "GOES-18" not in observer_name:
        raise RuntimeError(
            f"SPK ID {GOES18_SPK_ID} did not resolve to GOES-18: {observer_name}"
        )

    results = compute_visible_sky(
        earth_vectors,
        moon_vectors,
        args.earth_clearance,
        args.moon_clearance,
        args.moon_reference,
    )

    csv_path = args.output_prefix.with_suffix(".csv")
    plot_path = args.output_prefix.with_suffix(".png")
    write_results_csv(csv_path, earth_jd, earth_calendar, results)
    write_plot(
        plot_path,
        earth_calendar,
        results["visible_fraction"],
        observer_name,
        args.earth_clearance,
        args.moon_clearance,
        args.moon_reference,
    )

    fraction = results["visible_fraction"]
    percent = 100.0 * fraction
    print(f"Observer verified by Horizons: {observer_name}")
    print(f"Date range: {start} through {stop} UTC")
    print(f"Samples: {len(earth_jd)}")
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
    print(f"Wrote CSV: {csv_path.resolve()}")
    print(f"Wrote plot: {plot_path.resolve()}")


if __name__ == "__main__":
    main()