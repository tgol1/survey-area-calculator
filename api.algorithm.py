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
SUN_ID = 10
EARTH_MEAN_RADIUS_KM = 6371.0084
MOON_MEAN_RADIUS_KM = 1737.4
SUN_MEAN_RADIUS_KM = 695700.0
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


def _intersect_linear_intervals(
    intervals_1: list[tuple[float, float]],
    intervals_2: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Intersect two sets of non-wrapping intervals on [0, 2*pi]."""
    intersections: list[tuple[float, float]] = []
    for start_1, stop_1 in intervals_1:
        for start_2, stop_2 in intervals_2:
            start = max(start_1, start_2)
            stop = min(stop_1, stop_2)
            if stop > start:
                intersections.append((start, stop))
    return intersections


def _cap_longitude_intervals(
    theta: float,
    center_separation: float,
    center_longitude: float,
    cap_radius: float,
) -> list[tuple[float, float]]:
    """Return allowed longitude intervals for one cap at polar angle theta."""
    two_pi = 2.0 * math.pi
    denominator = math.sin(theta) * math.sin(center_separation)

    # When the cap center is at either polar axis, the entire latitude ring is
    # either inside or outside the cap.
    if abs(denominator) < 1e-14:
        cos_distance = (
            math.cos(theta) * math.cos(center_separation)
            + math.sin(theta) * math.sin(center_separation)
        )
        angular_distance = math.acos(float(np.clip(cos_distance, -1.0, 1.0)))
        return [(0.0, two_pi)] if angular_distance <= cap_radius else []

    q = (
        math.cos(cap_radius)
        - math.cos(theta) * math.cos(center_separation)
    ) / denominator

    if q <= -1.0:
        return [(0.0, two_pi)]
    if q >= 1.0:
        return []

    half_width = math.acos(float(np.clip(q, -1.0, 1.0)))
    start = (center_longitude - half_width) % two_pi
    stop = (center_longitude + half_width) % two_pi
    if start <= stop:
        return [(start, stop)]
    return [(0.0, stop), (start, two_pi)]


def spherical_cap_triple_intersection_area(
    centers: np.ndarray,
    radii: np.ndarray,
) -> float:
    """Solid angle common to three spherical caps, in steradians.

    The smallest cap is used as the polar integration domain. At each polar
    angle, the function intersects the longitude intervals admitted by the
    other two caps and integrates their combined length with Gauss-Legendre
    quadrature.
    """
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if centers.shape != (3, 3) or radii.shape != (3,):
        raise ValueError("Three cap centers and three cap radii are required.")
    if np.any(radii < 0.0) or np.any(radii >= math.pi):
        raise ValueError("Spherical-cap radii must be between 0 and 180 degrees.")

    norms = np.linalg.norm(centers, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("Spherical-cap center vectors cannot be zero.")
    centers = centers / norms[:, None]

    separations = np.arccos(
        np.clip(centers @ centers.T, -1.0, 1.0)
    )

    # A triple intersection is impossible if any pair of caps is disjoint.
    for first in range(3):
        for second in range(first + 1, 3):
            if separations[first, second] >= radii[first] + radii[second]:
                return 0.0

    # If one entire cap is contained in both others, it is the intersection.
    for index in range(3):
        if all(
            index == other
            or separations[index, other] + radii[index] <= radii[other]
            for other in range(3)
        ):
            return spherical_cap_area(float(radii[index]))

    base_index = int(np.argmin(radii))
    order = [base_index] + [index for index in range(3) if index != base_index]
    centers = centers[order]
    radii = radii[order]
    base_center = centers[0]

    # Define zero longitude using whichever other center has the largest
    # projection into the base cap's tangent plane.
    tangent_vectors = centers[1:] - (centers[1:] @ base_center)[:, None] * base_center
    tangent_norms = np.linalg.norm(tangent_vectors, axis=1)
    reference_index = int(np.argmax(tangent_norms))
    if tangent_norms[reference_index] > 1e-12:
        longitude_axis = tangent_vectors[reference_index] / tangent_norms[reference_index]
    else:
        coordinate_axis = np.eye(3)[int(np.argmin(np.abs(base_center)))]
        longitude_axis = np.cross(coordinate_axis, base_center)
        longitude_axis /= np.linalg.norm(longitude_axis)
    second_axis = np.cross(base_center, longitude_axis)

    other_separations: list[float] = []
    other_longitudes: list[float] = []
    for center in centers[1:]:
        separation = math.acos(float(np.clip(np.dot(base_center, center), -1.0, 1.0)))
        longitude = math.atan2(
            float(np.dot(center, second_axis)),
            float(np.dot(center, longitude_axis)),
        ) % (2.0 * math.pi)
        other_separations.append(separation)
        other_longitudes.append(longitude)

    # Split the polar integration at cap-boundary transitions so each
    # Gauss-Legendre interval remains smooth.
    boundaries = [0.0, float(radii[0])]
    for separation, radius in zip(other_separations, radii[1:]):
        for boundary in (abs(separation - radius), separation + radius):
            if 0.0 < boundary < radii[0]:
                boundaries.append(float(boundary))
    boundaries = sorted(set(boundaries))

    area = 0.0
    full_ring = [(0.0, 2.0 * math.pi)]
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        if upper <= lower:
            continue
        half_width = 0.5 * (upper - lower)
        midpoint = 0.5 * (upper + lower)
        thetas = midpoint + half_width * _GL_NODES
        integrand = np.empty_like(thetas)

        for index, theta in enumerate(thetas):
            allowed = full_ring
            for separation, longitude, radius in zip(
                other_separations,
                other_longitudes,
                radii[1:],
            ):
                cap_intervals = _cap_longitude_intervals(
                    float(theta), separation, longitude, float(radius)
                )
                allowed = _intersect_linear_intervals(allowed, cap_intervals)
                if not allowed:
                    break
            longitude_length = sum(stop - start for start, stop in allowed)
            integrand[index] = longitude_length * math.sin(float(theta))

        area += half_width * float(np.dot(_GL_WEIGHTS, integrand))

    return float(np.clip(area, 0.0, spherical_cap_area(float(radii[0]))))


def compute_visible_sky(
    earth_vectors_km: np.ndarray,
    moon_vectors_km: np.ndarray,
    sun_vectors_km: np.ndarray,
    earth_clearance_deg: float,
    moon_clearance_deg: float,
    moon_reference: str,
    sun_exclusion_deg: float,
) -> dict[str, np.ndarray]:
    """Calculate Earth-Moon-Sun cap geometry and visible sky fraction."""
    earth_distance = np.linalg.norm(earth_vectors_km, axis=1)
    moon_distance = np.linalg.norm(moon_vectors_km, axis=1)
    sun_distance = np.linalg.norm(sun_vectors_km, axis=1)

    if np.any(earth_distance <= EARTH_MEAN_RADIUS_KM):
        raise ValueError("Observer is not outside the Earth.")
    if np.any(moon_distance <= MOON_MEAN_RADIUS_KM):
        raise ValueError("Observer is not outside the Moon.")
    if np.any(sun_distance <= SUN_MEAN_RADIUS_KM):
        raise ValueError("Observer is not outside the Sun.")

    earth_apparent_radius = np.arcsin(EARTH_MEAN_RADIUS_KM / earth_distance)
    moon_apparent_radius = np.arcsin(MOON_MEAN_RADIUS_KM / moon_distance)
    sun_apparent_radius = np.arcsin(SUN_MEAN_RADIUS_KM / sun_distance)

    earth_cap_radius = earth_apparent_radius + math.radians(earth_clearance_deg)
    if moon_reference == "limb":
        moon_cap_radius = moon_apparent_radius + math.radians(moon_clearance_deg)
    else:
        moon_cap_radius = np.full_like(
            moon_apparent_radius, math.radians(moon_clearance_deg)
        )
    sun_cap_radius = np.full_like(
        sun_apparent_radius, math.radians(sun_exclusion_deg)
    )

    earth_unit = earth_vectors_km / earth_distance[:, None]
    moon_unit = moon_vectors_km / moon_distance[:, None]
    sun_unit = sun_vectors_km / sun_distance[:, None]

    earth_moon_separation = np.arccos(
        np.clip(np.einsum("ij,ij->i", earth_unit, moon_unit), -1.0, 1.0)
    )
    earth_sun_separation = np.arccos(
        np.clip(np.einsum("ij,ij->i", earth_unit, sun_unit), -1.0, 1.0)
    )
    moon_sun_separation = np.arccos(
        np.clip(np.einsum("ij,ij->i", moon_unit, sun_unit), -1.0, 1.0)
    )

    sample_count = len(earth_distance)
    earth_moon_overlap = np.empty(sample_count, dtype=float)
    earth_sun_overlap = np.empty(sample_count, dtype=float)
    moon_sun_overlap = np.empty(sample_count, dtype=float)
    triple_overlap = np.empty(sample_count, dtype=float)
    excluded_area = np.empty(sample_count, dtype=float)
    visible_fraction = np.empty(sample_count, dtype=float)

    for index in range(sample_count):
        earth_cap = float(earth_cap_radius[index])
        moon_cap = float(moon_cap_radius[index])
        sun_cap = float(sun_cap_radius[index])
        earth_area = spherical_cap_area(earth_cap)
        moon_area = spherical_cap_area(moon_cap)
        sun_area = spherical_cap_area(sun_cap)

        em_overlap = spherical_cap_intersection_area(
            earth_cap, moon_cap, float(earth_moon_separation[index])
        )
        es_overlap = spherical_cap_intersection_area(
            earth_cap, sun_cap, float(earth_sun_separation[index])
        )
        ms_overlap = spherical_cap_intersection_area(
            moon_cap, sun_cap, float(moon_sun_separation[index])
        )

        if em_overlap > 0.0 and es_overlap > 0.0 and ms_overlap > 0.0:
            three_way_overlap = spherical_cap_triple_intersection_area(
                np.vstack((earth_unit[index], moon_unit[index], sun_unit[index])),
                np.asarray((earth_cap, moon_cap, sun_cap)),
            )
        else:
            three_way_overlap = 0.0

        # Inclusion-exclusion for three caps:
        # |E union M union S| = E + M + S - EM - ES - MS + EMS.
        union_area = (
            earth_area
            + moon_area
            + sun_area
            - em_overlap
            - es_overlap
            - ms_overlap
            + three_way_overlap
        )
        union_area = float(np.clip(union_area, 0.0, FULL_SKY_SR))

        earth_moon_overlap[index] = em_overlap
        earth_sun_overlap[index] = es_overlap
        moon_sun_overlap[index] = ms_overlap
        triple_overlap[index] = three_way_overlap
        excluded_area[index] = union_area
        visible_fraction[index] = 1.0 - union_area / FULL_SKY_SR

    return {
        "earth_distance_km": earth_distance,
        "moon_distance_km": moon_distance,
        "sun_distance_km": sun_distance,
        "earth_moon_separation_rad": earth_moon_separation,
        "earth_sun_separation_rad": earth_sun_separation,
        "moon_sun_separation_rad": moon_sun_separation,
        "earth_apparent_radius_rad": earth_apparent_radius,
        "moon_apparent_radius_rad": moon_apparent_radius,
        "sun_apparent_radius_rad": sun_apparent_radius,
        "earth_cap_radius_rad": earth_cap_radius,
        "moon_cap_radius_rad": moon_cap_radius,
        "sun_cap_radius_rad": sun_cap_radius,
        "earth_moon_overlap_area_sr": earth_moon_overlap,
        "earth_sun_overlap_area_sr": earth_sun_overlap,
        "moon_sun_overlap_area_sr": moon_sun_overlap,
        "triple_overlap_area_sr": triple_overlap,
        "excluded_area_sr": excluded_area,
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
                "sun_distance_km",
                "earth_moon_separation_deg",
                "earth_sun_separation_deg",
                "moon_sun_separation_deg",
                "earth_apparent_radius_deg",
                "earth_exclusion_radius_deg",
                "moon_apparent_radius_deg",
                "moon_exclusion_radius_deg",
                "sun_apparent_radius_deg",
                "sun_exclusion_radius_deg",
                "earth_moon_cap_overlap_sr",
                "earth_sun_cap_overlap_sr",
                "moon_sun_cap_overlap_sr",
                "earth_moon_sun_triple_overlap_sr",
                "total_excluded_sky_sr",
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
                    f"{results['sun_distance_km'][index]:.6f}",
                    f"{degrees(results['earth_moon_separation_rad'][index]):.9f}",
                    f"{degrees(results['earth_sun_separation_rad'][index]):.9f}",
                    f"{degrees(results['moon_sun_separation_rad'][index]):.9f}",
                    f"{degrees(results['earth_apparent_radius_rad'][index]):.9f}",
                    f"{degrees(results['earth_cap_radius_rad'][index]):.9f}",
                    f"{degrees(results['moon_apparent_radius_rad'][index]):.9f}",
                    f"{degrees(results['moon_cap_radius_rad'][index]):.9f}",
                    f"{degrees(results['sun_apparent_radius_rad'][index]):.9f}",
                    f"{degrees(results['sun_cap_radius_rad'][index]):.9f}",
                    f"{results['earth_moon_overlap_area_sr'][index]:.12f}",
                    f"{results['earth_sun_overlap_area_sr'][index]:.12f}",
                    f"{results['moon_sun_overlap_area_sr'][index]:.12f}",
                    f"{results['triple_overlap_area_sr'][index]:.12f}",
                    f"{results['excluded_area_sr'][index]:.12f}",
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
    sun_exclusion_deg: float,
    sample_step: str,
) -> None:
    """Create a line plot with automatically spaced, readable UTC labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    times = [horizons_calendar_to_utc(value) for value in calendar_dates]
    percent = 100.0 * visible_fraction

    span_days = max(1.0, (times[-1] - times[0]).total_seconds() / 86400.0)
    figure_width = min(24.0, max(10.5, 6.0 + 0.5 * span_days))
    fig, ax = plt.subplots(
        figsize=(figure_width, 6.2),
        constrained_layout=True,
    )
    ax.plot(times, percent, color="#176B87", linewidth=1.5)

    spread = float(np.max(percent) - np.min(percent))
    padding = max(0.15, 0.12 * spread)
    ax.set_ylim(float(np.min(percent) - padding), float(np.max(percent) + padding))
    ax.set_ylabel("Visible sky (%)")
    ax.set_xlabel("Time (UTC)")
    ax.grid(True, which="major", color="#D7DEE5", linewidth=0.8, alpha=0.9)
    ax.grid(True, which="minor", color="#E8EDF2", linewidth=0.5, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    date_locator = mdates.AutoDateLocator(minticks=5, maxticks=10, tz=timezone.utc)
    ax.xaxis.set_major_locator(date_locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(date_locator, tz=timezone.utc))
    ax.xaxis.set_minor_locator(
        mdates.HourLocator(byhour=(6, 12, 18), tz=timezone.utc)
    )
    ax.tick_params(axis="x", which="major", labelrotation=30)
    for label in ax.get_xticklabels(which="major"):
        label.set_horizontalalignment("right")

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
        f"{moon_wording} + {moon_clearance_deg:g} deg; "
        f"Sun center + {sun_exclusion_deg:g} deg | "
        f"{sample_step} samples | {observer_name}",
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
            "Earth, Moon, and Sun ephemerides."
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
        default="5 min",
        help="Horizons time step (default: '5 min')",
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
        "--sun-exclusion",
        type=float,
        choices=(30.0, 45.0),
        help=(
            "Sun-centered exclusion radius in degrees (30 or 45); "
            "prompted for if omitted"
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


def prompt_for_sun_exclusion() -> float:
    """Prompt until the user chooses a 30- or 45-degree Sun exclusion."""
    print("\nSun exclusion-angle options:")
    print("  30 degrees from the Sun's center")
    print("  45 degrees from the Sun's center")
    while True:
        try:
            value = input("Choose Sun exclusion angle (30 or 45): ").strip()
        except EOFError as exc:
            raise SystemExit(
                "No interactive input was available. Use --sun-exclusion instead."
            ) from exc

        if value in {"30", "30.0"}:
            return 30.0
        if value in {"45", "45.0"}:
            return 45.0
        print("Invalid choice. Enter either 30 or 45.")


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
    sun_exclusion = args.sun_exclusion or prompt_for_sun_exclusion()

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
    print("Downloading Sun vectors from JPL Horizons...")
    sun_jd, sun_calendar, sun_vectors, sun_observer_name = fetch_vectors(
        SUN_ID, args.observer_spk, start, stop, args.step
    )

    for body_name, body_jd, body_calendar, body_observer in (
        ("Moon", moon_jd, moon_calendar, moon_observer_name),
        ("Sun", sun_jd, sun_calendar, sun_observer_name),
    ):
        if observer_name != body_observer:
            raise RuntimeError(
                f"Earth and {body_name} queries returned different observers."
            )
        if earth_jd.shape != body_jd.shape or not np.allclose(
            earth_jd, body_jd, rtol=0.0, atol=1e-9
        ):
            raise RuntimeError(
                f"Earth and {body_name} ephemeris times do not match."
            )
        if earth_calendar != body_calendar:
            raise RuntimeError(
                f"Earth and {body_name} calendar labels do not match."
            )

    if args.observer_spk == GOES18_SPK_ID and "GOES-18" not in observer_name:
        raise RuntimeError(
            f"SPK ID {GOES18_SPK_ID} did not resolve to GOES-18: {observer_name}"
        )

    results = compute_visible_sky(
        earth_vectors,
        moon_vectors,
        sun_vectors,
        args.earth_clearance,
        args.moon_clearance,
        args.moon_reference,
        sun_exclusion,
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
        sun_exclusion,
        args.step,
    )

    fraction = results["visible_fraction"]
    percent = 100.0 * fraction
    print(f"Observer verified by Horizons: {observer_name}")
    print(f"Date range: {start} through {stop} UTC")
    print(f"Sun exclusion angle: {sun_exclusion:g} degrees from Sun center")
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
