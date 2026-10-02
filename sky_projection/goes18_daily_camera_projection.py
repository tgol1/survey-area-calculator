#!/usr/bin/env python3
"""Create a GOES-18 survey-coverage map with a camera footprint.

The map integrates a 3-hour interval, a 24-hour interval, or both at
five-minute cadence.  For every sample, a fixed celestial direction is considered
available only when it lies outside the Earth, Moon, and Sun exclusion caps.
The resulting Mollweide map reports total unobscured time over the selected
interval.

A 24 degree by 24 degree TESS-like camera field is projected from a requested
right ascension and declination.  The field is constructed in the tangent
plane of the pointing direction, rather than drawn as a rectangle in plot
coordinates, so the outline correctly distorts near the celestial poles and
the Mollweide seam.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import math
from pathlib import Path
import sys

import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

UTC = timezone.utc
DEFAULT_OUTPUT = SCRIPT_DIRECTORY / "goes18_daily_camera_projection.png"
DEFAULT_OUTPUT_3H = SCRIPT_DIRECTORY / "goes18_3h_camera_projection.png"
DEFAULT_OUTPUT_24H = SCRIPT_DIRECTORY / "goes18_24h_camera_projection.png"
CAMERA_COLOR = "#FF2D95"
TRACK_COLORS = {
    "Earth": "#2563EB",
    "Moon": "#FFFFFF",
    "Sun": "#F59E0B",
}

try:
    from horizons_api_algorithm.goes18_visible_sky import (
        EARTH_ID,
        EARTH_MEAN_RADIUS_KM,
        GOES18_SPK_ID,
        MOON_ID,
        SUN_ID,
        fetch_vectors,
        horizons_calendar_to_utc,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import horizons_api_algorithm.goes18_visible_sky. Place "
        "this script in the repository's sky_projection folder."
    ) from exc


@dataclass(frozen=True)
class Ephemeris:
    """Time-matched GOES-18-centered body vectors."""

    times: tuple[datetime, ...]
    earth_vectors_km: np.ndarray
    moon_vectors_km: np.ndarray
    sun_vectors_km: np.ndarray
    observer_name: str


def parse_date(value: str) -> date:
    """Parse one YYYY-MM-DD UTC calendar date."""
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use YYYY-MM-DD.") from exc


def prompt_for_date() -> date:
    while True:
        try:
            return parse_date(input("Enter UTC start date (YYYY-MM-DD): "))
        except argparse.ArgumentTypeError as exc:
            print(f"Invalid date: {exc}")


def parse_clock_time(value: str) -> time:
    """Parse one HH:MM UTC clock time."""
    try:
        return datetime.strptime(value.strip(), "%H:%M").time()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use HH:MM in 24-hour UTC time.") from exc


def prompt_for_start_time() -> time:
    while True:
        try:
            return parse_clock_time(input("Enter UTC start time (HH:MM): "))
        except argparse.ArgumentTypeError as exc:
            print(f"Invalid time: {exc}")


def prompt_for_windows() -> tuple[int, ...]:
    while True:
        choice = input(
            "Choose integration output (3 hours, 24 hours, or both): "
        ).strip().lower()
        if choice in {"3", "3h", "3 hour", "3 hours"}:
            return (3,)
        if choice in {"24", "24h", "24 hour", "24 hours"}:
            return (24,)
        if choice in {"both", "3 and 24", "3,24"}:
            return (3, 24)
        print("Enter 3, 24, or both.")


def prompt_for_sun_exclusion() -> float:
    while True:
        try:
            angle = float(
                input("Choose Sun exclusion angle (30 or 45 degrees): ").strip()
            )
        except ValueError:
            print("Enter either 30 or 45.")
            continue
        if angle in (30.0, 45.0):
            return angle
        print("Enter either 30 or 45.")


def prompt_for_float(label: str, minimum: float, maximum: float) -> float:
    while True:
        try:
            value = float(input(label).strip())
        except ValueError:
            print(f"Enter a number from {minimum:g} through {maximum:g}.")
            continue
        if minimum <= value <= maximum:
            return value
        print(f"Enter a number from {minimum:g} through {maximum:g}.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create 3-hour, 24-hour, or both GOES-18 sky-coverage projections "
            "and overlay a TESS-like camera field."
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
        "--window-hours",
        type=int,
        choices=(3, 24),
        help="Generate one integration window; omit when using --both-windows",
    )
    parser.add_argument(
        "--both-windows",
        action="store_true",
        help="Generate both 3-hour and 24-hour maps from one ephemeris download",
    )
    parser.add_argument(
        "--sun-exclusion",
        type=float,
        choices=(30.0, 45.0),
        help="Sun-center exclusion radius; prompts when omitted",
    )
    parser.add_argument(
        "--camera-ra",
        type=float,
        help="Camera-center right ascension in degrees, 0 through 360",
    )
    parser.add_argument(
        "--camera-dec",
        type=float,
        help="Camera-center declination in degrees, -90 through +90",
    )
    parser.add_argument(
        "--camera-size",
        type=float,
        default=24.0,
        help="Square camera field width and height in degrees (default: 24)",
    )
    parser.add_argument(
        "--camera-roll",
        type=float,
        default=0.0,
        help="Camera roll east of celestial north in degrees (default: 0)",
    )
    parser.add_argument(
        "--step-minutes",
        type=int,
        default=5,
        help="Time-integration interval dividing the window evenly (default: 5)",
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
        help="Moon-center exclusion radius (default: 20 deg)",
    )
    parser.add_argument(
        "--observer-spk",
        type=int,
        default=GOES18_SPK_ID,
        help=f"Horizons observer SPK ID (default: {GOES18_SPK_ID})",
    )
    parser.add_argument(
        "--longitude-points",
        type=int,
        default=541,
        help="Sky-grid longitude samples (default: 541)",
    )
    parser.add_argument(
        "--latitude-points",
        type=int,
        default=271,
        help="Sky-grid latitude samples (default: 271)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path when generating one window",
    )
    parser.add_argument(
        "--output-3h",
        type=Path,
        default=DEFAULT_OUTPUT_3H,
        help="3-hour PNG path when using --both-windows",
    )
    parser.add_argument(
        "--output-24h",
        type=Path,
        default=DEFAULT_OUTPUT_24H,
        help="24-hour PNG path when using --both-windows",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output resolution (default: 220 dpi)",
    )
    return parser.parse_args()


def validate_arguments(
    args: argparse.Namespace,
) -> tuple[datetime, tuple[int, ...], float, float, float, float]:
    start_date = args.start_date or prompt_for_date()
    start_time = args.start_time or prompt_for_start_time()
    if args.both_windows and args.window_hours is not None:
        raise ValueError("Use either --both-windows or --window-hours, not both.")
    if args.both_windows:
        windows = (3, 24)
    elif args.window_hours is not None:
        windows = (args.window_hours,)
    else:
        windows = prompt_for_windows()
    sun_exclusion = args.sun_exclusion or prompt_for_sun_exclusion()
    camera_ra = (
        args.camera_ra
        if args.camera_ra is not None
        else prompt_for_float(
            "Enter camera-center right ascension (degrees): ", 0.0, 360.0
        )
    )
    camera_dec = (
        args.camera_dec
        if args.camera_dec is not None
        else prompt_for_float(
            "Enter camera-center declination (degrees): ", -90.0, 90.0
        )
    )

    if not 0.0 <= camera_ra <= 360.0:
        raise ValueError("--camera-ra must be between 0 and 360 degrees.")
    if not -90.0 <= camera_dec <= 90.0:
        raise ValueError("--camera-dec must be between -90 and +90 degrees.")
    if not math.isfinite(args.camera_roll):
        raise ValueError("--camera-roll must be a finite angle in degrees.")
    if not 0.0 < args.camera_size < 120.0:
        raise ValueError("--camera-size must be greater than 0 and below 120 degrees.")
    if args.step_minutes <= 0 or any(
        window_hours * 60 % args.step_minutes != 0 for window_hours in windows
    ):
        raise ValueError(
            "--step-minutes must be positive and divide every selected window evenly."
        )
    if not 0.0 <= args.earth_clearance <= 60.0:
        raise ValueError("--earth-clearance must be between 0 and 60 degrees.")
    if not 0.0 <= args.moon_clearance <= 60.0:
        raise ValueError("--moon-clearance must be between 0 and 60 degrees.")
    if args.longitude_points < 181 or args.latitude_points < 91:
        raise ValueError("The sky grid is too coarse; use at least 181 x 91.")
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72.")
    if len(windows) == 2:
        output_3h = args.output_3h.expanduser().resolve()
        output_24h = args.output_24h.expanduser().resolve()
        if output_3h == output_24h:
            raise ValueError("--output-3h and --output-24h must be different paths.")
    camera_roll = (args.camera_roll + 180.0) % 360.0 - 180.0
    start_datetime = datetime.combine(start_date, start_time, tzinfo=UTC)
    return (
        start_datetime,
        windows,
        sun_exclusion,
        camera_ra % 360.0,
        camera_dec,
        camera_roll,
    )


def fetch_ephemeris(
    start_datetime: datetime,
    window_hours: int,
    step_minutes: int,
    observer_spk: int,
) -> Ephemeris:
    """Download matching Earth, Moon, and Sun vectors from JPL Horizons."""
    stop_datetime = start_datetime + timedelta(hours=window_hours)
    start_text = start_datetime.strftime("%Y-%m-%d %H:%M")
    stop_text = stop_datetime.strftime("%Y-%m-%d %H:%M")
    step_text = f"{step_minutes} min"

    fetched: dict[str, tuple[np.ndarray, list[str], np.ndarray, str]] = {}
    for body_name, target_id in (
        ("Earth", EARTH_ID),
        ("Moon", MOON_ID),
        ("Sun", SUN_ID),
    ):
        print(
            f"Downloading {body_name} vectors from JPL Horizons "
            f"({start_datetime:%Y-%m-%d %H:%M} through "
            f"{stop_datetime:%Y-%m-%d %H:%M} UTC)..."
        )
        fetched[body_name] = fetch_vectors(
            target_id,
            observer_spk,
            start_text,
            stop_text,
            step_text,
        )

    earth_jd, earth_calendar, earth_vectors, observer_name = fetched["Earth"]
    for body_name in ("Moon", "Sun"):
        body_jd, body_calendar, _, body_observer = fetched[body_name]
        if earth_jd.shape != body_jd.shape or not np.allclose(
            earth_jd, body_jd, rtol=0.0, atol=1e-9
        ):
            raise RuntimeError(f"Earth and {body_name} time grids do not match.")
        if earth_calendar != body_calendar:
            raise RuntimeError(
                f"Earth and {body_name} calendar labels do not match."
            )
        if observer_name != body_observer:
            raise RuntimeError(f"Earth and {body_name} observers do not match.")

    times = tuple(horizons_calendar_to_utc(value) for value in earth_calendar)
    expected_samples = window_hours * 60 // step_minutes + 1
    if len(times) != expected_samples:
        raise RuntimeError(
            f"Horizons returned {len(times)} samples; expected {expected_samples}."
        )
    if observer_spk == GOES18_SPK_ID and "GOES-18" not in observer_name:
        raise RuntimeError(
            f"SPK {GOES18_SPK_ID} did not resolve to GOES-18: {observer_name}"
        )
    return Ephemeris(
        times=times,
        earth_vectors_km=earth_vectors,
        moon_vectors_km=fetched["Moon"][2],
        sun_vectors_km=fetched["Sun"][2],
        observer_name=observer_name,
    )


def sky_grid(
    longitude_points: int,
    latitude_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return RA-left Mollweide coordinates and flattened ICRF unit vectors."""
    longitudes = np.linspace(-math.pi, math.pi, longitude_points)
    latitudes = np.linspace(-math.pi / 2.0, math.pi / 2.0, latitude_points)
    longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)
    right_ascension = -longitude_grid
    cos_dec = np.cos(latitude_grid)
    vectors = np.stack(
        (
            cos_dec * np.cos(right_ascension),
            cos_dec * np.sin(right_ascension),
            np.sin(latitude_grid),
        ),
        axis=-1,
    )
    return (
        longitude_grid,
        latitude_grid,
        np.asarray(vectors.reshape(-1, 3), dtype=np.float32),
    )


def normalize_rows(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a vector table and return unit vectors and distances."""
    array = np.asarray(vectors, dtype=float)
    distances = np.linalg.norm(array, axis=1)
    if np.any(distances == 0.0) or not np.all(np.isfinite(distances)):
        raise ValueError("Ephemeris contains invalid body vectors.")
    return np.asarray(array / distances[:, None], dtype=np.float32), distances


def vectors_to_plot_track(vectors: np.ndarray) -> np.ndarray:
    """Convert ICRF vectors to RA-left Mollweide longitude and declination."""
    units, _ = normalize_rows(vectors)
    right_ascension = np.arctan2(units[:, 1], units[:, 0])
    declination = np.arcsin(np.clip(units[:, 2], -1.0, 1.0))
    longitude = (-right_ascension + math.pi) % (2.0 * math.pi) - math.pi
    return np.column_stack((longitude, declination))


def daily_coverage_map(
    grid_vectors: np.ndarray,
    earth_vectors: np.ndarray,
    moon_vectors: np.ndarray,
    sun_vectors: np.ndarray,
    earth_clearance_deg: float,
    moon_clearance_deg: float,
    sun_exclusion_deg: float,
    step_minutes: int,
    grid_shape: tuple[int, int],
    block_size: int = 24_000,
) -> np.ndarray:
    """Integrate unobscured time for every fixed sky direction."""
    earth_units, earth_distances = normalize_rows(earth_vectors)
    moon_units, _ = normalize_rows(moon_vectors)
    sun_units, _ = normalize_rows(sun_vectors)

    earth_radii = np.arcsin(EARTH_MEAN_RADIUS_KM / earth_distances)
    earth_radii += math.radians(earth_clearance_deg)
    moon_radii = np.full(
        len(moon_units), math.radians(moon_clearance_deg), dtype=float
    )
    sun_radii = np.full(
        len(sun_units), math.radians(sun_exclusion_deg), dtype=float
    )
    thresholds = (
        np.asarray(np.cos(earth_radii), dtype=np.float32),
        np.asarray(np.cos(moon_radii), dtype=np.float32),
        np.asarray(np.cos(sun_radii), dtype=np.float32),
    )

    point_count = grid_vectors.shape[0]
    visible_samples = np.empty(point_count, dtype=np.uint16)
    for block_start in range(0, point_count, block_size):
        block_stop = min(point_count, block_start + block_size)
        block = grid_vectors[block_start:block_stop].T
        earth_inside = earth_units @ block >= thresholds[0][:, None]
        moon_inside = moon_units @ block >= thresholds[1][:, None]
        sun_inside = sun_units @ block >= thresholds[2][:, None]
        visible_samples[block_start:block_stop] = np.count_nonzero(
            ~(earth_inside | moon_inside | sun_inside), axis=0
        )

    return (
        visible_samples.astype(np.float32) * (step_minutes / 60.0)
    ).reshape(grid_shape)


def solid_angle_weighted_mean(
    values: np.ndarray,
    latitude_grid: np.ndarray,
) -> float:
    """Return the full-sky equal-solid-angle mean of a gridded field."""
    weights = np.cos(latitude_grid)
    return float(np.sum(values * weights) / np.sum(weights))


def solid_angle_fraction(mask: np.ndarray, latitude_grid: np.ndarray) -> float:
    """Return the solid-angle fraction represented by a gridded mask."""
    weights = np.cos(latitude_grid)
    return float(np.sum(weights * mask) / np.sum(weights))


def camera_basis(
    right_ascension_deg: float,
    declination_deg: float,
    roll_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return camera boresight, image-right, and image-up unit vectors."""
    ra = math.radians(right_ascension_deg)
    dec = math.radians(declination_deg)
    center = np.asarray(
        [math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)]
    )
    east = np.asarray([-math.sin(ra), math.cos(ra), 0.0])
    north = np.asarray(
        [-math.sin(dec) * math.cos(ra), -math.sin(dec) * math.sin(ra), math.cos(dec)]
    )

    roll = math.radians(roll_deg)
    image_right = math.cos(roll) * east + math.sin(roll) * north
    image_up = -math.sin(roll) * east + math.cos(roll) * north
    return center, image_right, image_up


def camera_footprint_mask(
    grid_vectors: np.ndarray,
    right_ascension_deg: float,
    declination_deg: float,
    size_deg: float,
    roll_deg: float,
) -> np.ndarray:
    """Return pixels inside a square rectilinear camera field on the sphere."""
    center, image_right, image_up = camera_basis(
        right_ascension_deg, declination_deg, roll_deg
    )
    forward = grid_vectors @ center
    horizontal = grid_vectors @ image_right
    vertical = grid_vectors @ image_up
    tangent_limit = math.tan(math.radians(size_deg / 2.0))

    return (
        (forward > 0.0)
        & (np.abs(horizontal) <= tangent_limit * forward)
        & (np.abs(vertical) <= tangent_limit * forward)
    )


def vectors_to_plot_coordinates(vectors: np.ndarray) -> np.ndarray:
    """Convert ICRF vectors to RA-left Mollweide longitude and declination."""
    units, _ = normalize_rows(vectors)
    right_ascension = np.arctan2(units[:, 1], units[:, 0])
    declination = np.arcsin(np.clip(units[:, 2], -1.0, 1.0))
    longitude = (-right_ascension + math.pi) % (2.0 * math.pi) - math.pi
    return np.column_stack((longitude, declination))


def camera_boundary(
    right_ascension_deg: float,
    declination_deg: float,
    size_deg: float,
    roll_deg: float,
    points_per_edge: int = 100,
) -> np.ndarray:
    """Project the four tangent-plane detector edges onto the unit sphere."""
    center, image_right, image_up = camera_basis(
        right_ascension_deg, declination_deg, roll_deg
    )
    limit = math.tan(math.radians(size_deg / 2.0))
    forward = np.linspace(-limit, limit, points_per_edge, endpoint=False)
    backward = np.linspace(limit, -limit, points_per_edge, endpoint=False)
    x = np.concatenate(
        (
            forward,
            np.full(points_per_edge, limit),
            backward,
            np.full(points_per_edge, -limit),
            np.asarray([-limit]),
        )
    )
    y = np.concatenate(
        (
            np.full(points_per_edge, -limit),
            forward,
            np.full(points_per_edge, limit),
            backward,
            np.asarray([-limit]),
        )
    )
    vectors = center + x[:, None] * image_right + y[:, None] * image_up
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    return vectors_to_plot_coordinates(vectors)


def spherical_cap_boundary(
    center_vector: np.ndarray,
    radius_deg: float,
    point_count: int = 361,
) -> np.ndarray:
    """Return the boundary of an angular cap centered on an ICRF vector."""
    center = np.asarray(center_vector, dtype=float)
    center /= np.linalg.norm(center)
    reference = np.asarray([0.0, 0.0, 1.0])
    if abs(float(center @ reference)) > 0.95:
        reference = np.asarray([1.0, 0.0, 0.0])
    first = np.cross(reference, center)
    first /= np.linalg.norm(first)
    second = np.cross(center, first)
    angles = np.linspace(0.0, 2.0 * math.pi, point_count)
    radius = math.radians(radius_deg)
    vectors = (
        math.cos(radius) * center
        + math.sin(radius)
        * (
            np.cos(angles)[:, None] * first
            + np.sin(angles)[:, None] * second
        )
    )
    return vectors_to_plot_coordinates(vectors)


def plot_wrapped_line(
    axis: plt.Axes,
    coordinates: np.ndarray,
    *,
    color: str,
    linewidth: float,
    linestyle: str = "-",
    label: str | None = None,
    zorder: int = 5,
    outline: bool = True,
) -> None:
    """Plot a sky path without drawing a false line across the map seam."""
    longitudes = coordinates[:, 0]
    latitudes = coordinates[:, 1]
    breaks = np.flatnonzero(np.abs(np.diff(longitudes)) > math.pi) + 1
    segments = np.split(np.arange(len(coordinates)), breaks)
    label_pending = label
    for indices in segments:
        if len(indices) < 2:
            continue
        (line,) = axis.plot(
            longitudes[indices],
            latitudes[indices],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label_pending,
            zorder=zorder,
        )
        if outline:
            line.set_path_effects(
                [
                    path_effects.Stroke(
                        linewidth=linewidth + 1.5,
                        foreground="#111827",
                        alpha=0.88,
                    ),
                    path_effects.Normal(),
                ]
            )
        label_pending = None


def camera_statistics(
    coverage_hours: np.ndarray,
    camera_mask: np.ndarray,
    latitude_grid: np.ndarray,
    step_minutes: int,
    window_hours: int,
) -> dict[str, float]:
    """Summarize availability inside the projected detector footprint."""
    if not np.any(camera_mask):
        raise ValueError("The camera footprint did not intersect the sky grid.")
    weights = np.cos(latitude_grid)[camera_mask]
    values = coverage_hours[camera_mask]
    full_window = np.isclose(
        values,
        float(window_hours),
        atol=step_minutes / 120.0,
    )
    return {
        "mean": float(np.average(values, weights=weights)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "full_window_fraction": float(
            np.sum(weights[full_window]) / np.sum(weights)
        ),
    }


def write_figure(
    path: Path,
    start_datetime: datetime,
    window_hours: int,
    coverage_hours: np.ndarray,
    longitude_grid: np.ndarray,
    latitude_grid: np.ndarray,
    body_vectors: dict[str, np.ndarray],
    observer_name: str,
    sun_exclusion_deg: float,
    earth_clearance_deg: float,
    moon_clearance_deg: float,
    camera_ra_deg: float,
    camera_dec_deg: float,
    camera_size_deg: float,
    camera_roll_deg: float,
    camera_stats: dict[str, float],
    step_minutes: int,
    dpi: int,
) -> Path:
    """Write the selected-window Mollweide availability map and overlay."""
    figure = plt.figure(figsize=(14.0, 8.35))
    axis = figure.add_subplot(111, projection="mollweide")
    norm = mcolors.Normalize(vmin=0.0, vmax=float(window_hours))
    image = axis.pcolormesh(
        longitude_grid,
        latitude_grid,
        coverage_hours,
        cmap="turbo",
        norm=norm,
        shading="auto",
        rasterized=True,
        zorder=0,
    )
    contour_levels = tuple(
        float(window_hours) * fraction for fraction in (0.25, 0.50, 0.75)
    )
    contours = axis.contour(
        longitude_grid,
        latitude_grid,
        coverage_hours,
        levels=contour_levels,
        colors="white",
        linewidths=0.55,
        alpha=0.56,
        zorder=2,
    )
    axis.clabel(
        contours,
        fmt=lambda value: f"{value:g} h",
        inline=True,
        fontsize=7.0,
        colors="white",
    )

    sample_count = len(next(iter(body_vectors.values())))
    marker_interval_hours = 1 if window_hours == 3 else 6
    for body_name in ("Earth", "Moon", "Sun"):
        coordinates = vectors_to_plot_track(body_vectors[body_name])
        plot_wrapped_line(
            axis,
            coordinates,
            color=TRACK_COLORS[body_name],
            linewidth=1.35 if body_name == "Moon" else 1.05,
            label=f"{body_name} center track",
            zorder=5,
        )
        for hour in range(0, window_hours + 1, marker_interval_hours):
            index = min(round(hour * 60 / step_minutes), sample_count - 1)
            longitude, latitude = coordinates[index]
            axis.scatter(
                [longitude],
                [latitude],
                s=28 if body_name == "Moon" else 20,
                color=TRACK_COLORS[body_name],
                edgecolor="#111827",
                linewidth=0.7,
                zorder=7,
            )

    midpoint_minutes = window_hours * 60 / 2.0
    midpoint_index = min(
        round(midpoint_minutes / step_minutes),
        sample_count - 1,
    )
    midpoint_datetime = start_datetime + timedelta(minutes=midpoint_minutes)
    midpoint_label = midpoint_datetime.strftime("%Y-%m-%d %H:%M UTC")
    moon_reference_cap = spherical_cap_boundary(
        body_vectors["Moon"][midpoint_index], moon_clearance_deg
    )
    plot_wrapped_line(
        axis,
        moon_reference_cap,
        color="#FFFFFF",
        linewidth=1.25,
        linestyle="--",
        label=f"Moon exclusion at midpoint ({moon_clearance_deg:g}°)",
        zorder=6,
    )

    sunward_guide = spherical_cap_boundary(
        body_vectors["Sun"][midpoint_index], 60.0
    )
    plot_wrapped_line(
        axis,
        sunward_guide,
        color="#FFD400",
        linewidth=1.9,
        linestyle=":",
        label="60° from Sun at midpoint (planning guide)",
        zorder=7,
    )

    detector_outline = camera_boundary(
        camera_ra_deg,
        camera_dec_deg,
        camera_size_deg,
        camera_roll_deg,
    )
    plot_wrapped_line(
        axis,
        detector_outline,
        color=CAMERA_COLOR,
        linewidth=2.7,
        label=f"Camera field ({camera_size_deg:g}° × {camera_size_deg:g}°)",
        zorder=9,
    )
    camera_center = vectors_to_plot_coordinates(
        camera_basis(camera_ra_deg, camera_dec_deg, camera_roll_deg)[0][None, :]
    )[0]
    axis.scatter(
        [camera_center[0]],
        [camera_center[1]],
        marker="+",
        s=90,
        color=CAMERA_COLOR,
        linewidth=2.0,
        zorder=10,
    )

    axis.grid(True, color="white", linewidth=0.4, alpha=0.26, zorder=1)
    axis.set_longitude_grid(30)
    axis.set_latitude_grid(30)
    axis.set_xticklabels(
        ("10h", "8h", "6h", "4h", "2h", "0h", "22h", "20h", "18h", "16h", "14h")
    )
    axis.tick_params(labelsize=8.0, colors="#374151")

    colorbar_axis = figure.add_axes((0.075, 0.115, 0.85, 0.027))
    colorbar = figure.colorbar(
        image,
        cax=colorbar_axis,
        orientation="horizontal",
        ticks=np.linspace(0.0, float(window_hours), 7),
    )
    colorbar.set_label(
        f"Time unobscured during the selected {window_hours}-hour interval "
        "(hours)",
        fontsize=9.5,
        weight="bold",
    )
    colorbar.ax.tick_params(labelsize=8.5)

    legend_handles = [
        Line2D(
            [],
            [],
            color=TRACK_COLORS[name],
            marker="o",
            markeredgecolor="#111827",
            linewidth=1.5,
            markersize=5,
            label=f"{name} center track",
        )
        for name in ("Earth", "Moon", "Sun")
    ]
    legend_handles.extend(
        [
            Line2D(
                [],
                [],
                color="#FFFFFF",
                linestyle="--",
                linewidth=1.5,
                path_effects=[
                    path_effects.Stroke(linewidth=3.0, foreground="#111827"),
                    path_effects.Normal(),
                ],
                label=f"Moon exclusion at midpoint ({moon_clearance_deg:g}°)",
            ),
            Line2D(
                [],
                [],
                color="#FFD400",
                linestyle=":",
                linewidth=2.0,
                path_effects=[
                    path_effects.Stroke(linewidth=3.5, foreground="#111827"),
                    path_effects.Normal(),
                ],
                label="60° from Sun at midpoint (planning guide)",
            ),
            Line2D(
                [],
                [],
                color=CAMERA_COLOR,
                linewidth=2.8,
                label=f"Camera field ({camera_size_deg:g}° × {camera_size_deg:g}°)",
            ),
        ]
    )
    axis.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.105),
        ncol=3,
        fontsize=8.0,
        frameon=True,
        facecolor="white",
        framealpha=0.94,
    )

    stop_datetime = start_datetime + timedelta(hours=window_hours)
    figure.suptitle(
        f"NEO {window_hours}-hour sky-survey planning map",
        y=0.985,
        fontsize=17,
        weight="bold",
        color="#111827",
    )
    figure.text(
        0.5,
        0.943,
        f"{start_datetime:%Y-%m-%d %H:%M} to "
        f"{stop_datetime:%Y-%m-%d %H:%M} UTC  ·  "
        f"{step_minutes}-minute samples  ·  Sun exclusion {sun_exclusion_deg:g}°",
        ha="center",
        fontsize=10.0,
        color="#4B5563",
    )
    figure.text(
        0.018,
        0.025,
        f"Color = accumulated unobscured time. Tracks and {marker_interval_hours}-hour "
        "markers show body motion.\n"
        f"Dashed white Moon cap and dotted yellow 60° Sun guide use the interval "
        f"midpoint ({midpoint_label}); the Sun guide is not an added exclusion.\n"
        f"Earth limb + {earth_clearance_deg:g}°; observer = {observer_name}; "
        "RA increases toward the left.",
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#4B5563",
    )
    figure.text(
        0.982,
        0.025,
        f"Camera center: RA {camera_ra_deg:.2f}° "
        f"({camera_ra_deg / 15.0:.3f} h), Dec {camera_dec_deg:+.2f}°, "
        f"roll {camera_roll_deg:+.2f}°\n"
        f"Field mean {camera_stats['mean']:.2f} h "
        f"({100.0 * camera_stats['mean'] / window_hours:.1f}%)  ·  "
        f"min {camera_stats['minimum']:.2f} h  ·  "
        f"max {camera_stats['maximum']:.2f} h  ·  "
        f"Full-window field area "
        f"{100.0 * camera_stats['full_window_fraction']:.1f}%",
        ha="right",
        va="bottom",
        fontsize=8.2,
        color="#4B5563",
    )
    figure.subplots_adjust(left=0.035, right=0.965, top=0.89, bottom=0.26)

    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(resolved, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return resolved


def print_window_summary(
    *,
    start_datetime: datetime,
    window_hours: int,
    step_minutes: int,
    camera_ra: float,
    camera_dec: float,
    camera_roll: float,
    camera_size: float,
    statistics: dict[str, float],
    global_mean: float,
    global_full_window: float,
    output: Path,
) -> None:
    """Print one generated window's statistics and output location."""
    stop_datetime = start_datetime + timedelta(hours=window_hours)
    sample_count = window_hours * 60 // step_minutes
    print(f"\n{window_hours}-hour camera planning projection complete")
    print(
        f"Window: {start_datetime:%Y-%m-%d %H:%M} UTC through "
        f"{stop_datetime:%Y-%m-%d %H:%M} UTC"
    )
    print(
        f"Sampling: {sample_count} {step_minutes}-minute intervals "
        f"({sample_count + 1} ephemeris endpoints)"
    )
    print(
        f"Camera: RA {camera_ra:.6f} deg, Dec {camera_dec:+.6f} deg, "
        f"roll {camera_roll:+.6f} deg, "
        f"field {camera_size:g} x {camera_size:g} deg"
    )
    print(
        f"Camera-field access: min={statistics['minimum']:.6f} h, "
        f"mean={statistics['mean']:.6f} h, max={statistics['maximum']:.6f} h"
    )
    print(
        "Camera-field time-averaged availability: "
        f"{100.0 * statistics['mean'] / window_hours:.6f}%"
    )
    print(
        f"Camera field continuously accessible for {window_hours} h: "
        f"{100.0 * statistics['full_window_fraction']:.6f}% of field area"
    )
    print(
        f"Full-sky mean access={global_mean:.6f} h; full-sky continuous "
        f"{window_hours} h access={100.0 * global_full_window:.6f}%"
    )
    print(
        "Full-sky time-averaged availability: "
        f"{100.0 * global_mean / window_hours:.6f}%"
    )
    print(f"Wrote projection PNG: {output}")


def main() -> int:
    args = parse_arguments()
    try:
        (
            start_datetime,
            windows,
            sun_exclusion,
            camera_ra,
            camera_dec,
            camera_roll,
        ) = validate_arguments(args)
        ephemeris = fetch_ephemeris(
            start_datetime,
            window_hours=max(windows),
            step_minutes=args.step_minutes,
            observer_spk=args.observer_spk,
        )
        longitude_grid, latitude_grid, grid_vectors = sky_grid(
            args.longitude_points,
            args.latitude_points,
        )
        camera_mask = camera_footprint_mask(
            grid_vectors,
            camera_ra,
            camera_dec,
            args.camera_size,
            camera_roll,
        ).reshape(latitude_grid.shape)

        if len(windows) == 2:
            output_paths = {3: args.output_3h, 24: args.output_24h}
        else:
            output_paths = {windows[0]: args.output}

        for window_hours in windows:
            samples_per_window = window_hours * 60 // args.step_minutes
            coverage_hours = daily_coverage_map(
                grid_vectors,
                ephemeris.earth_vectors_km[:samples_per_window],
                ephemeris.moon_vectors_km[:samples_per_window],
                ephemeris.sun_vectors_km[:samples_per_window],
                args.earth_clearance,
                args.moon_clearance,
                sun_exclusion,
                args.step_minutes,
                latitude_grid.shape,
            )
            statistics = camera_statistics(
                coverage_hours,
                camera_mask,
                latitude_grid,
                args.step_minutes,
                window_hours,
            )
            body_vectors = {
                "Earth": ephemeris.earth_vectors_km[: samples_per_window + 1],
                "Moon": ephemeris.moon_vectors_km[: samples_per_window + 1],
                "Sun": ephemeris.sun_vectors_km[: samples_per_window + 1],
            }
            output = write_figure(
                output_paths[window_hours],
                start_datetime,
                window_hours,
                coverage_hours,
                longitude_grid,
                latitude_grid,
                body_vectors,
                ephemeris.observer_name,
                sun_exclusion,
                args.earth_clearance,
                args.moon_clearance,
                camera_ra,
                camera_dec,
                args.camera_size,
                camera_roll,
                statistics,
                args.step_minutes,
                args.dpi,
            )
            global_mean = solid_angle_weighted_mean(
                coverage_hours,
                latitude_grid,
            )
            global_full_window = solid_angle_fraction(
                np.isclose(
                    coverage_hours,
                    float(window_hours),
                    atol=args.step_minutes / 120.0,
                ),
                latitude_grid,
            )
            print_window_summary(
                start_datetime=start_datetime,
                window_hours=window_hours,
                step_minutes=args.step_minutes,
                camera_ra=camera_ra,
                camera_dec=camera_dec,
                camera_roll=camera_roll,
                camera_size=args.camera_size,
                statistics=statistics,
                global_mean=global_mean,
                global_full_window=global_full_window,
                output=output,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
