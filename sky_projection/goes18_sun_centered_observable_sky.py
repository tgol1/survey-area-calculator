#!/usr/bin/env python3
"""Plan camera fields and plot continuous sky access around the Sun track.

The target is the Sun-centered half of the celestial sphere between the
selected 30- or 45-degree Sun exclusion boundary and 90-degree solar
elongation. The script lays out unique TESS-like camera panels with minimal
overlap, samples every complete panel footprint, and finds the longest
continuous interval during which that panel avoids the Earth, Moon, and Sun.

Every panel is assigned once to its longest useful dwell category: 24 hours,
at least one hour, at least ten minutes, or unavailable. Cumulative panel
counts for the 10-minute, 1-hour, and 24-hour requirements are also reported.

The same ephemeris also produces the original three-panel line figure showing
the continuously observable fraction of the Sun-centered hemisphere versus
Sun exclusion angle for 10-minute, 1-hour, and 24-hour exposures.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
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
    camera_basis,
    camera_boundary,
    camera_footprint_mask,
    fetch_ephemeris,
    parse_clock_time,
    parse_date,
    plot_wrapped_line,
    prompt_for_date,
    prompt_for_start_time,
    prompt_for_sun_exclusion,
    sky_grid,
    spherical_cap_boundary,
    vectors_to_plot_coordinates,
)


TIME_STEP_MINUTES = 5
PLANNING_HOURS = 24
CAMERA_SIZE_DEG = 24.0
SCRIPT_REVISION = "2026-11-06-contained-panels-and-curves-v3"
DEFAULT_SKY_POINT_COUNT = 32_768
DEFAULT_EXPOSURE_SKY_POINT_COUNT = 8_192
DEFAULT_FOOTPRINT_SAMPLE_SIDE = 13
DEFAULT_OUTPUT_PREFIX = (
    SCRIPT_DIRECTORY
    / "visible_sky_data"
    / "goes18_sun_centered_observable_sky"
)
EXPOSURE_DURATIONS_MINUTES = (10, 60, 24 * 60)
DURATION_COLUMNS = {10: "10min", 60: "1hr", 24 * 60: "24hr"}
DURATION_LABELS = {10: "10 minutes", 60: "1 hour", 24 * 60: "24 hours"}

# Alternate rings are staggered by half a field. The innermost and outermost
# centers are one camera half-width from the radial boundaries, keeping each
# complete footprint between the selected Sun exclusion and 90 degrees. These
# are the same low-overlap layouts used by the annulus pointing-strategy tab.
PANEL_LAYOUTS: dict[int, tuple[tuple[float, int], ...]] = {
    30: ((42.0, 10), (60.0, 13), (78.0, 15)),
    45: ((57.0, 13), (78.0, 15)),
}

CATEGORY_ORDER = ("24 hours", "1 hour", "10 minutes", "unavailable")
CATEGORY_COLORS = {
    "24 hours": "#22C55E",
    "1 hour": "#38BDF8",
    "10 minutes": "#F59E0B",
    "unavailable": "#94A3B8",
}


@dataclass(frozen=True)
class CameraPanel:
    """One unique inertial 24 x 24 degree camera image."""

    number: int
    ring_number: int
    ring_field_number: int
    solar_elongation_deg: float
    sun_position_angle_deg: float
    right_ascension_deg: float
    declination_deg: float
    roll_deg: float


@dataclass(frozen=True)
class PanelDwellResult:
    """Longest continuous unobscured interval for one camera panel."""

    panel: CameraPanel
    max_dwell_minutes: int
    best_start_index: int | None
    best_stop_index: int | None
    safe_sample_count: int
    category: str


@dataclass(frozen=True)
class LayoutStatistics:
    """Numerical coverage and overlap of the fixed panel layout."""

    target_coverage_fraction: float
    overlap_fraction_of_union: float
    panel_union_fraction_of_full_sky: float
    target_fraction_of_full_sky: float


def fibonacci_sphere_points(point_count: int) -> np.ndarray:
    """Return deterministic, approximately equal-area full-sky directions."""
    indices = np.arange(point_count, dtype=float) + 0.5
    z = 1.0 - 2.0 * indices / point_count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    longitude = golden_angle * indices
    return np.column_stack(
        (radius * np.cos(longitude), radius * np.sin(longitude), z)
    )


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return one finite normalized Cartesian vector."""
    result = np.asarray(vector, dtype=float).copy()
    norm = float(np.linalg.norm(result))
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("Cannot normalize an invalid vector.")
    return result / norm


def vector_to_ra_dec(vector: np.ndarray) -> tuple[float, float]:
    """Convert one Cartesian direction to ICRF RA and declination."""
    unit = normalize(vector)
    right_ascension = math.degrees(math.atan2(unit[1], unit[0])) % 360.0
    declination = math.degrees(
        math.asin(float(np.clip(unit[2], -1.0, 1.0)))
    )
    return right_ascension, declination


def solar_tangent_basis(sun_vector: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return Sun unit, local celestial-east, and celestial-north vectors."""
    sun = normalize(sun_vector)
    sun_ra_deg, sun_dec_deg = vector_to_ra_dec(sun)
    sun_ra = math.radians(sun_ra_deg)
    sun_dec = math.radians(sun_dec_deg)
    east = np.asarray([-math.sin(sun_ra), math.cos(sun_ra), 0.0])
    north = np.asarray(
        [
            -math.sin(sun_dec) * math.cos(sun_ra),
            -math.sin(sun_dec) * math.sin(sun_ra),
            math.cos(sun_dec),
        ]
    )
    return sun, east, north


def calculate_camera_panels(
    reference_sun_vector: np.ndarray,
    sun_exclusion_deg: int,
) -> list[CameraPanel]:
    """Create the low-overlap panel lattice around the reference Sun."""
    if sun_exclusion_deg not in PANEL_LAYOUTS:
        raise ValueError("Sun exclusion must be 30 or 45 degrees.")
    sun, sun_east, sun_north = solar_tangent_basis(reference_sun_vector)
    panels: list[CameraPanel] = []
    panel_number = 1

    for ring_index, (radius_deg, field_count) in enumerate(
        PANEL_LAYOUTS[sun_exclusion_deg]
    ):
        phase_deg = 0.0 if ring_index % 2 == 0 else 180.0 / field_count
        radius = math.radians(radius_deg)
        for field_index in range(field_count):
            position_angle_deg = (
                phase_deg - field_index * 360.0 / field_count
            ) % 360.0
            position_angle = math.radians(position_angle_deg)
            radial = (
                math.cos(position_angle) * sun_north
                + math.sin(position_angle) * sun_east
            )
            center = normalize(
                math.cos(radius) * sun + math.sin(radius) * radial
            )

            increasing_angle_tangent = (
                -math.sin(position_angle) * sun_north
                + math.cos(position_angle) * sun_east
            )
            clockwise_tangent = normalize(-increasing_angle_tangent)
            right_ascension_deg, declination_deg = vector_to_ra_dec(center)
            center_ra = math.radians(right_ascension_deg)
            center_dec = math.radians(declination_deg)
            center_east = np.asarray(
                [-math.sin(center_ra), math.cos(center_ra), 0.0]
            )
            center_north = np.asarray(
                [
                    -math.sin(center_dec) * math.cos(center_ra),
                    -math.sin(center_dec) * math.sin(center_ra),
                    math.cos(center_dec),
                ]
            )
            roll_deg = math.degrees(
                math.atan2(
                    float(clockwise_tangent @ center_north),
                    float(clockwise_tangent @ center_east),
                )
            )
            roll_deg = (roll_deg + 180.0) % 360.0 - 180.0
            panels.append(
                CameraPanel(
                    number=panel_number,
                    ring_number=ring_index + 1,
                    ring_field_number=field_index + 1,
                    solar_elongation_deg=radius_deg,
                    sun_position_angle_deg=position_angle_deg,
                    right_ascension_deg=right_ascension_deg,
                    declination_deg=declination_deg,
                    roll_deg=roll_deg,
                )
            )
            panel_number += 1
    return panels


def sample_camera_footprint(
    panel: CameraPanel,
    sample_side: int,
) -> np.ndarray:
    """Return normalized directions sampling the complete square footprint."""
    if sample_side < 3 or sample_side % 2 == 0:
        raise ValueError("Footprint sample side must be an odd integer >= 3.")
    center, image_right, image_up = camera_basis(
        panel.right_ascension_deg,
        panel.declination_deg,
        panel.roll_deg,
    )
    tangent_limit = math.tan(math.radians(CAMERA_SIZE_DEG / 2.0))
    coordinates = np.linspace(-tangent_limit, tangent_limit, sample_side)
    horizontal, vertical = np.meshgrid(coordinates, coordinates)
    vectors = (
        center
        + horizontal.ravel()[:, None] * image_right
        + vertical.ravel()[:, None] * image_up
    )
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    return vectors


def body_geometry(
    earth_vectors_km: np.ndarray,
    moon_vectors_km: np.ndarray,
    sun_vectors_km: np.ndarray,
    earth_clearance_deg: float,
    moon_clearance_deg: float,
    moon_reference: str,
) -> tuple[np.ndarray, ...]:
    """Return body unit vectors and exclusion dot-product thresholds."""
    earth_vectors = np.asarray(earth_vectors_km, dtype=float)
    moon_vectors = np.asarray(moon_vectors_km, dtype=float)
    sun_vectors = np.asarray(sun_vectors_km, dtype=float)
    if not (
        earth_vectors.shape == moon_vectors.shape == sun_vectors.shape
        and earth_vectors.ndim == 2
        and earth_vectors.shape[1] == 3
    ):
        raise ValueError("Earth, Moon, and Sun vectors must be matching N-by-3 arrays.")

    earth_distance = np.linalg.norm(earth_vectors, axis=1)
    moon_distance = np.linalg.norm(moon_vectors, axis=1)
    sun_distance = np.linalg.norm(sun_vectors, axis=1)
    if np.any(earth_distance <= EARTH_MEAN_RADIUS_KM):
        raise ValueError("Invalid observer-to-Earth distance.")
    if np.any(moon_distance <= MOON_MEAN_RADIUS_KM):
        raise ValueError("Invalid observer-to-Moon distance.")
    if np.any(sun_distance <= 0.0):
        raise ValueError("Invalid observer-to-Sun distance.")

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
        moon_radius = np.full(len(moon_distance), math.radians(moon_clearance_deg))
    else:
        raise ValueError("moon_reference must be 'center' or 'limb'.")
    return (
        earth_unit,
        moon_unit,
        sun_unit,
        np.cos(earth_radius),
        np.cos(moon_radius),
    )


def longest_true_run(mask: np.ndarray) -> tuple[int, int | None, int | None]:
    """Return sample count and inclusive indices of the longest True run."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("The safe-time mask must be one-dimensional.")
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops_exclusive = np.flatnonzero(transitions == -1)
    if len(starts) == 0:
        return 0, None, None
    lengths = stops_exclusive - starts
    best = int(np.argmax(lengths))
    return (
        int(lengths[best]),
        int(starts[best]),
        int(stops_exclusive[best] - 1),
    )


def dwell_category(max_dwell_minutes: int) -> str:
    """Assign a panel to exactly one longest-useful-dwell category."""
    if max_dwell_minutes >= 24 * 60:
        return "24 hours"
    if max_dwell_minutes >= 60:
        return "1 hour"
    if max_dwell_minutes >= 10:
        return "10 minutes"
    return "unavailable"


def evaluate_panel_dwells(
    panels: list[CameraPanel],
    earth_vectors_km: np.ndarray,
    moon_vectors_km: np.ndarray,
    sun_vectors_km: np.ndarray,
    sun_exclusion_deg: float,
    earth_clearance_deg: float,
    moon_clearance_deg: float,
    moon_reference: str,
    footprint_sample_side: int,
) -> list[PanelDwellResult]:
    """Find each complete panel's longest continuously unobscured interval."""
    (
        earth_unit,
        moon_unit,
        sun_unit,
        earth_threshold,
        moon_threshold,
    ) = body_geometry(
        earth_vectors_km,
        moon_vectors_km,
        sun_vectors_km,
        earth_clearance_deg,
        moon_clearance_deg,
        moon_reference,
    )
    sun_threshold = math.cos(math.radians(sun_exclusion_deg))
    results: list[PanelDwellResult] = []

    for panel in panels:
        footprint = sample_camera_footprint(panel, footprint_sample_side)
        earth_excluded = np.max(earth_unit @ footprint.T, axis=1) >= earth_threshold
        moon_excluded = np.max(moon_unit @ footprint.T, axis=1) >= moon_threshold
        sun_excluded = np.max(sun_unit @ footprint.T, axis=1) >= sun_threshold
        safe = ~(earth_excluded | moon_excluded | sun_excluded)
        run_samples, start_index, stop_index = longest_true_run(safe)
        max_dwell_minutes = max(0, (run_samples - 1) * TIME_STEP_MINUTES)
        results.append(
            PanelDwellResult(
                panel=panel,
                max_dwell_minutes=max_dwell_minutes,
                best_start_index=start_index,
                best_stop_index=stop_index,
                safe_sample_count=int(np.count_nonzero(safe)),
                category=dwell_category(max_dwell_minutes),
            )
        )
    return results


def calculate_layout_statistics(
    panels: list[CameraPanel],
    reference_sun_vector: np.ndarray,
    sun_exclusion_deg: float,
    sky_point_count: int,
) -> LayoutStatistics:
    """Estimate target coverage and duplicate panel area on an equal-area grid."""
    sky_points = fibonacci_sphere_points(sky_point_count)
    multiplicity = np.zeros(sky_point_count, dtype=np.uint16)
    for panel in panels:
        multiplicity += camera_footprint_mask(
            sky_points,
            panel.right_ascension_deg,
            panel.declination_deg,
            CAMERA_SIZE_DEG,
            panel.roll_deg,
        ).astype(np.uint16)

    sun = normalize(reference_sun_vector)
    sun_dot = sky_points @ sun
    target = (
        (sun_dot >= 0.0)
        & (sun_dot <= math.cos(math.radians(sun_exclusion_deg)))
    )
    union = multiplicity > 0
    covered_target = union & target
    union_count = int(np.count_nonzero(union))
    target_count = int(np.count_nonzero(target))
    return LayoutStatistics(
        target_coverage_fraction=(
            float(np.count_nonzero(covered_target)) / target_count
            if target_count
            else 0.0
        ),
        overlap_fraction_of_union=(
            float(np.count_nonzero(multiplicity > 1)) / union_count
            if union_count
            else 0.0
        ),
        panel_union_fraction_of_full_sky=union_count / sky_point_count,
        target_fraction_of_full_sky=target_count / sky_point_count,
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
    sky_point_count: int = DEFAULT_EXPOSURE_SKY_POINT_COUNT,
    sky_block_size: int = 128,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, int]]:
    """Return continuous-access curves for the moving solar hemisphere."""
    (
        earth_unit,
        moon_unit,
        sun_unit,
        earth_threshold,
        moon_threshold,
    ) = body_geometry(
        earth_vectors_km,
        moon_vectors_km,
        sun_vectors_km,
        earth_clearance_deg,
        moon_clearance_deg,
        moon_reference,
    )
    if sky_point_count < 1_000:
        raise ValueError("Use at least 1,000 exposure-curve sky points.")
    if sky_block_size < 1:
        raise ValueError("Sky-grid block size must be positive.")

    sample_count = len(earth_unit)
    window_sizes = {
        duration: duration // TIME_STEP_MINUTES + 1
        for duration in EXPOSURE_DURATIONS_MINUTES
    }
    if sample_count < max(window_sizes.values()):
        raise ValueError("The ephemeris must span at least 24 hours.")

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
        midpoint_indices = np.arange(count, dtype=int) + (window_size - 1) // 2
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
            solar_hemisphere = midpoint_sun_units[duration] @ block.T >= 0.0
            hemisphere_counts[duration] += int(
                np.count_nonzero(solar_hemisphere)
            )
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


def write_exposure_curves_csv(
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
    """Write the three continuous-exposure curves and their metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stop_datetime = start_datetime + timedelta(hours=PLANNING_HOURS)
    fieldnames = [
        "script_revision",
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
                "script_revision": SCRIPT_REVISION,
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
                column = DURATION_COLUMNS[duration]
                percent = float(curves[duration][index])
                row[f"observable_fraction_{column}"] = f"{percent / 100.0:.12f}"
                row[f"observable_percent_{column}"] = f"{percent:.9f}"
            writer.writerow(row)


def write_exposure_curves_figure(
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
    """Restore the original three continuous-exposure line graphs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_index = int(round(selected_sun_exclusion_deg))
    duration_titles = {
        10: "10-minute continuous exposure",
        60: "1-hour continuous exposure",
        24 * 60: "24-hour continuous exposure",
    }
    figure, axes = plt.subplots(3, 1, figsize=(11.5, 12.2), sharex=True)
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
            f"{duration_titles[duration]} · "
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
    figure.subplots_adjust(
        left=0.11,
        right=0.97,
        bottom=0.07,
        top=0.80,
        hspace=0.40,
    )
    figure.suptitle(
        "Continuously observable sky in the Sun-centered hemisphere",
        x=0.11,
        y=0.98,
        ha="left",
        fontsize=15,
        weight="bold",
    )
    figure.text(
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
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def qualifying_panel_counts(
    results: list[PanelDwellResult],
) -> dict[int, int]:
    """Return cumulative counts meeting each continuous exposure duration."""
    return {
        duration: sum(result.max_dwell_minutes >= duration for result in results)
        for duration in EXPOSURE_DURATIONS_MINUTES
    }


def category_counts(results: list[PanelDwellResult]) -> dict[str, int]:
    """Return mutually exclusive longest-dwell category counts."""
    return {
        category: sum(result.category == category for result in results)
        for category in CATEGORY_ORDER
    }


def write_summary_csv(
    path: Path,
    results: list[PanelDwellResult],
    layout: LayoutStatistics,
    start_datetime: datetime,
    observer_name: str,
    ephemeris_source: str,
    sun_exclusion_deg: float,
    earth_clearance_deg: float,
    moon_clearance_deg: float,
    moon_reference: str,
    sky_point_count: int,
    footprint_sample_side: int,
) -> None:
    """Write one row summarizing the unique camera-panel calculation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(results)
    qualifying = qualifying_panel_counts(results)
    exclusive = category_counts(results)
    row: dict[str, str | int] = {
        "script_revision": SCRIPT_REVISION,
        "sun_exclusion_deg": f"{sun_exclusion_deg:.1f}",
        "selected_sun_exclusion": 1,
        "total_panel_count": total,
        "camera_size_deg": f"{CAMERA_SIZE_DEG:.1f}",
        "panel_overlap_fraction": f"{layout.overlap_fraction_of_union:.12f}",
        "panel_overlap_percent": f"{100.0 * layout.overlap_fraction_of_union:.9f}",
        "target_coverage_fraction": f"{layout.target_coverage_fraction:.12f}",
        "target_coverage_percent": f"{100.0 * layout.target_coverage_fraction:.9f}",
        "exclusive_24hr_count": exclusive["24 hours"],
        "exclusive_1hr_count": exclusive["1 hour"],
        "exclusive_10min_count": exclusive["10 minutes"],
        "unavailable_count": exclusive["unavailable"],
        "start_utc": start_datetime.isoformat(),
        "stop_utc": (start_datetime + timedelta(hours=PLANNING_HOURS)).isoformat(),
        "observer": observer_name,
        "ephemeris_source": ephemeris_source,
        "earth_clearance_deg": f"{earth_clearance_deg:.6f}",
        "moon_clearance_deg": f"{moon_clearance_deg:.6f}",
        "moon_reference": moon_reference,
        "layout_sky_point_count": sky_point_count,
        "footprint_sample_side": footprint_sample_side,
        "footprint_samples_per_panel": footprint_sample_side**2,
    }
    for duration in EXPOSURE_DURATIONS_MINUTES:
        column = DURATION_COLUMNS[duration]
        count = qualifying[duration]
        fraction = count / total if total else 0.0
        row[f"panel_count_{column}"] = count
        row[f"observable_fraction_{column}"] = f"{fraction:.12f}"
        row[f"observable_percent_{column}"] = f"{100.0 * fraction:.9f}"

    fieldnames = list(row.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def write_panel_csv(
    path: Path,
    results: list[PanelDwellResult],
    times: tuple[datetime, ...],
    reference_sun_vector: np.ndarray,
    sun_exclusion_deg: float,
    footprint_sample_side: int,
) -> None:
    """Write one schedulable row for every unique 24 x 24 degree image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sun_ra_deg, sun_dec_deg = vector_to_ra_dec(reference_sun_vector)
    fieldnames = [
        "script_revision", "panel_number", "ring_number", "ring_field_number",
        "nominal_solar_elongation_deg", "sun_position_angle_deg",
        "right_ascension_deg", "declination_deg", "roll_deg",
        "max_continuous_dwell_minutes", "max_continuous_dwell_hours",
        "longest_dwell_category", "best_dwell_start_utc",
        "best_dwell_stop_utc", "qualifies_10min", "qualifies_1hr",
        "qualifies_24hr", "safe_five_minute_samples", "camera_size_deg",
        "footprint_samples_tested", "sun_exclusion_deg",
        "reference_sun_ra_deg", "reference_sun_dec_deg",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            panel = result.panel
            start_text = (
                times[result.best_start_index].isoformat()
                if result.best_start_index is not None else ""
            )
            stop_text = (
                times[result.best_stop_index].isoformat()
                if result.best_stop_index is not None else ""
            )
            writer.writerow(
                {
                    "script_revision": SCRIPT_REVISION,
                    "panel_number": panel.number,
                    "ring_number": panel.ring_number,
                    "ring_field_number": panel.ring_field_number,
                    "nominal_solar_elongation_deg": f"{panel.solar_elongation_deg:.9f}",
                    "sun_position_angle_deg": f"{panel.sun_position_angle_deg:.9f}",
                    "right_ascension_deg": f"{panel.right_ascension_deg:.9f}",
                    "declination_deg": f"{panel.declination_deg:.9f}",
                    "roll_deg": f"{panel.roll_deg:.9f}",
                    "max_continuous_dwell_minutes": result.max_dwell_minutes,
                    "max_continuous_dwell_hours": f"{result.max_dwell_minutes / 60.0:.9f}",
                    "longest_dwell_category": result.category,
                    "best_dwell_start_utc": start_text,
                    "best_dwell_stop_utc": stop_text,
                    "qualifies_10min": int(result.max_dwell_minutes >= 10),
                    "qualifies_1hr": int(result.max_dwell_minutes >= 60),
                    "qualifies_24hr": int(result.max_dwell_minutes >= 24 * 60),
                    "safe_five_minute_samples": result.safe_sample_count,
                    "camera_size_deg": f"{CAMERA_SIZE_DEG:.1f}",
                    "footprint_samples_tested": footprint_sample_side**2,
                    "sun_exclusion_deg": f"{sun_exclusion_deg:.1f}",
                    "reference_sun_ra_deg": f"{sun_ra_deg:.9f}",
                    "reference_sun_dec_deg": f"{sun_dec_deg:.9f}",
                }
            )


def configure_sky_axis(axis: plt.Axes) -> None:
    """Apply astronomical RA-left Mollweide formatting."""
    axis.set_facecolor("#111827")
    axis.grid(True, color="white", linewidth=0.4, alpha=0.25, zorder=1)
    axis.set_longitude_grid(30)
    axis.set_latitude_grid(30)
    axis.set_xticklabels(
        ("10h", "8h", "6h", "4h", "2h", "0h",
         "22h", "20h", "18h", "16h", "14h")
    )
    axis.tick_params(labelsize=7.5, colors="#D1D5DB")


def write_figure(
    path: Path,
    results: list[PanelDwellResult],
    layout: LayoutStatistics,
    reference_sun_vector: np.ndarray,
    start_datetime: datetime,
    observer_name: str,
    ephemeris_source: str,
    sun_exclusion_deg: float,
    footprint_sample_side: int,
) -> None:
    """Create a panel map and cumulative duration-count chart."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(15.0, 11.4))
    grid = figure.add_gridspec(2, 1, height_ratios=(2.25, 1.0), hspace=0.30)
    sky_axis = figure.add_subplot(grid[0], projection="mollweide")
    bar_axis = figure.add_subplot(grid[1])
    configure_sky_axis(sky_axis)

    longitude_grid, latitude_grid, grid_vectors = sky_grid(541, 271)
    sun = normalize(reference_sun_vector)
    elongation_deg = np.degrees(
        np.arccos(np.clip(grid_vectors @ sun, -1.0, 1.0))
    ).reshape(longitude_grid.shape)
    target_mask = np.where(
        (elongation_deg >= sun_exclusion_deg) & (elongation_deg <= 90.0),
        1.0, np.nan,
    )
    sky_axis.pcolormesh(
        longitude_grid, latitude_grid, target_mask,
        cmap=ListedColormap(["#0F766E"]), shading="auto", alpha=0.48,
        rasterized=True, zorder=0,
    )

    plot_wrapped_line(
        sky_axis, spherical_cap_boundary(sun, sun_exclusion_deg),
        color="#FB923C", linewidth=2.0, linestyle="--",
        label=f"{sun_exclusion_deg:g}° Sun exclusion", zorder=7,
    )
    plot_wrapped_line(
        sky_axis, spherical_cap_boundary(sun, 90.0),
        color="#FDE047", linewidth=1.8, linestyle=":",
        label="90° Sun-centered boundary", zorder=7,
    )

    for result in results:
        panel = result.panel
        color = CATEGORY_COLORS[result.category]
        boundary = camera_boundary(
            panel.right_ascension_deg, panel.declination_deg,
            CAMERA_SIZE_DEG, panel.roll_deg, points_per_edge=32,
        )
        plot_wrapped_line(
            sky_axis, boundary, color=color, linewidth=1.05,
            zorder=6, outline=False,
        )
        center = camera_basis(
            panel.right_ascension_deg, panel.declination_deg, panel.roll_deg
        )[0]
        center_plot = vectors_to_plot_coordinates(center[None, :])[0]
        sky_axis.scatter(
            [center_plot[0]], [center_plot[1]], s=11, color=color,
            edgecolor="#111827", linewidth=0.35, zorder=8,
        )
        sky_axis.text(
            center_plot[0], center_plot[1], str(panel.number),
            fontsize=4.4, color="white", ha="center", va="bottom", zorder=9,
        )

    sun_plot = vectors_to_plot_coordinates(sun[None, :])[0]
    sky_axis.scatter(
        [sun_plot[0]], [sun_plot[1]], marker="*", s=155,
        color="#FACC15", edgecolor="#111827", linewidth=0.8, zorder=10,
    )
    sky_axis.set_title(
        f"{len(results)} unique 24° × 24° images · "
        f"estimated duplicate area {100.0 * layout.overlap_fraction_of_union:.2f}% · "
        f"target coverage {100.0 * layout.target_coverage_fraction:.1f}%",
        loc="left", fontsize=11.0, weight="bold", pad=15,
    )
    legend_handles = [
        Line2D([], [], color="#FB923C", linestyle="--", linewidth=2,
               label=f"{sun_exclusion_deg:g}° exclusion"),
        Line2D([], [], color="#FDE047", linestyle=":", linewidth=2,
               label="90° boundary"),
    ]
    legend_handles.extend(
        Line2D([], [], color=CATEGORY_COLORS[category], linewidth=2,
               label=category)
        for category in CATEGORY_ORDER
    )
    sky_axis.legend(
        handles=legend_handles, loc="lower left", bbox_to_anchor=(0.01, -0.04),
        ncol=3, fontsize=7.5, framealpha=0.90,
    )

    qualifying = qualifying_panel_counts(results)
    total = len(results)
    durations = list(EXPOSURE_DURATIONS_MINUTES)
    counts = [qualifying[duration] for duration in durations]
    labels = ["≥10 minutes", "≥1 hour", "24 hours"]
    colors = ["#F59E0B", "#38BDF8", "#22C55E"]
    bars = bar_axis.bar(labels, counts, color=colors, width=0.58)
    bar_axis.set_ylabel("Unique panels meeting duration")
    # Leave room above the annotations so they do not collide with the
    # left-aligned subplot title when every field qualifies.
    bar_axis.set_ylim(0, max(total + 6, 6))
    bar_axis.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.8)
    bar_axis.set_axisbelow(True)
    bar_axis.spines[["top", "right"]].set_visible(False)
    bar_axis.set_title(
        "Cumulative continuous-dwell qualification",
        loc="left", fontsize=11, weight="bold",
    )
    for bar, count in zip(bars, counts):
        fraction = count / total if total else 0.0
        bar_axis.text(
            bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.35,
            f"{count}/{total}\n{100.0 * fraction:.1f}%",
            ha="center", va="bottom", fontsize=9, weight="bold",
        )

    exclusive = category_counts(results)
    bar_axis.text(
        1.02, 0.94,
        "Exclusive longest-dwell assignment\n"
        + "\n".join(
            f"{category}: {exclusive[category]}" for category in CATEGORY_ORDER
        ),
        transform=bar_axis.transAxes, ha="left", va="top", fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#F8FAFC",
              "edgecolor": "#CBD5E1"},
    )

    figure.subplots_adjust(left=0.065, right=0.82, top=0.83, bottom=0.075)
    figure.suptitle(
        "GOES-18 Sun-centered camera-panel dwell plan",
        x=0.065, y=0.975, ha="left", fontsize=17, weight="bold",
    )
    figure.text(
        0.065, 0.935,
        f"{start_datetime:%Y-%m-%d %H:%M} to "
        f"{start_datetime + timedelta(hours=24):%Y-%m-%d %H:%M} UTC · "
        f"{observer_name}\n"
        f"Sun exclusion {sun_exclusion_deg:g}° · five-minute ephemeris · "
        f"{footprint_sample_side} × {footprint_sample_side} samples per complete panel · "
        f"source = {ephemeris_source}\n"
        "Panels are fixed in inertial RA/Dec and assigned to their longest "
        "continuous unobscured dwell; each panel appears only once on the map.",
        ha="left", va="top", fontsize=9.1, color="#4B5563",
        linespacing=1.35,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count low-overlap 24 x 24 degree camera panels in the "
            "Sun-centered observable sky, maximize each panel's dwell, and "
            "plot the original three continuous-exposure curves."
        )
    )
    parser.add_argument(
        "--date", dest="start_date", type=parse_date,
        help="UTC start date, YYYY-MM-DD; prompts when omitted",
    )
    parser.add_argument(
        "--start-time", type=parse_clock_time,
        help="UTC start time, HH:MM; prompts when omitted",
    )
    parser.add_argument(
        "--sun-exclusion", type=float, choices=(30.0, 45.0),
        help="Sun-center exclusion radius; prompts when omitted",
    )
    parser.add_argument(
        "--earth-clearance", type=float, default=20.0,
        help="Clearance beyond apparent Earth limb (default: 20 deg)",
    )
    parser.add_argument(
        "--moon-clearance", type=float, default=20.0,
        help="Moon exclusion clearance (default: 20 deg)",
    )
    parser.add_argument(
        "--moon-reference", choices=("center", "limb"), default="center",
        help="Measure Moon clearance from center or limb (default: center)",
    )
    parser.add_argument(
        "--ephemeris-source", choices=("tle", "horizons"), default="tle",
        help="GOES-18 ephemeris source (default: tle)",
    )
    parser.add_argument(
        "--observer-spk", type=int, default=GOES18_SPK_ID,
        help=f"Horizons observer SPK ID (default: {GOES18_SPK_ID})",
    )
    parser.add_argument(
        "--tle-file", type=Path,
        help="Optional local TLE file used with --ephemeris-source tle",
    )
    parser.add_argument(
        "--tle-source", choices=("auto", "space-track", "celestrak"),
        default="auto",
        help="Online TLE source when --tle-file is omitted (default: auto)",
    )
    parser.add_argument(
        "--allow-stale-tle", action="store_true",
        help="Allow stale historical TLEs; future dates continue automatically",
    )
    parser.add_argument(
        "--sky-points", type=int, default=DEFAULT_SKY_POINT_COUNT,
        help="Equal-area points used to estimate layout overlap (default: 32768)",
    )
    parser.add_argument(
        "--exposure-sky-points",
        type=int,
        default=DEFAULT_EXPOSURE_SKY_POINT_COUNT,
        help=(
            "Equal-area directions used for the three exposure curves "
            "(default: 8192)"
        ),
    )
    parser.add_argument(
        "--footprint-sample-side", type=int,
        default=DEFAULT_FOOTPRINT_SAMPLE_SIDE,
        help="Odd samples per panel side used for exclusion tests (default: 13)",
    )
    parser.add_argument(
        "--sky-block-size",
        type=int,
        default=128,
        help="Exposure-curve sky directions processed per block (default: 128)",
    )
    parser.add_argument(
        "--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX,
        help="Output path without extension",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        start_date = args.start_date or prompt_for_date()
        start_time = args.start_time or prompt_for_start_time()
        sun_exclusion = (
            args.sun_exclusion
            if args.sun_exclusion is not None
            else prompt_for_sun_exclusion()
        )
        sun_exclusion_int = int(round(sun_exclusion))
        if not 0.0 <= args.earth_clearance <= 60.0:
            raise ValueError("--earth-clearance must be between 0 and 60 degrees.")
        if not 0.0 <= args.moon_clearance <= 60.0:
            raise ValueError("--moon-clearance must be between 0 and 60 degrees.")
        if args.sky_points < 4_096:
            raise ValueError("--sky-points must be at least 4096.")
        if args.exposure_sky_points < 1_000:
            raise ValueError("--exposure-sky-points must be at least 1000.")
        if args.sky_block_size < 1:
            raise ValueError("--sky-block-size must be positive.")
        if args.footprint_sample_side < 3 or args.footprint_sample_side % 2 == 0:
            raise ValueError("--footprint-sample-side must be an odd integer >= 3.")

        start_datetime = datetime.combine(start_date, start_time, tzinfo=UTC)
        print(
            "GOES-18 Sun-centered camera-panel dwell calculation\n"
            f"Revision: {SCRIPT_REVISION}\n"
            f"Start: {start_datetime:%Y-%m-%d %H:%M} UTC\n"
            f"Interval: {PLANNING_HOURS} hours\n"
            f"Sun exclusion: {sun_exclusion:g} degrees\n"
            f"Camera: {CAMERA_SIZE_DEG:g} x {CAMERA_SIZE_DEG:g} degrees\n"
            f"Ephemeris source: {args.ephemeris_source}",
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
        midpoint_index = len(ephemeris.times) // 2
        reference_sun_vector = ephemeris.sun_vectors_km[midpoint_index]
        panels = calculate_camera_panels(reference_sun_vector, sun_exclusion_int)
        print(
            f"Generated {len(panels)} unique low-overlap camera panels.\n"
            "Testing every complete footprint at five-minute cadence...",
            flush=True,
        )
        results = evaluate_panel_dwells(
            panels,
            ephemeris.earth_vectors_km,
            ephemeris.moon_vectors_km,
            ephemeris.sun_vectors_km,
            sun_exclusion,
            args.earth_clearance,
            args.moon_clearance,
            args.moon_reference,
            args.footprint_sample_side,
        )
        layout = calculate_layout_statistics(
            panels, reference_sun_vector, sun_exclusion, args.sky_points
        )
        print(
            "Calculating the 10-minute, 1-hour, and 24-hour continuous "
            "observable-sky curves...",
            flush=True,
        )
        sun_angles, exposure_curves, exposure_window_counts = (
            sun_centered_exposure_sweep(
                ephemeris.earth_vectors_km,
                ephemeris.moon_vectors_km,
                ephemeris.sun_vectors_km,
                args.earth_clearance,
                args.moon_clearance,
                args.moon_reference,
                args.exposure_sky_points,
                args.sky_block_size,
            )
        )

        output_prefix = args.output_prefix.expanduser().resolve()
        plot_path = output_prefix.with_suffix(".png")
        summary_path = output_prefix.with_suffix(".csv")
        panels_path = output_prefix.parent / (output_prefix.name + "_panels.csv")
        exposure_plot_path = output_prefix.parent / (
            output_prefix.name + "_exposure_curves.png"
        )
        exposure_csv_path = output_prefix.parent / (
            output_prefix.name + "_exposure_curves.csv"
        )
        write_figure(
            plot_path, results, layout, reference_sun_vector, start_datetime,
            ephemeris.observer_name, args.ephemeris_source, sun_exclusion,
            args.footprint_sample_side,
        )
        write_summary_csv(
            summary_path, results, layout, start_datetime,
            ephemeris.observer_name, args.ephemeris_source, sun_exclusion,
            args.earth_clearance, args.moon_clearance, args.moon_reference,
            args.sky_points, args.footprint_sample_side,
        )
        write_panel_csv(
            panels_path, results, ephemeris.times, reference_sun_vector,
            sun_exclusion, args.footprint_sample_side,
        )
        write_exposure_curves_figure(
            exposure_plot_path,
            sun_angles,
            exposure_curves,
            start_datetime,
            ephemeris.observer_name,
            args.ephemeris_source,
            sun_exclusion,
            args.earth_clearance,
            args.moon_clearance,
            args.moon_reference,
            args.exposure_sky_points,
            exposure_window_counts,
        )
        write_exposure_curves_csv(
            exposure_csv_path,
            sun_angles,
            exposure_curves,
            start_datetime,
            ephemeris.observer_name,
            args.ephemeris_source,
            sun_exclusion,
            args.earth_clearance,
            args.moon_clearance,
            args.moon_reference,
            args.exposure_sky_points,
        )

        qualifying = qualifying_panel_counts(results)
        exclusive = category_counts(results)
        print(
            "\nUnique panel results:\n"
            f"  Total panels: {len(results)}\n"
            f"  Estimated duplicate panel area: "
            f"{100.0 * layout.overlap_fraction_of_union:.3f}%\n"
            f"  Target annulus covered by panel union: "
            f"{100.0 * layout.target_coverage_fraction:.3f}%",
            flush=True,
        )
        for duration in EXPOSURE_DURATIONS_MINUTES:
            count = qualifying[duration]
            print(
                f"  Panels supporting at least {DURATION_LABELS[duration]}: "
                f"{count}/{len(results)} "
                f"({100.0 * count / len(results):.3f}%)",
                flush=True,
            )
        print("\nExclusive longest-dwell assignment:", flush=True)
        for category in CATEGORY_ORDER:
            print(f"  {category}: {exclusive[category]}", flush=True)
        selected_angle_index = int(round(sun_exclusion))
        print(
            "\nContinuously observable solar-hemisphere fraction at "
            f"{sun_exclusion:g} degrees:",
            flush=True,
        )
        for duration in EXPOSURE_DURATIONS_MINUTES:
            percent = float(exposure_curves[duration][selected_angle_index])
            print(
                f"  {DURATION_LABELS[duration]}: "
                f"fraction={percent / 100.0:.9f}, percent={percent:.6f}%",
                flush=True,
            )
        print(f"\nWrote plot: {plot_path}", flush=True)
        print(f"Wrote summary: {summary_path}", flush=True)
        print(f"Wrote panel schedule: {panels_path}", flush=True)
        print(f"Wrote exposure curves: {exposure_plot_path}", flush=True)
        print(f"Wrote exposure-curve CSV: {exposure_csv_path}", flush=True)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
