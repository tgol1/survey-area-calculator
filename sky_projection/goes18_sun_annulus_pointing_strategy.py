#!/usr/bin/env python3
"""Plan low-overlap 24 x 24 degree images across the Sunward annulus.

One panel is one unique camera exposure with a fixed inertial right
ascension, declination, and roll. The target is the spherical annulus between
a 30- or 45-degree Sun exclusion boundary and 90 degrees from the Sun.

The layouts are the smallest tested staggered-ring arrangements that cover at
least 98.5 percent of the target on an equal-area grid while keeping each
complete camera footprint between the radial boundaries. The program also
calculates the maximum equal dwell per panel for either one complete pass or
two exact repeat passes within the selected time budget.
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

from horizons_api_algorithm.goes18_visible_sky import GOES18_SPK_ID  # noqa: E402
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
    sky_grid,
    spherical_cap_boundary,
    vectors_to_plot_coordinates,
)


SCRIPT_REVISION = "2026-11-06-low-overlap-auto-dwell-v2"
CAMERA_SIZE_DEG = 24.0
OUTER_SOLAR_ELONGATION_DEG = 90.0
TARGET_COVERAGE_GOAL = 0.985
DEFAULT_AVAILABLE_HOURS = 24.0
DEFAULT_VERIFICATION_POINTS = 32_768
DEFAULT_OUTPUT_PREFIX = (
    SCRIPT_DIRECTORY / "goes18_sun_annulus_pointing_strategy"
)
RING_COLORS = ("#FF2D95", "#38BDF8", "#F59E0B")

# Each field is tangent to its solar-elongation ring. The first and last ring
# centers are one camera half-width from the boundaries, so complete fields
# remain in the requested annulus. These layouts exceed 98.5% equal-area
# coverage with fewer panels and less overlap than the old gap-free plan.
RING_LAYOUTS: dict[int, tuple[tuple[float, int], ...]] = {
    30: ((42.0, 10), (60.0, 13), (78.0, 15)),  # 38 unique images
    45: ((57.0, 13), (78.0, 15)),              # 28 unique images
}


@dataclass(frozen=True)
class CameraPointing:
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
class LayoutMetrics:
    """Equal-area coverage diagnostics for one annulus layout."""

    target_coverage_fraction: float
    target_overlap_fraction: float
    target_uncovered_fraction: float
    outside_target_spill_fraction: float


@dataclass(frozen=True)
class StrategySummary:
    """Panel count, geometry, and exposure-time allocation."""

    exclusion_deg: int
    ring_count: int
    panel_count: int
    annulus_solid_angle_sr: float
    camera_solid_angle_sr: float
    theoretical_minimum_panels: int
    target_coverage_fraction: float
    target_overlap_fraction: float
    target_uncovered_fraction: float
    outside_target_spill_fraction: float
    available_hours: float
    slew_minutes: float
    maximum_one_pass_dwell_minutes: float
    maximum_two_pass_dwell_minutes: float
    one_pass_feasible: bool
    two_pass_feasible: bool


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return one normalized, finite Cartesian vector."""
    result = np.asarray(vector, dtype=float).copy()
    norm = float(np.linalg.norm(result))
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("Cannot normalize an invalid vector.")
    return result / norm


def vector_to_ra_dec(vector: np.ndarray) -> tuple[float, float]:
    """Convert one Cartesian direction into ICRF right ascension and Dec."""
    unit = normalize(vector)
    right_ascension = math.degrees(math.atan2(unit[1], unit[0])) % 360.0
    declination = math.degrees(
        math.asin(float(np.clip(unit[2], -1.0, 1.0)))
    )
    return right_ascension, declination


def solar_tangent_basis(sun_vector: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return Sun unit, celestial-east, and celestial-north unit vectors."""
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


def calculate_pointings(
    sun_vector: np.ndarray,
    exclusion_deg: int,
) -> list[CameraPointing]:
    """Create the selected staggered, low-overlap ring layout."""
    if exclusion_deg not in RING_LAYOUTS:
        raise ValueError("Sun exclusion must be 30 or 45 degrees.")
    sun, sun_east, sun_north = solar_tangent_basis(sun_vector)
    pointings: list[CameraPointing] = []
    panel_number = 1

    for ring_index, (radius_deg, field_count) in enumerate(
        RING_LAYOUTS[exclusion_deg]
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
            pointings.append(
                CameraPointing(
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
    return pointings


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


def verify_layout(
    sun_vector: np.ndarray,
    exclusion_deg: int,
    pointings: list[CameraPointing],
    point_count: int,
) -> LayoutMetrics:
    """Measure target coverage, overlap, gaps, and spill on an equal-area grid."""
    sky_points = fibonacci_sphere_points(point_count)
    sun = normalize(sun_vector)
    sun_dot = sky_points @ sun
    target = (
        (sun_dot >= math.cos(math.radians(OUTER_SOLAR_ELONGATION_DEG)))
        & (sun_dot <= math.cos(math.radians(exclusion_deg)))
    )
    multiplicity = np.zeros(point_count, dtype=np.uint16)
    for pointing in pointings:
        multiplicity += camera_footprint_mask(
            sky_points,
            pointing.right_ascension_deg,
            pointing.declination_deg,
            CAMERA_SIZE_DEG,
            pointing.roll_deg,
        ).astype(np.uint16)

    target_count = int(np.count_nonzero(target))
    covered_target = target & (multiplicity > 0)
    overlapped_target = target & (multiplicity > 1)
    union = multiplicity > 0
    union_count = int(np.count_nonzero(union))
    coverage = np.count_nonzero(covered_target) / target_count
    overlap = np.count_nonzero(overlapped_target) / target_count
    spill = np.count_nonzero(union & ~target) / union_count if union_count else 0.0
    return LayoutMetrics(
        target_coverage_fraction=float(coverage),
        target_overlap_fraction=float(overlap),
        target_uncovered_fraction=float(1.0 - coverage),
        outside_target_spill_fraction=float(spill),
    )


def camera_solid_angle(size_deg: float = CAMERA_SIZE_DEG) -> float:
    """Return the exact solid angle of a square rectilinear field."""
    tangent = math.tan(math.radians(size_deg / 2.0))
    return 4.0 * math.atan(
        tangent * tangent / math.sqrt(1.0 + 2.0 * tangent * tangent)
    )


def annulus_solid_angle(exclusion_deg: int) -> float:
    """Return solid angle between the exclusion radius and 90 degrees."""
    inner = math.radians(exclusion_deg)
    outer = math.radians(OUTER_SOLAR_ELONGATION_DEG)
    return 2.0 * math.pi * (math.cos(inner) - math.cos(outer))


def maximum_equal_dwell_minutes(
    panel_count: int,
    pass_count: int,
    slew_minutes: float,
    available_hours: float,
) -> float:
    """Return the maximum equal dwell that completes all requested passes."""
    observation_count = panel_count * pass_count
    transition_count = max(0, observation_count - 1)
    usable_minutes = available_hours * 60.0 - transition_count * slew_minutes
    return max(0.0, usable_minutes / observation_count)


def summarize_strategy(
    exclusion_deg: int,
    pointings: list[CameraPointing],
    layout: LayoutMetrics,
    slew_minutes: float,
    available_hours: float,
) -> StrategySummary:
    """Create geometry and maximum-dwell results for one target annulus."""
    panel_count = len(pointings)
    one_pass_dwell = maximum_equal_dwell_minutes(
        panel_count, 1, slew_minutes, available_hours
    )
    two_pass_dwell = maximum_equal_dwell_minutes(
        panel_count, 2, slew_minutes, available_hours
    )
    target_area = annulus_solid_angle(exclusion_deg)
    camera_area = camera_solid_angle()
    return StrategySummary(
        exclusion_deg=exclusion_deg,
        ring_count=len(RING_LAYOUTS[exclusion_deg]),
        panel_count=panel_count,
        annulus_solid_angle_sr=target_area,
        camera_solid_angle_sr=camera_area,
        theoretical_minimum_panels=math.ceil(target_area / camera_area),
        target_coverage_fraction=layout.target_coverage_fraction,
        target_overlap_fraction=layout.target_overlap_fraction,
        target_uncovered_fraction=layout.target_uncovered_fraction,
        outside_target_spill_fraction=layout.outside_target_spill_fraction,
        available_hours=available_hours,
        slew_minutes=slew_minutes,
        maximum_one_pass_dwell_minutes=one_pass_dwell,
        maximum_two_pass_dwell_minutes=two_pass_dwell,
        one_pass_feasible=one_pass_dwell > 0.0,
        two_pass_feasible=two_pass_dwell > 0.0,
    )


def write_summary_csv(
    path: Path,
    summaries: dict[int, StrategySummary],
    start_datetime: datetime,
    observer_name: str,
    ephemeris_source: str,
    verification_points: int,
) -> Path:
    """Write one strategy summary row for each exclusion angle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "script_revision", "sun_exclusion_deg", "outer_solar_elongation_deg",
        "annulus_definition", "annulus_solid_angle_sr", "ring_count",
        "ring_layout", "panel_count", "camera_size_deg",
        "camera_solid_angle_sr", "theoretical_area_lower_bound_panels",
        "coverage_goal_fraction", "target_coverage_fraction",
        "target_coverage_percent", "target_overlap_fraction",
        "target_overlap_percent", "target_uncovered_fraction",
        "target_uncovered_percent", "outside_target_spill_fraction",
        "outside_target_spill_percent", "available_hours",
        "slew_minutes_between_fields", "maximum_one_pass_dwell_minutes",
        "maximum_two_pass_dwell_minutes", "one_pass_feasible",
        "two_pass_feasible", "verification_point_count", "start_utc",
        "observer", "ephemeris_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for exclusion_deg in (30, 45):
            summary = summaries[exclusion_deg]
            layout_text = "; ".join(
                f"{radius:g}deg:{count}"
                for radius, count in RING_LAYOUTS[exclusion_deg]
            )
            writer.writerow(
                {
                    "script_revision": SCRIPT_REVISION,
                    "sun_exclusion_deg": exclusion_deg,
                    "outer_solar_elongation_deg": f"{OUTER_SOLAR_ELONGATION_DEG:.1f}",
                    "annulus_definition": (
                        f"{exclusion_deg} to {OUTER_SOLAR_ELONGATION_DEG:g} "
                        "deg solar elongation"
                    ),
                    "annulus_solid_angle_sr": f"{summary.annulus_solid_angle_sr:.12f}",
                    "ring_count": summary.ring_count,
                    "ring_layout": layout_text,
                    "panel_count": summary.panel_count,
                    "camera_size_deg": f"{CAMERA_SIZE_DEG:.1f}",
                    "camera_solid_angle_sr": f"{summary.camera_solid_angle_sr:.12f}",
                    "theoretical_area_lower_bound_panels": summary.theoretical_minimum_panels,
                    "coverage_goal_fraction": f"{TARGET_COVERAGE_GOAL:.6f}",
                    "target_coverage_fraction": f"{summary.target_coverage_fraction:.12f}",
                    "target_coverage_percent": f"{100.0 * summary.target_coverage_fraction:.9f}",
                    "target_overlap_fraction": f"{summary.target_overlap_fraction:.12f}",
                    "target_overlap_percent": f"{100.0 * summary.target_overlap_fraction:.9f}",
                    "target_uncovered_fraction": f"{summary.target_uncovered_fraction:.12f}",
                    "target_uncovered_percent": f"{100.0 * summary.target_uncovered_fraction:.9f}",
                    "outside_target_spill_fraction": f"{summary.outside_target_spill_fraction:.12f}",
                    "outside_target_spill_percent": f"{100.0 * summary.outside_target_spill_fraction:.9f}",
                    "available_hours": f"{summary.available_hours:.6f}",
                    "slew_minutes_between_fields": f"{summary.slew_minutes:.6f}",
                    "maximum_one_pass_dwell_minutes": f"{summary.maximum_one_pass_dwell_minutes:.9f}",
                    "maximum_two_pass_dwell_minutes": f"{summary.maximum_two_pass_dwell_minutes:.9f}",
                    "one_pass_feasible": int(summary.one_pass_feasible),
                    "two_pass_feasible": int(summary.two_pass_feasible),
                    "verification_point_count": verification_points,
                    "start_utc": start_datetime.isoformat(),
                    "observer": observer_name,
                    "ephemeris_source": ephemeris_source,
                }
            )
    return path


def scheduled_interval(
    start_datetime: datetime,
    observation_index: int,
    dwell_minutes: float,
    slew_minutes: float,
) -> tuple[datetime, datetime]:
    """Return one sequential dwell interval in an equal-duration schedule."""
    start = start_datetime + timedelta(
        minutes=observation_index * (dwell_minutes + slew_minutes)
    )
    return start, start + timedelta(minutes=dwell_minutes)


def write_pointings_csv(
    path: Path,
    pointings_by_exclusion: dict[int, list[CameraPointing]],
    summaries: dict[int, StrategySummary],
    start_datetime: datetime,
    reference_sun_vector: np.ndarray,
) -> Path:
    """Write coordinates and maximum one- and two-pass dwell schedules."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sun_ra_deg, sun_dec_deg = vector_to_ra_dec(reference_sun_vector)
    fieldnames = [
        "script_revision", "sun_exclusion_deg", "panel_number",
        "ring_number", "ring_field_number", "nominal_solar_elongation_deg",
        "sun_position_angle_deg", "right_ascension_deg", "declination_deg",
        "roll_deg", "one_pass_maximum_dwell_minutes", "one_pass_start_utc",
        "one_pass_stop_utc", "two_pass_maximum_dwell_minutes",
        "two_pass_first_start_utc", "two_pass_first_stop_utc",
        "two_pass_repeat_start_utc", "two_pass_repeat_stop_utc",
        "two_pass_revisit_delay_hours", "reference_sun_ra_deg",
        "reference_sun_dec_deg",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for exclusion_deg in (30, 45):
            pointings = pointings_by_exclusion[exclusion_deg]
            summary = summaries[exclusion_deg]
            one_dwell = summary.maximum_one_pass_dwell_minutes
            two_dwell = summary.maximum_two_pass_dwell_minutes
            panel_count = len(pointings)
            revisit_delay_hours = (
                panel_count * (two_dwell + summary.slew_minutes) / 60.0
            )
            for index, pointing in enumerate(pointings):
                one_start, one_stop = scheduled_interval(
                    start_datetime, index, one_dwell, summary.slew_minutes
                )
                two_first_start, two_first_stop = scheduled_interval(
                    start_datetime, index, two_dwell, summary.slew_minutes
                )
                two_repeat_start, two_repeat_stop = scheduled_interval(
                    start_datetime, panel_count + index,
                    two_dwell, summary.slew_minutes,
                )
                writer.writerow(
                    {
                        "script_revision": SCRIPT_REVISION,
                        "sun_exclusion_deg": exclusion_deg,
                        "panel_number": pointing.number,
                        "ring_number": pointing.ring_number,
                        "ring_field_number": pointing.ring_field_number,
                        "nominal_solar_elongation_deg": f"{pointing.solar_elongation_deg:.9f}",
                        "sun_position_angle_deg": f"{pointing.sun_position_angle_deg:.9f}",
                        "right_ascension_deg": f"{pointing.right_ascension_deg:.9f}",
                        "declination_deg": f"{pointing.declination_deg:.9f}",
                        "roll_deg": f"{pointing.roll_deg:.9f}",
                        "one_pass_maximum_dwell_minutes": f"{one_dwell:.9f}",
                        "one_pass_start_utc": one_start.isoformat(),
                        "one_pass_stop_utc": one_stop.isoformat(),
                        "two_pass_maximum_dwell_minutes": f"{two_dwell:.9f}",
                        "two_pass_first_start_utc": two_first_start.isoformat(),
                        "two_pass_first_stop_utc": two_first_stop.isoformat(),
                        "two_pass_repeat_start_utc": two_repeat_start.isoformat(),
                        "two_pass_repeat_stop_utc": two_repeat_stop.isoformat(),
                        "two_pass_revisit_delay_hours": f"{revisit_delay_hours:.9f}",
                        "reference_sun_ra_deg": f"{sun_ra_deg:.9f}",
                        "reference_sun_dec_deg": f"{sun_dec_deg:.9f}",
                    }
                )
    return path


def configure_sky_axis(axis: plt.Axes) -> None:
    """Apply astronomical RA-left Mollweide formatting."""
    axis.grid(True, color="white", linewidth=0.4, alpha=0.28, zorder=1)
    axis.set_longitude_grid(30)
    axis.set_latitude_grid(30)
    axis.set_xticklabels(
        ("10h", "8h", "6h", "4h", "2h", "0h", "22h", "20h", "18h", "16h", "14h")
    )
    axis.tick_params(labelsize=7.2, colors="#D1D5DB")
    axis.set_facecolor("#111827")


def write_figure(
    path: Path,
    start_datetime: datetime,
    sun_vector: np.ndarray,
    pointings_by_exclusion: dict[int, list[CameraPointing]],
    summaries: dict[int, StrategySummary],
    observer_name: str,
    ephemeris_source: str,
) -> Path:
    """Draw both optimized annulus layouts on one RA/Dec Mollweide figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    longitude_grid, latitude_grid, grid_vectors = sky_grid(541, 271)
    sun = normalize(sun_vector)
    elongation_deg = np.degrees(
        np.arccos(np.clip(grid_vectors @ sun, -1.0, 1.0))
    ).reshape(longitude_grid.shape)
    sun_plot = vectors_to_plot_coordinates(sun[None, :])[0]
    figure, axes = plt.subplots(
        2, 1, figsize=(15.0, 13.0), subplot_kw={"projection": "mollweide"}
    )
    figure.patch.set_facecolor("white")

    for axis, exclusion_deg in zip(axes, (30, 45)):
        configure_sky_axis(axis)
        target_mask = np.where(
            (elongation_deg >= exclusion_deg)
            & (elongation_deg <= OUTER_SOLAR_ELONGATION_DEG),
            1.0, np.nan,
        )
        axis.pcolormesh(
            longitude_grid, latitude_grid, target_mask,
            cmap=ListedColormap(["#0F766E"]), shading="auto", alpha=0.55,
            rasterized=True, zorder=0,
        )
        plot_wrapped_line(
            axis, spherical_cap_boundary(sun, exclusion_deg),
            color="#FB923C", linewidth=2.0, linestyle="--",
            label=f"{exclusion_deg}° exclusion", zorder=7,
        )
        plot_wrapped_line(
            axis, spherical_cap_boundary(sun, OUTER_SOLAR_ELONGATION_DEG),
            color="#FDE047", linewidth=1.8, linestyle=":",
            label="90° outer boundary", zorder=7,
        )

        for pointing in pointings_by_exclusion[exclusion_deg]:
            ring_color = RING_COLORS[
                (pointing.ring_number - 1) % len(RING_COLORS)
            ]
            boundary = camera_boundary(
                pointing.right_ascension_deg, pointing.declination_deg,
                CAMERA_SIZE_DEG, pointing.roll_deg, points_per_edge=30,
            )
            plot_wrapped_line(
                axis, boundary, color=ring_color, linewidth=0.85,
                zorder=5, outline=False,
            )
            center = camera_basis(
                pointing.right_ascension_deg,
                pointing.declination_deg,
                pointing.roll_deg,
            )[0]
            center_plot = vectors_to_plot_coordinates(center[None, :])[0]
            axis.scatter(
                [center_plot[0]], [center_plot[1]], s=9, color=ring_color,
                edgecolor="#111827", linewidth=0.25, zorder=8,
            )
            axis.text(
                center_plot[0], center_plot[1], str(pointing.number),
                fontsize=4.2, color="white", ha="center", va="bottom", zorder=9,
            )

        axis.scatter(
            [sun_plot[0]], [sun_plot[1]], marker="*", s=145,
            color="#FACC15", edgecolor="#111827", linewidth=0.9, zorder=10,
        )
        summary = summaries[exclusion_deg]
        axis.set_title(
            f"{exclusion_deg}°–90° annulus · {summary.panel_count} unique panels · "
            f"{100.0 * summary.target_coverage_fraction:.2f}% covered · "
            f"{100.0 * summary.target_overlap_fraction:.2f}% overlap\n"
            f"Maximum equal dwell in {summary.available_hours:g} h: "
            f"{summary.maximum_one_pass_dwell_minutes:.1f} min (one pass) · "
            f"{summary.maximum_two_pass_dwell_minutes:.1f} min (two passes)",
            loc="left", fontsize=10.8, weight="bold", pad=16,
        )

        legend_handles = [
            Line2D([], [], color="#FB923C", linestyle="--", linewidth=2.0,
                   label=f"{exclusion_deg}° exclusion"),
            Line2D([], [], color="#FDE047", linestyle=":", linewidth=2.0,
                   label="90° from Sun"),
        ]
        legend_handles.extend(
            Line2D(
                [], [], color=RING_COLORS[index % len(RING_COLORS)],
                linewidth=1.6, label=f"{radius:g}° ring: {field_count} panels",
            )
            for index, (radius, field_count) in enumerate(
                RING_LAYOUTS[exclusion_deg]
            )
        )
        axis.legend(
            handles=legend_handles, loc="lower left", bbox_to_anchor=(0.01, -0.03),
            fontsize=7.6, framealpha=0.88,
        )

    figure.subplots_adjust(
        left=0.055, right=0.98, top=0.84, bottom=0.08, hspace=0.32
    )
    figure.suptitle(
        "GOES-18 low-overlap Sun-annulus camera strategy",
        x=0.06, y=0.965, ha="left", fontsize=17, weight="bold",
    )
    first_summary = summaries[30]
    figure.text(
        0.06, 0.925,
        f"{start_datetime:%Y-%m-%d %H:%M UTC} · {observer_name} · "
        f"source = {ephemeris_source}\n"
        f"One panel = one fixed 24° × 24° image · "
        f"{first_summary.available_hours:g}-hour time budget · "
        f"{first_summary.slew_minutes:g}-minute slew between images\n"
        f"Layouts target at least {100.0 * TARGET_COVERAGE_GOAL:.1f}% coverage "
        "with minimum tested panel count; alternate rings are staggered.",
        ha="left", va="top", fontsize=9.4, color="#4B5563", linespacing=1.35,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan minimum-count, low-overlap 24-degree images for the "
            "30-to-90 and 45-to-90 degree Sun-centered annuli."
        )
    )
    parser.add_argument(
        "--date", dest="start_date", type=parse_date,
        help="UTC strategy date, YYYY-MM-DD; prompts when omitted",
    )
    parser.add_argument(
        "--start-time", type=parse_clock_time,
        help="UTC strategy start time, HH:MM; prompts when omitted",
    )
    parser.add_argument(
        "--slew-minutes", type=float, default=0.0,
        help="Transition time between consecutive images (default: 0)",
    )
    parser.add_argument(
        "--available-hours", type=float, default=DEFAULT_AVAILABLE_HOURS,
        help="Total time budget used to maximize equal dwell (default: 24)",
    )
    parser.add_argument(
        "--verification-points", type=int,
        default=DEFAULT_VERIFICATION_POINTS,
        help="Equal-area points used to verify coverage (default: 32768)",
    )
    parser.add_argument(
        "--ephemeris-source", choices=("tle", "horizons"), default="tle",
        help="Source for the GOES-18-centered Sun vector (default: tle)",
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
        "--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX,
        help="Output path without extension",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        start_date = args.start_date or prompt_for_date()
        start_time = args.start_time or prompt_for_start_time()
        if not 0.0 <= args.slew_minutes <= 24.0 * 60.0:
            raise ValueError("--slew-minutes must be between 0 and 1440.")
        if not 1.0 <= args.available_hours <= 24.0 * 365.0:
            raise ValueError("--available-hours must be between 1 and 8760.")
        if args.verification_points < 8_192:
            raise ValueError("--verification-points must be at least 8192.")

        start_datetime = datetime.combine(start_date, start_time, tzinfo=UTC)
        print(
            "GOES-18 low-overlap Sun-annulus pointing strategy\n"
            f"Revision: {SCRIPT_REVISION}\n"
            f"Start: {start_datetime:%Y-%m-%d %H:%M} UTC\n"
            f"Camera: {CAMERA_SIZE_DEG:g} x {CAMERA_SIZE_DEG:g} degrees\n"
            f"Coverage goal: {100.0 * TARGET_COVERAGE_GOAL:.1f}%\n"
            f"Available observing time: {args.available_hours:g} hours\n"
            f"Slew between images: {args.slew_minutes:g} minutes",
            flush=True,
        )
        ephemeris = fetch_ephemeris(
            start_datetime, window_hours=1, step_minutes=60,
            observer_spk=args.observer_spk,
            ephemeris_source=args.ephemeris_source,
            tle_file=args.tle_file, tle_source=args.tle_source,
            allow_stale_tle=args.allow_stale_tle,
        )
        reference_sun_vector = ephemeris.sun_vectors_km[0]

        pointings_by_exclusion: dict[int, list[CameraPointing]] = {}
        summaries: dict[int, StrategySummary] = {}
        for exclusion_deg in (30, 45):
            pointings = calculate_pointings(reference_sun_vector, exclusion_deg)
            layout = verify_layout(
                reference_sun_vector, exclusion_deg, pointings,
                args.verification_points,
            )
            if layout.target_coverage_fraction < TARGET_COVERAGE_GOAL:
                raise RuntimeError(
                    f"The {exclusion_deg}-degree layout verified at only "
                    f"{100.0 * layout.target_coverage_fraction:.3f}% coverage, "
                    f"below the {100.0 * TARGET_COVERAGE_GOAL:.1f}% goal."
                )
            pointings_by_exclusion[exclusion_deg] = pointings
            summaries[exclusion_deg] = summarize_strategy(
                exclusion_deg, pointings, layout,
                args.slew_minutes, args.available_hours,
            )

        output_prefix = args.output_prefix.expanduser().resolve()
        plot_path = output_prefix.with_suffix(".png")
        summary_path = output_prefix.parent / (output_prefix.name + "_summary.csv")
        pointings_path = output_prefix.parent / (output_prefix.name + "_pointings.csv")
        write_figure(
            plot_path, start_datetime, reference_sun_vector,
            pointings_by_exclusion, summaries,
            ephemeris.observer_name, args.ephemeris_source,
        )
        write_summary_csv(
            summary_path, summaries, start_datetime,
            ephemeris.observer_name, args.ephemeris_source,
            args.verification_points,
        )
        write_pointings_csv(
            pointings_path, pointings_by_exclusion, summaries,
            start_datetime, reference_sun_vector,
        )

        for exclusion_deg in (30, 45):
            summary = summaries[exclusion_deg]
            print(
                f"\n{exclusion_deg} to 90 degree observable annulus:\n"
                f"  Solid angle: {summary.annulus_solid_angle_sr:.6f} sr\n"
                f"  Rings: {summary.ring_count}\n"
                f"  Unique 24 x 24 panels: {summary.panel_count}\n"
                f"  Area-only lower bound: {summary.theoretical_minimum_panels} panels\n"
                f"  Equal-area target coverage: "
                f"{100.0 * summary.target_coverage_fraction:.6f}%\n"
                f"  Target covered more than once: "
                f"{100.0 * summary.target_overlap_fraction:.6f}%\n"
                f"  Maximum dwell for one complete pass: "
                f"{summary.maximum_one_pass_dwell_minutes:.6f} minutes\n"
                f"  Maximum dwell for two complete passes: "
                f"{summary.maximum_two_pass_dwell_minutes:.6f} minutes",
                flush=True,
            )
        print(f"\nWrote plot: {plot_path}", flush=True)
        print(f"Wrote summary: {summary_path}", flush=True)
        print(f"Wrote pointings and schedules: {pointings_path}", flush=True)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
