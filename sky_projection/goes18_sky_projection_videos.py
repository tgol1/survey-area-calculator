#!/usr/bin/env python3
"""Generate three GOES-18 sky-coverage MP4 planning products.

The long video spans six calendar months.  Each frame is a 24-hour integrated
sky-coverage map, and frame start times are separated by two days.  The short
video spans 48 hours.  Each frame is a three-hour integrated map, with
successive frame starts separated by three hours.  By default, every
integration uses five-minute vectors generated locally from the latest
GOES-18 TLE, SGP4, and Astropy's built-in Sun/Moon ephemeris.

The annulus-scan video contains two consecutive passes through 15 fixed
camera fields. Each field is centered 45 degrees from the Sun, neighboring
fields differ by 24 degrees in Sun-centered position angle, and each field
is held for one hour. The second 15 frames reuse the exact same inertial
RA, declination, and roll as the first pass so asteroid images can be
compared against an identical field of view.

The videos use H.264 with a yuv420p pixel format so browsers and Streamlit can
display them directly.  ``imageio-ffmpeg`` is preferred on hosted systems;
the script falls back to an ``ffmpeg`` executable already on PATH.
"""

from __future__ import annotations

import argparse
import calendar
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SIX_MONTH_OUTPUT = (
    SCRIPT_DIRECTORY / "neo_6month_24h_sky_coverage.mp4"
)
DEFAULT_48_HOUR_OUTPUT = SCRIPT_DIRECTORY / "neo_48h_3h_sky_coverage.mp4"
DEFAULT_ANNULUS_SCAN_OUTPUT = SCRIPT_DIRECTORY / "neo_30h_annulus_scan.mp4"
DEFAULT_ANNULUS_POINTINGS_OUTPUT = (
    SCRIPT_DIRECTORY / "neo_annulus_scan_pointings.csv"
)
LONG_WINDOW_HOURS = 24
ANNULUS_INNER_RADIUS_DEG = 30.0
ANNULUS_OUTER_RADIUS_DEG = 60.0
ANNULUS_CENTER_RADIUS_DEG = 45.0
ANNULUS_POINTING_COUNT = 15
ANNULUS_PASS_COUNT = 2
ANNULUS_DWELL_HOURS = 1
MAX_EPHEMERIS_SAMPLES_PER_BATCH = 8_000
VIDEO_SCRIPT_VERSION = "2026-10-18-annulus-scan-v3"

try:
    from sky_projection.goes18_daily_camera_projection import (
        CAMERA_COLOR,
        GOES18_SPK_ID,
        MAX_PLANNING_DATE,
        TRACK_COLORS,
        UTC,
        camera_basis,
        camera_boundary,
        camera_footprint_mask,
        camera_statistics,
        daily_coverage_map,
        fetch_ephemeris,
        parse_clock_time,
        parse_date,
        plot_wrapped_line,
        prompt_for_date,
        prompt_for_float,
        prompt_for_start_time,
        prompt_for_sun_exclusion,
        sky_grid,
        spherical_cap_boundary,
        vectors_to_plot_coordinates,
        vectors_to_plot_track,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import sky_projection.goes18_daily_camera_projection. "
        "Place this script in the repository's sky_projection folder."
    ) from exc


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a six-month 24-hour-map MP4 and a 48-hour "
            "three-hour-map MP4, plus a repeated 15-field Sun-annulus "
            "scan MP4, from GOES-18."
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
        "--step-minutes",
        type=int,
        default=5,
        help="Integration cadence in minutes (default: 5)",
    )
    parser.add_argument(
        "--ephemeris-source",
        choices=("tle", "horizons"),
        default="tle",
        help=(
            "Ephemeris source. Local TLE/SGP4 is the future-planning default; "
            "Horizons is retained for validation"
        ),
    )
    parser.add_argument(
        "--observer-spk",
        type=int,
        default=GOES18_SPK_ID,
        help=(
            "Horizons observer SPK ID, used only with "
            f"--ephemeris-source horizons (default: {GOES18_SPK_ID})"
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=6,
        help="Output video frames per second (default: 6)",
    )
    parser.add_argument(
        "--longitude-points",
        type=int,
        default=361,
        help="Sky-grid longitude samples (default: 361)",
    )
    parser.add_argument(
        "--latitude-points",
        type=int,
        default=181,
        help="Sky-grid latitude samples (default: 181)",
    )
    parser.add_argument(
        "--output-six-month",
        type=Path,
        default=DEFAULT_SIX_MONTH_OUTPUT,
        help="Six-month 24-hour-map MP4 output path",
    )
    parser.add_argument(
        "--output-48-hour",
        type=Path,
        default=DEFAULT_48_HOUR_OUTPUT,
        help="48-hour three-hour-map MP4 output path",
    )
    parser.add_argument(
        "--output-annulus-scan",
        type=Path,
        default=DEFAULT_ANNULUS_SCAN_OUTPUT,
        help="Thirty-hour repeated Sun-annulus scan MP4 output path",
    )
    parser.add_argument(
        "--output-annulus-pointings",
        type=Path,
        default=DEFAULT_ANNULUS_POINTINGS_OUTPUT,
        help="CSV output path for the 15 calculated RA/Dec/roll pointings",
    )
    parser.add_argument(
        "--scan-start-position-angle",
        type=float,
        default=0.0,
        help=(
            "Sun-centered position angle of field 1 in degrees; 0 is "
            "celestial north of the Sun (default: 0)"
        ),
    )
    parser.add_argument(
        "--scan-direction",
        choices=("clockwise", "counterclockwise"),
        default="clockwise",
        help="Scan direction on the RA-left Mollweide map (default: clockwise)",
    )

    # These defaults encode the assignment.  Keeping them configurable makes
    # short validation runs possible without changing production behavior.
    parser.add_argument("--long-months", type=int, default=6)
    parser.add_argument("--long-spacing-days", type=int, default=2)
    parser.add_argument("--short-span-hours", type=int, default=48)
    parser.add_argument("--short-window-hours", type=int, default=3)
    parser.add_argument("--short-spacing-hours", type=int, default=3)
    parser.add_argument("--max-long-frames", type=int)
    parser.add_argument("--max-short-frames", type=int)
    return parser.parse_args()


def validate_arguments(
    args: argparse.Namespace,
) -> tuple[datetime, float, float, float, float]:
    start_date = args.start_date or prompt_for_date()
    start_time = args.start_time or prompt_for_start_time()
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
    if not 0.0 <= args.earth_clearance <= 60.0:
        raise ValueError("--earth-clearance must be between 0 and 60 degrees.")
    if not 0.0 <= args.moon_clearance <= 60.0:
        raise ValueError("--moon-clearance must be between 0 and 60 degrees.")
    if args.step_minutes <= 0:
        raise ValueError("--step-minutes must be positive.")
    for hours in (
        LONG_WINDOW_HOURS,
        args.short_window_hours,
        ANNULUS_DWELL_HOURS,
    ):
        if hours <= 0 or hours * 60 % args.step_minutes != 0:
            raise ValueError(
                "Every integration window must be positive and evenly "
                "divisible by --step-minutes."
            )
    if args.long_months <= 0 or args.long_spacing_days <= 0:
        raise ValueError("Long-video duration and spacing must be positive.")
    if args.short_span_hours <= 0 or args.short_spacing_hours <= 0:
        raise ValueError("Short-video duration and spacing must be positive.")
    if args.short_window_hours > args.short_span_hours:
        raise ValueError("The short integration window exceeds its total span.")
    if args.fps < 1 or args.fps > 30:
        raise ValueError("--fps must be between 1 and 30.")
    if args.longitude_points < 181 or args.latitude_points < 91:
        raise ValueError("The sky grid is too coarse; use at least 181 x 91.")
    for name in ("max_long_frames", "max_short_frames"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")

    output_six_month = args.output_six_month.expanduser().resolve()
    output_48_hour = args.output_48_hour.expanduser().resolve()
    output_annulus_scan = args.output_annulus_scan.expanduser().resolve()
    output_annulus_pointings = args.output_annulus_pointings.expanduser().resolve()
    output_paths = {
        output_six_month,
        output_48_hour,
        output_annulus_scan,
        output_annulus_pointings,
    }
    if len(output_paths) != 4:
        raise ValueError("All video and pointing-table output paths must differ.")
    if not math.isfinite(args.scan_start_position_angle):
        raise ValueError("--scan-start-position-angle must be finite.")

    camera_roll = (args.camera_roll + 180.0) % 360.0 - 180.0
    start_datetime = datetime.combine(start_date, start_time, tzinfo=UTC)
    return (
        start_datetime,
        sun_exclusion,
        camera_ra % 360.0,
        camera_dec,
        camera_roll,
    )


def add_calendar_months(value: datetime, months: int) -> datetime:
    """Advance a datetime by whole calendar months, clamping the day."""
    absolute_month = value.year * 12 + (value.month - 1) + months
    year, zero_based_month = divmod(absolute_month, 12)
    month = zero_based_month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def build_long_frame_starts(
    start: datetime,
    months: int,
    spacing_days: int,
) -> list[datetime]:
    """Return 24-hour frame starts spanning the requested calendar months."""
    stop = add_calendar_months(start, months)
    starts: list[datetime] = []
    current = start
    while current + timedelta(hours=LONG_WINDOW_HOURS) <= stop:
        starts.append(current)
        current += timedelta(days=spacing_days)
    if not starts:
        raise ValueError("The six-month sequence did not contain a full frame.")
    return starts


def build_short_frame_starts(
    start: datetime,
    span_hours: int,
    window_hours: int,
    spacing_hours: int,
) -> list[datetime]:
    """Return short-sequence frame starts whose windows stay within the span."""
    starts: list[datetime] = []
    offset = 0
    while offset + window_hours <= span_hours:
        starts.append(start + timedelta(hours=offset))
        offset += spacing_hours
    if not starts:
        raise ValueError("The 48-hour sequence did not contain a full frame.")
    return starts


@dataclass(frozen=True)
class CameraPointing:
    """One fixed inertial camera field in the Sun-annulus scan."""

    number: int
    position_angle_deg: float
    right_ascension_deg: float
    declination_deg: float
    roll_deg: float


def vector_to_ra_dec(vector: np.ndarray) -> tuple[float, float]:
    """Convert one nonzero Cartesian vector to ICRF RA and declination."""
    unit = np.asarray(vector, dtype=float).copy()
    norm = float(np.linalg.norm(unit))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("Cannot convert an invalid vector to RA and declination.")
    unit /= norm
    right_ascension = math.degrees(math.atan2(unit[1], unit[0])) % 360.0
    declination = math.degrees(math.asin(float(np.clip(unit[2], -1.0, 1.0))))
    return right_ascension, declination


def calculate_annulus_pointings(
    sun_vector: np.ndarray,
    *,
    start_position_angle_deg: float = 0.0,
    direction: str = "clockwise",
) -> list[CameraPointing]:
    """Calculate 15 tangent-aligned fields on the 45-degree solar annulus.

    Position angle is measured from celestial north around the Sun. Because
    the plotting convention puts increasing RA to the left, decreasing the
    astronomical position angle moves clockwise on the displayed map.
    """
    sun = np.asarray(sun_vector, dtype=float).copy()
    sun_norm = float(np.linalg.norm(sun))
    if sun_norm == 0.0 or not np.isfinite(sun_norm):
        raise ValueError("The reference Sun vector is invalid.")
    sun /= sun_norm
    sun_ra_deg, sun_dec_deg = vector_to_ra_dec(sun)
    sun_ra = math.radians(sun_ra_deg)
    sun_dec = math.radians(sun_dec_deg)
    sun_east = np.asarray([-math.sin(sun_ra), math.cos(sun_ra), 0.0])
    sun_north = np.asarray(
        [
            -math.sin(sun_dec) * math.cos(sun_ra),
            -math.sin(sun_dec) * math.sin(sun_ra),
            math.cos(sun_dec),
        ]
    )

    if direction not in {"clockwise", "counterclockwise"}:
        raise ValueError("direction must be 'clockwise' or 'counterclockwise'.")
    direction_sign = -1.0 if direction == "clockwise" else 1.0
    radius = math.radians(ANNULUS_CENTER_RADIUS_DEG)
    spacing = 360.0 / ANNULUS_POINTING_COUNT
    pointings: list[CameraPointing] = []

    for index in range(ANNULUS_POINTING_COUNT):
        position_angle_deg = (
            start_position_angle_deg + direction_sign * index * spacing
        ) % 360.0
        position_angle = math.radians(position_angle_deg)
        radial = (
            math.cos(position_angle) * sun_north
            + math.sin(position_angle) * sun_east
        )
        center = math.cos(radius) * sun + math.sin(radius) * radial
        center /= np.linalg.norm(center)

        increasing_angle_tangent = (
            -math.sin(position_angle) * sun_north
            + math.cos(position_angle) * sun_east
        )
        scan_tangent = direction_sign * increasing_angle_tangent
        scan_tangent -= np.dot(scan_tangent, center) * center
        scan_tangent /= np.linalg.norm(scan_tangent)

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
                float(np.dot(scan_tangent, center_north)),
                float(np.dot(scan_tangent, center_east)),
            )
        )
        roll_deg = (roll_deg + 180.0) % 360.0 - 180.0
        pointings.append(
            CameraPointing(
                number=index + 1,
                position_angle_deg=position_angle_deg,
                right_ascension_deg=right_ascension_deg,
                declination_deg=declination_deg,
                roll_deg=roll_deg,
            )
        )

    return pointings


def write_annulus_pointing_table(
    path: Path,
    pointings: list[CameraPointing],
    start_datetime: datetime,
    reference_sun_vector: np.ndarray,
) -> Path:
    """Write the 15 unique fields and both one-hour dwell times to CSV."""
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    sun_ra_deg, sun_dec_deg = vector_to_ra_dec(reference_sun_vector)
    fieldnames = [
        "field_number",
        "sun_position_angle_deg",
        "right_ascension_deg",
        "declination_deg",
        "roll_deg",
        "nominal_solar_elongation_deg",
        "first_dwell_start_utc",
        "first_dwell_stop_utc",
        "repeat_dwell_start_utc",
        "repeat_dwell_stop_utc",
        "reference_sun_ra_deg",
        "reference_sun_dec_deg",
    ]
    with resolved.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, pointing in enumerate(pointings):
            first_start = start_datetime + timedelta(hours=index)
            repeat_start = start_datetime + timedelta(
                hours=index + ANNULUS_POINTING_COUNT
            )
            writer.writerow(
                {
                    "field_number": pointing.number,
                    "sun_position_angle_deg": f"{pointing.position_angle_deg:.9f}",
                    "right_ascension_deg": f"{pointing.right_ascension_deg:.9f}",
                    "declination_deg": f"{pointing.declination_deg:.9f}",
                    "roll_deg": f"{pointing.roll_deg:.9f}",
                    "nominal_solar_elongation_deg": (
                        f"{ANNULUS_CENTER_RADIUS_DEG:.9f}"
                    ),
                    "first_dwell_start_utc": first_start.isoformat(),
                    "first_dwell_stop_utc": (
                        first_start + timedelta(hours=1)
                    ).isoformat(),
                    "repeat_dwell_start_utc": repeat_start.isoformat(),
                    "repeat_dwell_stop_utc": (
                        repeat_start + timedelta(hours=1)
                    ).isoformat(),
                    "reference_sun_ra_deg": f"{sun_ra_deg:.9f}",
                    "reference_sun_dec_deg": f"{sun_dec_deg:.9f}",
                }
            )
    print(f"Wrote pointing table: {resolved}", flush=True)
    return resolved


def group_ephemeris_batches(
    frame_starts: Iterable[datetime],
    window_hours: int,
    step_minutes: int,
) -> list[list[datetime]]:
    """Group nearby frames into bounded-memory ephemeris calculations."""
    groups: list[list[datetime]] = []
    current: list[datetime] = []
    for frame_start in frame_starts:
        candidate_start = current[0] if current else frame_start
        candidate_stop = frame_start + timedelta(hours=window_hours)
        span_minutes = round(
            (candidate_stop - candidate_start).total_seconds() / 60.0
        )
        sample_count = span_minutes // step_minutes + 1
        if current and sample_count > MAX_EPHEMERIS_SAMPLES_PER_BATCH:
            groups.append(current)
            current = [frame_start]
        else:
            current.append(frame_start)
    if current:
        groups.append(current)
    return groups


def find_ffmpeg() -> str:
    """Return an ffmpeg executable suitable for H.264 MP4 output."""
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled:
            return bundled
    except (ImportError, RuntimeError):
        pass

    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    raise RuntimeError(
        "FFmpeg was not found. Install imageio-ffmpeg with "
        "'python -m pip install imageio-ffmpeg'."
    )


class Mp4Writer:
    """Stream fixed-size RGB frames into a browser-compatible H.264 MP4."""

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.stem}-",
            suffix=".mp4",
            dir=self.path.parent,
        )
        os.close(descriptor)
        self.temporary_path = Path(temporary_name)
        command = [
            find_ffmpeg(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.temporary_path),
        ]
        self.width = width
        self.height = height
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.closed = False

    def append(self, frame: np.ndarray) -> None:
        array = np.asarray(frame, dtype=np.uint8)
        if array.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Video frame has shape {array.shape}; expected "
                f"({self.height}, {self.width}, 3)."
            )
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg input stream is unavailable.")
        try:
            self.process.stdin.write(np.ascontiguousarray(array).tobytes())
        except BrokenPipeError as exc:
            stderr = (
                self.process.stderr.read().decode("utf-8", errors="replace")
                if self.process.stderr is not None
                else ""
            )
            raise RuntimeError(f"FFmpeg stopped while encoding: {stderr}") from exc

    def close(self) -> Path:
        if self.closed:
            return self.path
        self.closed = True
        if self.process.stdin is not None:
            self.process.stdin.close()
        stderr = (
            self.process.stderr.read().decode("utf-8", errors="replace")
            if self.process.stderr is not None
            else ""
        )
        return_code = self.process.wait()
        if return_code != 0:
            self.temporary_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"FFmpeg exited with status {return_code}: {stderr.strip()}"
            )
        os.replace(self.temporary_path, self.path)
        return self.path

    def abort(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.process.stdin is not None:
            self.process.stdin.close()
        self.process.terminate()
        self.process.wait()
        self.temporary_path.unlink(missing_ok=True)

    def __enter__(self) -> "Mp4Writer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def render_frame(
    *,
    coverage_hours: np.ndarray,
    longitude_grid: np.ndarray,
    latitude_grid: np.ndarray,
    body_vectors: dict[str, np.ndarray],
    observer_name: str,
    start_datetime: datetime,
    window_hours: int,
    sun_exclusion_deg: float,
    earth_clearance_deg: float,
    moon_clearance_deg: float,
    camera_ra_deg: float,
    camera_dec_deg: float,
    camera_size_deg: float,
    camera_roll_deg: float,
    camera_stats: dict[str, float],
    step_minutes: int,
    sequence_label: str,
    frame_number: int,
    frame_count: int,
    sun_inner_guide_deg: float | None = None,
    camera_sequence_note: str | None = None,
    width: int = 1280,
    height: int = 720,
) -> np.ndarray:
    """Render one fixed-size RGB animation frame."""
    dpi = 100
    figure = plt.figure(
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
        facecolor="white",
    )
    axis = figure.add_axes((0.055, 0.225, 0.89, 0.66), projection="mollweide")
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
    axis.contour(
        longitude_grid,
        latitude_grid,
        coverage_hours,
        levels=tuple(
            window_hours * fraction for fraction in (0.25, 0.50, 0.75)
        ),
        colors="white",
        linewidths=0.55,
        alpha=0.58,
        zorder=2,
    )
    # Matplotlib's clabel() can crash on contours crossing the Mollweide
    # seam. Keep the contour lines and describe their fixed levels below.

    sample_count = len(next(iter(body_vectors.values())))
    marker_interval_hours = 1 if window_hours <= 3 else 6
    for body_name in ("Earth", "Moon", "Sun"):
        coordinates = vectors_to_plot_track(body_vectors[body_name])
        plot_wrapped_line(
            axis,
            coordinates,
            color=TRACK_COLORS[body_name],
            linewidth=1.25 if body_name == "Moon" else 1.0,
            label=f"{body_name} center track",
            zorder=5,
        )
        for hour in range(0, window_hours + 1, marker_interval_hours):
            index = min(round(hour * 60 / step_minutes), sample_count - 1)
            longitude, latitude = coordinates[index]
            axis.scatter(
                [longitude],
                [latitude],
                s=22 if body_name == "Moon" else 17,
                color=TRACK_COLORS[body_name],
                edgecolor="#111827",
                linewidth=0.6,
                zorder=7,
            )

    midpoint_minutes = window_hours * 30
    midpoint_index = min(
        round(midpoint_minutes / step_minutes),
        sample_count - 1,
    )
    moon_cap = spherical_cap_boundary(
        body_vectors["Moon"][midpoint_index],
        moon_clearance_deg,
    )
    plot_wrapped_line(
        axis,
        moon_cap,
        color="#FFFFFF",
        linewidth=1.2,
        linestyle="--",
        label=f"Moon exclusion at midpoint ({moon_clearance_deg:g}°)",
        zorder=6,
    )
    sun_guide = spherical_cap_boundary(
        body_vectors["Sun"][midpoint_index],
        ANNULUS_OUTER_RADIUS_DEG,
    )
    plot_wrapped_line(
        axis,
        sun_guide,
        color="#FFD400",
        linewidth=1.7,
        linestyle=":",
        label=f"{ANNULUS_OUTER_RADIUS_DEG:g}° from Sun at midpoint",
        zorder=7,
    )
    if sun_inner_guide_deg is not None:
        sun_inner_guide = spherical_cap_boundary(
            body_vectors["Sun"][midpoint_index],
            sun_inner_guide_deg,
        )
        plot_wrapped_line(
            axis,
            sun_inner_guide,
            color="#FDE047",
            linewidth=1.7,
            linestyle="--",
            label=f"{sun_inner_guide_deg:g}° from Sun at midpoint",
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
        linewidth=2.5,
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
        s=78,
        color=CAMERA_COLOR,
        linewidth=1.9,
        zorder=10,
    )

    axis.grid(True, color="white", linewidth=0.4, alpha=0.27, zorder=1)
    axis.set_longitude_grid(30)
    axis.set_latitude_grid(30)
    axis.set_xticklabels(
        (
            "10h",
            "8h",
            "6h",
            "4h",
            "2h",
            "0h",
            "22h",
            "20h",
            "18h",
            "16h",
            "14h",
        )
    )
    axis.tick_params(labelsize=7.0, colors="#374151")

    colorbar_axis = figure.add_axes((0.09, 0.105, 0.82, 0.025))
    colorbar = figure.colorbar(
        image,
        cax=colorbar_axis,
        orientation="horizontal",
        ticks=np.linspace(0.0, float(window_hours), 7),
    )
    colorbar.set_label(
        f"Unobscured time during this {window_hours}-hour interval (hours)",
        fontsize=8.4,
        weight="bold",
    )
    colorbar.ax.tick_params(labelsize=7.2)

    legend_handles = [
        Line2D(
            [],
            [],
            color=TRACK_COLORS[name],
            marker="o",
            markeredgecolor="#111827",
            linewidth=1.4,
            markersize=4,
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
                linewidth=1.4,
                path_effects=[
                    path_effects.Stroke(linewidth=2.8, foreground="#111827"),
                    path_effects.Normal(),
                ],
                label=f"Moon midpoint cap ({moon_clearance_deg:g}°)",
            ),
            Line2D(
                [],
                [],
                color="#FFD400",
                linestyle=":",
                linewidth=1.8,
                path_effects=[
                    path_effects.Stroke(linewidth=3.2, foreground="#111827"),
                    path_effects.Normal(),
                ],
                label=f"{ANNULUS_OUTER_RADIUS_DEG:g}° Sun guide at midpoint",
            ),
            Line2D(
                [],
                [],
                color=CAMERA_COLOR,
                linewidth=2.5,
                label=f"Camera field ({camera_size_deg:g}° × {camera_size_deg:g}°)",
            ),
        ]
    )
    if sun_inner_guide_deg is not None:
        legend_handles.insert(
            -1,
            Line2D(
                [],
                [],
                color="#FDE047",
                linestyle="--",
                linewidth=1.8,
                path_effects=[
                    path_effects.Stroke(linewidth=3.2, foreground="#111827"),
                    path_effects.Normal(),
                ],
                label=f"{sun_inner_guide_deg:g}° Sun guide at midpoint",
            ),
        )
    axis.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        fontsize=7.0,
        frameon=True,
        facecolor="white",
        framealpha=0.94,
    )

    stop_datetime = start_datetime + timedelta(hours=window_hours)
    figure.suptitle(
        f"NEO {window_hours}-hour sky-survey coverage",
        y=0.975,
        fontsize=16,
        weight="bold",
        color="#111827",
    )
    if sun_inner_guide_deg is None:
        sun_guide_description = (
            f"Dotted yellow = {ANNULUS_OUTER_RADIUS_DEG:g}° Sun guide; "
        )
    else:
        sun_guide_description = (
            f"Dashed/dotted yellow = {sun_inner_guide_deg:g}°–"
            f"{ANNULUS_OUTER_RADIUS_DEG:g}° Sun annulus; "
        )
    figure.text(
        0.5,
        0.925,
        f"{sequence_label}  ·  Frame {frame_number:,}/{frame_count:,}  ·  "
        f"{start_datetime:%Y-%m-%d %H:%M} to "
        f"{stop_datetime:%Y-%m-%d %H:%M} UTC  ·  "
        f"Sun exclusion {sun_exclusion_deg:g}°",
        ha="center",
        fontsize=8.7,
        color="#4B5563",
    )
    figure.text(
        0.015,
        0.018,
        f"Five-minute integration; white contours = 25%, 50%, and 75%; "
        f"Earth limb + {earth_clearance_deg:g}°; observer = {observer_name}; "
        "RA increases left.\n"
        + sun_guide_description
        + "dashed white = Moon cap at the frame midpoint.",
        ha="left",
        va="bottom",
        fontsize=7.0,
        color="#4B5563",
    )
    figure.text(
        0.985,
        0.018,
        (f"{camera_sequence_note}\n" if camera_sequence_note else "")
        + f"Camera: RA {camera_ra_deg:.2f}°, Dec {camera_dec_deg:+.2f}°, "
        f"roll {camera_roll_deg:+.2f}°\n"
        f"Field mean {camera_stats['mean']:.2f} h "
        f"({100.0 * camera_stats['mean'] / window_hours:.1f}%)  ·  "
        f"full-window area "
        f"{100.0 * camera_stats['full_window_fraction']:.1f}%",
        ha="right",
        va="bottom",
        fontsize=7.0,
        color="#4B5563",
    )

    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    frame = np.ascontiguousarray(rgba[:, :, :3])
    plt.close(figure)
    if frame.shape != (height, width, 3):
        raise RuntimeError(
            f"Rendered frame has shape {frame.shape}; expected "
            f"({height}, {width}, 3)."
        )
    return frame


def generate_video(
    *,
    path: Path,
    frame_starts: list[datetime],
    window_hours: int,
    sequence_label: str,
    longitude_grid: np.ndarray,
    latitude_grid: np.ndarray,
    grid_vectors: np.ndarray,
    camera_mask: np.ndarray,
    sun_exclusion_deg: float,
    earth_clearance_deg: float,
    moon_clearance_deg: float,
    camera_ra_deg: float,
    camera_dec_deg: float,
    camera_size_deg: float,
    camera_roll_deg: float,
    step_minutes: int,
    observer_spk: int,
    ephemeris_source: str,
    fps: int,
) -> Path:
    """Fetch batched ephemerides, render the requested frames, and encode MP4."""
    batches = group_ephemeris_batches(
        frame_starts,
        window_hours,
        step_minutes,
    )
    total_frames = len(frame_starts)
    rendered_frames = 0
    print(
        f"\nGenerating {sequence_label}: {total_frames} frames in "
        f"{len(batches)} ephemeris batch(es)...",
        flush=True,
    )

    with Mp4Writer(path, width=1280, height=720, fps=fps) as writer:
        for batch_number, batch_starts in enumerate(batches, start=1):
            batch_start = batch_starts[0]
            batch_stop = batch_starts[-1] + timedelta(hours=window_hours)
            batch_hours = round(
                (batch_stop - batch_start).total_seconds() / 3600.0
            )
            print(
                f"Ephemeris batch {batch_number}/{len(batches)}: "
                f"{batch_start:%Y-%m-%d %H:%M} through "
                f"{batch_stop:%Y-%m-%d %H:%M} UTC",
                flush=True,
            )
            ephemeris = fetch_ephemeris(
                batch_start,
                window_hours=batch_hours,
                step_minutes=step_minutes,
                observer_spk=observer_spk,
                ephemeris_source=ephemeris_source,
            )

            samples_per_window = window_hours * 60 // step_minutes
            for frame_start in batch_starts:
                offset_samples = round(
                    (frame_start - batch_start).total_seconds()
                    / (60.0 * step_minutes)
                )
                sample_stop = offset_samples + samples_per_window
                track_stop = sample_stop + 1
                coverage_hours = daily_coverage_map(
                    grid_vectors,
                    ephemeris.earth_vectors_km[offset_samples:sample_stop],
                    ephemeris.moon_vectors_km[offset_samples:sample_stop],
                    ephemeris.sun_vectors_km[offset_samples:sample_stop],
                    earth_clearance_deg,
                    moon_clearance_deg,
                    sun_exclusion_deg,
                    step_minutes,
                    latitude_grid.shape,
                )
                statistics = camera_statistics(
                    coverage_hours,
                    camera_mask,
                    latitude_grid,
                    step_minutes,
                    window_hours,
                )
                body_vectors = {
                    "Earth": ephemeris.earth_vectors_km[
                        offset_samples:track_stop
                    ],
                    "Moon": ephemeris.moon_vectors_km[
                        offset_samples:track_stop
                    ],
                    "Sun": ephemeris.sun_vectors_km[
                        offset_samples:track_stop
                    ],
                }
                rendered_frames += 1
                frame = render_frame(
                    coverage_hours=coverage_hours,
                    longitude_grid=longitude_grid,
                    latitude_grid=latitude_grid,
                    body_vectors=body_vectors,
                    observer_name=ephemeris.observer_name,
                    start_datetime=frame_start,
                    window_hours=window_hours,
                    sun_exclusion_deg=sun_exclusion_deg,
                    earth_clearance_deg=earth_clearance_deg,
                    moon_clearance_deg=moon_clearance_deg,
                    camera_ra_deg=camera_ra_deg,
                    camera_dec_deg=camera_dec_deg,
                    camera_size_deg=camera_size_deg,
                    camera_roll_deg=camera_roll_deg,
                    camera_stats=statistics,
                    step_minutes=step_minutes,
                    sequence_label=sequence_label,
                    frame_number=rendered_frames,
                    frame_count=total_frames,
                )
                writer.append(frame)
                print(
                    f"Rendered frame {rendered_frames}/{total_frames}: "
                    f"{frame_start:%Y-%m-%d %H:%M} UTC",
                    flush=True,
                )

    resolved = path.expanduser().resolve()
    print(f"Wrote MP4: {resolved}", flush=True)
    return resolved


def generate_annulus_scan_video(
    *,
    path: Path,
    pointing_table_path: Path,
    start_datetime: datetime,
    longitude_grid: np.ndarray,
    latitude_grid: np.ndarray,
    grid_vectors: np.ndarray,
    earth_clearance_deg: float,
    moon_clearance_deg: float,
    camera_size_deg: float,
    step_minutes: int,
    observer_spk: int,
    ephemeris_source: str,
    fps: int,
    start_position_angle_deg: float,
    direction: str,
) -> tuple[Path, Path]:
    """Render two identical passes through 15 one-hour annulus fields."""
    total_frames = ANNULUS_POINTING_COUNT * ANNULUS_PASS_COUNT
    total_hours = total_frames * ANNULUS_DWELL_HOURS
    print(
        "\nGenerating 30°–60° Sun-annulus scan: "
        f"{total_frames} one-hour frames ({ANNULUS_PASS_COUNT} passes × "
        f"{ANNULUS_POINTING_COUNT} fields)...",
        flush=True,
    )
    ephemeris = fetch_ephemeris(
        start_datetime,
        window_hours=total_hours,
        step_minutes=step_minutes,
        observer_spk=observer_spk,
        ephemeris_source=ephemeris_source,
    )
    reference_sun_vector = ephemeris.sun_vectors_km[0].copy()
    pointings = calculate_annulus_pointings(
        reference_sun_vector,
        start_position_angle_deg=start_position_angle_deg,
        direction=direction,
    )
    samples_per_window = ANNULUS_DWELL_HOURS * 60 // step_minutes

    with Mp4Writer(path, width=1280, height=720, fps=fps) as writer:
        for frame_index in range(total_frames):
            field_index = frame_index % ANNULUS_POINTING_COUNT
            pass_number = frame_index // ANNULUS_POINTING_COUNT + 1
            pointing = pointings[field_index]
            offset_samples = frame_index * samples_per_window
            sample_stop = offset_samples + samples_per_window
            track_stop = sample_stop + 1
            frame_start = start_datetime + timedelta(
                hours=frame_index * ANNULUS_DWELL_HOURS
            )

            coverage_hours = daily_coverage_map(
                grid_vectors,
                ephemeris.earth_vectors_km[offset_samples:sample_stop],
                ephemeris.moon_vectors_km[offset_samples:sample_stop],
                ephemeris.sun_vectors_km[offset_samples:sample_stop],
                earth_clearance_deg,
                moon_clearance_deg,
                ANNULUS_INNER_RADIUS_DEG,
                step_minutes,
                latitude_grid.shape,
            )
            camera_mask = camera_footprint_mask(
                grid_vectors,
                pointing.right_ascension_deg,
                pointing.declination_deg,
                camera_size_deg,
                pointing.roll_deg,
            ).reshape(latitude_grid.shape)
            statistics = camera_statistics(
                coverage_hours,
                camera_mask,
                latitude_grid,
                step_minutes,
                ANNULUS_DWELL_HOURS,
            )
            body_vectors = {
                "Earth": ephemeris.earth_vectors_km[offset_samples:track_stop],
                "Moon": ephemeris.moon_vectors_km[offset_samples:track_stop],
                "Sun": ephemeris.sun_vectors_km[offset_samples:track_stop],
            }
            repeat_note = (
                f"Pass {pass_number}/{ANNULUS_PASS_COUNT} · "
                f"field {pointing.number}/{ANNULUS_POINTING_COUNT}"
            )
            if pass_number == 2:
                repeat_note += f" · exact repeat of frame {pointing.number}"
            frame = render_frame(
                coverage_hours=coverage_hours,
                longitude_grid=longitude_grid,
                latitude_grid=latitude_grid,
                body_vectors=body_vectors,
                observer_name=ephemeris.observer_name,
                start_datetime=frame_start,
                window_hours=ANNULUS_DWELL_HOURS,
                sun_exclusion_deg=ANNULUS_INNER_RADIUS_DEG,
                earth_clearance_deg=earth_clearance_deg,
                moon_clearance_deg=moon_clearance_deg,
                camera_ra_deg=pointing.right_ascension_deg,
                camera_dec_deg=pointing.declination_deg,
                camera_size_deg=camera_size_deg,
                camera_roll_deg=pointing.roll_deg,
                camera_stats=statistics,
                step_minutes=step_minutes,
                sequence_label="Sun-annulus survey · 1-hour dwell per field",
                frame_number=frame_index + 1,
                frame_count=total_frames,
                sun_inner_guide_deg=ANNULUS_INNER_RADIUS_DEG,
                camera_sequence_note=repeat_note,
            )
            writer.append(frame)
            print(
                f"Rendered annulus frame {frame_index + 1}/{total_frames}: "
                f"pass {pass_number}, field {pointing.number}, "
                f"{frame_start:%Y-%m-%d %H:%M} UTC",
                flush=True,
            )

    resolved_video = path.expanduser().resolve()
    print(f"Wrote MP4: {resolved_video}", flush=True)
    resolved_table = write_annulus_pointing_table(
        pointing_table_path,
        pointings,
        start_datetime,
        reference_sun_vector,
    )
    return resolved_video, resolved_table


def main() -> int:
    args = parse_arguments()
    try:
        (
            start_datetime,
            sun_exclusion,
            camera_ra,
            camera_dec,
            camera_roll,
        ) = validate_arguments(args)
        long_starts = build_long_frame_starts(
            start_datetime,
            args.long_months,
            args.long_spacing_days,
        )
        short_starts = build_short_frame_starts(
            start_datetime,
            args.short_span_hours,
            args.short_window_hours,
            args.short_spacing_hours,
        )
        if args.max_long_frames is not None:
            long_starts = long_starts[: args.max_long_frames]
        if args.max_short_frames is not None:
            short_starts = short_starts[: args.max_short_frames]

        latest_planning_sample = max(
            long_starts[-1] + timedelta(hours=LONG_WINDOW_HOURS),
            short_starts[-1] + timedelta(hours=args.short_window_hours),
            start_datetime
            + timedelta(
                hours=(
                    ANNULUS_POINTING_COUNT
                    * ANNULUS_PASS_COUNT
                    * ANNULUS_DWELL_HOURS
                )
            ),
        )
        if (
            args.ephemeris_source == "tle"
            and latest_planning_sample.date() > MAX_PLANNING_DATE
        ):
            raise ValueError(
                "Local TLE planning videos are supported through "
                f"{MAX_PLANNING_DATE.isoformat()}. Choose an earlier start date."
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

        print(
            "GOES-18 NEO sky-coverage video generation\n"
            f"Video script version: {VIDEO_SCRIPT_VERSION}\n"
            f"Start: {start_datetime:%Y-%m-%d %H:%M} UTC\n"
            f"Sun exclusion: {sun_exclusion:g} degrees\n"
            f"Ephemeris source: {args.ephemeris_source}\n"
            f"Long video: {len(long_starts)} frame(s), one 24-hour map every "
            f"{args.long_spacing_days} day(s)\n"
            f"Short video: {len(short_starts)} frame(s), one "
            f"{args.short_window_hours}-hour map every "
            f"{args.short_spacing_hours} hour(s)\n"
            f"Annulus scan: {ANNULUS_POINTING_COUNT * ANNULUS_PASS_COUNT} "
            f"frame(s), {ANNULUS_POINTING_COUNT} fields at "
            f"{360.0 / ANNULUS_POINTING_COUNT:g}° position-angle spacing, "
            f"two exact passes\n"
            f"Playback rate: {args.fps} fps",
            flush=True,
        )

        generate_video(
            path=args.output_six_month,
            frame_starts=long_starts,
            window_hours=LONG_WINDOW_HOURS,
            sequence_label="Six-month evolution · 2-day frame spacing",
            longitude_grid=longitude_grid,
            latitude_grid=latitude_grid,
            grid_vectors=grid_vectors,
            camera_mask=camera_mask,
            sun_exclusion_deg=sun_exclusion,
            earth_clearance_deg=args.earth_clearance,
            moon_clearance_deg=args.moon_clearance,
            camera_ra_deg=camera_ra,
            camera_dec_deg=camera_dec,
            camera_size_deg=args.camera_size,
            camera_roll_deg=camera_roll,
            step_minutes=args.step_minutes,
            observer_spk=args.observer_spk,
            ephemeris_source=args.ephemeris_source,
            fps=args.fps,
        )
        generate_annulus_scan_video(
            path=args.output_annulus_scan,
            pointing_table_path=args.output_annulus_pointings,
            start_datetime=start_datetime,
            longitude_grid=longitude_grid,
            latitude_grid=latitude_grid,
            grid_vectors=grid_vectors,
            earth_clearance_deg=args.earth_clearance,
            moon_clearance_deg=args.moon_clearance,
            camera_size_deg=args.camera_size,
            step_minutes=args.step_minutes,
            observer_spk=args.observer_spk,
            ephemeris_source=args.ephemeris_source,
            fps=args.fps,
            start_position_angle_deg=args.scan_start_position_angle,
            direction=args.scan_direction,
        )
        generate_video(
            path=args.output_48_hour,
            frame_starts=short_starts,
            window_hours=args.short_window_hours,
            sequence_label="48-hour evolution · 3-hour frame spacing",
            longitude_grid=longitude_grid,
            latitude_grid=latitude_grid,
            grid_vectors=grid_vectors,
            camera_mask=camera_mask,
            sun_exclusion_deg=sun_exclusion,
            earth_clearance_deg=args.earth_clearance,
            moon_clearance_deg=args.moon_clearance,
            camera_ra_deg=camera_ra,
            camera_dec_deg=camera_dec,
            camera_size_deg=args.camera_size,
            camera_roll_deg=camera_roll,
            step_minutes=args.step_minutes,
            observer_spk=args.observer_spk,
            ephemeris_source=args.ephemeris_source,
            fps=args.fps,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        "\nAll three sky-coverage videos and the annulus pointing table "
        "are complete.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
