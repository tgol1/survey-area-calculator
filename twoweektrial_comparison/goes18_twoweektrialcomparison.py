#!/usr/bin/env python3
"""Compare daily GOES-18 visible-sky profiles for two 14-day windows.

The program can use two manually entered date ranges or randomly choose two
non-overlapping 14-day windows. It queries the existing JPL Horizons algorithm
at one-hour cadence, plots every UTC day as a 24-point line, and emphasizes the
hour-by-hour mean profile for each window with a thicker line.

Place this file at:

    survey-area-calculator/algorithm_comparison/two_window_daily_variation.py

The existing JPL script must remain at:

    survey-area-calculator/horizons_api_algorithm/goes18_visible_sky.py

On the plot, hour 1 represents 00:00 UTC and hour 24 represents 23:00 UTC.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


UTC = timezone.utc
WINDOW_DAYS = 14
HOURS_PER_DAY = 24
WINDOW_SAMPLE_COUNT = WINDOW_DAYS * HOURS_PER_DAY

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from horizons_api_algorithm.goes18_visible_sky import (
        EARTH_ID,
        GOES18_SPK_ID,
        MOON_ID,
        SUN_ID,
        compute_visible_sky,
        fetch_vectors,
        horizons_calendar_to_utc,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import horizons_api_algorithm.goes18_visible_sky. "
        "Place this script in the repository's algorithm_comparison folder."
    ) from exc


@dataclass(frozen=True)
class DateWindow:
    """One exact 14-day interval with an exclusive stop date."""

    name: str
    start: date
    stop: date

    @property
    def display_range(self) -> str:
        """Return the inclusive calendar-date range shown to the user."""
        return f"{self.start.isoformat()} to {(self.stop - timedelta(days=1)).isoformat()}"


@dataclass(frozen=True)
class WindowResult:
    """Hourly visible-sky values arranged as 14 rows by 24 UTC hours."""

    window: DateWindow
    dates: tuple[date, ...]
    visible_fraction: np.ndarray

    @property
    def visible_percent(self) -> np.ndarray:
        return 100.0 * self.visible_fraction

    @property
    def mean_percent_by_hour(self) -> np.ndarray:
        return np.mean(self.visible_percent, axis=0)


def parse_arguments() -> argparse.Namespace:
    """Parse optional settings; omitted experiment settings are prompted."""
    current_year = datetime.now(UTC).year
    parser = argparse.ArgumentParser(
        description=(
            "Overlay hourly GOES-18 visible-sky profiles from two 14-day "
            "JPL Horizons windows."
        )
    )
    parser.add_argument(
        "--selection",
        choices=("manual", "random"),
        help="Choose manual date ranges or two random 14-day windows",
    )
    parser.add_argument("--window-1-start", help="First start date, YYYY-MM-DD")
    parser.add_argument("--window-1-stop", help="First exclusive stop date, YYYY-MM-DD")
    parser.add_argument("--window-2-start", help="Second start date, YYYY-MM-DD")
    parser.add_argument("--window-2-stop", help="Second exclusive stop date, YYYY-MM-DD")
    parser.add_argument(
        "--random-earliest",
        default=f"{current_year:04d}-01-01",
        help="Earliest random start date (default: first day of current UTC year)",
    )
    parser.add_argument(
        "--random-latest",
        default=f"{current_year + 1:04d}-01-01",
        help="Exclusive random-selection boundary (default: next January 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for reproducible random windows",
    )
    parser.add_argument(
        "--sun-exclusion",
        type=float,
        choices=(30.0, 45.0),
        help="Sun-centered exclusion radius; prompted if omitted",
    )
    parser.add_argument(
        "--earth-clearance",
        type=float,
        default=20.0,
        help="Earth-limb clearance in degrees (default: 20)",
    )
    parser.add_argument(
        "--moon-clearance",
        type=float,
        default=20.0,
        help="Moon-center clearance in degrees (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIRECTORY / "two_window_daily_variation.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=SCRIPT_DIRECTORY / "two_window_daily_variation.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def parse_date(value: str) -> date:
    """Parse a strict YYYY-MM-DD calendar date."""
    try:
        parsed = datetime.strptime(value.strip(), "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"Invalid date {value!r}; use YYYY-MM-DD, for example 2026-08-01."
        ) from exc
    return parsed.date()


def prompt_for_date(label: str) -> date:
    """Prompt repeatedly until a valid date is entered."""
    while True:
        try:
            return parse_date(input(f"{label} (YYYY-MM-DD): "))
        except ValueError as exc:
            print(exc)


def prompt_for_selection() -> str:
    """Prompt for manual or random window selection."""
    print("Date-window options:")
    print("  1. Enter two 14-day date ranges")
    print("  2. Randomly select two non-overlapping 14-day windows")
    while True:
        choice = input("Choose option 1 or 2: ").strip().lower()
        if choice in {"1", "manual", "m"}:
            return "manual"
        if choice in {"2", "random", "r"}:
            return "random"
        print("Please enter 1 for manual dates or 2 for random dates.")


def prompt_for_sun_exclusion() -> float:
    """Prompt until a 30- or 45-degree Sun exclusion is selected."""
    while True:
        value = input("Choose Sun exclusion angle (30 or 45 degrees): ").strip()
        if value in {"30", "30.0"}:
            return 30.0
        if value in {"45", "45.0"}:
            return 45.0
        print("Please enter either 30 or 45.")


def validate_window(name: str, start: date, stop: date) -> DateWindow:
    """Require an exact 14-day interval; the stop date is exclusive."""
    duration = stop - start
    if duration != timedelta(days=WINDOW_DAYS):
        raise ValueError(
            f"{name} must be exactly {WINDOW_DAYS} days: its stop date must "
            f"be {WINDOW_DAYS} days after its start date."
        )
    return DateWindow(name=name, start=start, stop=stop)


def windows_overlap(first: DateWindow, second: DateWindow) -> bool:
    """Return whether two half-open date intervals overlap."""
    return first.start < second.stop and second.start < first.stop


def collect_manual_windows(args: argparse.Namespace) -> tuple[DateWindow, DateWindow]:
    """Collect and validate two manually selected 14-day windows."""
    first_start = (
        parse_date(args.window_1_start)
        if args.window_1_start
        else prompt_for_date("Window 1 start date")
    )
    first_stop = (
        parse_date(args.window_1_stop)
        if args.window_1_stop
        else prompt_for_date("Window 1 stop date (14 days after start)")
    )
    second_start = (
        parse_date(args.window_2_start)
        if args.window_2_start
        else prompt_for_date("Window 2 start date")
    )
    second_stop = (
        parse_date(args.window_2_stop)
        if args.window_2_stop
        else prompt_for_date("Window 2 stop date (14 days after start)")
    )

    first = validate_window("Window 1", first_start, first_stop)
    second = validate_window("Window 2", second_start, second_stop)
    if windows_overlap(first, second):
        raise ValueError("The two 14-day windows must not overlap.")
    return first, second


def choose_random_windows(args: argparse.Namespace) -> tuple[DateWindow, DateWindow]:
    """Choose two non-overlapping 14-day windows from the configured bounds."""
    earliest = parse_date(args.random_earliest)
    latest = parse_date(args.random_latest)
    if latest - earliest < timedelta(days=2 * WINDOW_DAYS):
        raise ValueError(
            "The random-selection interval must contain at least 28 days so "
            "two non-overlapping 14-day windows can be selected."
        )

    last_start = latest - timedelta(days=WINDOW_DAYS)
    candidate_count = (last_start - earliest).days + 1
    candidates = [earliest + timedelta(days=index) for index in range(candidate_count)]
    generator = random.Random(args.seed) if args.seed is not None else random.SystemRandom()

    first_start = generator.choice(candidates)
    second_candidates = [
        candidate
        for candidate in candidates
        if abs((candidate - first_start).days) >= WINDOW_DAYS
    ]
    if not second_candidates:
        raise ValueError("Could not choose two non-overlapping random windows.")
    second_start = generator.choice(second_candidates)
    selected_starts = sorted((first_start, second_start))

    first = DateWindow(
        "Window 1",
        selected_starts[0],
        selected_starts[0] + timedelta(days=WINDOW_DAYS),
    )
    second = DateWindow(
        "Window 2",
        selected_starts[1],
        selected_starts[1] + timedelta(days=WINDOW_DAYS),
    )
    return first, second


def collect_settings(
    args: argparse.Namespace,
) -> tuple[tuple[DateWindow, DateWindow], float]:
    """Collect selection mode, two windows, and the Sun exclusion angle."""
    selection = args.selection or prompt_for_selection()
    windows = (
        collect_manual_windows(args)
        if selection == "manual"
        else choose_random_windows(args)
    )
    sun_exclusion = (
        args.sun_exclusion
        if args.sun_exclusion is not None
        else prompt_for_sun_exclusion()
    )
    return windows, sun_exclusion


def fetch_target_vectors(
    target_id: int,
    target_name: str,
    window: DateWindow,
) -> tuple[np.ndarray, list[str], np.ndarray, str]:
    """Fetch one target's hourly vectors for a selected window."""
    print(f"  Downloading {target_name} vectors...")
    return fetch_vectors(
        target_id,
        GOES18_SPK_ID,
        window.start.isoformat(),
        window.stop.isoformat(),
        "1 h",
    )


def calculate_window(
    window: DateWindow,
    earth_clearance: float,
    moon_clearance: float,
    sun_exclusion: float,
) -> tuple[WindowResult, str]:
    """Run the existing JPL geometry for one exact 14-day window."""
    print(f"\nRunning {window.name}: {window.display_range}")
    earth_jd, earth_calendar, earth_vectors, observer = fetch_target_vectors(
        EARTH_ID, "Earth", window
    )
    moon_jd, moon_calendar, moon_vectors, moon_observer = fetch_target_vectors(
        MOON_ID, "Moon", window
    )
    sun_jd, sun_calendar, sun_vectors, sun_observer = fetch_target_vectors(
        SUN_ID, "Sun", window
    )

    for target_name, target_jd, target_calendar, target_observer in (
        ("Moon", moon_jd, moon_calendar, moon_observer),
        ("Sun", sun_jd, sun_calendar, sun_observer),
    ):
        if not np.array_equal(earth_jd, target_jd):
            raise RuntimeError(f"Earth and {target_name} Julian dates do not match.")
        if earth_calendar != target_calendar:
            raise RuntimeError(f"Earth and {target_name} calendar labels do not match.")
        if target_observer != observer:
            raise RuntimeError(f"Earth and {target_name} observer names do not match.")

    if "GOES-18" not in observer:
        raise RuntimeError(f"Horizons did not resolve the observer as GOES-18: {observer}")

    results = compute_visible_sky(
        earth_vectors,
        moon_vectors,
        sun_vectors,
        earth_clearance,
        moon_clearance,
        "center",
        sun_exclusion,
    )
    timestamps = [horizons_calendar_to_utc(label) for label in earth_calendar]
    visible_by_time = {
        timestamp: float(results["visible_fraction"][index])
        for index, timestamp in enumerate(timestamps)
    }

    start_time = datetime.combine(window.start, datetime.min.time(), tzinfo=UTC)
    expected_times = [
        start_time + timedelta(hours=index) for index in range(WINDOW_SAMPLE_COUNT)
    ]
    missing_times = [timestamp for timestamp in expected_times if timestamp not in visible_by_time]
    if missing_times:
        raise RuntimeError(
            f"Horizons omitted {len(missing_times)} expected hourly sample(s); "
            f"first missing time: {missing_times[0].isoformat()}."
        )

    visible_fraction = np.asarray(
        [visible_by_time[timestamp] for timestamp in expected_times],
        dtype=float,
    ).reshape(WINDOW_DAYS, HOURS_PER_DAY)
    dates = tuple(window.start + timedelta(days=index) for index in range(WINDOW_DAYS))
    return WindowResult(window, dates, visible_fraction), observer


def write_results_csv(
    path: Path,
    results: tuple[WindowResult, WindowResult],
    sun_exclusion: float,
) -> Path:
    """Write all 672 hourly values used in the comparison plot."""
    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() != ".csv":
        resolved = resolved.with_suffix(".csv")
    resolved.parent.mkdir(parents=True, exist_ok=True)

    with resolved.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "window",
                "window_start_utc",
                "window_stop_exclusive_utc",
                "date_utc",
                "hour_index_1_to_24",
                "utc_hour_0_to_23",
                "sun_exclusion_deg",
                "visible_sky_fraction",
                "visible_sky_percent",
            ]
        )
        for result in results:
            for day_index, calendar_date in enumerate(result.dates):
                for hour_index in range(HOURS_PER_DAY):
                    fraction = float(result.visible_fraction[day_index, hour_index])
                    writer.writerow(
                        [
                            result.window.name,
                            result.window.start.isoformat(),
                            result.window.stop.isoformat(),
                            calendar_date.isoformat(),
                            hour_index + 1,
                            hour_index,
                            f"{sun_exclusion:g}",
                            f"{fraction:.12f}",
                            f"{100.0 * fraction:.9f}",
                        ]
                    )
    return resolved


def write_comparison_plot(
    path: Path,
    first: WindowResult,
    second: WindowResult,
    earth_clearance: float,
    moon_clearance: float,
    sun_exclusion: float,
    observer: str,
) -> Path:
    """Plot 28 daily profiles and two emphasized mean profiles."""
    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() != ".png":
        resolved = resolved.with_suffix(".png")
    resolved.parent.mkdir(parents=True, exist_ok=True)

    hours = np.arange(1, HOURS_PER_DAY + 1)
    first_color = "#176B87"
    second_color = "#D97706"
    figure, axis = plt.subplots(figsize=(13.0, 7.2), constrained_layout=True)

    for daily_percent in first.visible_percent:
        axis.plot(
            hours,
            daily_percent,
            color=first_color,
            linewidth=0.9,
            alpha=0.24,
            zorder=1,
        )
    for daily_percent in second.visible_percent:
        axis.plot(
            hours,
            daily_percent,
            color=second_color,
            linewidth=0.9,
            alpha=0.24,
            zorder=1,
        )

    axis.plot(
        hours,
        first.mean_percent_by_hour,
        color=first_color,
        linewidth=3.4,
        zorder=4,
    )
    axis.plot(
        hours,
        second.mean_percent_by_hour,
        color=second_color,
        linewidth=3.4,
        zorder=4,
    )

    legend_handles = [
        Line2D(
            [0], [0], color=first_color, linewidth=1.0, alpha=0.35,
            label=f"Window 1 daily profiles: {first.window.display_range}",
        ),
        Line2D(
            [0], [0], color=second_color, linewidth=1.0, alpha=0.35,
            label=f"Window 2 daily profiles: {second.window.display_range}",
        ),
        Line2D(
            [0], [0], color=first_color, linewidth=3.4,
            label="Window 1 hourly mean",
        ),
        Line2D(
            [0], [0], color=second_color, linewidth=3.4,
            label="Window 2 hourly mean",
        ),
    ]
    axis.legend(handles=legend_handles, loc="best", frameon=True, fontsize=9)
    axis.set_xlim(1, HOURS_PER_DAY)
    axis.set_xticks(hours)
    axis.set_xlabel("Hour of day (UTC): 1 = 00:00, 24 = 23:00")
    axis.set_ylabel("Visible sky (%)")
    figure.suptitle(
        "GOES-18 daily visible-sky variation across two 14-day windows",
        x=0.065,
        ha="left",
        fontsize=15,
        weight="bold",
    )
    axis.set_title(
        "Thin lines show individual UTC days; thick lines show each window's hourly mean\n"
        f"Earth limb + {earth_clearance:g} deg; "
        f"Moon center + {moon_clearance:g} deg; Sun center + "
        f"{sun_exclusion:g} deg | {observer}",
        loc="left",
        fontsize=9.5,
        color="#4B5563",
    )
    axis.grid(True, color="#D7DEE5", linewidth=0.8, alpha=0.85)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(resolved, dpi=180, facecolor="white")
    plt.close(figure)
    return resolved


def print_summary(result: WindowResult) -> None:
    """Print a concise summary for one selected window."""
    percent = result.visible_percent
    print(
        f"{result.window.name} ({result.window.display_range}): "
        f"min={np.min(percent):.6f}%, mean={np.mean(percent):.6f}%, "
        f"max={np.max(percent):.6f}%"
    )


def main() -> int:
    """Collect settings, run both JPL windows, and write CSV/PNG outputs."""
    args = parse_arguments()
    try:
        windows, sun_exclusion = collect_settings(args)
        print(f"\nSun exclusion angle: {sun_exclusion:g} degrees from Sun center")
        for window in windows:
            print(f"{window.name}: {window.display_range} (14 days)")

        first_result, first_observer = calculate_window(
            windows[0],
            args.earth_clearance,
            args.moon_clearance,
            sun_exclusion,
        )
        second_result, second_observer = calculate_window(
            windows[1],
            args.earth_clearance,
            args.moon_clearance,
            sun_exclusion,
        )
        if first_observer != second_observer:
            raise RuntimeError("The two Horizons runs returned different observers.")

        csv_path = write_results_csv(
            args.csv_output,
            (first_result, second_result),
            sun_exclusion,
        )
        plot_path = write_comparison_plot(
            args.output,
            first_result,
            second_result,
            args.earth_clearance,
            args.moon_clearance,
            sun_exclusion,
            first_observer,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("\nTwo-window daily-variation analysis complete")
    print_summary(first_result)
    print_summary(second_result)
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote PNG: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
