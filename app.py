#!/usr/bin/env python3
"""Streamlit interface for the GOES-18 visible-sky project."""

from __future__ import annotations

import csv
from datetime import date, time, timedelta
import io
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
README_PATH = PROJECT_ROOT / "README.md"
HORIZONS_SCRIPT = (
    PROJECT_ROOT / "horizons_api_algorithm" / "goes18_visible_sky.py"
)
TLE_SCRIPT = (
    PROJECT_ROOT / "tle_sgp4_algorithm" / "goes18_visible_sky_tle.py"
)
COMPARISON_SCRIPT = (
    PROJECT_ROOT
    / "algorithm_comparison"
    / "goes18_algorithm_comparison.py"
)
DAILY_VARIATION_SCRIPT = (
    PROJECT_ROOT
    / "twoweektrial_comparison"
    / "goes18_twoweektrialcomparison.py"
)
SKY_PROJECTION_SCRIPT = (
    PROJECT_ROOT
    / "sky_projection"
    / "goes18_visible_sky_projection.py"
)
TLE_FILE = (
    PROJECT_ROOT / "tle_sgp4_algorithm" / "goes18_2026-08-27.tle"
)

TLE_EPOCH = date(2026, 8, 27)
RECOMMENDED_TLE_WINDOW_DAYS = 14
MAX_HORIZONS_SPAN_DAYS = 34
DEFAULT_DATE_RANGE = (date(2026, 8, 26), date(2026, 8, 28))
DEFAULT_DAILY_WINDOW_1 = (date(2026, 1, 1), date(2026, 1, 15))
DEFAULT_DAILY_WINDOW_2 = (date(2026, 8, 1), date(2026, 8, 15))
DEFAULT_RANDOM_POOL = (date(2026, 1, 1), date(2027, 1, 1))
DAILY_WINDOW_DAYS = 14
DEFAULT_PROJECTION_SCENARIOS = (
    (date(2026, 1, 3), time(0, 0), 30),
    (date(2026, 1, 10), time(6, 0), 45),
    (date(2026, 1, 18), time(12, 0), 30),
    (date(2026, 1, 26), time(18, 0), 45),
)


st.set_page_config(
    page_title="GOES-18 Visible-Sky Calculator",
    page_icon="🛰️",
    layout="wide",
)


def normalize_date_range(selected_dates: object) -> tuple[date, date] | None:
    """Validate a Streamlit date-range widget value."""
    if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
        st.error("Select both a start date and an end date.")
        return None

    start, stop = selected_dates
    if not isinstance(start, date) or not isinstance(stop, date):
        st.error("The selected dates are invalid.")
        return None
    if stop <= start:
        st.error("The end date must be later than the start date.")
        return None
    return start, stop


def validate_exact_daily_window(
    start: date,
    stop: date,
    label: str,
) -> bool:
    """Require an exact 14-day half-open range for daily-profile analysis."""
    span_days = (stop - start).days
    if span_days == DAILY_WINDOW_DAYS:
        return True
    st.error(
        f"{label} is {span_days} days long. Its end date must be exactly "
        f"{DAILY_WINDOW_DAYS} days after its start date. The end date is an "
        "exclusive boundary."
    )
    return False


def daily_windows_overlap(
    first: tuple[date, date],
    second: tuple[date, date],
) -> bool:
    """Return whether two half-open manual daily-analysis windows overlap."""
    return first[0] < second[1] and second[0] < first[1]


def validate_random_pool(start: date, stop: date) -> bool:
    """Require enough dates for two non-overlapping random 14-day windows."""
    if stop - start >= timedelta(days=2 * DAILY_WINDOW_DAYS):
        return True
    st.error(
        "The random-selection pool must span at least 28 days so the script "
        "can select two non-overlapping 14-day windows."
    )
    return False


def validate_horizons_span(start: date, stop: date) -> bool:
    """Keep five-minute Horizons requests below the API row limit."""
    span_days = (stop - start).days
    if span_days > MAX_HORIZONS_SPAN_DAYS:
        st.error(
            "Select a date range of 34 days or less. The algorithms request "
            "five-minute ephemeris data from JPL Horizons."
        )
        return False
    return True


def show_tle_epoch_warning(start: date, stop: date, comparison: bool) -> None:
    """Warn when a requested range is far from the saved TLE epoch."""
    maximum_offset = max(
        abs((start - TLE_EPOCH).days),
        abs((stop - TLE_EPOCH).days),
    )
    if maximum_offset <= RECOMMENDED_TLE_WINDOW_DAYS:
        return

    consequence = (
        "the comparison graph may show large, structured differences that "
        "are caused by propagation far from the TLE epoch"
        if comparison
        else "the propagated satellite position and visible-sky results may be inaccurate"
    )
    st.warning(
        f"The selected range extends {maximum_offset} days from the "
        f"August 27, 2026 TLE epoch. Because this exceeds the recommended "
        f"±{RECOMMENDED_TLE_WINDOW_DAYS}-day window, {consequence}."
    )


def required_files_exist(paths: list[Path]) -> bool:
    """Show friendly errors for files missing from the deployed branch."""
    missing = [path.relative_to(PROJECT_ROOT) for path in paths if not path.is_file()]
    if not missing:
        return True

    st.error(
        "The deployed Git branch is missing: "
        + ", ".join(f"`{path}`" for path in missing)
    )
    return False


def run_command(command: list[str], timeout_seconds: int = 1_200) -> tuple[int, str]:
    """Run a fixed command and return its status and terminal text."""
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return 124, (
            stdout
            + ("\n" if stdout and stderr else "")
            + stderr
            + f"\nCalculation exceeded the {timeout_seconds}-second website limit."
        ).strip()
    except OSError as exc:
        return 1, f"Could not start the calculation: {exc}"

    output_parts = [part.strip() for part in (completed.stdout, completed.stderr) if part.strip()]
    return completed.returncode, "\n\n".join(output_parts)


def read_file_bytes(path: Path) -> bytes | None:
    """Read an output before its temporary directory is removed."""
    return path.read_bytes() if path.is_file() else None


def combine_projection_images(
    aitoff_bytes: bytes | None,
    mollweide_bytes: bytes | None,
) -> bytes | None:
    """Stack the two projection PNGs into one downloadable image."""
    if not aitoff_bytes or not mollweide_bytes:
        return None

    with Image.open(io.BytesIO(aitoff_bytes)) as aitoff_source:
        aitoff = aitoff_source.convert("RGB")
    with Image.open(io.BytesIO(mollweide_bytes)) as mollweide_source:
        mollweide = mollweide_source.convert("RGB")

    gap_pixels = 40
    output_width = max(aitoff.width, mollweide.width)
    output_height = aitoff.height + gap_pixels + mollweide.height
    combined = Image.new("RGB", (output_width, output_height), "white")
    combined.paste(aitoff, ((output_width - aitoff.width) // 2, 0))
    combined.paste(
        mollweide,
        (
            (output_width - mollweide.width) // 2,
            aitoff.height + gap_pixels,
        ),
    )

    output = io.BytesIO()
    combined.save(output, format="PNG", optimize=True)
    return output.getvalue()


def visible_sky_statistics(csv_bytes: bytes | None) -> dict[str, float | int]:
    """Calculate display statistics from one generated visible-sky CSV."""
    if not csv_bytes:
        return {}

    values: list[float] = []
    text = csv_bytes.decode("utf-8-sig")
    for row in csv.DictReader(io.StringIO(text)):
        raw_value = row.get("visible_sky_percent")
        if raw_value not in (None, ""):
            values.append(float(raw_value))

    if not values:
        return {}
    return {
        "samples": len(values),
        "minimum": min(values),
        "mean": sum(values) / len(values),
        "maximum": max(values),
    }


def save_result(
    state_key: str,
    return_code: int,
    log: str,
    images: dict[str, tuple[bytes | None, str]],
    downloads: dict[str, tuple[bytes | None, str, str]],
    statistics: dict[str, float | int] | None = None,
) -> None:
    """Keep generated bytes available across Streamlit reruns."""
    st.session_state[state_key] = {
        "return_code": return_code,
        "log": log,
        "images": images,
        "downloads": downloads,
        "statistics": statistics or {},
    }


def render_result(state_key: str) -> None:
    """Render saved calculation results directly below their form."""
    result = st.session_state.get(state_key)
    if not result:
        return

    with st.expander("Calculation log", expanded=result["return_code"] != 0):
        st.code(result["log"] or "No terminal output was produced.", language=None)

    if result["return_code"] != 0:
        st.error("The calculation failed. Review the calculation log above.")
        return

    st.success("Calculation complete")
    statistics = result["statistics"]
    if statistics:
        columns = st.columns(4)
        columns[0].metric("Samples", f"{int(statistics['samples']):,}")
        columns[1].metric("Minimum visible sky", f"{statistics['minimum']:.6f}%")
        columns[2].metric("Mean visible sky", f"{statistics['mean']:.6f}%")
        columns[3].metric("Maximum visible sky", f"{statistics['maximum']:.6f}%")

    for image_bytes, caption in result["images"].values():
        if image_bytes:
            st.image(image_bytes, caption=caption, use_container_width=True)

    available_downloads = [
        item for item in result["downloads"].values() if item[0] is not None
    ]
    if available_downloads:
        st.markdown("#### Downloads")
        columns = st.columns(min(3, len(available_downloads)))
        for index, (file_bytes, filename, mime_type) in enumerate(available_downloads):
            columns[index % len(columns)].download_button(
                label=f"Download {filename}",
                data=file_bytes,
                file_name=filename,
                mime=mime_type,
                key=f"{state_key}-{filename}",
            )


def run_horizons(start: date, stop: date, sun_angle: int) -> None:
    """Run the JPL Horizons method and retain its outputs."""
    if not required_files_exist([HORIZONS_SCRIPT]):
        return

    with tempfile.TemporaryDirectory(prefix="goes18-horizons-") as directory:
        prefix = Path(directory) / "goes18_visible_sky"
        command = [
            sys.executable,
            str(HORIZONS_SCRIPT),
            "--start",
            start.isoformat(),
            "--stop",
            stop.isoformat(),
            "--sun-exclusion",
            str(sun_angle),
            "--output-prefix",
            str(prefix),
        ]
        with st.spinner("Downloading JPL ephemerides and calculating visible sky..."):
            return_code, log = run_command(command)

        csv_path = prefix.with_suffix(".csv")
        csv_bytes = read_file_bytes(csv_path)
        save_result(
            "horizons_result",
            return_code,
            log,
            images={
                "main": (
                    read_file_bytes(prefix.with_suffix(".png")),
                    "JPL Horizons visible-sky coverage with a highlighted two-day detail section.",
                ),
                "monthly": (
                    read_file_bytes(prefix.parent / f"{prefix.name}_monthly_histogram.png"),
                    "JPL Horizons monthly mean visible-sky percentage.",
                ),
                "sun_sweep": (
                    read_file_bytes(prefix.parent / f"{prefix.name}_sun_exclusion_exposures.png"),
                    "JPL Horizons visible sky versus Sun exclusion angle for 10-minute, 1-hour, and 24-hour exposures.",
                ),
            },
            downloads={
                "csv": (csv_bytes, "goes18_visible_sky.csv", "text/csv"),
            },
            statistics=visible_sky_statistics(csv_bytes),
        )


def run_tle(start: date, stop: date, sun_angle: int) -> None:
    """Run the saved-TLE/SGP4 method and retain its outputs."""
    if not required_files_exist([TLE_SCRIPT, HORIZONS_SCRIPT, TLE_FILE]):
        return

    with tempfile.TemporaryDirectory(prefix="goes18-tle-") as directory:
        prefix = Path(directory) / "goes18_visible_sky_tle"
        command = [
            sys.executable,
            str(TLE_SCRIPT),
            "--start",
            start.isoformat(),
            "--stop",
            stop.isoformat(),
            "--sun-exclusion",
            str(sun_angle),
            "--tle-file",
            str(TLE_FILE),
            "--output-prefix",
            str(prefix),
        ]
        with st.spinner("Propagating the August 27, 2026 TLE with SGP4..."):
            return_code, log = run_command(command)

        csv_path = prefix.with_suffix(".csv")
        csv_bytes = read_file_bytes(csv_path)
        save_result(
            "tle_result",
            return_code,
            log,
            images={
                "main": (
                    read_file_bytes(prefix.with_suffix(".png")),
                    "TLE/SGP4 visible-sky coverage with a highlighted two-day detail section.",
                ),
                "monthly": (
                    read_file_bytes(prefix.parent / f"{prefix.name}_monthly_histogram.png"),
                    "TLE/SGP4 monthly mean visible-sky percentage.",
                ),
                "sun_sweep": (
                    read_file_bytes(prefix.parent / f"{prefix.name}_sun_exclusion_exposures.png"),
                    "TLE/SGP4 visible sky versus Sun exclusion angle for 10-minute, 1-hour, and 24-hour exposures.",
                ),
            },
            downloads={
                "csv": (csv_bytes, "goes18_visible_sky_tle.csv", "text/csv"),
                "ephemeris": (
                    read_file_bytes(prefix.parent / f"{prefix.name}_ephemeris.csv"),
                    "goes18_visible_sky_tle_ephemeris.csv",
                    "text/csv",
                ),
                "tle": (
                    read_file_bytes(prefix.parent / f"{prefix.name}_used.tle"),
                    "goes18_visible_sky_tle_used.tle",
                    "text/plain",
                ),
            },
            statistics=visible_sky_statistics(csv_bytes),
        )


def run_comparison(start: date, stop: date, sun_angle: int) -> None:
    """Run both algorithms and retain their comparison outputs."""
    if not required_files_exist(
        [COMPARISON_SCRIPT, HORIZONS_SCRIPT, TLE_SCRIPT, TLE_FILE]
    ):
        return

    with tempfile.TemporaryDirectory(prefix="goes18-comparison-") as directory:
        output_directory = Path(directory)
        comparison_png = output_directory / "goes18_algorithm_comparison.png"
        horizons_csv = output_directory / "goes18_visible_sky.csv"
        tle_csv = output_directory / "goes18_visible_sky_tle.csv"
        command = [
            sys.executable,
            str(COMPARISON_SCRIPT),
            "--start",
            start.isoformat(),
            "--stop",
            stop.isoformat(),
            "--sun-exclusion",
            str(sun_angle),
            "--horizons-csv",
            str(horizons_csv),
            "--tle-csv",
            str(tle_csv),
            "--tle-file",
            str(TLE_FILE),
            "--output",
            str(comparison_png),
        ]
        with st.spinner("Running both methods and generating the comparison..."):
            return_code, log = run_command(command)

        save_result(
            "comparison_result",
            return_code,
            log,
            images={
                "comparison": (
                    read_file_bytes(comparison_png),
                    "GOES-18 JPL Horizons versus TLE/SGP4 angular separation in arcseconds, with an empirical separation histogram.",
                ),
            },
            downloads={
                "horizons_csv": (
                    read_file_bytes(horizons_csv),
                    "comparison_horizons.csv",
                    "text/csv",
                ),
                "tle_csv": (
                    read_file_bytes(tle_csv),
                    "comparison_tle.csv",
                    "text/csv",
                ),
                "png": (
                    read_file_bytes(comparison_png),
                    "goes18_algorithm_comparison.png",
                    "image/png",
                ),
            },
        )


def run_daily_variation(
    selection: str,
    sun_angle: int,
    first_window: tuple[date, date] | None = None,
    second_window: tuple[date, date] | None = None,
    random_pool: tuple[date, date] | None = None,
) -> None:
    """Run the two-window JPL daily-variation analysis and retain its outputs."""
    if not required_files_exist([DAILY_VARIATION_SCRIPT, HORIZONS_SCRIPT]):
        return

    with tempfile.TemporaryDirectory(prefix="goes18-daily-variation-") as directory:
        output_directory = Path(directory)
        output_png = output_directory / "two_window_daily_variation.png"
        output_csv = output_directory / "two_window_daily_variation.csv"
        command = [
            sys.executable,
            str(DAILY_VARIATION_SCRIPT),
            "--selection",
            selection,
            "--sun-exclusion",
            str(sun_angle),
            "--output",
            str(output_png),
            "--csv-output",
            str(output_csv),
        ]
        if selection == "manual":
            if first_window is None or second_window is None:
                st.error("Both manual 14-day windows are required.")
                return
            command.extend(
                [
                    "--window-1-start",
                    first_window[0].isoformat(),
                    "--window-1-stop",
                    first_window[1].isoformat(),
                    "--window-2-start",
                    second_window[0].isoformat(),
                    "--window-2-stop",
                    second_window[1].isoformat(),
                ]
            )
        else:
            if random_pool is None:
                st.error("A random-selection date pool is required.")
                return
            command.extend(
                [
                    "--random-earliest",
                    random_pool[0].isoformat(),
                    "--random-latest",
                    random_pool[1].isoformat(),
                ]
            )

        with st.spinner(
            "Downloading hourly JPL ephemerides for both 14-day windows..."
        ):
            return_code, log = run_command(command)

        csv_bytes = read_file_bytes(output_csv)
        save_result(
            "daily_variation_result",
            return_code,
            log,
            images={
                "daily_variation": (
                    read_file_bytes(output_png),
                    "Hourly daily profiles for two 14-day JPL Horizons windows; thick lines show each window's mean profile.",
                ),
            },
            downloads={
                "csv": (
                    csv_bytes,
                    "two_window_daily_variation.csv",
                    "text/csv",
                ),
                "png": (
                    read_file_bytes(output_png),
                    "two_window_daily_variation.png",
                    "image/png",
                ),
            },
            statistics=visible_sky_statistics(csv_bytes),
        )


def run_sky_projections(
    scenarios: list[tuple[date, time, int]],
) -> None:
    """Generate Aitoff and Mollweide maps and combine them into one PNG."""
    if not required_files_exist([SKY_PROJECTION_SCRIPT, HORIZONS_SCRIPT]):
        return

    with tempfile.TemporaryDirectory(prefix="goes18-sky-projection-") as directory:
        prefix = Path(directory) / "goes18_visible_sky_projection_examples"
        command = [
            sys.executable,
            str(SKY_PROJECTION_SCRIPT),
            "--projection",
            "both",
            "--output-prefix",
            str(prefix),
        ]
        for scenario_date, scenario_time, sun_angle in scenarios:
            command.extend(
                [
                    "--scenario",
                    (
                        f"{scenario_date.isoformat()}T"
                        f"{scenario_time.strftime('%H:%M')},{sun_angle}"
                    ),
                ]
            )

        with st.spinner(
            "Downloading Earth, Moon, and Sun vectors and generating both projections..."
        ):
            return_code, log = run_command(command)

        aitoff_path = prefix.parent / f"{prefix.name}_aitoff.png"
        mollweide_path = prefix.parent / f"{prefix.name}_mollweide.png"
        csv_path = prefix.with_suffix(".csv")
        aitoff_bytes = read_file_bytes(aitoff_path)
        mollweide_bytes = read_file_bytes(mollweide_path)
        combined_bytes = (
            combine_projection_images(aitoff_bytes, mollweide_bytes)
            if return_code == 0
            else None
        )

        save_result(
            "projection_result",
            return_code,
            log,
            images={
                "combined": (
                    combined_bytes,
                    (
                        "GOES-18 observer-centered all-sky maps. The Aitoff "
                        "examples are shown first and the equal-area "
                        "Mollweide examples are shown below."
                    ),
                ),
            },
            downloads={
                "combined": (
                    combined_bytes,
                    "goes18_aitoff_mollweide_projection.png",
                    "image/png",
                ),
                "aitoff": (
                    aitoff_bytes,
                    "goes18_visible_sky_projection_aitoff.png",
                    "image/png",
                ),
                "mollweide": (
                    mollweide_bytes,
                    "goes18_visible_sky_projection_mollweide.png",
                    "image/png",
                ),
                "csv": (
                    read_file_bytes(csv_path),
                    "goes18_visible_sky_projection_scenarios.csv",
                    "text/csv",
                ),
            },
        )


st.title("GOES-18 Visible-Sky Calculator")
st.caption(
    "Interactive Earth, Moon, and Sun avoidance analysis for GOES-18 near 137°W"
)


with st.expander("About", expanded=True):
    st.markdown(
        """
        This project calculates the instantaneous fraction of the celestial sky
        available to an observer on GOES-18 after applying Earth, Moon, and Sun
        exclusion regions. It supports a NASA/JPL Horizons method, a local
        TLE/SGP4 method, and a direct comparison of the GOES-18 ephemeris
        directions derived from each method.

        The Earth clearance is measured 20° beyond the Earth limb, the Moon
        clearance is 20° from its center by default, and the selectable Sun
        exclusion radius is 30° or 45°.
        """
    )


with st.expander("README.md"):
    if README_PATH.is_file():
        st.markdown(README_PATH.read_text(encoding="utf-8"))
    else:
        st.info(
            "`README.md` was not found on the deployed branch. Add it to the "
            "repository root to display it here."
        )


method_tabs = st.tabs(
    [
        "JPL Horizons",
        "TLE/SGP4",
        "Algorithm comparison",
        "Two-week daily variation",
        "Aitoff–Mollweide sky maps",
    ]
)


with method_tabs[0]:
    st.markdown(
        "Downloads geometric Earth, Moon, and Sun vectors from NASA/JPL "
        "Horizons with GOES-18 as the observing center."
    )
    with st.form("horizons_form"):
        horizons_dates = st.date_input(
            "UTC date range",
            value=DEFAULT_DATE_RANGE,
            format="YYYY-MM-DD",
            key="horizons_dates",
        )
        horizons_sun_angle = st.radio(
            "Sun exclusion angle",
            options=(30, 45),
            index=1,
            horizontal=True,
            format_func=lambda angle: f"{angle}°",
            key="horizons_sun_angle",
        )
        horizons_submit = st.form_submit_button(
            "Run JPL Horizons calculation",
            type="primary",
        )

    if horizons_submit:
        selected = normalize_date_range(horizons_dates)
        if selected and validate_horizons_span(*selected):
            run_horizons(*selected, horizons_sun_angle)
    render_result("horizons_result")


with method_tabs[1]:
    st.info(
        "This website uses the saved GOES-18 TLE with an epoch of "
        "August 27, 2026. SGP4 is most reliable when propagated close to "
        "the TLE epoch."
    )
    with st.form("tle_form"):
        tle_dates = st.date_input(
            "UTC date range",
            value=DEFAULT_DATE_RANGE,
            format="YYYY-MM-DD",
            key="tle_dates",
        )
        tle_sun_angle = st.radio(
            "Sun exclusion angle",
            options=(30, 45),
            index=1,
            horizontal=True,
            format_func=lambda angle: f"{angle}°",
            key="tle_sun_angle",
        )
        tle_submit = st.form_submit_button(
            "Run TLE/SGP4 calculation",
            type="primary",
        )

    if tle_submit:
        selected = normalize_date_range(tle_dates)
        if selected:
            show_tle_epoch_warning(*selected, comparison=False)
            run_tle(*selected, tle_sun_angle)
    render_result("tle_result")


with method_tabs[2]:
    st.warning(
        "The comparison uses the saved GOES-18 TLE with an epoch of "
        "August 27, 2026. Selecting a date window far outside this epoch "
        "can produce inaccurate or misleading comparison graphs because "
        "TLE/SGP4 propagation error grows away from the element-set epoch. "
        "Use dates within approximately ±14 days whenever possible."
    )
    with st.form("comparison_form"):
        comparison_dates = st.date_input(
            "UTC date range",
            value=DEFAULT_DATE_RANGE,
            format="YYYY-MM-DD",
            key="comparison_dates",
        )
        comparison_sun_angle = st.radio(
            "Sun exclusion angle",
            options=(30, 45),
            index=1,
            horizontal=True,
            format_func=lambda angle: f"{angle}°",
            key="comparison_sun_angle",
        )
        comparison_submit = st.form_submit_button(
            "Run algorithm comparison",
            type="primary",
        )

    if comparison_submit:
        selected = normalize_date_range(comparison_dates)
        if selected and validate_horizons_span(*selected):
            show_tle_epoch_warning(*selected, comparison=True)
            run_comparison(*selected, comparison_sun_angle)
    render_result("comparison_result")


with method_tabs[3]:
    st.markdown(
        "Compare the hour-by-hour daily variation in visible sky across two "
        "14-day windows using the JPL Horizons method. Thin lines represent "
        "individual UTC days, while thick lines show each window's hourly mean."
    )
    st.info(
        "Every manually entered window must be exactly 14 days. The website "
        "and the calculation script both reject a window whose end date is not "
        "exactly 14 days after its start date. The end date is an exclusive boundary."
    )

    daily_selection_label = st.radio(
        "Date-window selection",
        options=("Manual 14-day windows", "Random 14-day windows"),
        horizontal=True,
        key="daily_variation_selection",
    )
    daily_selection = (
        "manual" if daily_selection_label.startswith("Manual") else "random"
    )

    with st.form("daily_variation_form"):
        if daily_selection == "manual":
            daily_window_1_dates = st.date_input(
                "Window 1 UTC date range",
                value=DEFAULT_DAILY_WINDOW_1,
                format="YYYY-MM-DD",
                key="daily_window_1_dates",
            )
            daily_window_2_dates = st.date_input(
                "Window 2 UTC date range",
                value=DEFAULT_DAILY_WINDOW_2,
                format="YYYY-MM-DD",
                key="daily_window_2_dates",
            )
        else:
            daily_random_pool = st.date_input(
                "UTC date pool for random selection",
                value=DEFAULT_RANDOM_POOL,
                format="YYYY-MM-DD",
                key="daily_random_pool",
                help=(
                    "The script randomly chooses two non-overlapping 14-day "
                    "windows within this range."
                ),
            )

        daily_sun_angle = st.radio(
            "Sun exclusion angle",
            options=(30, 45),
            index=1,
            horizontal=True,
            format_func=lambda angle: f"{angle}°",
            key="daily_variation_sun_angle",
        )
        daily_submit = st.form_submit_button(
            "Run two-week daily variation",
            type="primary",
        )

    if daily_submit:
        if daily_selection == "manual":
            first_window = normalize_date_range(daily_window_1_dates)
            second_window = normalize_date_range(daily_window_2_dates)
            if first_window and second_window:
                first_is_valid = validate_exact_daily_window(
                    *first_window,
                    "Window 1",
                )
                second_is_valid = validate_exact_daily_window(
                    *second_window,
                    "Window 2",
                )
                if first_is_valid and second_is_valid:
                    if daily_windows_overlap(first_window, second_window):
                        st.error("The two manual 14-day windows must not overlap.")
                    else:
                        run_daily_variation(
                            "manual",
                            daily_sun_angle,
                            first_window=first_window,
                            second_window=second_window,
                        )
        else:
            random_pool = normalize_date_range(daily_random_pool)
            if random_pool and validate_random_pool(*random_pool):
                run_daily_variation(
                    "random",
                    daily_sun_angle,
                    random_pool=random_pool,
                )
    render_result("daily_variation_result")


with method_tabs[4]:
    st.markdown(
        "Generate paired **Aitoff and Mollweide all-sky maps** for four "
        "user-selected UTC instants. Each map uses JPL Horizons vectors with "
        "GOES-18 as the observing center. The two map types are combined into "
        "one PNG, while the individual projections and plotted coordinates "
        "remain available as downloads."
    )

    with st.expander("How to read the sky maps", expanded=True):
        st.markdown(
            """
            - **Unshaded sky** is available for observation. Blue, gray, and
              orange show the Earth, Moon, and Sun exclusion regions,
              respectively. Overlapping colors indicate that two or more
              avoidance regions cover the same directions; the overlap is
              counted only once in the visible-sky calculation.
            - The **dark-blue inner disk** is the Earth's apparent physical
              disk. Its larger translucent blue region includes the required
              20° clearance beyond the Earth limb. The Moon and Sun are less
              than one degree across, so their plotted center markers are
              schematic; their much larger avoidance regions are drawn to
              their calculated angular radii.
            - The map is drawn from the **satellite's point of view**. Imagine
              GOES-18 at the center of a transparent celestial sphere, looking
              outward in every possible direction. GOES-18 is therefore not
              drawn as an object on the map—it is the observer at the origin.
              Each point represents a line of sight from the satellite, and
              the Earth, Moon, and Sun markers show where those bodies appear
              in the satellite-centered sky.
            - Changing the UTC instant changes GOES-18's inertial position and
              the apparent Moon and Sun directions, even though the satellite
              remains near its geostationary longitude. This moves and changes
              the overlap of the exclusion regions.
            - Right ascension is labeled in hours and **increases toward the
              left**, following astronomical sky-map convention. The left and
              right edges are the same celestial seam, so a cap split across
              both edges is one continuous region on the sphere.
            - The **Aitoff projection** provides a familiar whole-sky view with
              moderate shape distortion. The **Mollweide projection is
              equal-area**, so the relative shaded areas more directly
              represent excluded solid angle. Both flatten a sphere, so cap
              outlines can appear stretched near the map edges.
            """
        )

    st.info(
        "Four scenarios require twelve Horizons vector requests: Earth, Moon, "
        "and Sun for each instant. Generation can therefore take about one or "
        "two minutes."
    )

    with st.form("sky_projection_form"):
        projection_scenarios: list[tuple[date, time, int]] = []
        for index, (default_date, default_time, default_angle) in enumerate(
            DEFAULT_PROJECTION_SCENARIOS,
            start=1,
        ):
            st.markdown(f"#### Scenario {index}")
            scenario_columns = st.columns((1.2, 1.0, 0.9))
            scenario_date = scenario_columns[0].date_input(
                "UTC date",
                value=default_date,
                format="YYYY-MM-DD",
                key=f"projection_date_{index}",
            )
            scenario_time = scenario_columns[1].time_input(
                "UTC time",
                value=default_time,
                step=3_600,
                key=f"projection_time_{index}",
            )
            scenario_angle = scenario_columns[2].selectbox(
                "Sun exclusion angle",
                options=(30, 45),
                index=0 if default_angle == 30 else 1,
                format_func=lambda angle: f"{angle}°",
                key=f"projection_sun_angle_{index}",
            )
            projection_scenarios.append(
                (scenario_date, scenario_time, scenario_angle)
            )

        projection_submit = st.form_submit_button(
            "Generate combined Aitoff–Mollweide projection",
            type="primary",
        )

    if projection_submit:
        run_sky_projections(projection_scenarios)
    render_result("projection_result")
