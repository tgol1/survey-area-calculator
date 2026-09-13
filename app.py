from datetime import date, timedelta
from pathlib import Path
import subprocess
import sys
import tempfile

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
COMPARISON_SCRIPT = (
    PROJECT_ROOT
    / "algorithm_comparison"
    / "compare_algorithms.py"
)


st.set_page_config(
    page_title="GOES-18 Visible-Sky Calculator",
    page_icon="🛰️",
    layout="wide",
)

st.title("GOES-18 Visible-Sky Calculator")

st.write(
    "Compare visible-sky calculations produced using NASA/JPL "
    "Horizons and TLE/SGP4 ephemerides."
)

with st.form("visibility_settings"):
    date_range = st.date_input(
        "UTC date range",
        value=(
            date.today(),
            date.today() + timedelta(days=1),
        ),
        format="YYYY-MM-DD",
    )

    sun_exclusion = st.radio(
        "Sun exclusion angle",
        options=(30, 45),
        horizontal=True,
        format_func=lambda angle: f"{angle}°",
    )

    submitted = st.form_submit_button(
        "Run calculation",
        type="primary",
    )


if submitted:
    if len(date_range) != 2:
        st.error("Select both a start date and an end date.")
        st.stop()

    start_date, end_date = date_range

    if end_date <= start_date:
        st.error("The end date must be later than the start date.")
        st.stop()

    if (end_date - start_date).days > 34:
        st.error(
            "Select a range of 34 days or less for five-minute "
            "Horizons sampling."
        )
        st.stop()

    with tempfile.TemporaryDirectory() as temporary_directory:
        output_directory = Path(temporary_directory)

        comparison_png = output_directory / "comparison.png"
        horizons_csv = output_directory / "horizons.csv"
        tle_csv = output_directory / "tle.csv"

        command = [
            sys.executable,
            str(COMPARISON_SCRIPT),
            "--start",
            start_date.isoformat(),
            "--stop",
            end_date.isoformat(),
            "--sun-exclusion",
            str(sun_exclusion),
            "--horizons-csv",
            str(horizons_csv),
            "--tle-csv",
            str(tle_csv),
            "--output",
            str(comparison_png),
        ]

        with st.spinner("Running both algorithms..."):
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=900,
            )

        calculation_log = result.stdout

        if result.stderr:
            calculation_log += "\n" + result.stderr

        with st.expander("Calculation log"):
            st.code(calculation_log, language=None)

        if result.returncode != 0:
            st.error("The calculation failed. Review the log above.")
            st.stop()

        st.success("Calculation complete")

        st.image(
            comparison_png.read_bytes(),
            caption="Horizons and TLE/SGP4 comparison",
            use_container_width=True,
        )

        first_column, second_column = st.columns(2)

        with first_column:
            st.download_button(
                "Download Horizons CSV",
                data=horizons_csv.read_bytes(),
                file_name="goes18_horizons.csv",
                mime="text/csv",
            )

        with second_column:
            st.download_button(
                "Download TLE/SGP4 CSV",
                data=tle_csv.read_bytes(),
                file_name="goes18_tle.csv",
                mime="text/csv",
            )