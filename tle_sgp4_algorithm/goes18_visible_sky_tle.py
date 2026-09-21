#!/usr/bin/env python3
"""Calculate GOES-18 sky visibility with a TLE/SGP4 spacecraft trajectory.

GOES-18 is propagated locally with the standard SGP4 model and the SGP4 TEME
position is transformed to GCRS with Astropy.  By default, matching geometric
ICRF Moon and Sun vectors are downloaded from JPL Horizons with Earth's center
as the observer.  Subtracting the TLE/SGP4 GOES-18 position produces the
observer-relative vectors used by the visibility calculation.  This keeps the
solar-system model aligned with the Horizons algorithm while preserving the
TLE/SGP4 spacecraft trajectory that the comparison is intended to test.

An Astropy built-in Moon/Sun ephemeris remains available as an explicitly
selected offline option.  When Space-Track credentials are configured, the
program downloads GP_HISTORY records around the requested dates and propagates
every sample with the nearest TLE epoch.  Otherwise it uses CelesTrak's latest
TLE and retains the bundled element set as an offline fallback.

This file reuses the tested spherical-cap geometry, adaptive angular sampling,
CSV writer, and plot writer from ``goes18_visible_sky.py``.  Keep both Python
algorithm folders in the same repository.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.cookiejar import CookieJar
import os
from pathlib import Path
import sys
import warnings
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

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

# When this file is launched directly, Python normally searches only this
# ``tle_sgp4_algorithm`` directory. Add the repository root so the sibling
# ``horizons_api_algorithm`` package can always be imported. This is harmless
# when the script is launched with ``python -m`` because the root is then
# already present.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from horizons_api_algorithm.goes18_visible_sky import (
    ANGULAR_FINE_LIMITS_DEG,
    DEFAULT_EXPOSURE_SKY_POINTS,
    EARTH_ID,
    MOON_ID,
    SUN_ID,
    adaptive_sample_indices,
    compute_visible_sky,
    fetch_vectors,
    prompt_for_date,
    prompt_for_sun_exclusion,
    step_to_minutes,
    subset_results,
    validate_date_range,
    vector_right_ascension_declination,
    write_monthly_average_histogram,
    write_plot,
    write_results_csv,
    write_sun_exclusion_exposure_plot,
)


GOES18_NORAD_ID = 51850
CELESTRAK_TLE_URL = (
    "https://celestrak.org/NORAD/elements/"
    "gp.php?CATNR=51850&FORMAT=TLE"
)
SPACE_TRACK_BASE_URL = "https://www.space-track.org"
SPACE_TRACK_IDENTITY_ENV = "SPACETRACK_IDENTITY"
SPACE_TRACK_PASSWORD_ENV = "SPACETRACK_PASSWORD"
FINE_STEP_MINUTES = 5
DEFAULT_MAX_TLE_AGE_DAYS = 14.0
DEFAULT_FALLBACK_TLE_FILE = Path(__file__).with_name("goes18_2026-08-27.tle")
DEFAULT_TLE_CACHE_DIR = Path(__file__).with_name("tle_cache")
HISTORICAL_QUERY_PADDING_DAYS = 7
HORIZONS_BODY_BATCH_SAMPLES = 9000


@dataclass(frozen=True)
class TLERecord:
    """One validated GOES-18 two-line element set and its provenance."""

    name: str
    line1: str
    line2: str
    source: str

    def satellite(self) -> Satrec:
        """Build the SGP4 record represented by this element set."""
        satellite = Satrec.twoline2rv(self.line1, self.line2, WGS72)
        if satellite.satnum != GOES18_NORAD_ID:
            raise ValueError(
                f"TLE resolved to NORAD {satellite.satnum}, not GOES-18 "
                f"({GOES18_NORAD_ID})."
            )
        return satellite


def validate_tle_lines(line1: str, line2: str, source: str) -> None:
    """Validate that two element lines both describe GOES-18."""
    if not line1.startswith("1 ") or not line2.startswith("2 "):
        raise ValueError(f"Invalid two-line element set in {source}.")

    line1_satellite = line1[2:7].strip()
    line2_satellite = line2[2:7].strip()
    if line1_satellite != line2_satellite:
        raise ValueError(f"The two element lines in {source} describe different objects.")
    if int(line1_satellite) != GOES18_NORAD_ID:
        raise ValueError(
            f"Expected GOES-18 NORAD ID {GOES18_NORAD_ID}, "
            f"but {source} contains {line1_satellite}."
        )


def parse_tle_records(text: str, source: str) -> list[TLERecord]:
    """Extract every valid GOES-18 TLE from a text response or local file."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    records: list[TLERecord] = []
    seen: set[tuple[str, str]] = set()
    for index, line1 in enumerate(lines):
        if not line1.startswith("1 "):
            continue
        if index + 1 >= len(lines):
            raise ValueError(f"The second element line is missing from {source}.")
        line2 = lines[index + 1]
        validate_tle_lines(line1, line2, source)
        key = (line1, line2)
        if key in seen:
            continue
        seen.add(key)
        if index > 0 and not lines[index - 1].startswith(("1 ", "2 ")):
            name = lines[index - 1].removeprefix("0 ").strip()
        else:
            name = "GOES 18"
        records.append(TLERecord(name, line1, line2, source))

    if not records:
        raise ValueError(f"No two-line element set was found in {source}.")
    records.sort(key=lambda record: tle_epoch_jd(record.satellite()))
    return records


def parse_tle_text(text: str, source: str) -> tuple[str, str, str]:
    """Extract the first GOES-18 element set for backward compatibility."""
    record = parse_tle_records(text, source)[0]
    return record.name, record.line1, record.line2


def fetch_current_tle(url: str) -> list[TLERecord]:
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
    return parse_tle_records(text, url)


def read_tle_file(path: Path) -> list[TLERecord]:
    """Read and validate one or more GOES-18 TLEs from a local file."""
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Could not read TLE file {path}: {exc}") from exc
    return parse_tle_records(text, str(path.resolve()))


def tle_epoch_jd(satellite: Satrec) -> float:
    """Return a satellite record's UTC-like TLE epoch as a Julian date."""
    return float(satellite.jdsatepoch + satellite.jdsatepochF)


def save_tle_records(path: Path, records: list[TLERecord]) -> None:
    """Write one or more element sets in conventional three-line format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        f"{record.name}\n{record.line1}\n{record.line2}\n" for record in records
    )
    path.write_text(text, encoding="ascii")


def space_track_credentials() -> tuple[str, str] | None:
    """Read Space-Track credentials from environment variables, if configured."""
    identity = os.environ.get(SPACE_TRACK_IDENTITY_ENV, "").strip()
    password = os.environ.get(SPACE_TRACK_PASSWORD_ENV, "")
    if identity and password:
        return identity, password
    return None


def fetch_historical_tles(
    start: str,
    stop: str,
    cache_dir: Path,
    identity: str,
    password: str,
) -> list[TLERecord]:
    """Fetch and cache Space-Track GP_HISTORY records around a date range."""
    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    stop_date = datetime.strptime(stop, "%Y-%m-%d").date()
    query_start = start_date - timedelta(days=HISTORICAL_QUERY_PADDING_DAYS)
    query_stop = stop_date + timedelta(days=HISTORICAL_QUERY_PADDING_DAYS + 1)
    cache_path = cache_dir / (
        f"goes18_{query_start:%Y%m%d}_{query_stop:%Y%m%d}_gp_history.tle"
    )

    if cache_path.is_file():
        try:
            records = read_tle_file(cache_path)
            print(f"Using cached Space-Track TLE history: {cache_path.resolve()}")
            return records
        except (RuntimeError, ValueError) as exc:
            print(f"Ignoring invalid TLE cache {cache_path}: {exc}")

    login_url = f"{SPACE_TRACK_BASE_URL}/ajaxauth/login"
    query_url = (
        f"{SPACE_TRACK_BASE_URL}/basicspacedata/query/class/gp_history/"
        f"NORAD_CAT_ID/{GOES18_NORAD_ID}/"
        f"EPOCH/{query_start.isoformat()}--{query_stop.isoformat()}/"
        "orderby/EPOCH%20asc/format/tle/emptyresult/show"
    )
    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    login_data = urlencode({"identity": identity, "password": password}).encode()
    headers = {"User-Agent": "GOES18-TLE-sky/2.0"}
    try:
        login_request = Request(login_url, data=login_data, headers=headers)
        with opener.open(login_request, timeout=45) as response:
            login_reply = response.read().decode("utf-8", errors="replace")
        if "failed" in login_reply.lower() or "invalid" in login_reply.lower():
            raise RuntimeError("Space-Track rejected the configured credentials.")

        with opener.open(Request(query_url, headers=headers), timeout=60) as response:
            text = response.read().decode("ascii")
    except (HTTPError, URLError, TimeoutError, UnicodeDecodeError, OSError) as exc:
        raise RuntimeError(f"Could not download Space-Track TLE history: {exc}") from exc

    records = parse_tle_records(text, query_url)
    try:
        save_tle_records(cache_path, records)
        print(f"Cached Space-Track TLE history: {cache_path.resolve()}")
    except OSError as exc:
        print(f"Could not write TLE cache {cache_path}: {exc}")
    return records


def load_tle_records(
    path: Path | None,
    url: str,
    fallback_path: Path,
    start: str,
    stop: str,
    source_mode: str,
    cache_dir: Path,
) -> tuple[list[TLERecord], str]:
    """Select explicit, historical, current, or fallback GOES-18 elements."""
    if path is not None:
        return read_tle_file(path), str(path.resolve())

    if source_mode in {"auto", "space-track"}:
        credentials = space_track_credentials()
        if credentials is None:
            message = (
                "Space-Track credentials are not configured. Set "
                f"{SPACE_TRACK_IDENTITY_ENV} and {SPACE_TRACK_PASSWORD_ENV} "
                "to retrieve historical TLEs near the requested dates."
            )
            if source_mode == "space-track":
                raise RuntimeError(message)
            print(message)
        else:
            print("Downloading date-matched GOES-18 TLE history from Space-Track...")
            try:
                records = fetch_historical_tles(
                    start,
                    stop,
                    cache_dir,
                    credentials[0],
                    credentials[1],
                )
                return records, "Space-Track GP_HISTORY"
            except (RuntimeError, ValueError) as historical_error:
                if source_mode == "space-track":
                    raise
                print(f"Historical TLE download failed: {historical_error}")

    if source_mode in {"auto", "celestrak"}:
        print("Downloading the latest GOES-18 TLE from CelesTrak...")
        try:
            return fetch_current_tle(url), url
        except (RuntimeError, ValueError) as online_error:
            print(f"Online TLE download failed: {online_error}")

    if not fallback_path.is_file():
        raise RuntimeError(
            "No usable online TLE was found and the fallback TLE file does not "
            f"exist: {fallback_path}"
        )
    print(f"Using fallback GOES-18 TLE: {fallback_path.resolve()}")
    try:
        records = read_tle_file(fallback_path)
    except (RuntimeError, ValueError) as fallback_error:
        raise RuntimeError(
            "The online TLE download failed, and the fallback TLE could "
            f"not be loaded from {fallback_path}: {fallback_error}"
        ) from fallback_error
    return records, f"{fallback_path.resolve()} (offline fallback)"


def load_tle(
    path: Path | None,
    url: str,
    fallback_path: Path,
) -> tuple[str, str, str, str]:
    """Backward-compatible single-TLE loader used by older callers."""
    if path is not None:
        record = read_tle_file(path)[0]
        return record.name, record.line1, record.line2, str(path.resolve())
    try:
        record = fetch_current_tle(url)[0]
        return record.name, record.line1, record.line2, url
    except (RuntimeError, ValueError) as online_error:
        print(f"Online TLE download failed: {online_error}")
        print(f"Using fallback GOES-18 TLE: {fallback_path.resolve()}")
        try:
            record = read_tle_file(fallback_path)[0]
        except (RuntimeError, ValueError) as fallback_error:
            raise RuntimeError(
                "The online TLE download failed, and the fallback TLE could "
                f"not be loaded from {fallback_path}: {fallback_error}"
            ) from fallback_error
        return (
            record.name,
            record.line1,
            record.line2,
            f"{fallback_path.resolve()} (offline fallback)",
        )


def save_used_tle(path: Path, name: str, line1: str, line2: str) -> None:
    """Save one exact TLE; retained for compatibility with older callers."""
    save_tle_records(path, [TLERecord(name, line1, line2, str(path))])


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


def nearest_tle_indices(times: Time, satellites: list[Satrec]) -> np.ndarray:
    """Select the element-set epoch closest to every requested sample."""
    if not satellites:
        raise ValueError("At least one TLE satellite record is required.")
    epochs_jd = np.asarray([tle_epoch_jd(item) for item in satellites])
    distances = np.abs(np.asarray(times.utc.jd)[:, np.newaxis] - epochs_jd)
    return np.argmin(distances, axis=1).astype(int)


def enforce_tle_age_limit(
    times: Time,
    satellites: list[Satrec],
    selected_tle_indices: np.ndarray,
    max_age_days: float,
    tle_source: str,
    allow_stale_tle: bool,
) -> float:
    """Reject stale propagation unless the user explicitly accepts it."""
    epochs_jd = np.asarray([tle_epoch_jd(item) for item in satellites])
    selected_epochs = epochs_jd[selected_tle_indices]
    age_days = np.abs(np.asarray(times.utc.jd) - selected_epochs)
    maximum_age = float(np.max(age_days))
    if maximum_age > max_age_days:
        message = (
            "No sufficiently date-matched GOES-18 TLE is available. The "
            f"farthest requested sample is {maximum_age:.1f} days from its "
            f"nearest TLE epoch, exceeding the {max_age_days:g}-day limit.\n"
            f"Selected TLE source: {tle_source}\n"
            "Historical calculations require Space-Track GP_HISTORY data. "
            f"Set {SPACE_TRACK_IDENTITY_ENV} and {SPACE_TRACK_PASSWORD_ENV}, "
            "or provide a historical file with --tle-file. The script is "
            "stopping instead of producing a misleading JPL comparison."
        )
        if not allow_stale_tle:
            raise SystemExit(message)
        warnings.warn(
            message
            + " Stale propagation was explicitly enabled with "
            "--allow-stale-tle, so the resulting positions may be inaccurate.",
            RuntimeWarning,
            stacklevel=2,
        )
    return maximum_age


def fetch_horizons_geocentric_ephemeris(
    datetimes: list[datetime],
) -> tuple[np.ndarray, np.ndarray]:
    """Return geometric geocentric Moon and Sun ICRF vectors from Horizons.

    Horizons limits the number of rows returned by one request.  Split long
    five-minute grids into non-overlapping batches and verify that every
    returned Julian date matches the requested UTC grid before concatenating
    the vectors.
    """
    if not datetimes:
        raise ValueError("At least one datetime is required.")

    requested_jd = np.asarray(Time(datetimes, scale="utc").utc.jd, dtype=float)
    moon_chunks: list[np.ndarray] = []
    sun_chunks: list[np.ndarray] = []
    start_index = 0

    while start_index < len(datetimes):
        stop_index = min(
            start_index + HORIZONS_BODY_BATCH_SAMPLES,
            len(datetimes),
        )
        # A one-row Horizons request has identical start and stop times and is
        # rejected. Include a one-sample remainder in the preceding batch.
        if len(datetimes) - stop_index == 1:
            stop_index = len(datetimes)

        batch = datetimes[start_index:stop_index]
        start_label = batch[0].strftime("%Y-%m-%d %H:%M:%S")
        stop_label = batch[-1].strftime("%Y-%m-%d %H:%M:%S")
        expected_jd = requested_jd[start_index:stop_index]

        moon_jd, _, moon_vectors, _ = fetch_vectors(
            MOON_ID,
            EARTH_ID,
            start_label,
            stop_label,
            "5 min",
        )
        sun_jd, _, sun_vectors, _ = fetch_vectors(
            SUN_ID,
            EARTH_ID,
            start_label,
            stop_label,
            "5 min",
        )

        for body_name, returned_jd in (
            ("Moon", moon_jd),
            ("Sun", sun_jd),
        ):
            if returned_jd.shape != expected_jd.shape or not np.allclose(
                returned_jd,
                expected_jd,
                rtol=0.0,
                atol=1.0e-7,
            ):
                raise RuntimeError(
                    f"Horizons {body_name} time grid does not match the "
                    "requested five-minute UTC grid."
                )

        moon_chunks.append(np.asarray(moon_vectors, dtype=float))
        sun_chunks.append(np.asarray(sun_vectors, dtype=float))
        start_index = stop_index

    return np.vstack(moon_chunks), np.vstack(sun_chunks)


def propagate_tle_ephemeris(
    datetimes: list[datetime],
    satellites: list[Satrec],
    selected_tle_indices: np.ndarray,
    body_ephemeris: str = "horizons",
) -> dict[str, np.ndarray | Time]:
    """Generate GOES-18 and Earth/Moon/Sun vectors on the UTC time grid.

    SGP4 returns GOES-18 coordinates in TEME.  Astropy converts the spacecraft
    position to GCRS.  By default, Horizons supplies matching geometric ICRF
    Moon and Sun positions relative to Earth's center; ``astropy-builtin`` is
    available for offline operation.  Subtracting the GOES-18 position from
    those geocentric vectors produces the observer-relative vectors used by
    the sky geometry calculation.
    """
    times = Time(datetimes, scale="utc")
    sample_count = len(datetimes)
    if selected_tle_indices.shape != (sample_count,):
        raise ValueError("selected_tle_indices must contain one entry per sample.")

    errors = np.zeros(sample_count, dtype=np.uint8)
    teme_position = np.empty((sample_count, 3), dtype=float)
    teme_velocity = np.empty((sample_count, 3), dtype=float)
    for record_index in np.unique(selected_tle_indices):
        mask = selected_tle_indices == record_index
        subset_errors, subset_position, subset_velocity = satellites[
            int(record_index)
        ].sgp4_array(
            np.asarray(times.utc.jd1[mask], dtype=float),
            np.asarray(times.utc.jd2[mask], dtype=float),
        )
        errors[mask] = subset_errors
        teme_position[mask] = subset_position
        teme_velocity[mask] = subset_velocity
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
    # Hosted/offline environments can have a bundled IERS-A table whose
    # predictive rows are older than Astropy's default 30-day freshness
    # limit. Allow those bundled rows instead of failing the TEME-to-GCRS
    # transformation. This affects Earth-orientation interpolation only; it
    # does not remove the need to propagate close to the TLE epoch.
    iers.conf.auto_max_age = None
    teme_coordinates = TEME(
        CartesianRepresentation(teme_position.T * u.km),
        obstime=times,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="astropy.utils.iers")
        goes_gcrs = teme_coordinates.transform_to(GCRS(obstime=times))
    goes_gcrs_km = goes_gcrs.cartesian.xyz.to_value(u.km).T

    if body_ephemeris == "horizons":
        moon_geocentric_km, sun_geocentric_km = (
            fetch_horizons_geocentric_ephemeris(datetimes)
        )
    elif body_ephemeris == "astropy-builtin":
        # This analytical ephemeris requires no Horizons call and no
        # downloaded JPL BSP/SPK kernel, but it is not the matched-model mode.
        with solar_system_ephemeris.set("builtin"):
            moon_geocentric_km = (
                get_body("moon", times).cartesian.xyz.to_value(u.km).T
            )
            sun_geocentric_km = (
                get_body("sun", times).cartesian.xyz.to_value(u.km).T
            )
    else:
        raise ValueError(
            "body_ephemeris must be 'horizons' or 'astropy-builtin'."
        )

    earth_from_goes_km = -goes_gcrs_km
    moon_from_goes_km = moon_geocentric_km - goes_gcrs_km
    sun_from_goes_km = sun_geocentric_km - goes_gcrs_km
    tle_epochs_jd = np.asarray([tle_epoch_jd(item) for item in satellites])
    selected_epochs_jd = tle_epochs_jd[selected_tle_indices]

    return {
        "times": times,
        "tle_record_index": np.asarray(selected_tle_indices, dtype=int),
        "tle_epoch_jd": selected_epochs_jd,
        "tle_age_days": np.asarray(times.utc.jd) - selected_epochs_jd,
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
    tle_record_indices = np.asarray(ephemeris["tle_record_index"], dtype=int)
    tle_epochs_jd = np.asarray(ephemeris["tle_epoch_jd"], dtype=float)
    tle_ages_days = np.asarray(ephemeris["tle_age_days"], dtype=float)

    header = [
        "utc",
        "julian_date_ut",
        "tle_record_index",
        "tle_epoch_utc",
        "tle_age_days",
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
                    str(tle_record_indices[index]),
                    Time(
                        tle_epochs_jd[index], format="jd", scale="utc"
                    ).to_datetime(timezone=timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    f"{tle_ages_days[index]:.9f}",
                    *(f"{value:.9f}" for value in values),
                ]
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate GOES-18 visible sky using a TLE/SGP4 spacecraft "
            "trajectory and JPL Horizons Earth/Moon/Sun ephemerides."
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
            "Use this local GOES-18 TLE file instead of online selection; the "
            "file may contain multiple historical element sets"
        ),
    )
    parser.add_argument(
        "--tle-source",
        choices=("auto", "space-track", "celestrak"),
        default="auto",
        help=(
            "Online TLE source when --tle-file is omitted. 'auto' uses "
            "Space-Track GP_HISTORY when credentials are configured, then "
            "falls back to the latest CelesTrak TLE (default: auto)"
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
        "--tle-cache-dir",
        type=Path,
        default=DEFAULT_TLE_CACHE_DIR,
        help=(
            "Directory for cached Space-Track history responses "
            "(default: tle_cache beside this script)"
        ),
    )
    parser.add_argument(
        "--max-tle-age",
        type=float,
        default=DEFAULT_MAX_TLE_AGE_DAYS,
        help=(
            "Reject a requested time more than this many days from its nearest "
            "TLE epoch unless --allow-stale-tle is set (default: 14)"
        ),
    )
    parser.add_argument(
        "--allow-stale-tle",
        action="store_true",
        help=(
            "Allow propagation beyond --max-tle-age. This is intended only "
            "for diagnostics and can produce inaccurate comparisons."
        ),
    )
    parser.add_argument(
        "--body-ephemeris",
        choices=("horizons", "astropy-builtin"),
        default="horizons",
        help=(
            "Geocentric Moon/Sun model. 'horizons' matches the JPL algorithm "
            "(default); 'astropy-builtin' is an offline analytical fallback"
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
        default=(
            Path(__file__).resolve().parent
            / "visible_sky_data"
            / "goes18_visible_sky_tle"
        ),
        help=(
            "Output path without extension (default: visible_sky_data/"
            "goes18_visible_sky_tle beside this script)"
        ),
    )
    parser.add_argument(
        "--exposure-sky-points",
        type=int,
        default=DEFAULT_EXPOSURE_SKY_POINTS,
        help=(
            "Deterministic sky-grid size for the 0-to-90-degree exposure "
            "sweep (default: 8192; larger is more precise but slower)"
        ),
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
    if args.exposure_sky_points < 1000:
        raise SystemExit("--exposure-sky-points must be at least 1000.")

    if args.tle_file is not None:
        print(f"Reading GOES-18 TLE data from {args.tle_file}...")
    records, tle_source = load_tle_records(
        args.tle_file,
        args.tle_url,
        args.fallback_tle_file,
        start,
        stop,
        args.tle_source,
        args.tle_cache_dir,
    )
    satellites = [record.satellite() for record in records]

    datetimes = build_sample_datetimes(start, stop)
    astropy_times = Time(datetimes, scale="utc")
    selected_tle_indices = nearest_tle_indices(astropy_times, satellites)
    maximum_tle_age = enforce_tle_age_limit(
        astropy_times,
        satellites,
        selected_tle_indices,
        args.max_tle_age,
        tle_source,
        args.allow_stale_tle,
    )

    if args.body_ephemeris == "horizons":
        print(
            "Propagating GOES-18 with TLE/SGP4 and downloading matching "
            "geometric Moon/Sun vectors from JPL Horizons..."
        )
    else:
        print(
            "Propagating GOES-18 with TLE/SGP4 and generating Moon/Sun "
            "vectors with Astropy's offline built-in ephemeris..."
        )
    ephemeris = propagate_tle_ephemeris(
        datetimes,
        satellites,
        selected_tle_indices,
        args.body_ephemeris,
    )
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
    # The propagated spacecraft vector is already geocentric GCRS, so its
    # direction can be written directly as right ascension and declination.
    goes18_ra_deg, goes18_dec_deg = vector_right_ascension_declination(
        np.asarray(ephemeris["goes_gcrs_position_km"])
    )
    results["goes18_ra_deg"] = goes18_ra_deg
    results["goes18_dec_deg"] = goes18_dec_deg

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
    monthly_histogram_path = args.output_prefix.parent / (
        args.output_prefix.name + "_monthly_histogram.png"
    )
    exposure_sweep_path = args.output_prefix.parent / (
        args.output_prefix.name + "_sun_exclusion_exposures.png"
    )
    ephemeris_path = args.output_prefix.parent / (
        args.output_prefix.name + "_ephemeris.csv"
    )
    tle_path = args.output_prefix.parent / (args.output_prefix.name + "_used.tle")
    used_record_indices = sorted({int(value) for value in selected_tle_indices})
    used_records = [records[index] for index in used_record_indices]
    save_tle_records(tle_path, used_records)
    write_ephemeris_csv(ephemeris_path, datetimes, ephemeris)
    write_results_csv(csv_path, selected_jd, selected_labels, selected_results)

    used_satellites = [satellites[index] for index in used_record_indices]
    used_epoch_datetimes = [
        tle_epoch(satellite).to_datetime(timezone=timezone.utc)
        for satellite in used_satellites
    ]
    first_epoch_label = min(used_epoch_datetimes).strftime("%Y-%m-%d %H:%M UTC")
    last_epoch_label = max(used_epoch_datetimes).strftime("%Y-%m-%d %H:%M UTC")
    observer_name = "GOES-18 TLE/SGP4"
    sampling_description = f"5 min near angular alignments; {args.step} elsewhere"
    zoom_start, zoom_stop = write_plot(
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
        labels,
        results["visible_fraction"],
    )
    write_monthly_average_histogram(
        monthly_histogram_path,
        labels,
        results["visible_fraction"],
        observer_name,
        args.earth_clearance,
        args.moon_clearance,
        args.moon_reference,
        sun_exclusion,
    )
    print("Computing continuous-exposure Sun-angle sweep from 0 to 90 degrees...")
    write_sun_exclusion_exposure_plot(
        exposure_sweep_path,
        earth_vectors,
        moon_vectors,
        sun_vectors,
        observer_name,
        args.earth_clearance,
        args.moon_clearance,
        args.moon_reference,
        sun_exclusion,
        args.exposure_sky_points,
    )

    em_limit, es_limit, ms_limit = angular_limits_deg
    fraction = results["visible_fraction"]
    percent = 100.0 * fraction
    print(f"TLE source: {tle_source}")
    print(f"TLE object: GOES 18 (NORAD {GOES18_NORAD_ID})")
    print(
        "Earth/Moon/Sun model: "
        + (
            "JPL Horizons geometric ICRF vectors"
            if args.body_ephemeris == "horizons"
            else "Astropy built-in analytical ephemeris (offline mode)"
        )
    )
    print(f"TLE records downloaded/loaded: {len(records)}")
    print(f"TLE records actually used: {len(used_records)}")
    print(f"Used TLE epoch range: {first_epoch_label} through {last_epoch_label}")
    print(
        "Maximum distance from each sample to its selected TLE epoch: "
        f"{maximum_tle_age:.2f} days"
    )
    print(
        "SGP4 mode: "
        + (
            "deep-space"
            if all(satellite.method == "d" for satellite in used_satellites)
            else "mixed/near-Earth"
        )
    )
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
    print(
        "Two-day section included in the main plot: "
        f"{zoom_start:%Y-%m-%d %H:%M} through "
        f"{zoom_stop:%Y-%m-%d %H:%M} UTC"
    )
    print(f"Wrote monthly histogram: {monthly_histogram_path.resolve()}")
    print(f"Wrote Sun-exclusion exposure plot: {exposure_sweep_path.resolve()}")


if __name__ == "__main__":
    main()
