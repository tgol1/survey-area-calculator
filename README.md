## Survey Area Calculator v3
George Tolis

Last updated: 12 September 2026

## Changelog
Every new version number represents completion of additional assignments for project.

v1 - 14 August 2026
- Initial Release

v1.1 - 26 August 2026
- Changed step from one hour to five minutes between data points from JPL Horizons API.
- Script now uses line plot instead of histogram. README now reflects change.
- Renamed "main.py" to "api.algorithm.py" to reflect future update with additional algorithm independent of API.
- Major ticks now represent one day as opposed to using matplotlib's automatic bin spacing.
- Added note in "Function" section of README for intrepreting minimum and maximum values of fraction of visibility.

v1.2 - 27 August 2026
- Introduced Sun in calculation. User has option to select 30 or 45 degree exclusion zones with Sun in relation to Earth
- Function description in README changed to reflect additional features

v1.21 - 29 August 2026
- Amended graph formatting
- Graph uses exclusion zones to highlight specific areas and expand with 5 minute intervals

v2 - 30 August 2026
- Introduced additional script that uses TLE data for ephemeris as opposed to JPL Horizons API

v2.1 - 4 September 2026
- Organized JPL and SGP4 algorithms in respective folders.

v2.2 - 7 September 2026
- Added histogram displaying average visible sky percentage by month
- Added graphs for visible sky as a function of sun exclusion angle, using average times through 10 minute, 1 hour and 24 hour time periods.

v3 - 12 September 2026
- Script now highlights random two day section of visible sky data, showing finer scale
- Added script that runs both algorithms and compares values, highlighting Gaussian width, center and computational error.

v4 - 18 September 2026
- Removed Gaussian reference from algorithm comparison
- Angular difference in RA declination (in arc seconds) replaces y axis in lieu of direct comparison of visible sky fractions. Scripts have been updated to include RA declination in .csv file.
- Updated "Algorithm Comparison" section of README to reflect changes in the algorithm comparison. 
- Website reflects changes in algorithm comparison
- Introduced trial function: two 2 week sections picked manually or at random are compared with each other and the differences are plotted. Utilizes JPL algorithm.

## Function
Script computes the instantaneous visible fraction of sky from GOES-18 - a satellite positioned at 137W around Hawaii.

The script downloads geometric Earth, Moon and Sun position vectors from the
NASA/JPL Horizons API or a TLE data sheet, with GOES-18 as the observing center.  It then treats
the Earth, Moon and Sun avoidance regions as spherical caps and subtracts the
solid angle of their union from the full sky (4&pi steradians). For the TLE script, a copy of the used data will be saved in the directory.

The calculation is rotationally invariant: although the Earth direction
moves in a satellite-fixed celestial coordinate system, the instantaneous
fraction of sky depends only on cap sizes and Earth-Moon-Sun angular separation.

Minimum fractions of visible sky represent minimum visibility; earth, moon and sun exclusion regions do not overlap and both
independently block portions of the sky. Likewise, maximum fractions of visible sky represent maximum visibility; earth, moon and sun have minimal center-to-center separation. Plateaus at minimum visibility represent the duration that the Earth, Moon and Sun exclusion regions are completely separated.

## Algorithm Comparison

The compare_algorithms.py script compares the RA declination results produced by the NASA/JPL Horizons and TLE/SGP4 algorithms. It runs both methods using the same date range, sampling interval, and Sun-exclusion angle, then matches their output values at common UTC timestamps.

The difference is calculated as:

[
\text{Difference} =
\text{Horizons RA declination} -
\text{TLE/SGP4 RA declination}
]

The generated PNG contains two panels:

A time-domain dot plot showing how the difference between the two algorithms changes over time.
A normalized histogram showing the distribution of those differences, with a Gaussian curve based on the sample mean and standard deviation.

A mean difference near zero indicates little systematic bias between the algorithms. The standard deviation, (\sigma), describes the empirical spread of RA declinations and can be reported as the approximate one-standard-deviation computational disagreement. For example, (\sigma = 0.0072) percentage points means the two methods typically differ by approximately (0.0072) percentage points in calculated visible-sky coverage.

The histogram may not be perfectly Gaussian. Visible-sky plateaus can produce many differences close to zero, while orbital motion can create periodic or clustered residuals. Therefore, the Gaussian width should be interpreted as the spread of the visible-sky differences over the selected interval, rather than as a direct measurement of the satellite’s positional error in kilometers.

For a meaningful comparison, the selected date range should be close to the epoch of the TLE being used. Propagating a TLE far from its epoch can introduce large, structured errors that do not represent the normal short-term accuracy of SGP4. The TLE included is from 27 August 2026.


## How to Use
- Input date range as prompted in YYYY-MM-DD format.
- Input desired exclusion zone angle for Sun as prompted.
- Script outputs min, mean and max sky coverage in percent and fraction format. Table and line plot of sky coverage as a function of time are also generated by the script.

## Requirements

- Python 3.10 or newer
- NumPy — vector and angular calculations
- Matplotlib — graph generation
- SGP4 — propagates the GOES-18 TLE into position and velocity vectors

Install the required libraries with:

```bash
python -m pip install numpy matplotlib sgp4
```

Alternatively, install them from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

The script also uses Python standard-library modules including `argparse`, `csv`, `datetime`, `json`, `math`, `pathlib`, and `urllib`. These are included with Python and do not require separate installation.


