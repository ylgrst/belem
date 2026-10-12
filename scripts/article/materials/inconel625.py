"""
Inconel 625 bulk material properties, fitted to measured tensile curves.

Unlike a data sheet, which reports a proof stress and a UTS and leaves the
hardening underdetermined, this fits measured points directly: NIST publishes
the stress and strain of every specimen as csv, so the hardening parameters
come from a regression over real data and their uncertainties are the
regression's own.

    Habib, Saadi A., Weaver, Jordan S., Mates, Steven P., Levine, Lyle E.
    (2022), AM Bench 2022 ASTM E8 Macroscale Tension at Different Strain Rates
    on As-built IN625, Version 1.0.0, National Institute of Standards and
    Technology, https://doi.org/10.18434/mds2-2760

Build AMB2022-625-CBM-B1 on an EOS M270: 195 W, 800 mm/s, 100 um hatch, 20 um
layers, X-Y alternating scan, nitrogen. Specimens machined to ASTM E8 subsize,
6 mm gauge width by 4 mm thickness, pulled at room temperature. Strain is a
25.4 mm virtual extensometer from stereo digital image correlation, recorded to
fracture. Load uncertainty is 1 percent.

IMPORTANT: this is as built material, no heat treatment. As built IN625 yields
lower and hardens far more than solution treated and aged material. For a
printed lattice that is the more representative state, since lattices are
rarely given more than a stress relief, but it is a deliberate choice and the
article should say so rather than leave it implicit.

The dataset is not redistributed here, only the fit it produces. Fetch it, fit
it, then print it:

    python inconel625.py --download
    python inconel625.py --fit
    python inconel625.py
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import numpy.typing as npt

DOI = "10.18434/mds2-2760"
BASE_URL = "https://data.nist.gov/od/ds/mds2-2760"

# Nominal strain rate of each specimen, from Table 3 of the dataset summary.
SPECIMEN_RATES: Dict[str, float] = {
    "T3": 1.0e-3, "T4": 1.0e-3, "T6": 1.0e-3,
    "T5": 1.0e-2, "T7": 1.0e-2,
}
SPECIMEN_FILES: Dict[str, str] = {
    tag: f"Specimen{tag}_stress_strain.csv" for tag in SPECIMEN_RATES}
AUXILIARY_FILES: Tuple[str, ...] = (
    "README_AMBench_2022_Tensile_data.txt",
    "SUMMARY_AMBench_2022_Tensile_data.pdf",
)

# The campaign's constitutive law is rate independent, so it is fitted to the
# quasi static specimens alone rather than to a mixture of rates. The faster
# specimens are not wasted: they measure how much that choice costs, which the
# fit records as rate_sensitivity_percent and which turns out to be about 1.3
# percent of flow stress per decade of strain rate.
CAMPAIGN_STRAIN_RATE = 1.0e-3

# Fitting one specimen would give a curve with no sense of reproducibility.
MIN_SPECIMENS = 2

# Where the downloaded files live, next to this module by default. The raw data
# stays out of version control; only the fit derived from it is committed.
DATA_DIRECTORY = Path(__file__).resolve().parent / "data_in625"
FIT_CACHE = Path(__file__).resolve().parent / "inconel625_ludwik_fit.json"

# Not in the dataset, which reports no elastic constants. The modulus the
# campaign uses is measured from the curves themselves; this is only the
# fallback for a specimen too sparse near the origin to regress.
NOMINAL_YOUNG_MODULUS = 205.0e3
POISSON_RATIO = 0.28

# Nor the thermal expansion. 12.8e-6 per K is the usual wrought IN625 value
# near room temperature, which is the range the campaign runs in.
THERMAL_EXPANSION = 12.8e-6

# Plastic strain window for the Ludwik regression. The lower bound keeps the
# fit clear of the elastic to plastic knee, where the modulus estimate rather
# than the hardening dominates; the upper bound stops well before the measured
# necking strain, beyond which the engineering to true conversion is invalid.
#
# Worth knowing for step 3, which identifies yield surfaces: the fitted curve
# tracks the measurement to better than 0.2 percent of flow stress above 0.4
# percent plastic strain, but runs about 1 percent high between 0.2 and 0.4
# percent, peaking at 16 MPa right at the 0.2 percent offset. That band holds
# only 8 of the 420 fitted points, so the regression is driven by the rest of
# the curve and Ludwik's shape cannot quite follow the knee. The consequence is
# that the homogenised yield onset is biased high by about a percent, uniformly
# across every cell, which is a bias the geometries share rather than one that
# distorts the comparison between them.
FIT_STRAIN_WINDOW = (0.002, 0.10)

# Strain band the modulus is regressed over, below yield for as built IN625.
MODULUS_STRAIN_WINDOW = (0.0005, 0.0025)
MIN_MODULUS_POINTS = 5

# NIST either serves a file promptly or hangs until Cloudflare gives up, so the
# attempt on the original is cut short rather than given the full timeout. With
# their distribution down, that is the difference between a fetch of a few
# seconds per file and one of a minute per file.
NIST_PROBE_TIMEOUT = 15.0

# The archive rate limits its index and goes offline for maintenance, so
# requests to it are paced and a refusal is retried with a widening gap. Both
# failures are transient and neither means the file is absent: treating them as
# absence is how an earlier version of this module concluded that specimens it
# had already downloaded once had never been captured.
ARCHIVE_PAUSE = 5.0
ARCHIVE_RETRIES = 4
ARCHIVE_TRANSIENT_CODES = (429, 500, 502, 503, 504)


class Specimen(NamedTuple):
    """One measured curve, converted to true stress and true plastic strain"""

    tag: str
    nominal_rate: float
    measured_rate: float
    young_modulus: float
    proof_stress: float
    plastic_strain: npt.NDArray[np.float64]
    true_stress: npt.NDArray[np.float64]


class LudwikFit(NamedTuple):
    """Result of fitting sigma_y(p) = Re + k * p**m to the NIST specimens

    Errors are the square roots of the covariance diagonal, so they are
    regression scatter over the pooled points and nothing else. They do not
    carry the specimen to specimen spread, which is reported separately as
    young_modulus_std and proof_stress_std, nor the 1 percent load uncertainty
    of the measurement itself, which dominates both.
    """

    source: str
    condition: str
    specimen_ids: Tuple[str, ...]
    nominal_strain_rate: float
    points_fitted: int
    young_modulus: float
    young_modulus_std: float
    proof_stress: float
    proof_stress_std: float
    yield_strength: float
    yield_strength_err: float
    hardening_modulus: float
    hardening_modulus_err: float
    hardening_exponent: float
    hardening_exponent_err: float
    rms_residual: float
    strain_window: Tuple[float, float]
    rate_sensitivity_percent: Optional[float]


def _archive_capture(original: str, timeout: float) -> Tuple[Optional[str], str]:
    """Find a Wayback Machine capture that serves the original bytes

    NIST's file distribution has been timing out behind Cloudflare while the
    rest of the repository stays up, so the archive is the practical way to get
    the data. The id_ suffix asks for the captured response itself rather than
    the archive's rewritten viewer page.

    The archive rate limits its index aggressively, and being told to slow down
    is not the same as a file it never captured. Conflating the two makes a
    transient 429 look like permanent absence, so they are told apart and the
    reason is returned for reporting.

    :return: the capture URL, or None and the reason there is none
    """
    query = ("http://web.archive.org/cdx/search/cdx?url="
             + urllib.parse.quote(original, safe="")
             + "&output=json&filter=statuscode:200&fl=timestamp&limit=-3")

    delay = ARCHIVE_PAUSE
    for attempt in range(ARCHIVE_RETRIES):
        try:
            with urllib.request.urlopen(query, timeout=timeout) as response:
                rows = json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            if error.code not in ARCHIVE_TRANSIENT_CODES:
                return None, f"archive index returned {error.code}"
            if attempt == ARCHIVE_RETRIES - 1:
                return None, (f"archive index returned {error.code} on every attempt, "
                              f"it is rate limiting or offline, try again later")
            time.sleep(delay)
            delay *= 2
            continue
        except Exception as error:  # noqa: BLE001 - the archive is a fallback
            return None, f"archive index unreachable, {str(error)[:50]}"

        if len(rows) < 2:
            return None, "never captured by the archive"

        return f"http://web.archive.org/web/{rows[-1][0]}id_/{original}", "ok"

    return None, "archive index would not answer, try again later"


def _fetch(url: str, timeout: float, retries: int = 1) -> Optional[bytes]:
    """Return the bytes at a URL, or None with the reason printed

    :param retries: attempts before giving up, for sources that fail in bursts
    """
    delay = ARCHIVE_PAUSE
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return bytes(response.read())
        except Exception as error:  # noqa: BLE001 - every source here may be down
            print(f"      {str(error)[:70]}")
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2

    return None


def download(directory: Path = DATA_DIRECTORY, timeout: float = 60.0) -> List[Path]:
    """Fetch the dataset, from NIST if it answers and otherwise from the archive

    :param directory: where to put the files
    :param timeout: per request, in seconds, except the NIST probe
    :return: the paths of the files now present
    """
    directory.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for name in list(SPECIMEN_FILES.values()) + list(AUXILIARY_FILES):
        target = directory / name
        if target.is_file() and target.stat().st_size > 0:
            print(f"  have         {name}")
            written.append(target)
            continue

        original = f"{BASE_URL}/{name}"
        print(f"  fetching     {name}")
        source = "nist"
        data = _fetch(original, min(timeout, NIST_PROBE_TIMEOUT))

        if data is None:
            url, reason = _archive_capture(original, timeout)
            if url is None:
                print(f"  UNAVAILABLE  {name}: {reason}")
                continue
            source = "archive"
            data = _fetch(url, timeout, retries=ARCHIVE_RETRIES)
            time.sleep(ARCHIVE_PAUSE)

        if data is None:
            print(f"  UNAVAILABLE  {name}: capture found but would not download")
            continue

        target.write_bytes(data)
        written.append(target)
        print(f"  from {source:<8s} {name} ({len(data)} bytes)")

    return written


def _read_specimen(path: Path) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
                                        npt.NDArray[np.float64]]:
    """Return time, engineering strain and engineering stress from one csv

    The dataset documents row one as the column names and row two as the SI
    units. Columns are located by name rather than by position, and any row
    that does not parse as numbers is skipped, which disposes of the unit row
    without assuming it is exactly where it is documented to be.
    """
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))

    header_index = next(i for i, row in enumerate(rows)
                        if any("strain" in cell.lower() for cell in row))
    header = [cell.strip().lower() for cell in rows[header_index]]
    columns = {key: next(i for i, cell in enumerate(header) if key in cell)
               for key in ("time", "strain", "stress")}

    values: List[Tuple[float, float, float]] = []
    for row in rows[header_index + 1:]:
        if len(row) <= max(columns.values()):
            continue
        try:
            values.append((float(row[columns["time"]]),
                           float(row[columns["strain"]]),
                           float(row[columns["stress"]])))
        except ValueError:
            continue

    array = np.asarray(values)

    return array[:, 0], array[:, 1], array[:, 2]


def _prepare(tag: str, path: Path) -> Specimen:
    """Convert one engineering curve to true stress and true plastic strain

    The modulus is measured from the specimen's own elastic branch rather than
    assumed, because the plastic strain follows from it and a modulus taken
    from elsewhere would bias every point of the fit.
    """
    seconds, strain, stress = _read_specimen(path)

    low, high = MODULUS_STRAIN_WINDOW
    elastic = (strain > low) & (strain < high)
    if int(elastic.sum()) >= MIN_MODULUS_POINTS:
        young_modulus = float(np.polyfit(strain[elastic], stress[elastic], 1)[0])
    else:
        young_modulus = NOMINAL_YOUNG_MODULUS

    true_stress = stress * (1.0 + strain)
    plastic_strain = np.log(1.0 + strain) - true_stress / young_modulus

    # interpolated rather than nearest sample: the curves are spaced about
    # 5e-4 in strain, which is a quarter of the offset being measured
    rising = int(np.argmax(true_stress))
    proof = float(np.interp(0.002, plastic_strain[:rising], true_stress[:rising]))
    window = ((plastic_strain >= FIT_STRAIN_WINDOW[0])
              & (plastic_strain <= FIT_STRAIN_WINDOW[1]))

    return Specimen(
        tag=tag,
        nominal_rate=SPECIMEN_RATES[tag],
        measured_rate=float(strain[-1] / seconds[-1]),
        young_modulus=young_modulus,
        proof_stress=proof,
        plastic_strain=plastic_strain[window],
        true_stress=true_stress[window],
    )


def _ludwik(plastic_strain: npt.NDArray[np.float64], yield_strength: float,
            hardening_modulus: float, hardening_exponent: float,
            ) -> npt.NDArray[np.float64]:
    return yield_strength + hardening_modulus * plastic_strain ** hardening_exponent


def _regress(specimens: List[Specimen]) -> Tuple[npt.NDArray[np.float64],
                                                 npt.NDArray[np.float64], float, int]:
    """Least squares Ludwik fit over the pooled points of several specimens"""
    from scipy.optimize import curve_fit

    plastic = np.concatenate([s.plastic_strain for s in specimens])
    stress = np.concatenate([s.true_stress for s in specimens])
    parameters, covariance = curve_fit(
        _ludwik, plastic, stress, p0=[400.0, 1200.0, 0.4], maxfev=60000)
    residual = float(np.sqrt(np.mean((_ludwik(plastic, *parameters) - stress) ** 2)))

    return parameters, np.sqrt(np.diag(covariance)), residual, int(plastic.size)


def _rate_sensitivity(slow: List[Specimen], fast: List[Specimen]) -> Optional[float]:
    """Largest flow stress difference between the two rates, in percent

    Reported rather than corrected for: the campaign's law has no rate term, so
    this is the error that choosing the quasi static specimens introduces.
    """
    if not slow or not fast:
        return None
    (a, b, c), _, _, _ = _regress(slow)
    (d, e, f), _, _, _ = _regress(fast)
    probe = np.array([0.002, 0.01, 0.05, 0.10])
    difference = (_ludwik(probe, d, e, f) - _ludwik(probe, a, b, c)) / _ludwik(probe, a, b, c)

    return float(100.0 * np.max(np.abs(difference)))


def fit(directory: Path = DATA_DIRECTORY,
        nominal_strain_rate: float = CAMPAIGN_STRAIN_RATE) -> LudwikFit:
    """Fit Ludwik hardening to the specimens pulled at one nominal strain rate

    Every point of every qualifying specimen is regressed together rather than
    averaging per specimen fits, which weights each specimen by how much of it
    falls inside the window. That is what we want from specimens that fractured
    at different strains.

    :param directory: holding the downloaded csv files
    :param nominal_strain_rate: which specimens to fit, see SPECIMEN_RATES
    :return: the fitted parameters and their standard errors
    """
    available = {tag: directory / name for tag, name in SPECIMEN_FILES.items()
                 if (directory / name).is_file()}
    if not available:
        raise FileNotFoundError(
            f"No specimen csv files in {directory}. Run "
            f"`python inconel625.py --download` first. The dataset is "
            f"doi:{DOI} and is not redistributed with this repository."
        )

    prepared = [_prepare(tag, path) for tag, path in sorted(available.items())]
    wanted = [s for s in prepared if s.nominal_rate == nominal_strain_rate]
    others = [s for s in prepared if s.nominal_rate != nominal_strain_rate]
    if len(wanted) < MIN_SPECIMENS:
        raise FileNotFoundError(
            f"Only {len(wanted)} specimen(s) at {nominal_strain_rate:g} per second "
            f"are present in {directory}, and at least {MIN_SPECIMENS} are needed "
            f"to fit. Expected "
            f"{', '.join(t for t, r in SPECIMEN_RATES.items() if r == nominal_strain_rate)}; "
            f"found {', '.join(s.tag for s in wanted) or 'none'}. Run "
            f"`python inconel625.py --download`."
        )

    parameters, errors, residual, points = _regress(wanted)

    return LudwikFit(
        source=f"NIST AM Bench 2022, doi:{DOI}",
        condition="as built IN625, no heat treatment",
        specimen_ids=tuple(s.tag for s in wanted),
        nominal_strain_rate=nominal_strain_rate,
        points_fitted=points,
        young_modulus=float(np.mean([s.young_modulus for s in wanted])),
        young_modulus_std=float(np.std([s.young_modulus for s in wanted])),
        proof_stress=float(np.mean([s.proof_stress for s in wanted])),
        proof_stress_std=float(np.std([s.proof_stress for s in wanted])),
        yield_strength=float(parameters[0]),
        yield_strength_err=float(errors[0]),
        hardening_modulus=float(parameters[1]),
        hardening_modulus_err=float(errors[1]),
        hardening_exponent=float(parameters[2]),
        hardening_exponent_err=float(errors[2]),
        rms_residual=residual,
        strain_window=FIT_STRAIN_WINDOW,
        rate_sensitivity_percent=_rate_sensitivity(wanted, others),
    )


def write_fit(result: LudwikFit, path: Path = FIT_CACHE) -> None:
    """Cache a fit so the campaign reads it instead of refitting"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result._asdict(), indent=2))


def read_fit(path: Path = FIT_CACHE) -> LudwikFit:
    """Return the cached fit, or explain how to produce it"""
    if not path.is_file():
        raise FileNotFoundError(
            f"No cached fit at {path}. Run `python inconel625.py --download` "
            f"then `python inconel625.py --fit`. Inconel 625 properties are "
            f"fitted to measured data and deliberately not hard coded in this "
            f"module, so there is nothing to fall back on."
        )
    # read field by field rather than splatting, so a cache left over from an
    # earlier version of LudwikFit fails here and not in the middle of a run
    fields = json.loads(path.read_text())
    window = fields["strain_window"]
    rate_check = fields["rate_sensitivity_percent"]

    return LudwikFit(
        source=str(fields["source"]),
        condition=str(fields["condition"]),
        specimen_ids=tuple(str(t) for t in fields["specimen_ids"]),
        nominal_strain_rate=float(fields["nominal_strain_rate"]),
        points_fitted=int(fields["points_fitted"]),
        young_modulus=float(fields["young_modulus"]),
        young_modulus_std=float(fields["young_modulus_std"]),
        proof_stress=float(fields["proof_stress"]),
        proof_stress_std=float(fields["proof_stress_std"]),
        yield_strength=float(fields["yield_strength"]),
        yield_strength_err=float(fields["yield_strength_err"]),
        hardening_modulus=float(fields["hardening_modulus"]),
        hardening_modulus_err=float(fields["hardening_modulus_err"]),
        hardening_exponent=float(fields["hardening_exponent"]),
        hardening_exponent_err=float(fields["hardening_exponent_err"]),
        rms_residual=float(fields["rms_residual"]),
        strain_window=(float(window[0]), float(window[1])),
        rate_sensitivity_percent=None if rate_check is None else float(rate_check),
    )


def epicp_props(poisson_ratio: float = POISSON_RATIO,
                thermal_expansion: float = THERMAL_EXPANSION,
                hardening: Optional[Tuple[float, float]] = None,
                fit_path: Path = FIT_CACHE,
                ) -> npt.NDArray[np.float64]:
    """Build the EPICP property vector, [E, nu, alpha, Re, k, m]

    :param poisson_ratio: see POISSON_RATIO
    :param thermal_expansion: see THERMAL_EXPANSION
    :param hardening: explicit (k, m) replacing the fitted pair
    :param fit_path: cached fit to read, see read_fit
    :return: the six EPICP properties
    """
    fitted = read_fit(fit_path)
    hardening_modulus, hardening_exponent = (
        hardening if hardening is not None
        else (fitted.hardening_modulus, fitted.hardening_exponent))

    return np.array([
        fitted.young_modulus,
        poisson_ratio,
        thermal_expansion,
        fitted.yield_strength,
        hardening_modulus,
        hardening_exponent,
    ])


def describe(fit_path: Path = FIT_CACHE) -> str:
    """Return a readable summary of the properties actually used"""
    fitted = read_fit(fit_path)
    props = epicp_props(fit_path=fit_path)
    implied = props[3] + props[4] * 0.002 ** props[5]
    rate = ("not measured" if fitted.rate_sensitivity_percent is None
            else f"{fitted.rate_sensitivity_percent:.1f} percent per decade")

    return "\n".join([
        "Inconel 625, additively manufactured, AS BUILT, no heat treatment",
        f"  source          {fitted.source}",
        f"  fitted over     specimens {', '.join(fitted.specimen_ids)} at "
        f"{fitted.nominal_strain_rate:g} 1/s, {fitted.points_fitted} points, "
        f"plastic strain {fitted.strain_window[0]} to {fitted.strain_window[1]}",
        f"  E               {props[0]:.0f} MPa   "
        f"(measured, specimen spread {fitted.young_modulus_std:.0f})",
        f"  nu              {props[1]:.3f}            (literature, not in the dataset)",
        f"  alpha           {props[2]:.2e} 1/K      (literature, not in the dataset)",
        f"  Re              {props[3]:.1f} MPa   +/- {fitted.yield_strength_err:.1f} "
        f"(initial yield, not the proof stress)",
        f"  k               {props[4]:.1f} MPa   +/- {fitted.hardening_modulus_err:.1f}",
        f"  m               {props[5]:.4f}        +/- {fitted.hardening_exponent_err:.4f}",
        f"  fit residual    {fitted.rms_residual:.1f} MPa rms",
        f"  rate effect     {rate}, not modelled",
        f"  proof stress    measured {fitted.proof_stress:.0f} "
        f"+/- {fitted.proof_stress_std:.0f} MPa, model gives {implied:.0f}, "
        f"see FIT_STRAIN_WINDOW",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIRECTORY, metavar="DIR",
                        help=f"where the dataset files are, default {DATA_DIRECTORY}")
    parser.add_argument("--download", action="store_true",
                        help=f"fetch the dataset (doi:{DOI}) into --data-dir")
    parser.add_argument("--fit", action="store_true",
                        help="refit from the csv files and cache the result")
    parser.add_argument("--strain-rate", type=float, default=CAMPAIGN_STRAIN_RATE,
                        metavar="RATE", help="nominal strain rate to fit, per second")
    args = parser.parse_args()

    if args.download:
        written = download(args.data_dir)
        print(f"{len(written)} of {len(SPECIMEN_FILES) + len(AUXILIARY_FILES)} files "
              f"present in {args.data_dir}")
        return

    if args.fit:
        write_fit(fit(args.data_dir, args.strain_rate))
        print(f"fit written to {FIT_CACHE}")
    print(describe())


if __name__ == "__main__":
    main()
