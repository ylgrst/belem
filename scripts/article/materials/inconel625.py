"""
Inconel 625 bulk material properties, fitted to measured tensile curves.

Unlike the Inconel 718 module, which fits a digitised figure, this one fits
measured data directly: NIST publishes the stress and strain of every specimen
as csv, so the hardening parameters come from a regression over real points and
their uncertainties are the regression's own.

    AM Bench 2022 ASTM E8 Macroscale Tension at Different Strain Rates on
    As-built IN625, National Institute of Standards and Technology,
    doi:10.18434/mds2-2760

Five ASTM E8 subsize specimens, stereo digital image correlation with a virtual
extensometer, engineering stress and strain recorded to fracture. Three were
pulled at a nominal 0.001 per second and two at 0.01.

IMPORTANT: this is as built material, not solution treated and aged. As built
IN625 yields lower and hardens far more than aged material. For a printed
lattice that is arguably the more representative state, since lattices are
often used stress relieved at most, but it is a deliberate choice and the
article should say so rather than leave it implicit. The Inconel 718 module
describes aged material, so the two are not interchangeable as a sensitivity
check on the alloy alone.

The dataset is not redistributed here. Fetch it, fit it, then print it:

    python inconel625.py --download
    python inconel625.py --fit
    python inconel625.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import numpy.typing as npt

DOI = "10.18434/mds2-2760"
BASE_URL = "https://data.nist.gov/od/ds/mds2-2760"
SPECIMEN_FILES: Tuple[str, ...] = tuple(
    f"SpecimenT{n}_stress_strain.csv" for n in (3, 4, 5, 6, 7))
AUXILIARY_FILES: Tuple[str, ...] = (
    "README_AMBench_2022_Tensile_data.txt",
    "SUMMARY_AMBench_2022_Tensile_data.pdf",
)

# Where the downloaded csv files live, next to this module by default.
DATA_DIRECTORY = Path(__file__).resolve().parent / "data_in625"

# Cached fit, written by --fit. The campaign reads this rather than refitting on
# every import, so the numbers a run used are recorded rather than recomputed.
FIT_CACHE = DATA_DIRECTORY / "ludwik_fit.json"

# Neither of these is in the NIST dataset, which reports no elastic constants
# beyond what the curves themselves contain. The modulus used by the campaign is
# measured from the elastic portion of each curve; this nominal wrought value is
# only the fallback for a specimen too sparse near the origin to regress.
NOMINAL_YOUNG_MODULUS = 205.0e3
POISSON_RATIO = 0.28

# Nor the thermal expansion. 12.8e-6 per K is the usual wrought IN625 value near
# room temperature, which is the range the campaign runs in.
THERMAL_EXPANSION = 12.8e-6

# Plastic strain window for the Ludwik regression. The lower bound keeps the fit
# clear of the elastic to plastic knee, where the modulus estimate rather than
# the hardening dominates; the upper bound stops before necking invalidates the
# engineering to true conversion.
FIT_STRAIN_WINDOW = (0.002, 0.10)

# Strain band the modulus is regressed over, below yield for as built IN625.
MODULUS_STRAIN_WINDOW = (0.0005, 0.002)
MIN_MODULUS_POINTS = 5


class LudwikFit(NamedTuple):
    """Result of fitting sigma_y(p) = Re + k * p**m to the NIST specimens

    Errors are the square roots of the covariance diagonal, so they are
    regression scatter over the pooled points and nothing else: they do not
    carry the specimen to specimen spread, which is reported separately as
    young_modulus_std, nor any systematic error of the measurement itself.
    """

    source: str
    condition: str
    specimens: int
    points_fitted: int
    young_modulus: float
    young_modulus_std: float
    yield_strength: float
    yield_strength_err: float
    hardening_modulus: float
    hardening_modulus_err: float
    hardening_exponent: float
    hardening_exponent_err: float
    rms_residual: float
    strain_window: Tuple[float, float]


def download(directory: Path = DATA_DIRECTORY, timeout: float = 120.0) -> List[Path]:
    """Fetch the NIST csv files, with the readme and summary alongside them

    NIST's download endpoint has been returning gateway timeouts; if this fails,
    the same files can be pulled by hand from the landing page for doi:DOI and
    dropped into the directory, which is all the fit needs.

    :param directory: where to put them
    :param timeout: per file, in seconds
    :return: the paths actually written
    """
    import urllib.request

    directory.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for name in SPECIMEN_FILES + AUXILIARY_FILES:
        target = directory / name
        try:
            with urllib.request.urlopen(f"{BASE_URL}/{name}", timeout=timeout) as response:
                target.write_bytes(response.read())
        except Exception as error:  # noqa: BLE001 - report every file, keep going
            print(f"  FAILED      {name}: {error}")
            continue
        written.append(target)
        print(f"  downloaded  {name} ({target.stat().st_size} bytes)")

    return written


def _read_specimen(path: Path) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return engineering strain and engineering stress from one NIST csv

    The columns are located by name rather than by position: the dataset
    documents a time, a strain and a stress column but promises no order.
    """
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))

    header_index = next(i for i, row in enumerate(rows)
                        if any("strain" in cell.lower() for cell in row))
    header = [cell.strip().lower() for cell in rows[header_index]]
    strain_column = next(i for i, cell in enumerate(header) if "strain" in cell)
    stress_column = next(i for i, cell in enumerate(header) if "stress" in cell)

    strain: List[float] = []
    stress: List[float] = []
    for row in rows[header_index + 1:]:
        if len(row) <= max(strain_column, stress_column):
            continue
        try:
            strain.append(float(row[strain_column]))
            stress.append(float(row[stress_column]))
        except ValueError:
            # the files carry unit rows and blank trailing fields
            continue

    return np.asarray(strain), np.asarray(stress)


def _young_modulus(strain: npt.NDArray[np.float64],
                   stress: npt.NDArray[np.float64]) -> float:
    """Slope of the initial linear portion, regressed over MODULUS_STRAIN_WINDOW"""
    low, high = MODULUS_STRAIN_WINDOW
    window = (strain > low) & (strain < high)
    if int(window.sum()) < MIN_MODULUS_POINTS:
        return NOMINAL_YOUNG_MODULUS
    slope = np.polyfit(strain[window], stress[window], 1)[0]

    return float(slope)


def _true_plastic(strain: npt.NDArray[np.float64], stress: npt.NDArray[np.float64],
                  young_modulus: float,
                  ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Convert one engineering curve to true plastic strain and true stress"""
    true_stress = stress * (1.0 + strain)
    true_strain = np.log(1.0 + strain)
    plastic_strain = true_strain - true_stress / young_modulus

    return plastic_strain, true_stress


def _ludwik(plastic_strain: npt.NDArray[np.float64], yield_strength: float,
            hardening_modulus: float, hardening_exponent: float,
            ) -> npt.NDArray[np.float64]:
    return yield_strength + hardening_modulus * plastic_strain ** hardening_exponent


def fit(directory: Path = DATA_DIRECTORY) -> LudwikFit:
    """Fit Ludwik hardening to every specimen of the dataset, pooled

    Each engineering curve is converted to true stress and true plastic strain
    using that specimen's own measured modulus, then every point inside
    FIT_STRAIN_WINDOW is regressed together. Pooling rather than averaging the
    per specimen fits weights each specimen by how much of it lies in the
    window, which is what we want when the specimens fractured at different
    strains.

    :param directory: holding the downloaded csv files
    :return: the fitted parameters and their standard errors
    """
    from scipy.optimize import curve_fit

    paths = [directory / name for name in SPECIMEN_FILES]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of the {len(paths)} NIST csv files are not in "
            f"{directory}: {', '.join(missing)}. Run "
            f"`python inconel625.py --download` first. The dataset is "
            f"doi:{DOI} and is not redistributed with this repository."
        )

    plastic: List[npt.NDArray[np.float64]] = []
    true_stress: List[npt.NDArray[np.float64]] = []
    moduli: List[float] = []
    for path in paths:
        strain, stress = _read_specimen(path)
        modulus = _young_modulus(strain, stress)
        moduli.append(modulus)

        plastic_strain, stress_true = _true_plastic(strain, stress, modulus)
        low, high = FIT_STRAIN_WINDOW
        window = (plastic_strain >= low) & (plastic_strain <= high)
        plastic.append(plastic_strain[window])
        true_stress.append(stress_true[window])

    all_plastic = np.concatenate(plastic)
    all_stress = np.concatenate(true_stress)

    parameters, covariance = curve_fit(
        _ludwik, all_plastic, all_stress, p0=[400.0, 1200.0, 0.4], maxfev=60000)
    errors = np.sqrt(np.diag(covariance))
    residual = float(np.sqrt(np.mean((_ludwik(all_plastic, *parameters) - all_stress) ** 2)))

    return LudwikFit(
        source=f"NIST AM Bench 2022, doi:{DOI}",
        condition="as built IN625",
        specimens=len(paths),
        points_fitted=int(all_plastic.size),
        young_modulus=float(np.mean(moduli)),
        young_modulus_std=float(np.std(moduli)),
        yield_strength=float(parameters[0]),
        yield_strength_err=float(errors[0]),
        hardening_modulus=float(parameters[1]),
        hardening_modulus_err=float(errors[1]),
        hardening_exponent=float(parameters[2]),
        hardening_exponent_err=float(errors[2]),
        rms_residual=residual,
        strain_window=FIT_STRAIN_WINDOW,
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

    return LudwikFit(
        source=str(fields["source"]),
        condition=str(fields["condition"]),
        specimens=int(fields["specimens"]),
        points_fitted=int(fields["points_fitted"]),
        young_modulus=float(fields["young_modulus"]),
        young_modulus_std=float(fields["young_modulus_std"]),
        yield_strength=float(fields["yield_strength"]),
        yield_strength_err=float(fields["yield_strength_err"]),
        hardening_modulus=float(fields["hardening_modulus"]),
        hardening_modulus_err=float(fields["hardening_modulus_err"]),
        hardening_exponent=float(fields["hardening_exponent"]),
        hardening_exponent_err=float(fields["hardening_exponent_err"]),
        rms_residual=float(fields["rms_residual"]),
        strain_window=(float(window[0]), float(window[1])),
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
    proof = props[3] + props[4] * 0.002 ** props[5]

    return "\n".join([
        "Inconel 625, additively manufactured, AS BUILT, not heat treated",
        f"  source          {fitted.source}",
        f"  fitted over     {fitted.specimens} specimens, {fitted.points_fitted} points, "
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
        f"  check           implied 0.2 percent proof stress {proof:.0f} MPa",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIRECTORY, metavar="DIR",
                        help=f"where the NIST csv files are, default {DATA_DIRECTORY}")
    parser.add_argument("--download", action="store_true",
                        help=f"fetch the dataset (doi:{DOI}) into --data-dir")
    parser.add_argument("--fit", action="store_true",
                        help="refit from the csv files and cache the result")
    args = parser.parse_args()

    if args.download:
        written = download(args.data_dir)
        print(f"{len(written)} of {len(SPECIMEN_FILES) + len(AUXILIARY_FILES)} files "
              f"written to {args.data_dir}")
        return

    cache = args.data_dir / FIT_CACHE.name
    if args.fit:
        result = fit(args.data_dir)
        write_fit(result, cache)
        print(f"fit written to {cache}")
    print(describe(cache))


if __name__ == "__main__":
    main()
