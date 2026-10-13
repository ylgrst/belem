"""
Inconel 718 bulk material properties, fitted to measured tensile curves.

The hardening comes from a regression over measured points rather than from a
data sheet, so its uncertainties are the regression's own:

    Hrabe, N., Benzing, J., Moser, N., Derimow, N., Kafka, O., Saville, A.,
    Allen, C. (2025), AMB2025-02 Macroscale Quasi-Static Tensile Tests of
    PBF-L IN718, National Institute of Standards and Technology,
    https://doi.org/10.18434/mds2-3735

Eight continuum-but-miniature specimens were excised by EDM from the 2.5 mm
legs of one AMB2022-01 build (AMB2022-718-AMMT-B7-P4), about 1.25 mm wide and
1.03 mm thick, and pulled per ASTM E8 at a nominal 1e-3 per second with a 3 mm
gauge length contact extensometer. The files are the testing machine's own
output: time, crosshead displacement, extensometer displacement and force, from
which engineering stress and strain follow using each specimen's measured
cross section.

AS BUILT, no heat treatment, which matches materials/inconel625.py so the two
alloys can be compared at the same condition.

IMPORTANT, THE MEASURED MODULUS IS NOT A MATERIAL PROPERTY. These curves give
an elastic modulus of about 124 GPa, which NIST reports as the answer to its
own challenge and which this module reproduces to 1.3 GPa. That is roughly two
thirds of what IN718 actually is, because a 3 mm contact extensometer on a 1 mm
thick specimen measures its own compliance along with the specimen's. NIST says
as much in the dataset: "Material property data such as elastic mechanical
properties are not provided." So the hardening is taken from here and the
modulus from the Renishaw sheet, see YOUNG_MODULUS_NOTE below.

The dataset is not redistributed here, only the fit it produces:

    python inconel718.py --download
    python inconel718.py --fit
    python inconel718.py
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import numpy.typing as npt

DOI = "10.18434/mds2-3735"

# The documented download endpoint is https://data.nist.gov/od/ds/mds2-3735/,
# which only redirects to the object store below. That redirector has been
# answering 504 while the store itself serves in under a second, so the store is
# addressed directly. If the redirector is working again, either route is fine
# and both return identical bytes.
BASE_URL = "https://nist-oar-cache.s3.amazonaws.com/prd/fst4/mds2-3735"
OFFICIAL_URL = "https://data.nist.gov/od/ds/mds2-3735"

ARCHIVE_NAME = "answers-raw tensile data.zip"
README_NAME = "readme.pdf"
ANSWERS_NAME = "AMB2025-02 prediction answers.xlsx"

# Specimen folders inside the archive.
SPECIMEN_TAGS: Tuple[str, ...] = tuple(f"MT{n}" for n in range(1, 9))

# Gauge length of the contact extensometer, mm, from the dataset readme.
GAUGE_LENGTH = 3.0

# Nominal strain rate of the test, per second, from the dataset readme. Every
# specimen was pulled at this rate, so unlike the IN625 dataset there is no
# rate comparison to be made here.
NOMINAL_STRAIN_RATE = 1.0e-3

# What NIST publishes as the answer to the challenge, mean and sample standard
# deviation over the eight specimens. The fit checks itself against these: the
# module derives them from the raw files and refuses to go on if it cannot
# reproduce them, which is what makes the parsing of a machine specific format
# trustworthy rather than merely plausible.
NIST_REPORTED: Dict[str, Tuple[float, float]] = {
    "young_modulus_gpa": (123.6, 5.9),
    "yield_strength": (690.7, 3.1),
    "ultimate_tensile_strength": (951.6, 9.2),
    "strain_at_uts": (0.247, 0.020),
}

# How far the module's own reduction may sit from the published answer before it
# treats the parse as broken. Generous against the published scatter, tight
# enough to catch a wrong column, a wrong gauge length or a wrong area.
REPRODUCTION_TOLERANCE = 0.02

# Elastic properties for the EPICP law. These do NOT come from the dataset, for
# the reason given at the top of this module.
#
# The Renishaw RenAM 500 data sheet for Inconel 718, part no. H-5800-6794-03-A,
# issued 08.2024, page 6, reports 184 GPa for as built horizontal material and
# 187 GPa for as built vertical, measured to ASTM E8 on machined specimens with
# a sample size of 6. The horizontal value is used, matching the 0.2 percent
# proof stress of 735 MPa and UTS of 1051 MPa on the same line, both of which
# sit close to the 691 MPa and 952 MPa measured here on a different build.
YOUNG_MODULUS = 184.0e3
YOUNG_MODULUS_NOTE = (
    "Renishaw RenAM 500 data sheet H-5800-6794-03-A page 6, as built horizontal. "
    "The dataset's own 124 GPa is an artefact of miniature specimen "
    "extensometry and is not used."
)

# Not in either source. NASA CR-123774 Fig 7.21 gives Poisson's ratio against
# temperature for Alloy 718, about 0.29 at room temperature, which is also the
# standard wrought value.
POISSON_RATIO = 0.29

# Page 2 of the Renishaw sheet gives a wrought coefficient of 12e-6 to 16e-6 per
# K over 25 to 760 degrees C. NASA CR-123774 is more specific, 13.1e-6 over 20
# to 100 degrees C, which is the range the campaign runs in.
THERMAL_EXPANSION = 13.0e-6

DATA_DIRECTORY = Path(__file__).resolve().parent / "data_in718"
FIT_CACHE = Path(__file__).resolve().parent / "inconel718_ludwik_fit.json"

# Plastic strain window for the Ludwik regression, the same as the IN625 module
# uses so the two alloys are fitted on equal terms. The upper bound is well
# inside the measured uniform elongation, which ends at 0.247 total strain.
#
# As in the IN625 module, the fitted curve runs high at the knee: about 1.7
# percent over the measurement between 0.2 and 0.4 percent plastic strain,
# against better than 0.5 percent everywhere above that. The homogenised yield
# onset is biased high to the same degree, uniformly across every cell.
FIT_STRAIN_WINDOW = (0.002, 0.10)

# Engineering stress band the elastic slope is regressed over, below yield.
MODULUS_STRESS_WINDOW = (150.0, 450.0)


class Specimen(NamedTuple):
    """One measured curve, reduced to true stress and true plastic strain"""

    tag: str
    width: float
    thickness: float
    apparent_modulus: float
    yield_strength: float
    ultimate_tensile_strength: float
    strain_at_uts: float
    plastic_strain: npt.NDArray[np.float64]
    true_stress: npt.NDArray[np.float64]


class LudwikFit(NamedTuple):
    """Result of fitting sigma_y(p) = Re + k * p**m to the NIST specimens"""

    source: str
    condition: str
    specimen_ids: Tuple[str, ...]
    nominal_strain_rate: float
    points_fitted: int
    apparent_modulus: float
    apparent_modulus_std: float
    measured_yield_strength: float
    measured_yield_strength_std: float
    measured_uts: float
    measured_uts_std: float
    yield_strength: float
    yield_strength_err: float
    hardening_modulus: float
    hardening_modulus_err: float
    hardening_exponent: float
    hardening_exponent_err: float
    rms_residual: float
    strain_window: Tuple[float, float]
    reproduces_nist: bool


def download(directory: Path = DATA_DIRECTORY, timeout: float = 120.0) -> List[Path]:
    """Fetch the dataset and unpack the specimen folders

    :param directory: where to put the files
    :param timeout: per request, in seconds
    :return: the paths of the files now present
    """
    directory.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for name in (ARCHIVE_NAME, README_NAME, ANSWERS_NAME):
        target = directory / name
        if target.is_file() and target.stat().st_size > 0:
            print(f"  have        {name}")
            written.append(target)
            continue

        url = f"{BASE_URL}/{urllib.parse.quote(name)}"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                target.write_bytes(response.read())
        except Exception as error:  # noqa: BLE001 - report and try the next file
            print(f"  FAILED      {name}: {str(error)[:70]}")
            continue
        written.append(target)
        print(f"  downloaded  {name} ({target.stat().st_size} bytes)")

    archive = directory / ARCHIVE_NAME
    if archive.is_file():
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(directory / "raw")
        print(f"  unpacked    {ARCHIVE_NAME}")

    return written


def _specimen_directory(directory: Path, tag: str) -> Optional[Path]:
    """Locate one specimen's folder inside the unpacked archive"""
    matches = sorted((directory / "raw").glob(f"**/IN_718_{tag}/specimen.dat"))

    return matches[0].parent if matches else None


def _read_specimen(path: Path) -> Tuple[float, float, npt.NDArray[np.float64]]:
    """Read one MTS specimen.dat file

    The format is the testing machine's own: a header of tab separated label and
    value pairs carrying the measured cross section, then a data block whose
    first row names the channels and whose second row gives their units.

    :return: width and thickness in mm, and the rows of the data block
    """
    lines = path.read_text(errors="replace").splitlines()

    width = thickness = None
    start = None
    for index, line in enumerate(lines):
        if line.startswith("Specimen Width"):
            width = float(line.split("\t")[1])
        elif line.startswith("Specimen Thickness"):
            thickness = float(line.split("\t")[1])
        elif line.startswith("Time\t"):
            columns = [c.strip().lower() for c in line.split("\t")]
            start = index + 2  # skip the units row
            break

    if width is None or thickness is None or start is None:
        raise ValueError(f"{path} is not an MTS specimen file this module understands")

    extensometer = next(i for i, c in enumerate(columns) if "extensometer" in c)
    force = next(i for i, c in enumerate(columns) if "force" in c)

    rows: List[Tuple[float, float]] = []
    for line in lines[start:]:
        parts = line.split("\t")
        if len(parts) <= max(extensometer, force):
            continue
        try:
            rows.append((float(parts[extensometer]), float(parts[force])))
        except ValueError:
            continue

    return width, thickness, np.asarray(rows)


def _prepare(tag: str, directory: Path) -> Specimen:
    """Reduce one specimen to true stress and true plastic strain

    The elastic slope is measured from the specimen's own curve and then used to
    subtract the elastic part. That is deliberate even though the slope is too
    low to be the material's modulus: if the extensometer adds a compliance
    proportional to load, the same wrong slope appears in both the measurement
    and the subtraction and cancels, leaving the plastic strain right. Using a
    handbook modulus here instead would leave that error in.
    """
    width, thickness, rows = _read_specimen(directory / "specimen.dat")
    strain = rows[:, 0] / GAUGE_LENGTH
    stress = rows[:, 1] / (width * thickness)

    low, high = MODULUS_STRESS_WINDOW
    band = (stress > low) & (stress < high)
    modulus, intercept = np.polyfit(strain[band], stress[band], 1)
    strain = strain + intercept / modulus  # drop the toe onto the origin

    offset = modulus * (strain - 0.002)
    crossed = np.flatnonzero((stress < offset) & (strain > 0.002))
    index = int(crossed[0]) if crossed.size else int(np.argmax(stress))
    yield_strength = float(np.interp(
        0.0, (offset - stress)[index - 1:index + 1][::-1],
        stress[index - 1:index + 1][::-1])) if index else float(stress[0])

    peak = int(np.argmax(stress))
    true_stress = stress * (1.0 + strain)
    plastic_strain = np.log(1.0 + strain) - true_stress / modulus
    window = ((plastic_strain >= FIT_STRAIN_WINDOW[0])
              & (plastic_strain <= FIT_STRAIN_WINDOW[1]))

    return Specimen(
        tag=tag,
        width=width,
        thickness=thickness,
        apparent_modulus=float(modulus),
        yield_strength=yield_strength,
        ultimate_tensile_strength=float(stress[peak]),
        strain_at_uts=float(strain[peak]),
        plastic_strain=plastic_strain[window],
        true_stress=true_stress[window],
    )


def _ludwik(plastic_strain: npt.NDArray[np.float64], yield_strength: float,
            hardening_modulus: float, hardening_exponent: float,
            ) -> npt.NDArray[np.float64]:
    return yield_strength + hardening_modulus * plastic_strain ** hardening_exponent


def _check_against_nist(specimens: List[Specimen]) -> Tuple[bool, List[str]]:
    """Compare this module's reduction with the answers NIST publishes

    :return: whether every quantity agrees, and a line per quantity for printing
    """
    measured = {
        "young_modulus_gpa": [s.apparent_modulus / 1000.0 for s in specimens],
        "yield_strength": [s.yield_strength for s in specimens],
        "ultimate_tensile_strength": [s.ultimate_tensile_strength for s in specimens],
        "strain_at_uts": [s.strain_at_uts for s in specimens],
    }

    agrees = True
    report: List[str] = []
    for key, (published, published_std) in NIST_REPORTED.items():
        mean = float(np.mean(measured[key]))
        deviation = abs(mean - published) / published
        ok = deviation <= REPRODUCTION_TOLERANCE
        agrees = agrees and ok
        report.append(
            f"    {key:<26s} {mean:9.3f} against {published:9.3f} "
            f"+/- {published_std:.3f}  {100 * deviation:5.2f}%  {'ok' if ok else 'MISMATCH'}"
        )

    return agrees, report


def fit(directory: Path = DATA_DIRECTORY, verbose: bool = False) -> LudwikFit:
    """Fit Ludwik hardening to every specimen of the dataset, pooled

    :param directory: holding the downloaded and unpacked dataset
    :param verbose: print the comparison against NIST's published answers
    :return: the fitted parameters and their standard errors
    """
    from scipy.optimize import curve_fit

    found = {tag: _specimen_directory(directory, tag) for tag in SPECIMEN_TAGS}
    missing = [tag for tag, path in found.items() if path is None]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of the {len(SPECIMEN_TAGS)} specimen folders are not "
            f"under {directory / 'raw'}: {', '.join(missing)}. Run "
            f"`python inconel718.py --download` first. The dataset is doi:{DOI} "
            f"and is not redistributed with this repository."
        )

    specimens = [_prepare(tag, path) for tag, path in found.items() if path is not None]

    agrees, report = _check_against_nist(specimens)
    if verbose or not agrees:
        print("  reduction against NIST's published answers:")
        print("\n".join(report))
    if not agrees:
        raise ValueError(
            "This module's reduction of the raw files does not reproduce the "
            "answers NIST publishes for the same specimens, so the parsing is "
            "wrong and the fit would be meaningless. See the comparison above."
        )

    plastic = np.concatenate([s.plastic_strain for s in specimens])
    stress = np.concatenate([s.true_stress for s in specimens])
    parameters, covariance = curve_fit(
        _ludwik, plastic, stress, p0=[600.0, 1200.0, 0.5], maxfev=60000)
    errors = np.sqrt(np.diag(covariance))
    residual = float(np.sqrt(np.mean((_ludwik(plastic, *parameters) - stress) ** 2)))

    return LudwikFit(
        source=f"NIST AM Bench 2025, AMB2025-02, doi:{DOI}",
        condition="as built IN718, no heat treatment",
        specimen_ids=tuple(s.tag for s in specimens),
        nominal_strain_rate=NOMINAL_STRAIN_RATE,
        points_fitted=int(plastic.size),
        apparent_modulus=float(np.mean([s.apparent_modulus for s in specimens])),
        apparent_modulus_std=float(np.std([s.apparent_modulus for s in specimens], ddof=1)),
        measured_yield_strength=float(np.mean([s.yield_strength for s in specimens])),
        measured_yield_strength_std=float(
            np.std([s.yield_strength for s in specimens], ddof=1)),
        measured_uts=float(np.mean([s.ultimate_tensile_strength for s in specimens])),
        measured_uts_std=float(
            np.std([s.ultimate_tensile_strength for s in specimens], ddof=1)),
        yield_strength=float(parameters[0]),
        yield_strength_err=float(errors[0]),
        hardening_modulus=float(parameters[1]),
        hardening_modulus_err=float(errors[1]),
        hardening_exponent=float(parameters[2]),
        hardening_exponent_err=float(errors[2]),
        rms_residual=residual,
        strain_window=FIT_STRAIN_WINDOW,
        reproduces_nist=agrees,
    )


def write_fit(result: LudwikFit, path: Path = FIT_CACHE) -> None:
    """Cache a fit so the campaign reads it instead of refitting"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result._asdict(), indent=2))


def read_fit(path: Path = FIT_CACHE) -> LudwikFit:
    """Return the cached fit, or explain how to produce it"""
    if not path.is_file():
        raise FileNotFoundError(
            f"No cached fit at {path}. Run `python inconel718.py --download` "
            f"then `python inconel718.py --fit`. Inconel 718 hardening is fitted "
            f"to measured data and deliberately not hard coded in this module, "
            f"so there is nothing to fall back on."
        )
    # read field by field rather than splatting, so a cache left over from an
    # earlier version of LudwikFit fails here and not in the middle of a run
    fields = json.loads(path.read_text())
    window = fields["strain_window"]

    return LudwikFit(
        source=str(fields["source"]),
        condition=str(fields["condition"]),
        specimen_ids=tuple(str(t) for t in fields["specimen_ids"]),
        nominal_strain_rate=float(fields["nominal_strain_rate"]),
        points_fitted=int(fields["points_fitted"]),
        apparent_modulus=float(fields["apparent_modulus"]),
        apparent_modulus_std=float(fields["apparent_modulus_std"]),
        measured_yield_strength=float(fields["measured_yield_strength"]),
        measured_yield_strength_std=float(fields["measured_yield_strength_std"]),
        measured_uts=float(fields["measured_uts"]),
        measured_uts_std=float(fields["measured_uts_std"]),
        yield_strength=float(fields["yield_strength"]),
        yield_strength_err=float(fields["yield_strength_err"]),
        hardening_modulus=float(fields["hardening_modulus"]),
        hardening_modulus_err=float(fields["hardening_modulus_err"]),
        hardening_exponent=float(fields["hardening_exponent"]),
        hardening_exponent_err=float(fields["hardening_exponent_err"]),
        rms_residual=float(fields["rms_residual"]),
        strain_window=(float(window[0]), float(window[1])),
        reproduces_nist=bool(fields["reproduces_nist"]),
    )


def epicp_props(young_modulus: float = YOUNG_MODULUS,
                poisson_ratio: float = POISSON_RATIO,
                thermal_expansion: float = THERMAL_EXPANSION,
                hardening: Optional[Tuple[float, float]] = None,
                fit_path: Path = FIT_CACHE,
                ) -> npt.NDArray[np.float64]:
    """Build the EPICP property vector, [E, nu, alpha, Re, k, m]

    :param young_modulus: see YOUNG_MODULUS, not taken from the dataset
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
        young_modulus,
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

    return "\n".join([
        "Inconel 718, additively manufactured, AS BUILT, no heat treatment",
        f"  source          {fitted.source}",
        f"  fitted over     specimens {', '.join(fitted.specimen_ids)} at "
        f"{fitted.nominal_strain_rate:g} 1/s, {fitted.points_fitted} points, "
        f"plastic strain {fitted.strain_window[0]} to {fitted.strain_window[1]}",
        f"  E               {props[0]:.0f} MPa   ({YOUNG_MODULUS_NOTE.split('.')[0]})",
        f"  nu              {props[1]:.3f}            (literature, not in the dataset)",
        f"  alpha           {props[2]:.2e} 1/K      (literature, not in the dataset)",
        f"  Re              {props[3]:.1f} MPa   +/- {fitted.yield_strength_err:.1f} "
        f"(initial yield, not the proof stress)",
        f"  k               {props[4]:.1f} MPa   +/- {fitted.hardening_modulus_err:.1f}",
        f"  m               {props[5]:.4f}        +/- {fitted.hardening_exponent_err:.4f}",
        f"  fit residual    {fitted.rms_residual:.1f} MPa rms",
        f"  measured        0.2 percent proof {fitted.measured_yield_strength:.1f} "
        f"+/- {fitted.measured_yield_strength_std:.1f} MPa, "
        f"UTS {fitted.measured_uts:.1f} +/- {fitted.measured_uts_std:.1f} MPa",
        f"  check           reduction reproduces NIST's published answers: "
        f"{'yes' if fitted.reproduces_nist else 'NO'}",
        f"  note            the dataset's own modulus, "
        f"{fitted.apparent_modulus / 1000:.1f} GPa, is miniature specimen "
        f"extensometry and is not used",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIRECTORY, metavar="DIR",
                        help=f"where the dataset files are, default {DATA_DIRECTORY}")
    parser.add_argument("--download", action="store_true",
                        help=f"fetch the dataset (doi:{DOI}) into --data-dir")
    parser.add_argument("--fit", action="store_true",
                        help="refit from the raw files and cache the result")
    args = parser.parse_args()

    if args.download:
        written = download(args.data_dir)
        print(f"{len(written)} of 3 files present in {args.data_dir}")
        return

    if args.fit:
        write_fit(fit(args.data_dir, verbose=True))
        print(f"fit written to {FIT_CACHE}")
    print(describe())


if __name__ == "__main__":
    main()
