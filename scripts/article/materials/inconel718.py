"""
Inconel 718 bulk material properties for the article campaign.

Values are transcribed from the Renishaw RenAM 500 series material data sheet
for Inconel 718, part no. H-5800-6794-03-A, issued 08.2024. Page numbers below
refer to that document.

Two quantities the EPICP law needs are not in the data sheet and are derived or
taken from literature. Both are flagged in place and are easy to override.
"""
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
import numpy.typing as npt


class ParameterSet(NamedTuple):
    """One Renishaw parameter set, in one build orientation, one heat treatment

    :param label: human readable identifier
    :param page: page of the data sheet the values come from
    :param condition: heat treatment condition
    :param orientation: build orientation of the test samples
    :param young_modulus: modulus of elasticity in MPa
    :param yield_strength: 0.2 percent proof stress in MPa
    :param ultimate_tensile_strength: engineering UTS in MPa
    :param elongation_after_fracture: total engineering elongation, dimensionless
    """

    label: str
    page: int
    condition: str
    orientation: str
    young_modulus: float
    yield_strength: float
    ultimate_tensile_strength: float
    elongation_after_fracture: float


# Only the sets that report a modulus of elasticity are listed: the EPICP law
# needs one, and the 30 um modulated and the 90 and 120 um sets do not give it
# for both orientations.
PARAMETER_SETS: Dict[str, ParameterSet] = {
    "30um_cw_sta_xy": ParameterSet(
        "30 um, single laser, continuous wave, solution treated and aged, horizontal",
        5, "solution treated and aged", "horizontal (XY)", 207.0e3, 1350.0, 1539.0, 0.17),
    "30um_cw_sta_z": ParameterSet(
        "30 um, single laser, continuous wave, solution treated and aged, vertical",
        5, "solution treated and aged", "vertical (Z)", 190.0e3, 1256.0, 1448.0, 0.24),
    "60um_cw_asbuilt_xy": ParameterSet(
        "60 um, single laser, continuous wave, as built, horizontal",
        6, "as built", "horizontal (XY)", 184.0e3, 735.0, 1051.0, 0.33),
    "60um_cw_asbuilt_z": ParameterSet(
        "60 um, single laser, continuous wave, as built, vertical meander",
        6, "as built", "vertical (Z) meander", 187.0e3, 632.0, 1002.0, 0.36),
    "60um_cw_sta_xy": ParameterSet(
        "60 um, single laser, continuous wave, solution treated and aged, horizontal",
        6, "solution treated and aged", "horizontal (XY)", 195.0e3, 1275.0, 1480.0, 0.18),
    "60um_cw_sta_z": ParameterSet(
        "60 um, single laser, continuous wave, solution treated and aged, vertical meander",
        6, "solution treated and aged", "vertical (Z) meander", 198.0e3, 1201.0, 1440.0, 0.18),
}

# The campaign runs on one bulk material. The 60 um solution treated and aged
# horizontal set is the default: 60 um is the common production layer thickness,
# and that page is the only one reporting both as built and heat treated values
# in both orientations, so the choice can be revisited without changing source.
DEFAULT_PARAMETER_SET = "60um_cw_sta_xy"

# Not in the Renishaw data sheet. NASA CR-123774 Fig 7.21 gives Poisson's ratio
# against temperature for Alloy 718; the room temperature value reads about 0.29.
# The scan has lost the decimal point on that axis, but 0.29 is also the standard
# wrought value and is consistent with Fig 7.241, which puts G near 77 GPa.
POISSON_RATIO = 0.29

# Page 2 of the Renishaw sheet gives a wrought coefficient of 12e-6 to 16e-6 per K
# over 25 to 760 degrees C. NASA CR-123774 is more specific, 13.1e-6 over 20 to
# 100 degrees C, which is the range the campaign runs in.
THERMAL_EXPANSION = 13.0e-6

# Hardening of the EPICP law, sigma_y(p) = Re + k * p**m.
#
# The Renishaw data sheet reports no hardening law, only a proof stress, a UTS
# and an elongation. Those three cannot determine three Ludwik parameters: a
# Considere construction over them leaves one degree of freedom, and starting
# the solver from different guesses reproduces both reported values exactly
# while giving m anywhere from 0.07 to 0.43. A measured curve is needed.
#
# NASA CR-123774 Fig 7.4121 is a room temperature tensile curve for aged Alloy
# 718. Digitising it and fitting Ludwik to true stress against true plastic
# strain gives the values below, root mean square residual 5.8 MPa over nine
# points. As a check, the fit implies a 0.2 percent offset proof stress of
# 1273 MPa against the 1275 MPa Renishaw reports, from an independent source
# fifty years apart.
#
# Re is the initial yield surface, not the proof stress. Passing the data
# sheet's 1275 MPa in as Re, as an earlier version of this module did, puts the
# model's own proof stress at 1346 MPa, 5.6 percent above the intended target.
#
# Two caveats. The curve is wrought rather than additively manufactured;
# Gallmeyer et al. (arXiv:1909.12290) Fig 3 plots wrought IN718 among LPBF
# IN718 heat treated curves and they lie together, which is what justifies the
# substitution. And the uncertainties are regression scatter only, not the
# error of reading a 1972 scan by eye.
YIELD_STRENGTH = 1192.6        # Re, MPa, +/- 26.4 from the fit
HARDENING_MODULUS = 1113.6     # k,  MPa, +/- 53.4
HARDENING_EXPONENT = 0.4236    # m,       +/- 0.0441

HARDENING_REFERENCE = (
    "NASA CR-123774, Materials Data Handbook: Inconel Alloy 718, 2nd edition, "
    "Muraca and Whittick, April 1972, Fig 7.4121, room temperature curve"
)


def epicp_props(
    parameter_set: str = DEFAULT_PARAMETER_SET,
    poisson_ratio: float = POISSON_RATIO,
    thermal_expansion: float = THERMAL_EXPANSION,
    hardening: Optional[Tuple[float, float]] = None,
) -> npt.NDArray[np.float64]:
    """Build the EPICP property vector for one Renishaw parameter set

    Simcoon's EPICP law takes [E, nu, alpha, Re, k, m]. The modulus comes from
    the Renishaw sheet, the hardening from the NASA curve fit above.

    :param parameter_set: key into PARAMETER_SETS, selects the modulus
    :param poisson_ratio: see POISSON_RATIO
    :param thermal_expansion: see THERMAL_EXPANSION
    :param hardening: explicit (k, m) replacing the NASA fit
    :return: the six EPICP properties
    """
    if parameter_set not in PARAMETER_SETS:
        raise ValueError(
            f"Unknown parameter set '{parameter_set}'. "
            f"Available: {', '.join(sorted(PARAMETER_SETS))}"
        )
    material = PARAMETER_SETS[parameter_set]
    hardening_modulus, hardening_exponent = (
        hardening if hardening is not None
        else (HARDENING_MODULUS, HARDENING_EXPONENT))

    return np.array([
        material.young_modulus,
        poisson_ratio,
        thermal_expansion,
        YIELD_STRENGTH,
        hardening_modulus,
        hardening_exponent,
    ])


def describe(parameter_set: str = DEFAULT_PARAMETER_SET) -> str:
    """Return a readable summary of the properties actually used"""
    material = PARAMETER_SETS[parameter_set]
    props = epicp_props(parameter_set)
    proof = props[3] + props[4] * 0.002 ** props[5]

    return "\n".join([
        "Inconel 718, additively manufactured, solution treated and aged",
        f"  modulus from    Renishaw RenAM 500 data sheet H-5800-6794-03-A, page {material.page}",
        f"  hardening from  {HARDENING_REFERENCE}",
        f"  parameter set   {material.label}",
        f"  E               {props[0]:.0f} MPa",
        f"  nu              {props[1]:.3f}",
        f"  alpha           {props[2]:.2e} 1/K",
        f"  Re              {props[3]:.1f} MPa   (initial yield, not the proof stress)",
        f"  k               {props[4]:.1f} MPa",
        f"  m               {props[5]:.4f}",
        f"  check           implied 0.2 percent proof stress {proof:.0f} MPa, "
        f"data sheet reports {material.yield_strength:.0f}",
    ])


if __name__ == "__main__":
    print(describe())
