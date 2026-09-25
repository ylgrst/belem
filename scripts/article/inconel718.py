"""
Inconel 718 bulk material properties for the article campaign.

Values are transcribed from the Renishaw RenAM 500 series material data sheet
for Inconel 718, part no. H-5800-6794-03-A, issued 08.2024. Page numbers below
refer to that document.

Two quantities the EPICP law needs are not in the data sheet and are derived or
taken from literature. Both are flagged in place and are easy to override.
"""
from typing import Dict, NamedTuple, Tuple

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

# NOT IN THE DATA SHEET. Poisson ratio of wrought Inconel 718 is 0.29 to 0.30 in
# the literature; 0.3 is used here, as in the existing belem scripts.
POISSON_RATIO = 0.3

# NOT IN THE DATA SHEET as a single value. Page 2 gives a wrought coefficient of
# thermal expansion of 12e-6 to 16e-6 per K over 25 to 760 degrees C. The low end
# of that band applies near room temperature, which is where the campaign runs.
THERMAL_EXPANSION = 13.0e-6

# Fraction of the elongation after fracture taken as uniform elongation, i.e. the
# strain at which necking starts.
#
# THIS IS AN ASSUMPTION, NOT A DATA SHEET VALUE, and it is the single least
# certain input of the campaign. The data sheet reports elongation after
# fracture, which includes post necking deformation, while the Considere
# construction below needs the uniform elongation. For a precipitation hardened
# nickel superalloy the uniform part is roughly half to two thirds of the total.
# Set it to 1.0 to use the full reported elongation, which maximises the fitted
# hardening. Better: replace k and m outright with values fitted to a measured
# stress strain curve, via epicp_props(..., hardening=(k, m)).
UNIFORM_ELONGATION_FRACTION = 0.6


def fit_ludwik_hardening(
    young_modulus: float,
    yield_strength: float,
    ultimate_tensile_strength: float,
    uniform_elongation: float,
) -> Tuple[float, float]:
    """Fit the Ludwik hardening of the EPICP law to a tensile test summary

    EPICP hardens as sigma_y(p) = Re + k * p**m, with p the accumulated plastic
    strain. Two conditions at the onset of necking determine k and m:

    * the true stress there equals Re + k * p_u**m
    * the Considere criterion, d(sigma)/d(epsilon) = sigma, which for this law
      reads k * m * p_u**(m - 1) = sigma_u

    Dividing one by the other gives m directly, then k follows.

    :param young_modulus: bulk modulus in MPa
    :param yield_strength: 0.2 percent proof stress in MPa
    :param ultimate_tensile_strength: engineering UTS in MPa
    :param uniform_elongation: engineering strain at the UTS, dimensionless
    :return: the pair (k, m), k in MPa and m dimensionless
    """
    if uniform_elongation <= 0.0:
        raise ValueError("uniform_elongation must be strictly positive")

    true_stress_at_necking = ultimate_tensile_strength * (1.0 + uniform_elongation)
    if true_stress_at_necking <= yield_strength:
        raise ValueError(
            "the true stress at necking must exceed the yield strength; "
            f"got {true_stress_at_necking:.1f} MPa against {yield_strength:.1f} MPa"
        )

    true_strain_at_necking = np.log(1.0 + uniform_elongation)
    plastic_strain_at_necking = true_strain_at_necking - true_stress_at_necking / young_modulus
    if plastic_strain_at_necking <= 0.0:
        raise ValueError("the plastic strain at necking came out non positive")

    hardening_exponent = (
        plastic_strain_at_necking * true_stress_at_necking
        / (true_stress_at_necking - yield_strength)
    )
    hardening_modulus = (
        (true_stress_at_necking - yield_strength)
        / plastic_strain_at_necking ** hardening_exponent
    )

    return float(hardening_modulus), float(hardening_exponent)


def epicp_props(
    parameter_set: str = DEFAULT_PARAMETER_SET,
    poisson_ratio: float = POISSON_RATIO,
    thermal_expansion: float = THERMAL_EXPANSION,
    uniform_elongation_fraction: float = UNIFORM_ELONGATION_FRACTION,
    hardening: Tuple[float, float] = None,
) -> npt.NDArray[np.float64]:
    """Build the EPICP property vector for one Renishaw parameter set

    Simcoon's EPICP law takes [E, nu, alpha, Re, k, m].

    :param parameter_set: key into PARAMETER_SETS
    :param poisson_ratio: not a data sheet value, see POISSON_RATIO
    :param thermal_expansion: not a single data sheet value, see THERMAL_EXPANSION
    :param uniform_elongation_fraction: see UNIFORM_ELONGATION_FRACTION
    :param hardening: explicit (k, m) that bypasses the Considere fit entirely
    :return: the six EPICP properties
    """
    if parameter_set not in PARAMETER_SETS:
        raise ValueError(
            f"Unknown parameter set '{parameter_set}'. "
            f"Available: {', '.join(sorted(PARAMETER_SETS))}"
        )
    material = PARAMETER_SETS[parameter_set]

    if hardening is None:
        hardening = fit_ludwik_hardening(
            young_modulus=material.young_modulus,
            yield_strength=material.yield_strength,
            ultimate_tensile_strength=material.ultimate_tensile_strength,
            uniform_elongation=uniform_elongation_fraction * material.elongation_after_fracture,
        )
    hardening_modulus, hardening_exponent = hardening

    return np.array([
        material.young_modulus,
        poisson_ratio,
        thermal_expansion,
        material.yield_strength,
        hardening_modulus,
        hardening_exponent,
    ])


def describe(parameter_set: str = DEFAULT_PARAMETER_SET) -> str:
    """Return a readable summary of the properties actually used"""
    material = PARAMETER_SETS[parameter_set]
    props = epicp_props(parameter_set)
    uniform_elongation = UNIFORM_ELONGATION_FRACTION * material.elongation_after_fracture

    # reconstruct the UTS the fitted law predicts, as a sanity check
    plastic_strain = np.log(1.0 + uniform_elongation) - props[3] / props[0]
    true_stress = props[3] + props[4] * plastic_strain ** props[5]
    predicted_uts = true_stress / (1.0 + uniform_elongation)

    return "\n".join([
        f"Inconel 718, Renishaw RenAM 500 series data sheet H-5800-6794-03-A, page {material.page}",
        f"  parameter set     {material.label}",
        f"  condition         {material.condition}, {material.orientation}",
        f"  E                 {props[0]:.0f} MPa        (data sheet)",
        f"  nu                {props[1]:.3f}             (literature, not in the data sheet)",
        f"  alpha             {props[2]:.2e} 1/K      (data sheet band 12e-6 to 16e-6)",
        f"  Re                {props[3]:.0f} MPa         (data sheet yield strength)",
        f"  k                 {props[4]:.1f} MPa        (fitted, not in the data sheet)",
        f"  m                 {props[5]:.4f}            (fitted, not in the data sheet)",
        f"  fit inputs        UTS {material.ultimate_tensile_strength:.0f} MPa, "
        f"elongation after fracture {100 * material.elongation_after_fracture:.0f} percent",
        f"                    uniform elongation taken as {100 * uniform_elongation:.1f} percent "
        f"({UNIFORM_ELONGATION_FRACTION:.2f} of it)",
        f"  fit check         law predicts UTS {predicted_uts:.0f} MPa "
        f"against {material.ultimate_tensile_strength:.0f} MPa reported",
    ])


if __name__ == "__main__":
    print(describe())
