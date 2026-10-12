"""
Bulk materials available to the article campaign.

One module per alloy, each exposing the same three names so the campaign can be
switched from one to the other by name alone:

    THERMAL_EXPANSION   coefficient the homogenised law of step 4 carries over
    epicp_props()       the six EPICP properties, [E, nu, alpha, Re, k, m]
    describe()          a readable summary, printed in every run's log

The two alloys are not drop in substitutes for each other physically. Inconel
718 is solution treated and aged, with a high yield and modest hardening;
Inconel 625 is as built, with a much lower yield and far more hardening
capacity. Which one a result came from therefore has to be reported, which is
why describe() is printed by the FEA step rather than kept to this module.

Select one in article_config.MATERIAL, not here, so that every step of the
campaign and every shard reads the same setting.
"""
from __future__ import annotations

import importlib
from typing import Callable, Protocol, Tuple, cast

import numpy as np
import numpy.typing as npt

MATERIALS: Tuple[str, ...] = ("inconel718", "inconel625")


class Material(Protocol):
    """The interface every material module provides

    Declared as callable attributes rather than as methods because the objects
    matched against this are modules, whose functions take no implicit self.
    """

    THERMAL_EXPANSION: float
    epicp_props: Callable[[], npt.NDArray[np.float64]]
    describe: Callable[[], str]


def get(name: str) -> Material:
    """Import one material module by name

    :param name: a key of MATERIALS
    :return: the module, as a Material
    """
    if name not in MATERIALS:
        raise ValueError(
            f"Unknown material '{name}'. Available: {', '.join(MATERIALS)}"
        )

    return cast(Material, importlib.import_module(f"{__name__}.{name}"))
