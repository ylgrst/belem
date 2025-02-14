from enum import IntEnum

class ResultsColumnHeader(IntEnum):
    PHASE = 0
    BLOCK = 1
    STEP = 2
    INCREMENT = 3
    TIME = 4
    TEMPERATURE = 5
    SPECIFIC_HEAT = 6
    PRODUCED_HEAT = 7
    E11 = 8
    E22 = 9
    E33 = 10
    E12 = 11
    E13 = 12
    E23 = 13
    S11 = 14
    S22 = 15
    S33 = 16
    S12 = 17
    S13 = 18
    S23 = 19
    TOTAL_DEFORMATION_ENERGY = 20
    REVERSIBLE_DEFORMATION_ENERGY = 21
    IRREVERSIBLE_DEFORMATION_ENERGY = 22
    DISSIPATED_DEFORMATION_ENERGY = 23

class InputColumnHeader(IntEnum):
    INCREMENT = 0
    TIME = 1
    STRAIN = 2
    STRESS = 3