"""
File driven identification cost function.

Vendored from Simcoon's C++ ``calc_cost``, which was removed in simcoon 2.0
(``simcoon.identify.calc_cost`` replaces it, but takes in-memory arrays and a
different set of metrics). This reimplementation keeps the original file based
behaviour so that the ``data/files_exp.inp``, ``data/files_weights.inp`` and
``data/files_num.inp`` input files keep driving the cost exactly as before.
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

# Simcoon's parameter.hpp threshold below which a weight is treated as zero
IOTA = 1.0e-12


class _OptiData:
    """Column selection and data of one experimental or numerical result file"""

    def __init__(self) -> None:
        self.name: str = ""
        self.ndata: int = 0
        self.ninfo: int = 0
        self.ncolumns: int = 0
        self.skiplines: int = 0
        self.c_data: npt.NDArray[np.int_] = np.zeros(0, dtype=int)
        self.data: npt.NDArray[np.float64] = np.zeros((0, 0))

    def construct_c_data(self) -> None:
        self.c_data = np.zeros(self.ninfo, dtype=int)

    def read(self, folder: str) -> None:
        """Reads the columns listed in c_data from folder/name"""
        path = folder.rstrip("/") + "/" + self.name
        with open(path) as file:
            lines = file.read().splitlines()

        self.ndata = sum(1 for line in lines if line.strip() != "") - self.skiplines
        if self.ndata < 0:
            self.ndata = 0

        # Simcoon reads a flat whitespace separated token stream, ndata rows of
        # ncolumns values each, after dropping the first skiplines lines
        tokens = " ".join(lines[self.skiplines:]).split()
        n_values = self.ndata * self.ncolumns
        values = np.asarray(tokens[:n_values], dtype=float)
        if values.size < n_values:
            self.ndata = values.size // self.ncolumns
            values = values[: self.ndata * self.ncolumns]

        self.data = values.reshape(self.ndata, self.ncolumns)[:, self.c_data]


class _TokenStream:
    """Whitespace separated token reader mimicking C++ ifstream extraction"""

    def __init__(self, path: str) -> None:
        with open(path) as file:
            self._tokens = file.read().split()
        self._index = 0

    def next(self) -> Optional[str]:
        if self._index >= len(self._tokens):
            return None
        token = self._tokens[self._index]
        self._index += 1
        return token

    def next_int(self, default: int = 0) -> int:
        token = self.next()
        return default if token is None else int(token)

    def next_float(self, default: float = 0.0) -> float:
        token = self.next()
        return default if token is None else float(token)


def _read_data_exp(nfiles: int, data_dir: str) -> List[_OptiData]:
    stream = _TokenStream(data_dir.rstrip("/") + "/files_exp.inp")
    datas = [_OptiData() for _ in range(nfiles)]

    stream.next()  # #Name_of_the_exp_files
    for data in datas:
        data.name = stream.next() or ""

    stream.next()  # #EXP_Nb_columns_in_files
    for data in datas:
        data.ncolumns = stream.next_int()

    stream.next()  # #EXP_Nb_columns_to_identify
    for data in datas:
        data.ninfo = stream.next_int()
        data.construct_c_data()

    stream.next()  # #EXP_colums_to_identify
    for data in datas:
        for j in range(data.ninfo):
            data.c_data[j] = stream.next_int()
            if not 0 <= data.c_data[j] <= data.ncolumns:
                raise ValueError(
                    f"Column {data.c_data[j]} to identify is out of range for "
                    f"experimental file '{data.name}' ({data.ncolumns} columns)"
                )

    # Simcoon's writer does not emit a skiplines section, in which case the
    # stream runs out and skiplines stays at 0
    stream.next()
    for data in datas:
        data.skiplines = stream.next_int()

    return datas


def _read_data_weights(
    nfiles: int, data_dir: str, data_exp: List[_OptiData]
) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.float64],
           List[npt.NDArray[np.float64]], List[_OptiData]]:
    stream = _TokenStream(data_dir.rstrip("/") + "/files_weights.inp")

    weight_types = np.zeros(3, dtype=int)
    weight_files = np.zeros(nfiles)
    weight_cols: List[npt.NDArray[np.float64]] = [np.zeros(0) for _ in range(nfiles)]

    weights = []
    for exp in data_exp:
        weight = _OptiData()
        weight.name = exp.name
        weight.ndata = exp.ndata
        weight.ninfo = exp.ninfo
        weight.ncolumns = exp.ncolumns
        weight.skiplines = exp.skiplines
        weight.construct_c_data()
        weights.append(weight)

    # Weight type 1: one weight per data file
    stream.next()
    stream.next()
    weight_types[0] = stream.next_int()
    if weight_types[0] == 0:
        stream.next()
    elif weight_types[0] == 1:
        stream.next()
        for i in range(nfiles):
            weight_files[i] = stream.next_float()
    else:
        raise ValueError("Weight type 1 (weight per file) must be 0 or 1")

    # Weight type 2: one weight per data column
    stream.next()
    stream.next()
    weight_types[1] = stream.next_int()
    if weight_types[1] in (0, 1):
        stream.next()
    elif weight_types[1] in (2, 3):
        stream.next()
        for i in range(nfiles):
            weight_cols[i] = np.zeros(weights[i].ninfo)
            for j in range(weights[i].ninfo):
                weight_cols[i][j] = stream.next_float()
    else:
        raise ValueError("Weight type 2 (weight per column) must be 0, 1, 2 or 3")

    # Weight type 3: one weight per data point, read from a column of the exp file
    stream.next()
    stream.next()
    weight_types[2] = stream.next_int()
    if weight_types[2] == 0:
        stream.next()
    elif weight_types[2] == 1:
        stream.next()
        for weight in weights:
            for j in range(weight.ninfo):
                weight.c_data[j] = stream.next_int()
                if not 0 < weight.c_data[j] <= weight.ncolumns:
                    raise ValueError(
                        f"Weight column {weight.c_data[j]} is out of range for "
                        f"file '{weight.name}' ({weight.ncolumns} columns)"
                    )
    else:
        raise ValueError("Weight type 3 (weight per point) must be 0 or 1")

    return weight_types, weight_files, weight_cols, weights


def _read_data_num(nfiles: int, data_dir: str, data_exp: List[_OptiData]) -> List[_OptiData]:
    stream = _TokenStream(data_dir.rstrip("/") + "/files_num.inp")
    datas = [_OptiData() for _ in range(nfiles)]

    stream.next()  # NUMNb_columnsinfiles
    for i, data in enumerate(datas):
        data.ncolumns = stream.next_int()
        data.ninfo = data_exp[i].ninfo
        data.construct_c_data()

    stream.next()  # NUMNb_colums_to_identify
    for data in datas:
        for j in range(data.ninfo):
            data.c_data[j] = stream.next_int()
            if not 0 < data.c_data[j] <= data.ncolumns:
                raise ValueError(
                    f"Column {data.c_data[j]} to identify is out of range for a "
                    f"numerical file with {data.ncolumns} columns"
                )

    # As for files_exp.inp, a missing skiplines section leaves skiplines at 0
    stream.next()
    for data in datas:
        data.skiplines = stream.next_int()

    return datas


def _calc_v(datas: List[_OptiData], data_exp: List[_OptiData], sizev: int) -> npt.NDArray[np.float64]:
    """Flattens the identified columns, zero padding files shorter than the exp ones"""
    v = np.zeros(sizev)
    z = 0
    for data, exp in zip(datas, data_exp):
        n_common = min(data.ndata, exp.ndata)
        for column in range(exp.ninfo):
            v[z:z + n_common] = data.data[:n_common, column]
            z += exp.ndata  # the tail stays at zero
    return v


def _calc_w(sizev: int, weight_types: npt.NDArray[np.int_], weight_files: npt.NDArray[np.float64],
            weight_cols: List[npt.NDArray[np.float64]], weights: List[_OptiData],
            data_exp: List[_OptiData]) -> npt.NDArray[np.float64]:
    w = np.ones(sizev)

    # Weight per file
    if weight_types[0] == 1:
        z = 0
        for i, exp in enumerate(data_exp):
            for _ in range(exp.ninfo):
                w[z:z + exp.ndata] *= weight_files[i]
                z += exp.ndata

    # Weight per column
    if weight_types[1] in (1, 2, 3):
        z = 0
        for i, exp in enumerate(data_exp):
            for column in range(exp.ninfo):
                if weight_types[1] == 3:
                    factor = weight_cols[i][column]
                else:
                    denom = np.sum(exp.data[:, column] ** 2)
                    factor = (1.0 / denom if weight_types[1] == 1
                              else weight_cols[i][column] / denom)
                w[z:z + exp.ndata] *= factor
                z += exp.ndata

    # Weight per data point
    if weight_types[2] == 1:
        z = 0
        for i, exp in enumerate(data_exp):
            for column in range(exp.ninfo):
                w[z:z + exp.ndata] *= np.abs(weights[i].data[:exp.ndata, column])
                z += exp.ndata

    return w


def _calc_c(vexp: npt.NDArray[np.float64], vnum: npt.NDArray[np.float64],
            w: npt.NDArray[np.float64]) -> float:
    if vnum.size < vexp.size:
        vnum = np.zeros(vexp.size)
    significant = w > IOTA

    return float(np.sum(((vexp[significant] - vnum[significant]) ** 2) * w[significant]))


def calc_cost(num_data_names: Sequence[str], data_dir: str = "data",
              exp_data_dir: str = "exp_data", num_data_dir: str = "num_data") -> float:
    """
    Computes the weighted least squares cost between experimental and numerical results

    :param num_data_names: names of the numerical result files, in the same order as the
    experimental files listed in files_exp.inp
    :param data_dir: directory holding files_exp.inp, files_weights.inp and files_num.inp
    :param exp_data_dir: directory holding the experimental data files
    :param num_data_dir: directory holding the numerical result files
    """
    nfiles = len(num_data_names)

    data_exp = _read_data_exp(nfiles, data_dir)
    for exp in data_exp:
        exp.read(exp_data_dir)

    weight_types, weight_files, weight_cols, weights = _read_data_weights(
        nfiles, data_dir, data_exp
    )
    if weight_types[2] == 1:
        for weight in weights:
            weight.read(exp_data_dir)

    data_num = _read_data_num(nfiles, data_dir, data_exp)
    for data, name in zip(data_num, num_data_names):
        data.name = name
        data.read(num_data_dir)

    sizev = sum(exp.ndata * exp.ninfo for exp in data_exp)

    vexp = _calc_v(data_exp, data_exp, sizev)
    vnum = _calc_v(data_num, data_exp, sizev)
    w = _calc_w(sizev, weight_types, weight_files, weight_cols, weights, data_exp)

    return _calc_c(vexp, vnum, w)
