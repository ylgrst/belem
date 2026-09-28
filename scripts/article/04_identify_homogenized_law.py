"""
Step 4 of the article campaign: identify the homogenised law for every cell.

Adapted from
scripts/non_linear_homogenization/homogenized_law_identification.py, extended to
sweep the whole campaign and to shard across machines.

For each geometry and density this fits an EPCHG law, one isotropic and two
kinematic hardening terms, to the three cyclic load cases. The elastic part comes
from the effective stiffness tensor of step 2 and the yield surface from the
criterion parameters of step 3, so both must have run first.

    python 04_identify_homogenized_law.py --root /data/article --shard 0 --shards 4 --workers 4

Each identification is a differential evolution run that calls the simcoon solver
thousands of times, so this step is CPU bound and parallelises well. Note that
--workers multiplies with the optimiser's own --de-workers.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import article_config as cfg
import inconel718

# EPCHG hardening structure: one isotropic term (Q, b) and two kinematic ones
# (C_1, D_1, C_2, D_2), on top of the yield stress. This is the structure the
# existing belem example uses.
N_ISO_HARD = 1
N_KIN_HARD = 2

# Search bounds for the seven parameters, as multiples of the cell's own stress
# at 0.2 percent plastic strain. Expressing them relative to a measured stress is
# what lets one set of bounds cover 5 percent and 50 percent dense cells, whose
# homogenised yield stresses differ by more than an order of magnitude.
#
# Entries are (name, low, high, scales_with_stress). The dimensionless rate
# parameters b, D_1 and D_2 keep the absolute bounds of the original script.
PARAMETER_BOUNDS: Tuple[Tuple[str, float, float, bool], ...] = (
    ("sigmaY", 0.20, 1.20, True),
    ("Q", 0.01, 1.00, True),
    ("b", 1.0, 1000.0, False),
    ("C_1", 1.0, 400.0, True),
    ("D_1", 5.0, 1000.0, False),
    ("C_2", 0.5, 150.0, True),
    ("D_2", 10.0, 1000.0, False),
)

# Plastic strain, in percent, at which the reference stress above is read.
REFERENCE_PLASTIC_STRAIN = 0.2


def _reference_stress(directory: Path) -> float:
    """von Mises stress of the tension case at the reference plastic strain

    Used only to scale the search bounds, so a rough value is enough; it falls
    back to the last point of the curve when the cell never reaches the threshold.
    """
    import numpy as np

    stress = np.loadtxt(directory / "tension" / "vm_stress.txt")
    plastic = np.loadtxt(directory / "tension" / "vm_plastic_strain.txt")
    above = np.flatnonzero(plastic > REFERENCE_PLASTIC_STRAIN)
    index = int(above[0]) if above.size else len(plastic) - 1

    return float(stress[index])


def identify_one(job: cfg.Job, root: Path, force: bool, criterion: str,
                 maxiter: int, popsize: int, tol: float, de_workers: int) -> dict:
    """Identify the homogenised law of one geometry at one density"""
    import numpy as np
    from simcoon import simmit as sim
    from simcoon.parameter import Parameter

    from belem.fem.data import Data
    from belem.fem.identification import (
        plot_graph,
        plot_nrmse,
        prepare_epchg_identification,
        run_epchg_identification,
    )
    from belem.utils import InputColumnHeader, ResultsColumnHeader

    directory = job.directory(root)
    identification_dir = directory / "identification"
    if not force and cfg.is_done(identification_dir, "identification"):
        return {"job": job.label, "status": "skipped"}

    stiffness_file = directory / "effective_stiffness_tensor.txt"
    criterion_file = directory / f"{criterion}_params.txt"
    for required in (stiffness_file, criterion_file):
        if not required.is_file():
            raise FileNotFoundError(f"{required} is missing, run steps 2 and 3 first")

    started = time.time()

    # The linear homogenisation of step 2 already ran at the real bulk modulus,
    # so these are the effective moduli of the cell, with no rescaling needed.
    effective_young, effective_poisson, effective_shear = sim.L_cubic_props(
        np.loadtxt(stiffness_file))
    elastic_params = np.array([
        float(effective_young[0]),
        float(effective_poisson[0]),
        float(effective_shear[0]),
        inconel718.THERMAL_EXPANSION,
    ])
    criterion_params = np.atleast_1d(np.loadtxt(criterion_file))

    reference_stress = _reference_stress(directory)
    parameters = [
        Parameter(index, (low * (reference_stress if scales else 1.0),
                          high * (reference_stress if scales else 1.0)),
                  f"@{index}p", ["material.dat"])
        for index, (_, low, high, scales) in enumerate(PARAMETER_BOUNDS)
    ]

    data = []
    for load_case in cfg.IDENTIFICATION_LOAD_CASES:
        case_dir = directory / load_case
        # the postprocessing writes strain in percent, the identification wants it
        # dimensionless
        data.append(Data(
            control=np.loadtxt(case_dir / "strain.txt") / 100.0,
            observation=np.loadtxt(case_dir / "stress_component.txt"),
        ))

    # tension and biaxial tension are compared on S11, shear on S12
    columns_to_compare = [[ResultsColumnHeader.S11],
                          [ResultsColumnHeader.S11],
                          [ResultsColumnHeader.S12]]

    basedir = str(identification_dir) + "/"
    prepare_epchg_identification(
        data_to_identify=data,
        list_columns_to_compare=columns_to_compare,
        parameters_to_optimize=parameters,
        elastic_params=elastic_params,
        n_iso_hard=N_ISO_HARD,
        n_kin_hard=N_KIN_HARD,
        criteria=criterion,
        criteria_params=criterion_params,
        basedir=basedir,
    )

    # the cost function reads data/, exp_data/ and num_data/ relative to the
    # current directory, so the identification must run from inside basedir, and
    # each parallel identification therefore needs its own directory
    with cfg.run_in_directory(identification_dir):
        identified = run_epchg_identification(
            parameters_to_optimize=parameters,
            elastic_params=elastic_params,
            n_iso_hard=N_ISO_HARD,
            n_kin_hard=N_KIN_HARD,
            criteria=criterion,
            criteria_params=criterion_params,
            path_dir=basedir + "data/",
            num_dir=basedir + "num_data/",
            results_dir=basedir + "results_id/",
            popsize=popsize,
            tol=tol,
            maxiter=maxiter,
            workers=de_workers,
            disp=False,
        )

        labels = ["tension", "biaxial tension", "shear"]
        plot_columns = [[ResultsColumnHeader.E11, ResultsColumnHeader.S11],
                        [ResultsColumnHeader.E11, ResultsColumnHeader.S11],
                        [ResultsColumnHeader.E12, ResultsColumnHeader.S12]]
        exp_columns = [[InputColumnHeader.STRAIN, InputColumnHeader.STRESS]] * 3
        plot_graph(sim_list=labels, ident_data_columns_to_plot=plot_columns,
                   exp_data_columns_to_plot=exp_columns,
                   path_results_id=basedir + "results_id/", path_exp=basedir + "exp_data/")
        plot_nrmse(sim_list=labels, ident_data_columns_to_plot=plot_columns,
                   exp_data_columns_to_plot=exp_columns,
                   path_results_id=basedir + "results_id/", path_exp=basedir + "exp_data/")

    np.savetxt(identification_dir / "identified_parameters.txt", identified)
    info = {
        "geometry": job.geometry.slug,
        "density_percent": job.density_percent,
        "criterion": criterion,
        "effective_young": float(effective_young[0]),
        "effective_poisson": float(effective_poisson[0]),
        "effective_shear": float(effective_shear[0]),
        "reference_stress": reference_stress,
        "parameters": {name: float(value)
                       for (name, _, _, _), value in zip(PARAMETER_BOUNDS, identified)},
        "seconds": time.time() - started,
    }
    (identification_dir / "identification_summary.json").write_text(json.dumps(info, indent=2))
    cfg.write_status(identification_dir, "identification", info)

    return {"job": job.label, "status": "done", **info}


def _worker(payload: Tuple) -> dict:
    job, root, force, criterion, maxiter, popsize, tol, de_workers = payload
    try:
        return identify_one(job, root, force, criterion, maxiter, popsize, tol, de_workers)
    except Exception:
        return {"job": job.label, "status": "failed", "error": traceback.format_exc()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg.add_common_arguments(parser)
    parser.add_argument("--criterion", default="dfa",
                        choices=["hill", "dfa", "anisotropic"],
                        help="yield criterion identified in step 3")
    parser.add_argument("--maxiter", type=int, default=100,
                        help="differential evolution generations")
    parser.add_argument("--popsize", type=int, default=10,
                        help="differential evolution population multiplier")
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="differential evolution relative tolerance")
    parser.add_argument("--de-workers", type=int, default=1,
                        help="processes inside one differential evolution run; "
                             "multiplies with --workers")
    args = parser.parse_args()

    jobs = cfg.select_jobs(args.shard, args.shards, args.only, args.densities)
    cfg.report_selection(jobs, args, "law identification")
    if args.dry_run:
        return

    payloads = [(job, args.root, args.force, args.criterion,
                 args.maxiter, args.popsize, args.tol, args.de_workers) for job in jobs]
    failures: List[dict] = []
    done = 0

    if args.workers <= 1:
        for payload in payloads:
            done += 1
            _report(_worker(payload), done, len(jobs), failures)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                                 max_tasks_per_child=1) as pool:
            futures = [pool.submit(_worker, p) for p in payloads]
            for future in as_completed(futures):
                done += 1
                _report(future.result(), done, len(jobs), failures)

    print(f"\nidentified {len(jobs) - len(failures)} of {len(jobs)}, {len(failures)} failed")
    if failures:
        report = args.root / f"identification_failures_shard{args.shard}.log"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n\n".join(f"{f['job']}\n{f['error']}" for f in failures))
        print(f"failure details written to {report}")
        sys.exit(1)


def _report(result: dict, done: int, total: int, failures: List[dict]) -> None:
    if result["status"] == "failed":
        failures.append(result)
        print(f"[{done}/{total}] FAILED  {result['job']}", flush=True)
    elif result["status"] == "skipped":
        print(f"[{done}/{total}] skipped {result['job']}", flush=True)
    else:
        print(f"[{done}/{total}] {result['job']:<44s} "
              f"sigmaY {result['parameters']['sigmaY']:8.2f} MPa  "
              f"{result['seconds']:7.1f}s", flush=True)


if __name__ == "__main__":
    main()
