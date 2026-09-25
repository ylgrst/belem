"""
Step 2 of the article campaign: run every FEA computation.

Adapted from scripts/non_linear_homogenization/run_all_fea_computations.py,
extended to sweep the whole campaign and to shard across machines.

For each geometry and density this runs

* one linear homogenisation, giving the effective stiffness tensor that step 4
  needs to build the elastic part of the homogenised law
* the nine non linear load cases: six monotonic ones that feed the yield surface
  identification, and three cyclic ones that feed the homogenised law

The bulk material is Inconel 718 from the Renishaw data sheet, see inconel718.py.

Sharding is per computation rather than per geometry, so a machine that draws
several expensive load cases is not left behind. Across four machines:

    python 02_run_all_fea_computations.py --root /data/article --shard 0 --shards 4 --workers 4
    python 02_run_all_fea_computations.py --root /data/article --shard 1 --shards 4 --workers 4
    ...

Pick --workers to suit memory rather than core count: each process holds a full
factorised stiffness matrix.
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, NamedTuple, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import article_config as cfg
import inconel718

LINEAR_TASK = "linear_homogenisation"


class Task(NamedTuple):
    """One computation: either the linear homogenisation, or one load case"""

    job: cfg.Job
    load_case: str

    @property
    def label(self) -> str:
        return f"{self.job.geometry.slug} {self.job.density_percent}% {self.load_case}"


def build_tasks(jobs: List[cfg.Job], load_cases: List[str], with_linear: bool) -> List[Task]:
    """Expand jobs into individual computations

    The linear homogenisation comes first for each job: it is by far the cheapest
    and step 4 cannot start without it.
    """
    tasks: List[Task] = []
    for job in jobs:
        if with_linear:
            tasks.append(Task(job, LINEAR_TASK))
        for load_case in load_cases:
            tasks.append(Task(job, load_case))
    return tasks


def _mesh_file(task: Task, root: Path) -> Path:
    mesh = task.job.directory(root) / "mesh" / "remeshed.vtk"
    if not mesh.is_file():
        raise FileNotFoundError(
            f"{mesh} is missing, run 01_generate_meshes.py for this geometry first"
        )
    return mesh


def run_linear(task: Task, root: Path) -> dict:
    """Effective stiffness tensor of the unit cell

    Run at the real bulk modulus rather than at a reference value and rescaled,
    so the tensor written out is directly the effective stiffness of the cell.
    """
    import numpy as np
    from belem.fem.fea import run_linear_homogenization

    directory = task.job.directory(root)
    output = directory / "effective_stiffness_tensor.txt"
    props = inconel718.epicp_props()

    started = time.time()
    # the working directory is per job: fedoo writes scratch files into it
    with cfg.run_in_directory(directory / "scratch_linear"):
        stiffness = run_linear_homogenization(
            mesh_filename=str(_mesh_file(task, root)),
            young_modulus=float(props[0]),
            poisson_ratio=float(props[1]),
        )
    np.savetxt(output, stiffness)

    return {"seconds": time.time() - started}


def run_load_case(task: Task, root: Path) -> dict:
    """One non linear load case, written where step 3 expects to find it"""
    from belem.fem.fea import Load, run_fea_computation

    directory = task.job.directory(root) / task.load_case
    props = inconel718.epicp_props()
    loads = [Load("Dirichlet", node_id, variables, values)
             for node_id, variables, values in cfg.LOAD_CASES[task.load_case]]

    started = time.time()
    # each computation gets its own working directory: fedoo writes a temporary
    # _mesh_.npz into the current directory while building the fdz archive, and
    # two computations sharing a directory will destroy each other's copy
    with cfg.run_in_directory(directory):
        run_fea_computation(
            mesh_filename=str(_mesh_file(task, root)),
            material_law="EPICP",
            props=props,
            results_dir=str(directory),
            output_file_name=task.load_case,
            load_list=loads,
        )

    return {"seconds": time.time() - started}


def run_one(task: Task, root: Path, force: bool) -> dict:
    """Run one computation, skipping it when its output is already present"""
    if task.load_case == LINEAR_TASK:
        marker_dir = task.job.directory(root)
        step = "linear"
    else:
        marker_dir = task.job.directory(root) / task.load_case
        step = f"fea_{task.load_case}"

    if not force and cfg.is_done(marker_dir, step):
        return {"task": task.label, "status": "skipped"}

    if task.load_case == LINEAR_TASK:
        info = run_linear(task, root)
    else:
        info = run_load_case(task, root)

    cfg.write_status(marker_dir, step, {"task": task.label, **info})

    return {"task": task.label, "status": "done", **info}


def _worker(payload: Tuple[Task, Path, bool]) -> dict:
    task, root, force = payload
    try:
        return run_one(task, root, force)
    except Exception:
        return {"task": task.label, "status": "failed", "error": traceback.format_exc()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg.add_common_arguments(parser)
    parser.add_argument("--load-cases", nargs="+", default=list(cfg.LOAD_CASES),
                        choices=list(cfg.LOAD_CASES), metavar="NAME",
                        help="restrict to these load cases")
    parser.add_argument("--no-linear", action="store_true",
                        help="skip the linear homogenisation, which step 4 requires")
    args = parser.parse_args()

    jobs = cfg.select_jobs(0, 1, args.only, args.densities)
    tasks = build_tasks(jobs, args.load_cases, not args.no_linear)
    # shard the computations, not the geometries, so the load balances better
    mine = tasks[args.shard::args.shards]

    print(f"FEA: {len(mine)} of {len(tasks)} computations "
          f"(shard {args.shard + 1} of {args.shards}, {args.workers} worker process(es))",
          flush=True)
    print(inconel718.describe(), flush=True)
    if args.dry_run:
        for task in mine:
            print(f"    {task.label}")
        return

    payloads = [(task, args.root, args.force) for task in mine]
    failures: List[dict] = []
    done = 0

    if args.workers <= 1:
        for payload in payloads:
            done += 1
            _report(_worker(payload), done, len(mine), failures)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = [pool.submit(_worker, p) for p in payloads]
            for future in as_completed(futures):
                done += 1
                _report(future.result(), done, len(mine), failures)

    print(f"\ncompleted {len(mine) - len(failures)} of {len(mine)}, {len(failures)} failed")
    if failures:
        report = args.root / f"fea_failures_shard{args.shard}.log"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n\n".join(f"{f['task']}\n{f['error']}" for f in failures))
        print(f"failure details written to {report}")
        sys.exit(1)


def _report(result: dict, done: int, total: int, failures: List[dict]) -> None:
    if result["status"] == "failed":
        failures.append(result)
        print(f"[{done}/{total}] FAILED  {result['task']}", flush=True)
    elif result["status"] == "skipped":
        print(f"[{done}/{total}] skipped {result['task']}", flush=True)
    else:
        print(f"[{done}/{total}] {result['task']:<58s} {result['seconds']:8.1f}s", flush=True)


if __name__ == "__main__":
    main()
