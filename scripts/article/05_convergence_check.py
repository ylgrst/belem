"""
Mesh convergence check for the article campaign.

Checking convergence on every geometry, density and load case would cost more
than the campaign it is meant to justify. This exploits the fact that what drives
mesh convergence here is how well the ligament cross section is resolved, which
shows up just as sharply in the linear effective stiffness as in the plastic
response, and a linear homogenisation costs seconds rather than minutes.

So the default sweep is exhaustive in geometry and density but linear only: every
cell, remeshed at each requested scale, homogenised, and the effective cubic
moduli compared between successive scales. That is cheap enough to cover the
whole campaign rather than a sample.

    python 05_convergence_check.py --root /data/convergence

Measured on a body centred cubic cell at 30 percent, effective Young modulus
against the finest mesh tested:

    scale 6.00    10 809 nodes    1.69 percent
    scale 4.00    10 965 nodes    3.53 percent
    scale 3.00    11 116 nodes    5.17 percent
    scale 2.00    13 813 nodes    4.42 percent
    scale 1.50    26 953 nodes    2.37 percent
    scale 1.25    45 339 nodes    1.23 percent
    scale 1.00    88 075 nodes    reference

Scales 6, 4 and 3 all land near 11 000 nodes because the curvature of the ligament
surfaces floors the element size, but they do not agree with each other: only the
element distribution changes, and the modulus scatters over 7 percent between
them. That region is noise rather than a plateau, and scale 6 sitting closest to
the converged value is luck. Monotonic convergence sets in from about scale 2.

The effective shear modulus converges far faster than the Young modulus, 0.54
percent against 4.42 at scale 2, so the Young modulus is what binds. Note that
the Young modulus is still drifting monotonically at scale 1, which is
belem-private's own resolution, so that resolution is itself roughly one to two
percent from converged rather than exact.

Then spot check the non linear response where the linear result converged
slowest, which is where a coarse mesh will actually bite:

    python 05_convergence_check.py --root /data/convergence --mesh-scales 2 1.25 \
        --nonlinear-load-case tension --only gyroid_sheet --densities 5

--report rereads a finished sweep and writes the comparison without recomputing.

Sharding and --workers behave as in the other steps.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import multiprocessing
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import article_config as cfg

HERE = Path(__file__).resolve().parent


def _load(stem: str):
    """Import one of the numbered step modules, whose names are not identifiers"""
    path = next(HERE.glob(f"{stem}*.py"))
    spec = importlib.util.spec_from_file_location(f"step_{stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scale_root(root: Path, mesh_scale: float) -> Path:
    """Each mesh scale gets its own tree, so the steps below can be reused as is"""
    return Path(root) / f"scale{mesh_scale:g}"


def run_one(job: cfg.Job, root: Path, mesh_scale: float, force: bool,
            nonlinear_load_case: str, discard_meshes: bool = False) -> dict:
    """Mesh one cell at one scale and homogenise it, optionally one load case too"""
    meshes = _load("01")
    fea = _load("02")

    target = scale_root(root, mesh_scale)
    started = time.time()

    # Nothing left to do when the moduli are already out and no load case is
    # wanted. This is what makes a resumed sweep cheap: --discard-meshes removes
    # the mesh, so without this the rerun would rebuild every mesh it had just
    # deleted purely to skip the homogenisation that follows.
    stiffness = job.directory(target) / "effective_stiffness_tensor.txt"
    mesh_info_file = job.directory(target) / "mesh" / "mesh_info.json"
    if (not force and not nonlinear_load_case
            and stiffness.is_file() and mesh_info_file.is_file()):
        info = json.loads(mesh_info_file.read_text())
        return {"job": job.label, "status": "skipped", "mesh_scale": mesh_scale,
                "n_nodes": info["n_nodes"], "achieved_density": info["achieved_density"],
                "seconds": 0.0, "meshed": "skipped"}

    # a discarded cell keeps its mesh done marker, so force a rebuild when the
    # mesh the linear homogenisation needs is no longer on disk
    remeshed = job.directory(target) / "mesh" / "remeshed.vtk"
    needs_mesh = force or not remeshed.is_file()
    mesh_result = meshes.generate_one(job, target, needs_mesh, mesh_scale)

    linear_task = fea.Task(job, fea.LINEAR_TASK)
    if force or not cfg.is_done(job.directory(target), "linear"):
        fea.run_linear(linear_task, target)
        cfg.write_status(job.directory(target), "linear", {"mesh_scale": mesh_scale})

    if nonlinear_load_case:
        marker = job.directory(target) / nonlinear_load_case
        if force or not cfg.is_done(marker, f"fea_{nonlinear_load_case}"):
            fea.run_load_case(fea.Task(job, nonlinear_load_case), target)
            cfg.write_status(marker, f"fea_{nonlinear_load_case}",
                             {"mesh_scale": mesh_scale})

    info = json.loads((job.directory(target) / "mesh" / "mesh_info.json").read_text())

    if discard_meshes:
        # a sweep keeps only the moduli and the mesh statistics, so the geometry
        # and the meshes themselves are dead weight: one fine TPMS cell is about
        # 266 MB, and the full sweep would need hundreds of gigabytes to hold
        # meshes nothing reads again
        mesh_dir = job.directory(target) / "mesh"
        for leftover in ("shape.step", "mesh.vtk", "remeshed.vtk"):
            (mesh_dir / leftover).unlink(missing_ok=True)
        for scratch in (job.directory(target) / "scratch_linear",):
            if scratch.is_dir():
                for item in scratch.iterdir():
                    item.unlink()
                scratch.rmdir()

    return {
        "job": job.label,
        "status": "done",
        "mesh_scale": mesh_scale,
        "n_nodes": info["n_nodes"],
        "achieved_density": info["achieved_density"],
        "seconds": time.time() - started,
        "meshed": mesh_result["status"],
    }


def _worker(payload: Tuple) -> dict:
    job, root, mesh_scale, force, nonlinear, discard = payload
    try:
        return run_one(job, root, mesh_scale, force, nonlinear, discard)
    except Exception:
        return {"job": job.label, "mesh_scale": mesh_scale,
                "status": "failed", "error": traceback.format_exc()}


def cubic_moduli(stiffness_file: Path) -> Tuple[float, float, float]:
    """Effective Young modulus, Poisson ratio and shear modulus of a cubic cell"""
    import numpy as np
    from simcoon import simmit as sim

    # L_cubic_props returns a (3, 1) array, hence the indexing
    young, poisson, shear = sim.L_cubic_props(np.loadtxt(stiffness_file))

    return float(young[0]), float(poisson[0]), float(shear[0])


def build_report(jobs: Sequence[cfg.Job], root: Path,
                 mesh_scales: Sequence[float]) -> List[dict]:
    """Compare the effective moduli across mesh scales, finest scale as reference

    A small mesh scale means a fine mesh, so the finest available scale is the
    reference and every coarser one is reported as a relative deviation from it.
    """
    finest = min(mesh_scales)
    rows: List[dict] = []

    for job in jobs:
        reference = None
        reference_file = job.directory(scale_root(root, finest)) / "effective_stiffness_tensor.txt"
        if reference_file.is_file():
            reference = cubic_moduli(reference_file)

        for mesh_scale in sorted(mesh_scales, reverse=True):
            directory = job.directory(scale_root(root, mesh_scale))
            stiffness_file = directory / "effective_stiffness_tensor.txt"
            mesh_info = directory / "mesh" / "mesh_info.json"
            if not stiffness_file.is_file() or not mesh_info.is_file():
                continue

            young, poisson, shear = cubic_moduli(stiffness_file)
            info = json.loads(mesh_info.read_text())
            row = {
                "geometry": job.geometry.slug,
                "density_percent": job.density_percent,
                "mesh_scale": mesh_scale,
                "n_nodes": info["n_nodes"],
                "achieved_density": round(info["achieved_density"], 5),
                "young": round(young, 3),
                "poisson": round(poisson, 5),
                "shear": round(shear, 3),
            }
            if reference is not None and mesh_scale != finest:
                row["young_deviation"] = round(abs(young - reference[0]) / reference[0], 5)
                row["shear_deviation"] = round(abs(shear - reference[2]) / reference[2], 5)
            else:
                row["young_deviation"] = 0.0
                row["shear_deviation"] = 0.0
            rows.append(row)

    return rows


def write_report(rows: List[dict], root: Path, tolerance: float) -> None:
    """Write the comparison table and the coarsest scale that meets the tolerance"""
    if not rows:
        print("no results found to report on")
        return

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    fields = ["geometry", "density_percent", "mesh_scale", "n_nodes", "achieved_density",
              "young", "poisson", "shear", "young_deviation", "shear_deviation"]
    table = root / "convergence.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # coarsest scale, so cheapest, that stays within tolerance of the finest one
    acceptable: Dict[Tuple[str, int], float] = {}
    worst: Dict[Tuple[str, int], float] = {}
    for row in rows:
        key = (row["geometry"], row["density_percent"])
        deviation = max(row["young_deviation"], row["shear_deviation"])
        worst[key] = max(worst.get(key, 0.0), deviation)
        if deviation <= tolerance:
            acceptable[key] = max(acceptable.get(key, 0.0), row["mesh_scale"])

    summary = root / "convergence_summary.txt"
    lines = [
        f"Coarsest mesh scale within {100 * tolerance:.1f} percent of the finest one",
        f"for the effective Young and shear moduli. {len(worst)} cells.",
        "",
        f"{'geometry':<34s} {'density':>8s} {'scale':>7s} {'worst dev':>10s}",
    ]
    for key in sorted(worst):
        chosen = acceptable.get(key)
        lines.append(f"{key[0]:<34s} {key[1]:>7d}% "
                     f"{('%.4g' % chosen) if chosen else 'none':>7s} "
                     f"{100 * worst[key]:>9.2f}%")

    unconverged = [k for k in worst if k not in acceptable]
    lines += ["", f"{len(unconverged)} cells met the tolerance at no tested scale."]
    if unconverged:
        lines.append("Those need a finer scale, or a non linear spot check:")
        lines += [f"    {g} at {d} percent" for g, d in sorted(unconverged)]
    if acceptable:
        lines += ["",
                  f"Coarsest scale acceptable everywhere it converged: "
                  f"{min(acceptable.values()):g}",
                  "Use that as --mesh-scale for the campaign, and report it."]

    summary.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\ntable written to {table}")
    print(f"summary written to {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg.add_common_arguments(parser)
    parser.add_argument("--mesh-scales", type=float, nargs="+",
                        default=[2.0, 1.5, 1.25, 1.0], metavar="SCALE",
                        help="mesh scales to compare; smaller is finer, and the finest "
                             "one is the reference. The knob saturates above about 2: "
                             "the curvature of the ligament surfaces sets a floor on the "
                             "element size, so scales 3, 4 and 6 all give a mesh of much "
                             "the same size, but with arbitrarily different element "
                             "distributions and results scattering over several percent. "
                             "The useful range lies between 1 and 2.")
    parser.add_argument("--nonlinear-load-case", default="", metavar="NAME",
                        choices=[""] + list(cfg.LOAD_CASES),
                        help="also run this load case at each scale, for spot checks")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="relative deviation accepted on the effective moduli")
    parser.add_argument("--discard-meshes", action="store_true",
                        help="delete the CAD and the meshes once the moduli are out. "
                             "A sweep reads none of them again and a fine TPMS cell is "
                             "about 266 MB, so a full sweep otherwise needs hundreds of "
                             "gigabytes. Incompatible with --nonlinear-load-case reruns, "
                             "which need the mesh to still be there.")
    parser.add_argument("--report", action="store_true",
                        help="only rebuild the comparison from results already on disk")
    args = parser.parse_args()

    jobs = cfg.select_jobs(args.shard, args.shards, args.only, args.densities)

    if args.report:
        write_report(build_report(jobs, args.root, args.mesh_scales),
                     args.root, args.tolerance)
        return

    # coarsest first: the cheap scales finish early and expose failures sooner
    payloads = [(job, args.root, scale, args.force, args.nonlinear_load_case,
                 args.discard_meshes)
                for scale in sorted(args.mesh_scales, reverse=True)
                for job in jobs]

    print(f"convergence: {len(jobs)} cells at {len(args.mesh_scales)} mesh scales "
          f"= {len(payloads)} runs "
          f"(shard {args.shard + 1} of {args.shards}, {args.workers} worker process(es))",
          flush=True)
    if args.nonlinear_load_case:
        print(f"including the {args.nonlinear_load_case} load case at each scale", flush=True)
    if args.dry_run:
        for job, _, scale, _, _ in payloads:
            print(f"    {job.geometry.slug:<34s} {job.density_percent:>2d}%  scale {scale:g}")
        return

    failures: List[dict] = []
    done = 0
    if args.workers <= 1:
        for payload in payloads:
            done += 1
            _report(_worker(payload), done, len(payloads), failures)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                                 max_tasks_per_child=1) as pool:
            futures = [pool.submit(_worker, p) for p in payloads]
            for future in as_completed(futures):
                done += 1
                _report(future.result(), done, len(payloads), failures)

    print(f"\ncompleted {len(payloads) - len(failures)} of {len(payloads)}, "
          f"{len(failures)} failed\n")
    if failures:
        report = args.root / f"convergence_failures_shard{args.shard}.log"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n\n".join(f"{f['job']} scale {f['mesh_scale']}\n{f['error']}"
                                      for f in failures))
        print(f"failure details written to {report}")

    write_report(build_report(jobs, args.root, args.mesh_scales), args.root, args.tolerance)
    if failures:
        sys.exit(1)


def _report(result: dict, done: int, total: int, failures: List[dict]) -> None:
    if result["status"] == "failed":
        failures.append(result)
        print(f"[{done}/{total}] FAILED  {result['job']} scale {result['mesh_scale']:g}",
              flush=True)
    elif result["status"] == "skipped":
        print(f"[{done}/{total}] skipped {result['job']} scale {result['mesh_scale']:g}",
              flush=True)
    else:
        print(f"[{done}/{total}] {result['job']:<44s} scale {result['mesh_scale']:<4g} "
              f"{result['n_nodes']:>7d} nodes  {result['seconds']:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
