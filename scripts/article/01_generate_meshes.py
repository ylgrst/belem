"""
Step 1 of the article campaign: generate one periodic mesh per geometry and density.

Follows the meshing recipe of belem-private's lattice_cadgen and
tpms_skeletal_cadgen: build the unit cell as CAD, export it to STEP, mesh it
periodically with gmsh, then remesh with mmg while keeping the periodicity, which
is what makes the mesh usable with fedoo's periodic boundary conditions.

Run one shard per machine or container, for example across four machines:

    python 01_generate_meshes.py --root /data/article --shard 0 --shards 4 --workers 8
    python 01_generate_meshes.py --root /data/article --shard 1 --shards 4 --workers 8
    ...

Jobs already meshed are skipped, so an interrupted run is resumed by repeating
the same command.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.resources
import json
import multiprocessing
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import article_config as cfg

# belem ships the radius to density fit for the strut lattices measured on a 4 mm
# cell. Density depends only on the ratio of strut radius to cell size, so the
# radius for another cell size is the fitted one scaled by the size ratio.
REFERENCE_LATTICE_CELL_SIZE = 4.0

# The fit gives a good first radius but not an exact density. One or two
# multiplicative corrections on the measured volume close the gap; density grows
# roughly as the square of the radius for slender struts, which is what the
# correction exponent below assumes.
DENSITY_TOLERANCE = 0.002
MAX_DENSITY_REFINEMENTS = 3


def _lattice_radius_fit(name: str) -> object:
    """Return belem's fitted radius(density) model for one lattice family"""
    import belem
    from belem.utils import compute_density_offset_polyfit_from_dataset_json_file

    dataset = (importlib.resources.files(belem) / "data"
               / "radius_densities_lattice_cell_size_4.json").as_posix()
    return compute_density_offset_polyfit_from_dataset_json_file(dataset)[name]


def _build_lattice(job: cfg.Job) -> Tuple[object, float, dict]:
    """Build a strut lattice at the requested density

    :return: the CAD shape, the density achieved, and the parameters used
    """
    lattice_class = getattr(importlib.import_module("belem.latticegen"), job.geometry.name)
    cell_volume = cfg.CELL_SIZE ** 3

    radius = float(_lattice_radius_fit(job.geometry.name)(job.density))
    radius *= cfg.CELL_SIZE / REFERENCE_LATTICE_CELL_SIZE

    shape = None
    achieved = float("nan")
    for _ in range(MAX_DENSITY_REFINEMENTS):
        shape = lattice_class(strut_radius=radius, cell_size=cfg.CELL_SIZE,
                              center=cfg.CENTER).generate()
        achieved = shape.Volume() / cell_volume
        if abs(achieved - job.density) <= DENSITY_TOLERANCE:
            break
        # density scales roughly with the square of the radius
        radius *= (job.density / achieved) ** 0.5

    return shape, achieved, {"strut_radius": radius}


def _build_tpms(job: cfg.Job) -> Tuple[object, float, dict]:
    """Build a TPMS part at the requested density

    microgen solves the surface offset that gives the requested density, so no
    external offset to density dataset is needed and any cell size works.
    """
    from microgen import Tpms
    from microgen.shape import surface_functions

    surface_function = getattr(surface_functions, job.geometry.name)
    tpms = Tpms(
        surface_function=surface_function,
        density=job.density,
        cell_size=cfg.CELL_SIZE,
        resolution=cfg.TPMS_RESOLUTION,
    )
    shape = tpms.generate(type_part=job.geometry.part_type)
    achieved = shape.Volume() / cfg.CELL_SIZE ** 3

    return shape, achieved, {"offset": float(tpms.offset),
                             "resolution": cfg.TPMS_RESOLUTION}


def generate_one(job: cfg.Job, root: Path, force: bool, mesh_scale: float = 1.0) -> dict:
    """Mesh a single geometry at a single density

    Runs entirely inside the job directory: gmsh and mmg both drop intermediate
    files into the working directory, so parallel jobs must not share one.
    """
    directory = job.directory(root)
    mesh_dir = directory / "mesh"
    if not force and cfg.is_done(mesh_dir, "mesh"):
        return {"job": job.label, "status": "skipped"}

    import cadquery as cq
    import pyvista as pv
    from microgen import Phase, Rve, mesh_periodic
    from microgen.remesh import remesh_keeping_periodicity_for_fem

    started = time.time()
    with cfg.run_in_directory(mesh_dir):
        if job.geometry.kind == "lattice":
            shape, achieved, parameters = _build_lattice(job)
            mesh_size = cfg.LATTICE_MESH_SIZE_RATIO * cfg.CELL_SIZE * mesh_scale
            remesh_hmax = cfg.LATTICE_REMESH_HMAX_RATIO * cfg.CELL_SIZE * mesh_scale
        else:
            shape, achieved, parameters = _build_tpms(job)
            mesh_size = cfg.TPMS_MESH_SIZE_RATIO * cfg.CELL_SIZE * mesh_scale
            remesh_hmax = cfg.TPMS_REMESH_HMAX_RATIO * cfg.CELL_SIZE * mesh_scale

        cq.exporters.export(shape, "shape.step")

        rve = Rve(dim=cfg.CELL_SIZE, center=cfg.CENTER)
        mesh_periodic(mesh_file="shape.step", rve=rve, list_phases=[Phase(shape)],
                      size=mesh_size, order=1, output_file="mesh.vtk")

        remeshed = remesh_keeping_periodicity_for_fem(input_mesh=pv.read("mesh.vtk"),
                                                      hmax=remesh_hmax)
        remeshed.save("remeshed.vtk")

        info = {
            "geometry": job.geometry.slug,
            "kind": job.geometry.kind,
            "surface_or_class": job.geometry.name,
            "part_type": job.geometry.part_type,
            "cell_size": cfg.CELL_SIZE,
            "target_density": job.density,
            "achieved_density": achieved,
            "density_error": achieved - job.density,
            "mesh_scale": mesh_scale,
            "mesh_size": mesh_size,
            "remesh_hmax": remesh_hmax,
            "n_nodes": int(remeshed.n_points),
            "n_cells": int(remeshed.n_cells),
            "mesh_volume": float(remeshed.volume),
            "seconds": time.time() - started,
            **parameters,
        }
        Path("mesh_info.json").write_text(json.dumps(info, indent=2))

    cfg.write_status(mesh_dir, "mesh", info)

    return {"job": job.label, "status": "done", **info}


def _worker(payload: Tuple[cfg.Job, Path, bool, float]) -> dict:
    job, root, force, mesh_scale = payload
    try:
        return generate_one(job, root, force, mesh_scale)
    except Exception:
        return {"job": job.label, "status": "failed", "error": traceback.format_exc()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg.add_common_arguments(parser)
    parser.add_argument("--mesh-scale", type=float, default=1.0, metavar="FACTOR",
                        help="multiply the gmsh size and the mmg hmax by FACTOR. "
                             "1.0 reproduces belem-private's resolution, which gives "
                             "roughly 90k nodes for a strut lattice and 380k for a TPMS "
                             "sheet on a 5 mm cell. Raise it to make the campaign "
                             "tractable, and report the value used.")
    args = parser.parse_args()

    jobs = cfg.select_jobs(args.shard, args.shards, args.only, args.densities)
    cfg.report_selection(jobs, args, "mesh generation")
    if args.dry_run:
        return

    payloads = [(job, args.root, args.force, args.mesh_scale) for job in jobs]
    failures = []
    done = 0

    if args.workers <= 1:
        results = (_worker(p) for p in payloads)
        for result in results:
            done += 1
            _report(result, done, len(jobs), failures)
    else:
        # spawn rather than fork: gmsh holds global state that does not survive
        # being forked into several processes
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = [pool.submit(_worker, p) for p in payloads]
            for future in as_completed(futures):
                done += 1
                _report(future.result(), done, len(jobs), failures)

    print(f"\nmeshed {len(jobs) - len(failures)} of {len(jobs)}, {len(failures)} failed")
    if failures:
        report = args.root / f"mesh_failures_shard{args.shard}.log"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n\n".join(f"{f['job']}\n{f['error']}" for f in failures))
        print(f"failure details written to {report}")
        sys.exit(1)


def _report(result: dict, done: int, total: int, failures: list) -> None:
    if result["status"] == "failed":
        failures.append(result)
        print(f"[{done}/{total}] FAILED  {result['job']}", flush=True)
    elif result["status"] == "skipped":
        print(f"[{done}/{total}] skipped {result['job']}", flush=True)
    else:
        print(f"[{done}/{total}] {result['job']:<44s} "
              f"density {100 * result['achieved_density']:5.2f}% "
              f"(target {100 * result['target_density']:.0f}%)  "
              f"{result['n_nodes']:>7d} nodes  {result['seconds']:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
