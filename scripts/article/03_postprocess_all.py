"""
Step 3 of the article campaign: postprocess every FEA computation.

Adapted from
scripts/non_linear_homogenization/postprocess_nonlinear_homogenization.py,
extended to sweep the whole campaign and to shard across machines.

For each geometry and density this reads the nine fdz result files, reduces them
to the stress, strain, von Mises and plastic strain arrays, writes the response
and hardening plots, then identifies the yield surface parameters that step 4
consumes.

    python 03_postprocess_all.py --root /data/article --shard 0 --shards 4 --workers 8

This step is cheap next to the FEA but reads every result file, so it is worth
running it where the results live rather than over a network share.
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

# Yield criterion identified from the six monotonic load cases. "dfa" is what the
# existing belem example uses; "hill" and "anisotropic" are the alternatives.
CRITERION = "dfa"


def postprocess_one(job: cfg.Job, root: Path, force: bool, criterion: str) -> dict:
    """Reduce one geometry and density, and identify its yield surface"""
    import numpy as np
    from belem.fem.postprocess import (
        identify_plasticity_criterion_parameters,
        plot_all_hardening_from_all_results,
        plot_all_stress_strain_from_all_results,
        plot_all_vm_stress_vm_strain_from_all_results,
        plot_criteria_shear_yield_surface,
        plot_criteria_yield_surface,
        postprocess_all_homogenization_computations,
    )

    directory = job.directory(root)
    if not force and cfg.is_done(directory, "postprocess"):
        return {"job": job.label, "status": "skipped"}

    missing = [name for name in cfg.LOAD_CASES
               if not (directory / name / f"{name}.fdz").is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing FEA results for {', '.join(missing)}; run step 2 first"
        )

    started = time.time()
    # postprocess_all_homogenization_computations builds its paths as
    # basedir + typesim + "/", so basedir must carry the trailing separator
    basedir = str(directory) + "/"

    with cfg.run_in_directory(directory):
        results = postprocess_all_homogenization_computations(basedir)

        plot_all_stress_strain_from_all_results(results, basedir + "all_stress_strain.png")
        plot_all_vm_stress_vm_strain_from_all_results(
            results, basedir + "all_vm_stress_vm_strain.png")
        plot_all_hardening_from_all_results(results, basedir + "all_hardening.png")

        criterion_params = identify_plasticity_criterion_parameters(
            criterion=criterion, all_results_dict=results)
        np.savetxt(basedir + f"{criterion}_params.txt", criterion_params)

        plot_criteria_yield_surface(
            criterion=criterion, all_results_dict=results,
            criteria_params=criterion_params,
            figname=basedir + f"{criterion}_yield_surface.png")
        plot_criteria_shear_yield_surface(
            criterion=criterion, all_results_dict=results,
            criteria_params=criterion_params,
            figname=basedir + f"{criterion}_shear_yield_surface.png")

    info = {
        "geometry": job.geometry.slug,
        "density_percent": job.density_percent,
        "criterion": criterion,
        "criterion_params": [float(p) for p in criterion_params],
        "final_stress_component": {
            name: float(results[name]["stress_component"][-1]) for name in results},
        "seconds": time.time() - started,
    }
    (directory / "postprocess_summary.json").write_text(json.dumps(info, indent=2))
    cfg.write_status(directory, "postprocess", info)

    return {"job": job.label, "status": "done", **info}


def _worker(payload: Tuple[cfg.Job, Path, bool, str]) -> dict:
    job, root, force, criterion = payload
    try:
        return postprocess_one(job, root, force, criterion)
    except Exception:
        return {"job": job.label, "status": "failed", "error": traceback.format_exc()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg.add_common_arguments(parser)
    parser.add_argument("--criterion", default=CRITERION,
                        choices=["hill", "dfa", "anisotropic"],
                        help="yield criterion to identify")
    args = parser.parse_args()

    jobs = cfg.select_jobs(args.shard, args.shards, args.only, args.densities)
    cfg.report_selection(jobs, args, "postprocessing")
    if args.dry_run:
        return

    payloads = [(job, args.root, args.force, args.criterion) for job in jobs]
    failures: List[dict] = []
    done = 0

    if args.workers <= 1:
        for payload in payloads:
            done += 1
            _report(_worker(payload), done, len(jobs), failures)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = [pool.submit(_worker, p) for p in payloads]
            for future in as_completed(futures):
                done += 1
                _report(future.result(), done, len(jobs), failures)

    print(f"\npostprocessed {len(jobs) - len(failures)} of {len(jobs)}, "
          f"{len(failures)} failed")
    if failures:
        report = args.root / f"postprocess_failures_shard{args.shard}.log"
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
        print(f"[{done}/{total}] {result['job']:<44s} {result['seconds']:7.1f}s", flush=True)


if __name__ == "__main__":
    main()
