# Article campaign

A full factorial homogenisation study: every strut based lattice and every TPMS
part microgen offers, in a 5 mm cubic RVE, at ten densities from 5 to 50 percent,
in Inconel 718.

| | count |
| --- | --- |
| strut lattices | 5 |
| TPMS surfaces x parts | 11 x 3 = 33 |
| geometries | 38 |
| densities | 10 (5 to 50 percent in steps of 5) |
| meshes | 380 |
| linear homogenisations | 380 |
| non linear FEA runs | 3420 (9 load cases each) |
| law identifications | 380 |

`python article_config.py` prints the campaign, `python inconel718.py` prints the
material.

## The four steps

Run them in order; each reads what the previous one wrote.

| script | produces |
| --- | --- |
| `01_generate_meshes.py` | CAD, periodic mesh, remeshed mesh, achieved density |
| `02_run_all_fea_computations.py` | effective stiffness tensor, 9 fdz result files |
| `03_postprocess_all.py` | reduced arrays, response and hardening plots, yield surface |
| `04_identify_homogenized_law.py` | identified EPCHG parameters, fit quality plots |

`05_convergence_check.py` sits beside them rather than in the chain. Checking
convergence on every geometry, density and load case would cost more than the
campaign it justifies, so it exploits the fact that mesh convergence here is set
by how well the ligament cross section is resolved, which shows in the linear
effective stiffness just as sharply as in the plastic response and costs seconds
per run instead of minutes. The default sweep is therefore exhaustive in geometry
and density but linear only, which is cheap enough to cover every cell rather than
a sample. `--nonlinear-load-case` adds a load case per scale for spot checks where
the linear result converged slowest.

Steps 2, 3 and 4 refuse to start on a cell whose inputs are missing, and name what
to run first.

## Layout

```
<root>/<geometry>/density<NN>/
    mesh/shape.step  mesh.vtk  remeshed.vtk  mesh_info.json
    effective_stiffness_tensor.txt
    <load case>/<load case>.fdz          FEA results
    <load case>/<array>.txt              reduced arrays
    dfa_params.txt                       identified yield surface
    all_stress_strain.png  all_hardening.png  ...
    identification/                      identified law and its inputs
```

The `<load case>` subdirectories and the reduced array names are exactly what
`belem.fem.postprocess.postprocess_all_homogenization_computations` expects, so
that function is reused unchanged.

## Running in parallel

Every script takes `--shard i --shards n`. Each worker enumerates the same
campaign and takes a disjoint slice of it, so no coordination, queue or shared
database is needed: the only thing the workers share is the output tree.

Across four machines or containers, with a shared or synced `--root`:

```
python 02_run_all_fea_computations.py --root /data/article --shard 0 --shards 4 --workers 6
python 02_run_all_fea_computations.py --root /data/article --shard 1 --shards 4 --workers 6
python 02_run_all_fea_computations.py --root /data/article --shard 2 --shards 4 --workers 6
python 02_run_all_fea_computations.py --root /data/article --shard 3 --shards 4 --workers 6
```

`--workers` adds process level parallelism inside one shard.

Details that matter:

* Steps 1, 3 and 4 shard over the 380 geometry and density pairs. Step 2 shards
  over the 3800 individual computations, because one geometry's nine load cases
  are far from equal in cost and splitting them balances better.
* Sharding is round robin, not contiguous blocks, so the expensive high density
  cells spread evenly over the workers.
* Every job records a `.<step>.done` marker once its real output is on disk.
  Rerunning the same command skips finished work, so an interrupted or crashed
  run is resumed by repeating it. `--force` redoes work anyway.
* A failure never stops the shard. Failures are collected, the traceback is
  written to `<root>/<step>_failures_shard<i>.log`, and the exit status is non
  zero so a batch scheduler notices.
* Each computation runs in its own working directory. This is not cosmetic: fedoo
  writes a temporary `_mesh_.npz` into the current directory while building an fdz
  archive, so two computations sharing a directory silently destroy each other's
  temporary file and both fail.
* Workers are spawned, not forked, because gmsh keeps global state that does not
  survive being forked.

Useful for a partial or test run: `--only PATTERN` (repeatable, matches the
geometry slug), `--densities 30 40`, and `--dry-run` to see the work list.

## Cost, and the mesh resolution knob

Measured on one 24 core machine, for a body centred cubic cell at 30 percent:

| `--mesh-scale` | nodes | mesh | effective E, deviation | one monotonic load case | one cyclic load case |
| --- | --- | --- | --- | --- | --- |
| 1.0 (belem-private resolution) | 88 075 | 35 s | reference | not measured | not measured |
| 1.25 | 45 339 | 19 s | 1.23 percent | not measured | not measured |
| 1.5 | 26 953 | 12 s | 2.37 percent | not measured | not measured |
| 2.0 | 13 813 | 8 s | 4.42 percent | not measured | not measured |
| 4.0 | 10 965 | 5 s | not compared | about 10 min | about 30 min |

At `--mesh-scale 4` one geometry and density costs roughly 2.5 core hours for its
ten computations, so the 380 cells come to of order 950 core hours: about two days
on one 24 core machine, or half a day spread over four of them.

At `--mesh-scale 1` the meshes are eight times larger and a direct solver scales
worse than linearly, so the same campaign is one to two orders of magnitude more
expensive. TPMS cells are larger still, 380 000 nodes for a gyroid sheet against
88 000 for the lattice.

Two things the table shows that are worth knowing before choosing:

* **The knob saturates above about 2.** The curvature of the ligament surfaces
  puts a floor under the element size, so scales 3, 4 and 6 all produce much the
  same mesh, around 11 000 nodes. Almost all of the useful range is between 1
  and 2.
* **Scale 1 is not converged either.** The effective Young modulus is still
  drifting monotonically at belem-private's own resolution, so that resolution is
  itself roughly one to two percent away from a converged value. The effective
  shear modulus converges much faster, 0.54 percent at scale 2 against 4.42 for
  the Young modulus, so the Young modulus is what binds.

For an article comparing geometries against each other, a bias of a percent or
two that every cell shares matters far less than the resolution being the same
everywhere. **Choose `--mesh-scale` deliberately, keep it fixed across the whole
campaign, and report it.** `05_convergence_check.py` measures where to put it.

## Assumptions worth reviewing

* **Hardening parameters.** The EPICP law needs `k` and `m`, which the Renishaw
  data sheet does not report. `inconel718.py` fits them with a Considere
  construction from the yield strength, the UTS and the elongation, assuming the
  uniform elongation is 60 percent of the elongation after fracture. That fraction
  is an estimate, and it is the least certain input of the campaign. Override it,
  or pass measured `(k, m)` directly via `epicp_props(hardening=...)`.
* **Poisson ratio and thermal expansion** are not in the data sheet either;
  0.3 and 13e-6 per K come from the literature and from the data sheet's wrought
  band respectively.
* **Identification bounds** scale with each cell's own stress at 0.2 percent
  plastic strain, which is what lets one set of bounds span a 5 percent and a
  50 percent dense cell. The multipliers are in `PARAMETER_BOUNDS` in step 4.
* **Honeycomb variants.** microgen also ships `honeycomb_gyroid`,
  `honeycomb_schwarz_p`, `honeycomb_schwarz_d`, `honeycomb_schoen_iwp` and
  `honeycomb_lidinoid`. They are extruded two dimensional sections rather than
  genuinely triply periodic surfaces, so they are left out of `TPMS_SURFACES`.
  Add them there if the article should cover them.
