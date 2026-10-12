# Article campaign

A full factorial homogenisation study: every strut based lattice and every TPMS
part microgen offers, in a 5 mm cubic RVE, at ten densities from 5 to 50 percent,
in an additively manufactured nickel superalloy.

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

`python article_config.py` prints the campaign, `python materials/inconel718.py`
prints the material.

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

* **Above about scale 2 the knob stops working, and the results turn into
  noise.** The curvature of the ligament surfaces puts a floor under the element
  size, so scales 3, 4 and 6 all land near 11 000 nodes. They do not agree with
  each other though: the effective Young modulus scatters from 5 625 to 6 018 MPa
  across those three, a spread of 7 percent, because only the element
  distribution changes and it changes arbitrarily. Scale 6 happens to sit closer
  to the converged value than scale 3 does, which is luck rather than
  convergence. Treat anything above 2 as unusable, not merely coarse. Monotonic
  convergence only sets in from about scale 2 downward, so the useful range is
  between 1 and 2.
* **Scale 1 is not converged either.** The effective Young modulus is still
  drifting monotonically at belem-private's own resolution, so that resolution is
  itself roughly one to two percent away from a converged value. The effective
  shear modulus converges much faster, 0.54 percent at scale 2 against 4.42 for
  the Young modulus, so the Young modulus is what binds.

For an article comparing geometries against each other, a bias of a percent or
two that every cell shares matters far less than the resolution being the same
everywhere. **Choose `--mesh-scale` deliberately, keep it fixed across the whole
campaign, and report it.** `05_convergence_check.py` measures where to put it.

## Bulk material

Set `MATERIAL` in `article_config.py`; every step and every shard reads it from
there, and step 2 prints the properties it used into its log. The default is
`inconel625`.

| | `inconel625` (default) | `inconel718` |
| --- | --- | --- |
| condition | as built | solution treated and aged |
| modulus | 195 744 MPa, measured per specimen | 195 000 MPa, Renishaw data sheet |
| hardening | Ludwik fit to NIST AM Bench 2022 | Ludwik fit to NASA CR-123774 |
| Re, k, m | 816.4 MPa, 1523.5 MPa, 0.7535 | 1192.6 MPa, 1113.6 MPa, 0.4236 |
| source type | measured AM curves | digitised wrought figure |
| ready to use | yes, fit is committed | yes |

**They are not condition matched.** IN625 here is as built and IN718 is aged, so
running both varies heat treatment as well as alloy and the pair is not an alloy
comparison. As built is the campaign's choice because it is the state a printed
lattice is normally used in, and the only state for which measured AM tensile
curves are published. The as built IN718 counterpart exists, NIST AMB2025-02
(doi:10.18434/mds2-3735, eight ASTM E8 miniature specimens at 1e-3 per second),
but NIST's file distribution returns gateway timeouts and the Internet Archive
never captured it, so that module is blocked on the data rather than on work.

### Refitting Inconel 625

The fit is committed as `materials/inconel625_ludwik_fit.json`, so nothing needs
fetching for a normal run. To reproduce or change it:

    python materials/inconel625.py --download
    python materials/inconel625.py --fit

The dataset (doi:10.18434/mds2-2760) is not redistributed here. `--download`
takes it from NIST when their distribution is up and from the Internet Archive
when it is not, which is currently the only route that works. The committed fit
uses specimens T3 and T4, the two quasi-static ones the archive holds; T6, also
quasi-static, was never captured.

Each engineering curve is converted to true stress and true plastic strain with
that specimen's own measured modulus, and Ludwik is regressed over the pooled
points between 0.2 and 10 percent plastic strain. Pooling weights each specimen
by how much of it falls in the window, which is what we want from specimens that
fractured at different strains. Residual is 3.7 MPa rms over 420 points.

Two numbers worth carrying into the article. The dataset's faster specimen
(1e-2 per second) puts the strain rate sensitivity at 1.4 percent of flow stress
per decade, which bounds the error of using a rate independent law. And the
fitted curve tracks the measurement to better than 0.2 percent above 0.4 percent
plastic strain but runs about 1 percent high below that, peaking at 16 MPa right
at the 0.2 percent offset, so the homogenised yield onset is biased high by
about a percent uniformly across every cell.

## Assumptions worth reviewing

* **Hardening parameters.** The EPICP law needs `k` and `m`. For Inconel 718
  these come from fitting Ludwik to the room temperature tensile curve of
  NASA CR-123774 Fig 7.4121, because the Renishaw data sheet reports only a proof
  stress, a UTS and an elongation, and those three do not determine three Ludwik
  parameters: a Considere construction over them leaves one degree of freedom and
  admits `m` anywhere from 0.07 to 0.43. The fit is checked against the data sheet
  by an independent route, its implied 0.2 percent proof stress being 1273 MPa
  against the 1275 MPa reported. It is however a wrought curve standing in for a
  printed one. Pass measured `(k, m)` via `epicp_props(hardening=...)` to override.
* **Poisson ratio and thermal expansion** are in neither source; 0.29 and
  13e-6 per K come from NASA CR-123774 and are consistent with the data sheet's
  wrought band.
* **Identification bounds** scale with each cell's own stress at 0.2 percent
  plastic strain, which is what lets one set of bounds span a 5 percent and a
  50 percent dense cell. The multipliers are in `PARAMETER_BOUNDS` in step 4.
* **Honeycomb variants.** microgen also ships `honeycomb_gyroid`,
  `honeycomb_schwarz_p`, `honeycomb_schwarz_d`, `honeycomb_schoen_iwp` and
  `honeycomb_lidinoid`. They are extruded two dimensional sections rather than
  genuinely triply periodic surfaces, so they are left out of `TPMS_SURFACES`.
  Add them there if the article should cover them.
