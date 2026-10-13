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
prints the material. Both are as built, see Bulk material below.

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
`inconel718`.

Both alloys are as built PBF-LB material whose hardening is fitted to measured
AM tensile curves published by NIST, at the same condition, test standard and
strain rate, from the same institution. Switching between them therefore varies
the alloy and nothing else, so the pair is a genuine alloy comparison.

| | `inconel718` (default) | `inconel625` |
| --- | --- | --- |
| dataset | AMB2025-02, doi:10.18434/mds2-3735 | AM Bench 2022, doi:10.18434/mds2-2760 |
| specimens | 8, ASTM E8 miniature | 3, ASTM E8 subsize |
| strain rate | 1e-3 1/s | 1e-3 1/s |
| modulus | 184 000 MPa, Renishaw sheet | 195 973 MPa, measured |
| Re, k, m | 685.5, 1146.4 MPa, 0.5621 | 817.2, 1531.1 MPa, 0.7540 |
| measured proof, UTS | 690 MPa, 952 MPa | 814 MPa, 1037 MPa |
| points fitted | 17 683 | 632 |
| residual | 7.2 MPa rms | 3.8 MPa rms |

Each module fetches its own dataset and caches the fit, which is committed, so a
normal run needs no network:

    python materials/inconel718.py --download && python materials/inconel718.py --fit
    python materials/inconel625.py --download && python materials/inconel625.py --fit

Neither dataset is redistributed here.

### Where the elastic moduli come from

They are not symmetric, and the reason matters. Inconel 625 was measured on ASTM
E8 subsize specimens with a 6 mm gauge width and stereo DIC over a 25.4 mm
extensometer, so its modulus is measured from the curves themselves.

Inconel 718 was measured on specimens about 1.25 mm wide and 1 mm thick with a
3 mm contact extensometer, which measures its own compliance along with the
specimen's. The curves give 124 GPa, which is what NIST publishes as its own
challenge answer and which this module reproduces to 1.3 GPa, but it is about
two thirds of what IN718 actually is and NIST states plainly that elastic
properties are not provided by the dataset. The modulus therefore comes from the
Renishaw RenAM 500 sheet, 184 GPa for as built horizontal material, whose proof
stress and UTS on the same line (735 and 1051 MPa) sit close to the 691 and
952 MPa measured here on a different build.

That compliance does not contaminate the hardening. The elastic part is
subtracted using each specimen's own apparent slope, so a compliance
proportional to load appears in the measurement and the subtraction alike and
cancels, leaving the plastic strain right. Using a handbook modulus for the
subtraction would have left the error in.

### Checks the modules run on themselves

`inconel718.py` re-derives the modulus, 0.2 percent proof stress, UTS and strain
at UTS from the raw machine files and compares all four against the answers NIST
publishes for the same specimens. It refuses to produce a fit if any disagrees by
more than 2 percent, which is what makes trusting a parse of a vendor specific
format reasonable. Current agreement is 1.02, 0.13, 0.00 and 0.06 percent.

`inconel625.py` has no published answers to check against, so it reports the
measured proof stress beside the model's own and the strain rate sensitivity,
1.3 percent of flow stress per decade, measured from the dataset's faster
specimen.

### Known bias at the yield knee

Both fits run high where the curve bends: about 1.7 percent over the measurement
between 0.2 and 0.4 percent plastic strain for IN718 and about 1 percent for
IN625, against better than 0.5 percent everywhere above that. Ludwik cannot
follow a sharp knee and the regression is driven by the far larger number of
points at higher strain. Step 3 identifies yield surfaces, so the homogenised
yield onset carries that same bias, uniformly across every cell rather than
differently between geometries.

## Assumptions worth reviewing

* **Specimen size.** Both datasets are small. Inconel 718 is 1.25 mm wide by
  1 mm thick, excised from the legs of a larger build; Inconel 625 is ASTM E8
  subsize at 6 by 4 mm. For a lattice whose struts are a fraction of a
  millimetre this is arguably closer than a bulk specimen would be, since the
  grain structure and defect population are those of a thin printed section
  rather than of a block, but it is a property of the available data rather than
  a choice, and no measured AM curve on a full size specimen was found for
  either alloy in the as built condition. Pass measured `(k, m)` via
  `epicp_props(hardening=...)` to override either fit.
* **Poisson ratio and thermal expansion** are in none of the sources. 0.29 and
  13e-6 per K for IN718 come from NASA CR-123774 and are consistent with the
  Renishaw sheet's wrought band; 0.28 and 12.8e-6 for IN625 are the usual
  wrought values near room temperature.
* **Identification bounds** scale with each cell's own stress at 0.2 percent
  plastic strain, which is what lets one set of bounds span a 5 percent and a
  50 percent dense cell. The multipliers are in `PARAMETER_BOUNDS` in step 4.
* **Honeycomb variants.** microgen also ships `honeycomb_gyroid`,
  `honeycomb_schwarz_p`, `honeycomb_schwarz_d`, `honeycomb_schoen_iwp` and
  `honeycomb_lidinoid`. They are extruded two dimensional sections rather than
  genuinely triply periodic surfaces, so they are left out of `TPMS_SURFACES`.
  Add them there if the article should cover them.
