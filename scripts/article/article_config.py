"""
Shared configuration for the article campaign.

Every step of the campaign derives its work list from this module, so the four
scripts always agree on what the campaign contains and where each result goes.

The campaign is a full factorial: every geometry at every density, and for each
of those one linear homogenisation plus nine non linear load cases.

Layout on disk, rooted at the directory given by --root:

    <root>/<geometry>/density<NN>/
        mesh/
            shape.step                      CAD of the unit cell
            mesh.vtk                        periodic mesh
            remeshed.vtk                    remeshed, this is what the FEA reads
            mesh_info.json                  density actually achieved, node count
        effective_stiffness_tensor.txt      linear homogenisation
        <load case>/<load case>.fdz         non linear FEA results
        <load case>/<array>.txt             postprocessed arrays
        dfa_params.txt                      identified yield surface
        identification/                     homogenised law identification

The <load case> subdirectories and the postprocessed file names are the layout
belem.fem.postprocess.postprocess_all_homogenization_computations expects, so
that function is reused as is.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Sequence, Tuple

# --------------------------------------------------------------------------
# campaign definition
# --------------------------------------------------------------------------

# Side of the cubic RVE, in mm.
CELL_SIZE = 5.0
CENTER = (0.0, 0.0, 0.0)

# 5 percent to 50 percent in 5 percent steps. Built with integers so the
# directory names and the float densities cannot drift apart.
DENSITY_PERCENTS: Tuple[int, ...] = tuple(range(5, 55, 5))

# Strut based lattices, from belem.latticegen.
STRUT_LATTICES: Tuple[str, ...] = (
    "BodyCenteredCubic",
    "Cuboctahedron",
    "Kelvin",
    "OctetTruss",
    "TruncatedOctahedron",
)

# The triply periodic minimal surfaces microgen provides, by the snake_case name
# of their surface function in microgen.shape.surface_functions.
#
# microgen also ships honeycomb_gyroid, honeycomb_schwarz_p, honeycomb_schwarz_d,
# honeycomb_schoen_iwp and honeycomb_lidinoid. Those are extruded two dimensional
# sections rather than genuinely triply periodic surfaces, so they are left out;
# add them here if the article should cover them.
TPMS_SURFACES: Tuple[str, ...] = (
    "gyroid",
    "schwarz_p",
    "schwarz_d",
    "neovius",
    "schoen_iwp",
    "schoen_frd",
    "fischer_koch_s",
    "pmy",
    "honeycomb",
    "lidinoid",
    "split_p",
)

# Each TPMS yields three solids.
TPMS_PART_TYPES: Tuple[str, ...] = ("sheet", "upper skeletal", "lower skeletal")

# --------------------------------------------------------------------------
# meshing
# --------------------------------------------------------------------------

# belem-private meshes a 4 mm cell with gmsh size 0.08 and mmg hmax 0.08 for the
# strut lattices, and size 0.06 with hmax 0.05 for the TPMS. Those are expressed
# here as fractions of the cell size so a 5 mm cell keeps the same number of
# elements across the cell rather than a coarser mesh.
LATTICE_MESH_SIZE_RATIO = 0.08 / 4.0
LATTICE_REMESH_HMAX_RATIO = 0.08 / 4.0
TPMS_MESH_SIZE_RATIO = 0.06 / 4.0
TPMS_REMESH_HMAX_RATIO = 0.05 / 4.0

# Grid resolution microgen uses to sample the implicit TPMS function.
TPMS_RESOLUTION = 50

# --------------------------------------------------------------------------
# load cases
# --------------------------------------------------------------------------

# (constraint driver node id, variables, values) for belem.fem.fea.Load.
# Node 0 drives (E_xx, E_yy, E_zz), node 1 drives (E_xy, E_xz, E_yz), and the
# shear values are engineering shear angles, hence 0.1 for 5 percent shear.
_TENSION = ([0], ["DispX"], [0.05])
_BIAXIAL_TENSION = ([0], ["DispX", "DispY"], [0.05, 0.05])
_COMPRESSION = ([0], ["DispX"], [-0.05])
_BIAXIAL_COMPRESSION = ([0], ["DispX", "DispY"], [-0.05, -0.05])
_TENCOMP = ([0], ["DispX", "DispY"], [0.05, -0.05])
_SHEAR = ([1], ["DispX"], [0.1])
_TENSION_ZERO = ([0], ["DispX"], [0.0])
_BIAXIAL_TENSION_ZERO = ([0], ["DispX", "DispY"], [0.0, 0.0])
_SHEAR_ZERO = ([1], ["DispX"], [0.0])

# The monotonic cases feed the yield surface identification, the cyclic ones feed
# the homogenised law identification.
LOAD_CASES: Dict[str, Tuple[Tuple, ...]] = {
    "tension": (_TENSION,),
    "biaxial_tension": (_BIAXIAL_TENSION,),
    "compression": (_COMPRESSION,),
    "biaxial_compression": (_BIAXIAL_COMPRESSION,),
    "tencomp": (_TENCOMP,),
    "shear": (_SHEAR,),
    "tension_cycle": (_TENSION, _TENSION_ZERO, _TENSION),
    "biaxial_tension_cycle": (_BIAXIAL_TENSION, _BIAXIAL_TENSION_ZERO, _BIAXIAL_TENSION),
    "shear_cycle": (_SHEAR, _SHEAR_ZERO, _SHEAR),
}

# Load cases the homogenised law identification consumes, in the order the
# identification expects its path_id files to be written.
IDENTIFICATION_LOAD_CASES: Tuple[str, ...] = (
    "tension_cycle",
    "biaxial_tension_cycle",
    "shear_cycle",
)

# --------------------------------------------------------------------------
# geometries and jobs
# --------------------------------------------------------------------------


class Geometry(NamedTuple):
    """One unit cell family, independent of density

    :param kind: either "lattice" or "tpms"
    :param name: lattice class name, or TPMS surface function name
    :param part_type: TPMS part, empty for a strut lattice
    """

    kind: str
    name: str
    part_type: str = ""

    @property
    def slug(self) -> str:
        """Filesystem safe identifier, unique across the campaign"""
        if self.kind == "lattice":
            return self.name
        return f"{self.name}_{self.part_type.replace(' ', '_')}"

    @property
    def label(self) -> str:
        """Readable identifier for logs"""
        if self.kind == "lattice":
            return self.name
        return f"{self.name} ({self.part_type})"


class Job(NamedTuple):
    """One geometry at one density, the unit of work every script shards on"""

    geometry: Geometry
    density_percent: int

    @property
    def density(self) -> float:
        return self.density_percent / 100.0

    @property
    def label(self) -> str:
        return f"{self.geometry.label} at {self.density_percent} percent"

    def directory(self, root: Path) -> Path:
        return Path(root) / self.geometry.slug / f"density{self.density_percent:02d}"


def iter_geometries() -> Iterator[Geometry]:
    """Every geometry of the campaign, strut lattices first"""
    for name in STRUT_LATTICES:
        yield Geometry("lattice", name)
    for surface in TPMS_SURFACES:
        for part_type in TPMS_PART_TYPES:
            yield Geometry("tpms", surface, part_type)


def iter_jobs() -> Iterator[Job]:
    """Every (geometry, density) pair, in a stable order

    The order is what makes sharding reproducible: every machine enumerates the
    same list and takes a disjoint slice of it, so no coordination is needed.
    """
    for geometry in iter_geometries():
        for density_percent in DENSITY_PERCENTS:
            yield Job(geometry, density_percent)


def select_jobs(
    shard: int = 0,
    shards: int = 1,
    only: Sequence[str] = (),
    densities: Sequence[int] = (),
) -> List[Job]:
    """Return this worker's slice of the campaign

    :param shard: index of this worker, from 0 to shards - 1
    :param shards: total number of workers splitting the campaign
    :param only: keep jobs whose geometry slug contains any of these substrings
    :param densities: keep only these densities, in percent
    """
    if shards < 1:
        raise ValueError("shards must be at least 1")
    if not 0 <= shard < shards:
        raise ValueError(f"shard must be in [0, {shards}), got {shard}")

    jobs = list(iter_jobs())
    if only:
        jobs = [j for j in jobs if any(pattern in j.geometry.slug for pattern in only)]
    if densities:
        jobs = [j for j in jobs if j.density_percent in densities]

    # round robin rather than contiguous blocks: cost grows with density, so
    # striding keeps the cheap and expensive jobs evenly spread over the workers
    return jobs[shard::shards]


# --------------------------------------------------------------------------
# command line plumbing shared by the four scripts
# --------------------------------------------------------------------------


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the campaign arguments every step understands"""
    parser.add_argument("--root", type=Path, required=True,
                        help="campaign root directory, shared by all four steps")
    parser.add_argument("--shard", type=int, default=0,
                        help="index of this worker, from 0 to --shards minus 1")
    parser.add_argument("--shards", type=int, default=1,
                        help="how many workers split the campaign")
    parser.add_argument("--workers", type=int, default=1,
                        help="processes to run in parallel inside this worker")
    parser.add_argument("--only", action="append", default=[], metavar="PATTERN",
                        help="restrict to geometries whose slug contains PATTERN, repeatable")
    parser.add_argument("--densities", type=int, nargs="+", default=[], metavar="PERCENT",
                        help="restrict to these densities, in percent")
    parser.add_argument("--force", action="store_true",
                        help="redo jobs whose output is already present")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the work this invocation would do, then stop")


def report_selection(jobs: Sequence[Job], args: argparse.Namespace, step: str) -> None:
    """Print what this invocation is about to do"""
    total = len(list(iter_jobs()))
    print(f"{step}: {len(jobs)} of {total} campaign jobs "
          f"(shard {args.shard + 1} of {args.shards}, {args.workers} worker process(es))",
          flush=True)
    if args.dry_run:
        for job in jobs:
            print(f"    {job.geometry.slug:<34s} density {job.density_percent:>2d}%  "
                  f"-> {job.directory(args.root)}")


def write_status(directory: Path, step: str, payload: Dict[str, object]) -> None:
    """Record that a step finished, so a rerun can skip it

    Written last, once the real outputs are on disk, so a job interrupted midway
    is retried rather than silently treated as complete.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f".{step}.done").write_text(json.dumps(payload, indent=2, default=str))


def is_done(directory: Path, step: str) -> bool:
    """Whether the given step already completed in this directory"""
    return (Path(directory) / f".{step}.done").is_file()


def run_in_directory(directory: Path):
    """Context manager that runs a job from inside its own directory

    fedoo writes a temporary _mesh_.npz into the current working directory while
    assembling an fdz archive, so two jobs sharing a working directory will
    clobber each other's temporary file and fail. Every parallel job therefore
    gets its own working directory.
    """
    from contextlib import contextmanager

    @contextmanager
    def _cd(target: Path):
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        previous = Path.cwd()
        os.chdir(target)
        try:
            yield target
        finally:
            os.chdir(previous)

    return _cd(directory)


if __name__ == "__main__":
    geometries = list(iter_geometries())
    jobs = list(iter_jobs())
    print(f"{len(geometries)} geometries "
          f"({len(STRUT_LATTICES)} strut lattices, "
          f"{len(TPMS_SURFACES)} TPMS x {len(TPMS_PART_TYPES)} parts)")
    print(f"{len(DENSITY_PERCENTS)} densities: "
          f"{', '.join(str(d) for d in DENSITY_PERCENTS)} percent")
    print(f"{len(jobs)} meshes, "
          f"{len(jobs)} linear homogenisations, "
          f"{len(jobs) * len(LOAD_CASES)} non linear FEA runs")
    print()
    for geometry in geometries:
        print(f"    {geometry.slug}")
