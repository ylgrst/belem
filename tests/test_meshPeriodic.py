import pytest
from microgen import Box, meshPeriodic, Phase, Rve
import cadquery as cq
import fedoo as fd
from pathlib import Path

@pytest.fixture(scope="function")
def tmp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_dir_name = "test_tmp_dir"
    return tmp_path_factory.mktemp(tmp_dir_name)


@pytest.fixture(scope="function")
def tmp_step_filename(tmp_dir: Path) -> str:
    return (tmp_dir / "shape.step").as_posix()

@pytest.fixture(scope="function")
def tmp_vtk_filename(tmp_dir:Path) -> str:
    return (tmp_dir / "shape.vtk").as_posix()

def test_given_box_meshPeriodic_mesh_must_be_periodic(tmp_step_filename, tmp_vtk_filename):

    # Arrange

    shape = Box(center=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0), dim_x=1.0, dim_y=1.0, dim_z=1.0).generate()
    cq.exporters.export(shape, tmp_step_filename)
    phase = [Phase(shape)]
    rve = Rve(dim_x=1.0, dim_y=1.0, dim_z=1.0, center=(0.0, 0.0, 0.0))

    # Act

    meshPeriodic(tmp_step_filename, rve=rve, listPhases=phase, size=0.05, order=1, output_file=tmp_vtk_filename)
    mesh = fd.Mesh.read(tmp_vtk_filename)

    assert mesh.is_periodic(tol=1e-5)