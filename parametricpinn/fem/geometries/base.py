import gmsh
from dolfinx.io.gmshio import model_to_mesh
from dolfinx.mesh import Mesh
from mpi4py import MPI

from parametricpinn.fem.base import GMesh, _join_output_path
from parametricpinn.io import ProjectDirectory


def save_gmesh(
    output_subdir: str, save_to_input_dir: bool, project_directory: ProjectDirectory
) -> None:
    file_name = "mesh.msh"
    output_path = _join_output_path(
        project_directory, file_name, output_subdir, save_to_input_dir
    )
    gmsh.write(str(output_path))

def load_mesh_from_gmsh_model(gmesh: GMesh, geometric_dim: int) -> Mesh:
    mpi_rank = 0
    mesh, cell_tags, facet_tags = model_to_mesh(
        gmesh, MPI.COMM_WORLD, mpi_rank, gdim=geometric_dim
    )
    return mesh