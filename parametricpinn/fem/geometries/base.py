import gmsh
from dolfinx.io import XDMFFile
from dolfinx.io.gmshio import model_to_mesh
from mpi4py import MPI

from parametricpinn.fem.base import DMesh, DMeshTags, GMesh, join_output_path
from parametricpinn.io import ProjectDirectory


def save_gmesh(
    output_subdir: str, save_to_input_dir: bool, project_directory: ProjectDirectory
) -> None:
    file_name = "mesh.msh"
    output_path = join_output_path(
        project_directory, file_name, output_subdir, save_to_input_dir
    )
    gmsh.write(str(output_path))


def load_mesh_from_gmsh_model(gmesh: GMesh, geometric_dim: int) -> DMesh:
    mpi_rank = 0
    mesh, cell_tags, facet_tags = model_to_mesh(
        gmesh, MPI.COMM_WORLD, mpi_rank, gdim=geometric_dim
    )
    return mesh


def save_boundary_tags_as_xdmf(
    boundary_tags: DMeshTags,
    mesh: DMesh,
    output_subdir: str,
    project_directory: ProjectDirectory,
    save_to_input_dir: bool = False,
) -> None:
    file_name = "boundary_tags.xdmf"
    output_path = join_output_path(
        project_directory, file_name, output_subdir, save_to_input_dir
    )
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    with XDMFFile(mesh.comm, output_path, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(boundary_tags)
