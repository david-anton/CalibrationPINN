from typing import Callable, TypeAlias

import gmsh
import numpy as np
import numpy.typing as npt
from dolfinx.io import XDMFFile
from dolfinx.io.gmshio import model_to_mesh
from dolfinx.mesh import locate_entities_boundary
from mpi4py import MPI

from parametricpinn.fem.base import DMesh, DMeshTags, GMesh, join_output_path
from parametricpinn.io import ProjectDirectory
from parametricpinn.types import NPArray

FacetsDim: TypeAlias = int
LocateFacetFunc: TypeAlias = Callable[[NPArray], npt.NDArray[np.bool_]]
FacetTag: TypeAlias = int
Boundary: TypeAlias = tuple[FacetTag, LocateFacetFunc]
BoundaryList: TypeAlias = list[Boundary]
SortedFacetIndices: TypeAlias = npt.NDArray[np.int32]
SortedFacetTags: TypeAlias = npt.NDArray[np.int32]


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
        xdmf.write_meshtags(boundary_tags, mesh.geometry)


def list_sorted_facet_indices_and_tags(
    boundaries: BoundaryList, mesh: DMesh, bc_facets_dim: FacetsDim
) -> tuple[SortedFacetIndices, SortedFacetTags]:
    facet_indices_list: list[npt.NDArray[np.int32]] = []
    facet_tags_list: list[npt.NDArray[np.int32]] = []
    for tag, locator_func in boundaries:
        located_facet_indices = locate_entities_boundary(
            mesh, bc_facets_dim, locator_func
        )
        facet_indices_list.append(located_facet_indices)
        facet_tags_list.append(np.full_like(located_facet_indices, tag))
    facet_indices = np.hstack(facet_indices_list).astype(np.int32)
    facet_tags = np.hstack(facet_tags_list).astype(np.int32)
    sorting_index_array = np.argsort(facet_indices)
    sorted_facet_indices = facet_indices[sorting_index_array]
    sorted_facet_tags = facet_tags[sorting_index_array]
    return sorted_facet_indices, sorted_facet_tags
