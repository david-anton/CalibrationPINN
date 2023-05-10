import dolfinx
from dataclasses import dataclass
import gmsh
import sys
from typing import Optional, Any, Callable
from parametricpinn.io import ProjectDirectory
from parametricpinn.settings import Settings
from typing import TypeAlias
from dolfinx.io.gmshio import model_to_mesh
from dolfinx import fem
from dolfinx.fem import (
    locate_dofs_topological,
    Constant,
    dirichletbc,
    DirichletBCMetaClass,
)
from dolfinx.mesh import Mesh, locate_entities, meshtags
from dolfinx.io import XDMFFile
import ufl
from ufl import (
    as_matrix,
    inv,
    as_vector,
    as_tensor,
    nabla_grad,
    Measure,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    div,
    dot,
    dx,
    grad,
    inner,
    lhs,
    rhs,
    Argument,
)
from mpi4py import MPI
import numpy as np
from parametricpinn.types import NPArray
from petsc4py.PETSc import ScalarType
from dolfinx.fem.petsc import LinearProblem


GMesh: TypeAlias = gmsh.model
GOutDimAndTags: TypeAlias = list[tuple[int, int]]
GOutDimAndTagsMap: TypeAlias = list[GOutDimAndTags, list[Any]]
GGeometry: TypeAlias = tuple[GOutDimAndTags, GOutDimAndTagsMap]
UFLOperator: TypeAlias = ufl.core.operator.Operator
UFLSigmaFunc: TypeAlias = Callable[[TrialFunction], UFLOperator]
UFLEpsilonFunc: TypeAlias = Callable[[TrialFunction], UFLOperator]


@dataclass
class PWHSimulationConfig:
    model: str
    youngs_modulus: float
    poissons_ratio: float
    edge_length: float
    radius: float
    volume_force_x: float
    volume_force_y: float
    traction_left_x: float
    traction_left_y: float
    element_family: str = "Lagrange"
    element_degree: int = 1
    resolution: float = 10


Config: TypeAlias = PWHSimulationConfig

geometric_dim = 2


def run_one_simulation(
    config: Config, output_subdir: str, project_directory: ProjectDirectory
) -> None:
    gmsh.initialize()
    gmesh = _generate_gmesh(config, output_subdir, project_directory)
    mesh = _load_mesh_from_gmsh_model(gmesh)
    gmsh.finalize()
    _simulate_once(mesh, config, output_subdir, project_directory)


def _generate_gmesh(
    config: Config, output_subdir: str, project_directory: ProjectDirectory
) -> GMesh:
    length = config.edge_length
    radius = config.radius
    resolution = config.resolution
    geometry_kernel = gmsh.model.occ  # Use Open CASCADE as geometry kernel

    def create_geometry() -> GGeometry:
        gmsh.model.add("domain")
        plate = geometry_kernel.add_rectangle(0, 0, 0, -length, length)
        hole = geometry_kernel.add_disk(0, 0, 0, radius, radius)
        return geometry_kernel.cut([(2, plate)], [(2, hole)])

    def tag_physical_enteties(geometry: GGeometry) -> None:
        geometry_kernel.synchronize()
        solid_surface_marker = 1

        def tag_solid_surface() -> None:
            surface = geometry_kernel.getEntities(dim=2)
            assert surface == geometry[0]
            print(surface)
            gmsh.model.addPhysicalGroup(
                surface[0][0], [surface[0][1]], solid_surface_marker
            )
            gmsh.model.setPhysicalName(surface[0][0], solid_surface_marker, "Solid")

        tag_solid_surface()

    def configure_mesh() -> None:
        gmsh.model.mesh.setSizeCallback(mesh_size_callback)

    def mesh_size_callback(
        dim: int, tag: int, x: float, y: float, z: float, lc: float
    ) -> float:
        return resolution

    def generate_mesh() -> None:
        geometry_kernel.synchronize()
        gmsh.model.mesh.generate(geometric_dim)

    def save_mesh() -> None:
        output_path = project_directory.create_output_file_path(
            file_name="mesh.msh", subdir_name=output_subdir
        )
        gmsh.write(str(output_path))

    geometry = create_geometry()
    tag_physical_enteties(geometry)
    configure_mesh()
    generate_mesh()
    save_mesh()

    return gmsh.model


def _load_mesh_from_gmsh_model(gmesh: GMesh) -> Mesh:
    rank = 0  # The rank the Gmsh model is initialized on.
    mesh, cell_tags, facet_tags = model_to_mesh(
        gmesh, MPI.COMM_WORLD, rank, gdim=geometric_dim
    )
    return mesh


def _simulate_once(
    mesh: Mesh,
    config: Config,
    output_subdir: str,
    project_directory: ProjectDirectory,
) -> None:
    model = config.model
    youngs_modulus = config.youngs_modulus
    poissons_ratio = config.poissons_ratio
    length = config.edge_length
    radius = config.radius
    volume_force_x = config.volume_force_x
    volume_force_y = config.volume_force_y
    traction_top_x = 0.0
    traction_top_y = 0.0
    traction_left_x = config.traction_left_x
    traction_left_y = config.traction_left_y
    traction_hole_x = 0.0
    traction_hole_y = 0.0
    element_family = config.element_family
    element_degree = config.element_degree

    element = ufl.VectorElement(element_family, mesh.ufl_cell(), element_degree)
    func_space = fem.FunctionSpace(mesh, element)
    facets_dim = mesh.topology.dim - 1

    x = SpatialCoordinate(mesh)
    T_top = Constant(mesh, (ScalarType((traction_top_x, traction_top_y))))
    T_left = Constant(mesh, (ScalarType((traction_left_x, traction_left_y))))
    T_hole = Constant(mesh, (ScalarType((traction_hole_x, traction_hole_y))))
    u_D_bc_x_right = ScalarType(0)
    u_D_bc_y_bottom = ScalarType(0)
    f = Constant(mesh, (ScalarType((volume_force_x, volume_force_y))))
    u = TrialFunction(func_space)
    w = TestFunction(func_space)

    def sigma_and_epsilon_factory() -> tuple[UFLSigmaFunc, UFLEpsilonFunc]:
        compliance_matrix = None
        if model == "plane stress":
            compliance_matrix = (1 / youngs_modulus) * as_matrix(
                [
                    [1, -poissons_ratio, 0],
                    [-poissons_ratio, 1, 0],
                    [0, 0, 2 * (1 + poissons_ratio)],
                ]
            )
        elif model == "plane strain":
            compliance_matrix = (1 / youngs_modulus) * as_matrix(
                [
                    [
                        1 - poissons_ratio**2,
                        -poissons_ratio * (1 + poissons_ratio),
                        0,
                    ],
                    [
                        -poissons_ratio * (1 + poissons_ratio),
                        1 - poissons_ratio**2,
                        0,
                    ],
                    [0, 0, 2 * (1 + poissons_ratio)],
                ]
            )
        else:
            raise TypeError(f"Unknown model: {model}")

        elasticity_matrix = inv(compliance_matrix)

        def sigma(u: TrialFunction) -> UFLOperator:
            return _sigma_voigt_to_matrix(
                dot(elasticity_matrix, _epsilon_matrix_to_voigt(epsilon(u)))
            )

        def _epsilon_matrix_to_voigt(eps: UFLOperator) -> UFLOperator:
            return as_vector([eps[0, 0], eps[1, 1], 2 * eps[0, 1]])

        def _sigma_voigt_to_matrix(sig: UFLOperator) -> UFLOperator:
            return as_tensor([[sig[0], sig[2]], [sig[2], sig[1]]])

        def epsilon(u: TrialFunction) -> UFLOperator:
            return 0.5 * (nabla_grad(u) + nabla_grad(u).T)

        return sigma, epsilon

    class BoundaryCondition:
        def __init__(self, type, marker, value):
            self._type = type
            if type == "Dirichlet_x":
                facet = boundary_tags.find(marker)
                dofs = locate_dofs_topological(func_space, facets_dim, facet)
                self._bc = dirichletbc(value, dofs, func_space.sub(0))
            elif type == "Dirichlet_y":
                facet = boundary_tags.find(marker)
                dofs = locate_dofs_topological(func_space, facets_dim, facet)
                self._bc = dirichletbc(value, dofs, func_space.sub(1))
            elif type == "Neumann":
                self._bc = inner(value, w) * ds(marker)
            else:
                raise TypeError("Unknown boundary condition: {0:s}".format(type))

        @property
        def bc(self):
            return self._bc

        @property
        def type(self):
            return self._type

    def define_boundary_conditions() -> list[BoundaryCondition]:
        boundaries = [
            (0, lambda x: np.isclose(x[1], length)),  # top
            (1, lambda x: np.isclose(x[0], 0)),  # right
            (
                2,
                lambda x: np.isclose(
                    np.sqrt(np.square(x[0]) + np.square(x[1])), radius
                ),
            ),  # hole
            (3, lambda x: np.isclose(x[1], 0)),  # bottom
            (4, lambda x: np.isclose(x[0], -length)),  # left
        ]

        facet_indices, facet_markers = [], []
        for marker, locator_func in boundaries:
            _facet_indices = locate_entities(mesh, facets_dim, locator_func)
            facet_indices.append(_facet_indices)
            facet_markers.append(np.full_like(_facet_indices, marker))
        facet_indices = np.hstack(facet_indices).astype(np.int32)
        facet_markers = np.hstack(facet_markers).astype(np.int32)
        sorted_facet_indices = np.argsort(facet_indices)
        boundary_tags = meshtags(
            mesh,
            facets_dim,
            facet_indices[sorted_facet_indices],
            facet_markers[sorted_facet_indices],
        )

        output_path = project_directory.create_output_file_path(
            file_name="boundary_tags.xdmf", subdir_name=output_subdir
        )
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        with XDMFFile(mesh.comm, output_path, "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_meshtags(boundary_tags)

        ds = Measure("ds", domain=mesh, subdomain_data=boundary_tags)

        boundary_conditions = [
            BoundaryCondition("Neumann", 0, T_top),
            BoundaryCondition("Dirichlet_x", 1, u_D_bc_x_right),
            BoundaryCondition("Neumann", 2, T_hole),
            BoundaryCondition("Dirichlet_y", 3, u_D_bc_y_bottom),
            BoundaryCondition("Neumann", 4, T_left),
        ]

        return boundary_conditions

    sigma, epsilon = sigma_and_epsilon_factory()
    boundary_conditions = define_boundary_conditions()

    F = inner(sigma(u), epsilon(w)) * dx - inner(w, f) * dx

    bcs = []
    for condition in boundary_conditions:
        if condition.type == "Dirichlet":
            bcs.append(condition.bc)
        else:
            F += condition.bc

    a = lhs(F)
    L = rhs(F)
    problem = LinearProblem(
        a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    uh = problem.solve()


if __name__ == "__main__":
    settings = Settings()
    project_directory = ProjectDirectory(settings)
    simulation_config = PWHSimulationConfig(
        model="plane stress",
        youngs_modulus=210000.0,
        poissons_ratio=0.3,
        edge_length=100,
        radius=10,
        volume_force_x=0.0,
        volume_force_y=0.0,
        traction_left_x=-100.0,
        traction_left_y=0.0,
    )
    run_one_simulation(
        config=simulation_config,
        output_subdir="mesh_test",
        project_directory=project_directory,
    )

    # surface_geometry = geometry_kernel.getEntities(dim=2)
    # assert surface_geometry == geometry[0]
    # surface_geometry = surface_geometry[0]
    # field = gmsh.model.mesh.field
    # constant_meshsize_field = field.add("Constant")
    # field.setNumbers(constant_meshsize_field, "SurfacesList", [surface_geometry[1]])
    # field.setNumber(constant_meshsize_field, "IncludeBoundary", 1)
    # field.setNumber(constant_meshsize_field, "VIn", mesh_resolution)
    # field.setNumber(constant_meshsize_field, "VOut", mesh_resolution)

    # gmsh.option.setNumber("Mesh.MeshSizeMin", 3)
    # gmsh.option.setNumber("Mesh.MeshSizeMax", 3)

    # gmsh.model.mesh.setSize(gmsh.model.getEntities(0), resolution)
