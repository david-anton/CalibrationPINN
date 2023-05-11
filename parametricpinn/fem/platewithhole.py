import dolfinx
from dataclasses import dataclass
import gmsh
import sys
from typing import Optional, Any, Callable, Union, TypeAlias
from parametricpinn.io import ProjectDirectory
from parametricpinn.settings import Settings
from dolfinx.io.gmshio import model_to_mesh
from dolfinx import fem
from dolfinx.fem import (
    Function,
    FunctionSpace,
    TensorFunctionSpace,
    VectorFunctionSpace,
    locate_dofs_topological,
    Constant,
    dirichletbc,
    DirichletBCMetaClass,
)
from dolfinx.mesh import Mesh, locate_entities, meshtags
from dolfinx.io import XDMFFile
import ufl
from ufl import (
    VectorElement,
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
import numpy.typing as npt
from parametricpinn.types import NPArray
import petsc4py
from petsc4py.PETSc import ScalarType
from dolfinx.fem.petsc import LinearProblem


GMesh: TypeAlias = gmsh.model
GOutDimAndTags: TypeAlias = list[tuple[int, int]]
GOutDimAndTagsMap: TypeAlias = list[Union[GOutDimAndTags, list[Any]]]
GGeometry: TypeAlias = tuple[GOutDimAndTags, GOutDimAndTagsMap]
DMesh: TypeAlias = dolfinx.mesh.Mesh
DFunction: TypeAlias = dolfinx.fem.Function
DFunctionSpace: TypeAlias = dolfinx.fem.FunctionSpace
DTestFunction: TypeAlias = ufl.TestFunction
DConstant: TypeAlias = dolfinx.fem.Constant
DDofs: TypeAlias = npt.NDArray[np.int32]
DMeshTags: TypeAlias = Any # dolfinx.mesh.MeshTags
DDirichletBC: TypeAlias = dolfinx.fem.DirichletBCMetaClass
UFLOperator: TypeAlias = ufl.core.operator.Operator
UFLMeasure: TypeAlias = ufl.Measure
UFLSigmaFunc: TypeAlias = Callable[[TrialFunction], UFLOperator]
UFLEpsilonFunc: TypeAlias = Callable[[TrialFunction], UFLOperator]
PETScScalarType: TypeAlias = petsc4py.PETSc.ScalarType


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
BCValue: TypeAlias = Union[DConstant, PETScScalarType]

geometric_dim = 2


def run_one_simulation(
    config: Config, output_subdir: str, project_directory: ProjectDirectory
) -> None:
    mesh = _generate_mesh(config, output_subdir, project_directory)
    _simulate_once(mesh, config, output_subdir, project_directory)


def _generate_mesh(
    config: Config,
    output_subdir: str,
    project_directory: ProjectDirectory,
    save_mesh: bool = True,
) -> DMesh:
    gmsh.initialize()
    gmesh = _generate_gmesh(config)
    if save_mesh:
        _save_gmesh(output_subdir, project_directory)
    mesh = _load_mesh_from_gmsh_model(gmesh)
    gmsh.finalize()
    return mesh


def _generate_gmesh(config: Config) -> GMesh:
    length = config.edge_length
    radius = config.radius
    resolution = config.resolution
    geometry_kernel = gmsh.model.occ
    solid_marker = 1

    def create_geometry() -> GGeometry:
        gmsh.model.add("domain")
        plate = geometry_kernel.add_rectangle(0, 0, 0, -length, length)
        hole = geometry_kernel.add_disk(0, 0, 0, radius, radius)
        return geometry_kernel.cut([(2, plate)], [(2, hole)])

    def tag_physical_enteties(geometry: GGeometry) -> None:
        geometry_kernel.synchronize()

        def tag_solid_surface() -> None:
            surface = geometry_kernel.getEntities(dim=2)
            assert surface == geometry[0]
            gmsh.model.addPhysicalGroup(surface[0][0], [surface[0][1]], solid_marker)
            gmsh.model.setPhysicalName(surface[0][0], solid_marker, "Solid")

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

    geometry = create_geometry()
    tag_physical_enteties(geometry)
    configure_mesh()
    generate_mesh()

    return gmsh.model


def _save_gmesh(output_subdir: str, project_directory: ProjectDirectory) -> None:
    output_path = project_directory.create_output_file_path(
        file_name="mesh.msh", subdir_name=output_subdir
    )
    gmsh.write(str(output_path))


def _load_mesh_from_gmsh_model(gmesh: GMesh) -> Mesh:
    mpi_rank = 0
    mesh, cell_tags, facet_tags = model_to_mesh(
        gmesh, MPI.COMM_WORLD, mpi_rank, gdim=geometric_dim
    )
    return mesh


class NeumannBC:
    def __init__(
        self, tag: int, value: BCValue, measure: UFLMeasure, test_func: DTestFunction
    ) -> None:
        self.bc = inner(value, test_func) * measure(tag)


class DirichletBC:
    def __init__(
        self, dofs: DDofs, value: BCValue, dim: int, func_space: DFunctionSpace
    ) -> None:
        self.bc = dirichletbc(value, dofs, func_space.sub(dim))


BoundaryConditions: TypeAlias = list[Union[DirichletBC, NeumannBC]]


def _simulate_once(
    mesh: Mesh,
    config: Config,
    output_subdir: str,
    project_directory: ProjectDirectory,
    save_meta_data: bool = False,
) -> None:
    model = config.model
    youngs_modulus = config.youngs_modulus
    poissons_ratio = config.poissons_ratio
    length = config.edge_length
    radius = config.radius
    volume_force_x = config.volume_force_x
    volume_force_y = config.volume_force_y
    traction_left_x = config.traction_left_x
    traction_left_y = config.traction_left_y
    traction_top_x = traction_top_y = 0.0
    traction_hole_x = traction_hole_y = 0.0
    element_family = config.element_family
    element_degree = config.element_degree

    T_top = Constant(mesh, (ScalarType((traction_top_x, traction_top_y))))
    T_hole = Constant(mesh, (ScalarType((traction_hole_x, traction_hole_y))))
    T_left = Constant(mesh, (ScalarType((traction_left_x, traction_left_y))))
    u_x_right = ScalarType(0.0)
    u_y_bottom = ScalarType(0.0)
    f = Constant(mesh, (ScalarType((volume_force_x, volume_force_y))))
    bc_facets_dim = mesh.topology.dim - 1

    tag_top = 0
    tag_right = 1
    tag_hole = 2
    tag_bottom = 3
    tag_left = 4
    locate_top_facet = lambda x: np.isclose(x[1], length)
    locate_right_facet = lambda x: np.isclose(x[0], 0.0)
    locate_hole_facet = lambda x: np.isclose(
        np.sqrt(np.square(x[0]) + np.square(x[1])), radius
    )
    locate_bottom_facet = lambda x: np.isclose(x[1], 0.0)
    locate_left_facet = lambda x: np.isclose(x[0], -length)

    element = VectorElement(element_family, mesh.ufl_cell(), element_degree)
    func_space = FunctionSpace(mesh, element)
    x = SpatialCoordinate(mesh)
    u = TrialFunction(func_space)
    w = TestFunction(func_space)

    def sigma_and_epsilon_factory() -> tuple[UFLSigmaFunc, UFLEpsilonFunc]:
        compliance_matrix = None
        if model == "plane stress":
            compliance_matrix = (1 / youngs_modulus) * as_matrix(
                [
                    [1.0, -poissons_ratio, 0.0],
                    [-poissons_ratio, 1.0, 0.0],
                    [0.0, 0.0, 2 * (1.0 + poissons_ratio)],
                ]
            )
        elif model == "plane strain":
            compliance_matrix = (1 / youngs_modulus) * as_matrix(
                [
                    [
                        1.0 - poissons_ratio**2,
                        -poissons_ratio * (1.0 + poissons_ratio),
                        0.0,
                    ],
                    [
                        -poissons_ratio * (1.0 + poissons_ratio),
                        1.0 - poissons_ratio**2,
                        0.0,
                    ],
                    [0.0, 0.0, 2 * (1.0 + poissons_ratio)],
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

    def tag_boundaries() -> DMeshTags:
        boundaries = [
            (tag_top, locate_top_facet),
            (tag_right, locate_right_facet),
            (tag_hole, locate_hole_facet),
            (tag_bottom, locate_bottom_facet),
            (tag_left, locate_left_facet),
        ]

        facet_indices_list: list[npt.NDArray[np.int32]] = []
        facet_tags_list: list[npt.NDArray[np.int32]] = []
        for tag, locator_func in boundaries:
            _facet_indices = locate_entities(mesh, bc_facets_dim, locator_func)
            facet_indices_list.append(_facet_indices)
            facet_tags_list.append(np.full_like(_facet_indices, tag))
        facet_indices = np.hstack(facet_indices_list).astype(np.int32)
        facet_tags = np.hstack(facet_tags_list).astype(np.int32)
        sorted_facet_indices = np.argsort(facet_indices)
        return meshtags(
            mesh,
            bc_facets_dim,
            facet_indices[sorted_facet_indices],
            facet_tags[sorted_facet_indices],
        )

    def save_boundary_tags(boundary_tags: DMeshTags) -> None:
        output_path = project_directory.create_output_file_path(
            file_name="boundary_tags.xdmf", subdir_name=output_subdir
        )
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        with XDMFFile(mesh.comm, output_path, "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_meshtags(boundary_tags)

    def define_boundary_conditions(
        boundary_tags: DMeshTags,
    ) -> BoundaryConditions:
        facet_right = boundary_tags.find(tag_right)
        dofs_right = locate_dofs_topological(func_space, bc_facets_dim, facet_right)
        facet_bottom = boundary_tags.find(tag_bottom)
        dofs_bottom = locate_dofs_topological(func_space, bc_facets_dim, facet_bottom)

        return [
            NeumannBC(tag=tag_top, value=T_top, measure=ds, test_func=w),
            DirichletBC(dofs=dofs_right, value=u_x_right, dim=0, func_space=func_space),
            NeumannBC(tag=tag_hole, value=T_hole, measure=ds, test_func=w),
            DirichletBC(
                dofs=dofs_bottom, value=u_y_bottom, dim=1, func_space=func_space
            ),
            NeumannBC(tag=tag_left, value=T_left, measure=ds, test_func=w),
        ]

    def apply_boundary_conditions(
        boundary_conditions: BoundaryConditions, F: UFLOperator
    ) -> tuple[list[DDirichletBC], UFLOperator]:
        dirichlet_bcs = []
        for condition in boundary_conditions:
            if isinstance(condition, DirichletBC):
                dirichlet_bcs.append(condition.bc)
            else:
                F += condition.bc
        return dirichlet_bcs, F

    boundary_tags = tag_boundaries()
    if save_meta_data:
        save_boundary_tags(boundary_tags)

    ds = Measure("ds", domain=mesh, subdomain_data=boundary_tags)

    boundary_conditions = define_boundary_conditions(boundary_tags)
    sigma, epsilon = sigma_and_epsilon_factory()

    F = inner(sigma(u), epsilon(w)) * dx - inner(w, f) * dx

    dirichlet_bcs, F = apply_boundary_conditions(boundary_conditions, F)

    a = lhs(F)
    L = rhs(F)
    problem = LinearProblem(
        a, L, bcs=dirichlet_bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    uh = problem.solve()

    def conpute_strain(uh: DFunction) -> DFunction:
        _, epsilon = sigma_and_epsilon_factory()
        tensor_space = TensorFunctionSpace(mesh, element_family, element_degree)
        strain = epsilon(uh)

    @dataclass
    class SimulationResults:
        coordinates_x: NPArray
        coordinates_y: NPArray
        displacements_x: NPArray
        displacements_y: NPArray
        max_strain: float

    @dataclass 
    class SimulationConfiguration:
        youngs_modulus: float
        poissons_ratio: float
        element_family: str
        element_degree: int
        mesh_resolution: float


    def compile_output(uh: DFunction) -> tuple[SimulationResults, SimulationConfiguration]:
        coordinates = mesh.geometry.x
        coordinates_x = coordinates[:, 0]
        coordinates_y = coordinates[:, 1]

        displacements = uh.x.array.reshape((-1, mesh.geometry.dim))
        displacements_x = displacements[:, 0]
        displacements_y = displacements[:, 1]

        print(f"x-coordinates: {coordinates_x.size}")
        print(f"y-coordinates: {coordinates_y.size}")
        print(f"x-displacements: {displacements_x.size}")
        print(f"y-displacements: {displacements_y.size}")

    compile_output(uh)


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
