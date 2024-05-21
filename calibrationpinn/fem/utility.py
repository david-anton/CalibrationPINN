import numpy as np
from dolfinx import geometry

from calibrationpinn.fem.base import DFunction
from calibrationpinn.types import NPArray


def evaluate_function(function: DFunction, points: NPArray) -> NPArray:
    # See: https://jsdokken.com/dolfinx-tutorial/chapter1/membrane_code.html#making-curve-plots-throughout-the-domain
    mesh = function.function_space.mesh
    bounding_box_tree = geometry.bb_tree(mesh, mesh.topology.dim)

    cells = []
    points_on_process = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions_points(bounding_box_tree, points)
    # Frome the candidates, choose the cells that actually contains the point
    colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, points)
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_process.append(point)
            cells.append(colliding_cells.links(i)[0])
    points_on_process = np.array(points_on_process, dtype=np.float64)

    return function.eval(points_on_process, cells)
