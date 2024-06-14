import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pyvista
from os.path import join
from utils.utils import getFiles
import pickle

def pyvistarise(verts, triangles):
    return pyvista.PolyData(verts, np.insert(triangles, 0, 3, axis=1), deep=True, n_faces=len(triangles))

source_dir = "./neuromorph_adapted/data/eg1/brainstem_test__epoch600_steps3/corrs/"
for i in range(2):#25):
    if i % 5 == i // 5:
        continue # skip self-correspondences

    with open(join(source_dir, f"seq_{i}.pkl"), "rb") as f:
        seq = pickle.load(f)

    verts_x = seq["X"]["verts"]
    verts_y = seq["Y"]["verts"]
    triangles_x = seq["X"]["triangles"]
    triangles_y = seq["Y"]["triangles"]
    inter_verts = seq["inter_verts"]
    assignment = seq["assignment"]
    assignmentinv = seq["assignmentinv"]

    # offset
    verts_y = verts_y + np.array([0,0,50])

    pyv_mesh_x = pyvistarise(verts_x, triangles_x)
    pyv_mesh_y = pyvistarise(verts_y, triangles_y)

    pyvista.global_theme.background = 'white'
    plotter = pyvista.Plotter()

    #plotter.add_mesh(pyv_mesh_x, show_edges=True, line_width=1, edge_color=[0,0,0,1], color="white", opacity=1.)
    #plotter.add_mesh(pyv_mesh_y, show_edges=True, line_width=1, edge_color=[0,0,0,1], color="white", opacity=1.)

    # Forward assignment
    # set colors on target mesh corresponding with xyz position
    colors_y = verts_y - np.min(verts_y, axis=0)
    colors_y = colors_y / np.max(colors_y, axis=0)
    colors_y = np.concatenate([colors_y, np.ones((len(colors_y), 1))], axis=1)
    # set colors on source mesh with the hard corrspondence assignment
    colors_x = colors_y[assignment]

    # Inverse assignment
    # # set colors on target mesh corresponding with xyz position
    # colors_x = verts_x - np.min(verts_x, axis=0)
    # colors_x = colors_x / np.max(colors_x, axis=0)
    # colors_x = np.concatenate([colors_x, np.ones((len(colors_x), 1))], axis=1)
    # # set colors on source mesh with the hard corrspondence assignment
    # colors_y = colors_x[assignmentinv]

    # add points
    plotter.add_points(verts_y, opacity=1., point_size=30, render_points_as_spheres=True, scalars=colors_y, rgb=True)
    plotter.add_points(verts_x, opacity=1., point_size=30, render_points_as_spheres=True, scalars=colors_x, rgb=True)

    plotter.store_image = True
    plotter.camera.position = (-0.7589108043722953, -214.16258771704162, 4.519541281732801)
    plotter.camera.focal_point = (-6.069699287414551, 0.7501001358032227, 24.872499465942383)
    plotter.camera.up = (0.9995088598428464, 0.022648078794458057, 0.021658800118505014)
    camera_pos = plotter.show(screenshot="./temp.png", window_size=[1600, 1000], auto_close=True, return_cpos=True)
    #print(camera_pos)