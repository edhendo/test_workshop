import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pyvista
from os.path import join
from utils.utils import getFiles
import pickle
import time

def pyvistarise(verts, triangles):
    return pyvista.PolyData(verts, np.insert(triangles, 0, 3, axis=1), deep=True, n_faces=len(triangles))

source_dir = "./neuromorph_adapted/data/eg1/brainstem_test__epoch600_steps3/corrs/"
for i in range(4,5):#25):
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
    inter_verts = np.transpose(inter_verts, (2,0,1))

    # offset
    verts_y = verts_y + np.array([0,0,50])

    pyv_verts_x = pyvista.PolyData(verts_x)
    pyv_verts_y = pyvista.PolyData(verts_y)

    pyvista.global_theme.background = 'white'
    plotter = pyvista.Plotter(off_screen=True, notebook=False)
    plotter.open_gif("test.gif")
    #plotter.add_mesh(pyv_mesh_x, show_edges=True, line_width=1, edge_color=[0,0,0,1], color="white", opacity=1.)
    #plotter.add_mesh(pyv_mesh_y, show_edges=True, line_width=1, edge_color=[0,0,0,1], color="white", opacity=1.)

    # Forward assignment
    # set colors on target mesh corresponding with xyz position
    colors_y = verts_y - np.min(verts_y, axis=0)
    colors_y = colors_y / np.max(colors_y, axis=0)
    colors_y = np.concatenate([colors_y, np.ones((len(colors_y), 1))], axis=1)
    # set colors on source mesh with the hard corrspondence assignment
    colors_x = colors_y[assignment]

    # add points
    plotter.add_points(verts_y, opacity=1., point_size=30, render_points_as_spheres=True, scalars=colors_y, rgb=True)
    plotter.add_points(pyv_verts_x, opacity=1., point_size=30, render_points_as_spheres=True, scalars=colors_x, rgb=True)

    if True:
        plotter.camera.position = (-0.7589108043722953, -214.16258771704162, 4.519541281732801)
        plotter.camera.focal_point = (-6.069699287414551, 0.7501001358032227, 24.872499465942383)
        plotter.camera.up = (0.9995088598428464, 0.022648078794458057, 0.021658800118505014)
    if True:
        plotter.camera.position = (53.82657717981272, 53.24561569745179, -175.8426412000512)
        plotter.camera.focal_point = (-6.069699287414551, 0.7501001358032227, 24.872499465942383)
        plotter.camera.up = (0.9605782283268545, -0.051265353454879696, 0.27324225661413987)
    
    for _ in range(5):
        plotter.write_frame()
    
    inter_vert_prev = verts_x.copy()
    for inter_vert in inter_verts:
        increments = 8
        for pp in range(1, increments):
            inter = inter_vert_prev * (increments-pp) / increments + inter_vert * pp / increments
            plotter.update_coordinates(inter, mesh=pyv_verts_x, render=True)
            plotter.write_frame()
        plotter.update_coordinates(inter_vert, mesh=pyv_verts_x, render=True)
        plotter.write_frame()
        inter_vert_prev = inter_vert.copy()

    for _ in range(5):
        plotter.write_frame()

    plotter.close()