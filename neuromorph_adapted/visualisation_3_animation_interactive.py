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

#source_dir = "./neuromorph_adapted/data/eg1/brainstem_test__epoch600_steps3/corrs/"
#source_dir = "M:/meshes_for_viz/corrs/"
#source_dir = "M:/meshes_for_viz/corrs_wMix/"
source_dir = "M:/meshes_for_viz/corrs_latest/" 
#source_dir = "M:/meshes_for_viz/corrs_spinal_cord/"
#source_dir = "M:/meshes_for_viz/brainstem_rigid_reg_corrs/zero_CoM_high_ARAP/"
#source_dir = "M:/meshes_for_viz/brainstem_rigid_reg_corrs/break_it/"
'''
# Interesting cases:    sequence: 7 x: ['opt_parotid_lt_0522c0457'] y: ['opt_parotid_lt_TCGA-CV-5977']  - acc. to no acc. smush
  (corrs_latest/)       sequence: 9 x: ['opt_parotid_lt_0522c0457'] y: ['opt_parotid_lt_0522c0708']     - acc. to no acc. better?
                        sequence: 11 x: ['opt_parotid_lt_TCGA-CV-5977'] y: ['opt_parotid_lt_0522c0457'] - no acc. to acc. quite well handled
                        sequence: 15 x: ['opt_parotid_lt_0522c0845'] y: ['opt_parotid_lt_0522c0669']    - large volume change
                        sequence: 27 x: ['opt_parotid_rt_TCGA-CV-A6JO'] y: ['opt_parotid_rt_0522c0427'] - forced acc? bit strange
'''
for i in range(len(getFiles(source_dir))):
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
    fname_x = seq["fname_x"][0]
    fname_y = seq["fname_y"][0]

    if fname_x == fname_y:
        continue # skip self-correspondences

    # offset
    verts_y = verts_y + np.array([0,-50,50])

    pyv_verts_x = pyvista.PolyData(verts_x)
    pyv_verts_y = pyvista.PolyData(verts_y)

    pyvista.global_theme.background = 'white'
    plotter = pyvista.Plotter(off_screen=False, notebook=False)
    
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

    # plotter.camera.position = (-0.7589108043722953, -214.16258771704162, 4.519541281732801)
    # plotter.camera.focal_point = (-6.069699287414551, 0.7501001358032227, 24.872499465942383)
    # plotter.camera.up = (0.9995088598428464, 0.022648078794458057, 0.021658800118505014)

    def run_interp():
        inter_vert_prev = verts_x.copy()
        for inter_vert in inter_verts:
            increments = 3#8                3 for 7 time steps, 8 for 3 time steps
            for pp in range(1, increments):
                time.sleep(0.1)
                inter = inter_vert_prev * (increments-pp) / increments + inter_vert * pp / increments
                plotter.update_coordinates(inter, mesh=pyv_verts_x, render=True)
            plotter.update_coordinates(inter_vert, mesh=pyv_verts_x, render=True)
            inter_vert_prev = inter_vert.copy()

    def reset():
        plotter.update_coordinates(verts_x, mesh=pyv_verts_x, render=True)

    print(f"Showing sequence: {i} x: {fname_x} y: {fname_y}")
    plotter.add_key_event("i", run_interp)
    plotter.add_key_event("r", reset)
    plotter.show(auto_close=True)