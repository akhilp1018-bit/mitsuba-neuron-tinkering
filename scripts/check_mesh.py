import trimesh
import glob
import os

files = glob.glob("neuron/*.ply")
if not files:
    files = glob.glob("../neuron/*.ply")

for f in sorted(files):
    mesh = trimesh.load(f, force="mesh")
    print(os.path.basename(f))
    print("  watertight:", mesh.is_watertight)
    print("  euler:", mesh.euler_number)
    print("  vertices:", len(mesh.vertices))
    print("  faces:", len(mesh.faces))
    print()
    
#inorder to repair mesh
"""mesh.remove_unreferenced_vertices()
mesh.remove_duplicate_faces()
mesh.remove_degenerate_faces()

trimesh.repair.fix_normals(mesh)
trimesh.repair.fill_holes(mesh)

print("watertight after repair:", mesh.is_watertight)

mesh.export("neuron/h01_mesh_repaired.ply")"""